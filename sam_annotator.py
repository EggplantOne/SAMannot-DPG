import cv2
import torch
import time
import numpy as np
import os
from label import Label
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import re
import shutil
class SAM_Annotator:
    def __init__(self, model_type="vit_b", model_cfg_path = None, ckpt_path=None):
        self.model_loaded = False
        self.model_loading = False
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model_cfg_path = model_cfg_path
        self.labels = []
        self.object_id_to_group_id = {}
        self.blocking_frames = []
        self.current_block = 0
        self.media_files = []
        self.sam_extract_dir = "sam_temp_dir"  # 默认值，由 Annotator 同步覆盖
        self.propagation_blocks = {}
        self.loading_stages = [
            "Loading model weights",
            "Setup complete"
        ]
        self.current_stage = 0
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.autocast_dtype = torch.float16
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:  # Ampere+
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = torch.device('xpu')
            self.autocast_dtype = torch.bfloat16  # XPU has poor float16 precision
            print(f'Using Intel XPU: {torch.xpu.get_device_name(0)}')
        else:
            self.device = torch.device('cpu')
            self.autocast_dtype = torch.float32
            print('Warning: no GPU found, SAM2 will be slow')
        self.curr_img_idx = -1
        self.inference_state = None
        self.tracking_init = False
        self.object_ids = {}
        self.frame_names = None
    def load_model(self, status_callback):
        if self.model_loaded or self.model_loading:
            return True
        self.model_loading = True
        try:
            start_time = time.time()
            self._update_loading_stage(status_callback)
            self.predictor = build_sam2_video_predictor(self.model_cfg_path, self.ckpt_path).to(device=self.device)
            self.image_predictor = SAM2ImagePredictor(self.predictor)
            self._update_loading_stage(status_callback)
            self.model_loaded = True
            self.model_loading = False
            load_time = time.time() - start_time
            if status_callback:
                status_callback(f"SAM2 loaded successfully in {load_time:.1f}s (using {self.device})")
            return True
        except Exception as e:
            if status_callback:
                status_callback(f"Error loading SAM2: {str(e)}")
            self.model_loading = False
            return False
    def _update_loading_stage(self, status_callback):
        if self.current_stage < len(self.loading_stages):
            stage_text = self.loading_stages[self.current_stage]
            if status_callback:
                progress = f"[{self.current_stage+1}/{len(self.loading_stages)}]"
                status_callback(f"Loading SAM2 model {progress} {stage_text}")
            self.current_stage += 1
    def is_model_loaded(self):
        return self.model_loaded
    def _find_most_recent_prompt(self,label,idx):
        most_recent_prompt = -1
        for pt in label.pts.get(self.current_block, []):
            if pt.idx <= idx:
                most_recent_prompt = np.amax([most_recent_prompt,pt.idx])
        return most_recent_prompt
    def _find_most_recent_prompt_box(self,label,idx):
        most_recent_prompt = -1
        for box in label.boxes.get(self.current_block, []):
            if box.idx <= idx:
                most_recent_prompt = np.amax([most_recent_prompt,box.idx])
        return most_recent_prompt
    def _preprocess_label(self,label,idx,flag=0):
        pt_coords = []
        pt_labels = []
        boxes = []
        most_recent_prompt = -1
        most_recent_prompt_box = -1
        if flag == 1:
            most_recent_prompt = self._find_most_recent_prompt(label,idx)
        for pt in label.pts.get(self.current_block, []):
            if flag == 0 and pt.idx == self.curr_img_idx:
                pt_coords.append([pt.x, pt.y])
                pt_labels.append(pt.pt_type)
            if flag == 1 and pt.idx == most_recent_prompt:
                pt_coords.append([pt.x, pt.y])
                pt_labels.append(pt.pt_type)
        if flag == 1:
            most_recent_prompt_box = self._find_most_recent_prompt_box(label,idx)
        for box in label.boxes.get(self.current_block, []):
            if flag == 0 and box.idx == self.curr_img_idx:
                boxes.append([box.fx,box.fy,box.x,box.y])
            if flag == 1 and box.idx== most_recent_prompt_box:
                boxes.append([box.fx,box.fy,box.x,box.y])
            if len(boxes) > 0:
                boxes = [boxes[0]]
        pt_coords = np.array(pt_coords)
        pt_labels = np.array(pt_labels)
        boxes = np.array(boxes)
        return pt_coords, pt_labels, boxes, most_recent_prompt, most_recent_prompt_box
    def _get_object_id(self, group_id):
        key = (group_id, self.current_block)
        if key not in self.object_ids:
            self.object_ids[key] = len(self.object_ids) + 1
            self.object_id_to_group_id[self.object_ids[key]] = group_id
        return self.object_ids[key]
    def init_inference_state(self, path, status_callback=None):
        if not self.model_loaded:
            return False
        try:
            self.inference_state = self.predictor.init_state(video_path=path)
            self.tracking_init = True
            if status_callback:
                status_callback(f"SAM2 tracking initialized successfully!")
            return True
        except Exception as e:
            if status_callback:
                status_callback(f"Error initializing SAM2 tracking: {str(e)}")
            return False
    def _find_nearest_prompt_frame(self, frame_idx):
        """Find the nearest prompt frame <= frame_idx. Returns None if no prompt exists."""
        all_prompt_frames = set()
        for label in self.labels:
            for pt in label.pts.get(self.current_block, []):
                all_prompt_frames.add(pt.idx)
            for box in label.boxes.get(self.current_block, []):
                all_prompt_frames.add(box.idx)
        candidates = [f for f in all_prompt_frames if f <= frame_idx]
        return max(candidates) if candidates else None

    @torch.inference_mode()
    def generate_mask_for_frame(self, idx, flag=0):
        """Single frame mask generation using SAM2ImagePredictor."""
        if not self.model_loaded:
            print("[SAM] model not loaded")
            return {}
        try:
            current_path = self.media_files[idx]
            image = cv2.imread(current_path)
            if image is None:
                print(f"[SAM] cannot read image: {current_path}")
                return {}
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with torch.autocast(self.device.type, dtype=self.autocast_dtype):
                self.image_predictor.set_image(image_rgb)
                results = {}
                frame_results = {}
                for label in self.labels:
                    pt_coords, pt_labels, boxes, _, _ = self._preprocess_label(label, idx, flag)
                    has_pts = pt_coords.shape[0] != 0
                    has_boxes = boxes.shape[0] != 0
                    if not has_pts and not has_boxes:
                        continue
                    masks, scores, _ = self.image_predictor.predict(
                        point_coords=pt_coords if has_pts else None,
                        point_labels=pt_labels if has_pts else None,
                        box=boxes[0] if has_boxes else None,
                        multimask_output=False,
                    )
                    frame_results[label.group_id] = masks[:1]
                if not frame_results:
                    return {}
                results[idx] = frame_results
                return results
        except Exception as e:
            print(f"Error generating mask: {str(e)}")
            import traceback; traceback.print_exc()
            return {}
    @torch.inference_mode()
    def propagate(self, direction, start_frame_idx, end_frame_idx = None, progress_callback = None, flag = 1):
        if not self.tracking_init:
            print("Tracking not initialized!")
            return {}
        try:
            self.predictor.reset_state(self.inference_state)
            with torch.autocast(self.device.type, dtype=self.autocast_dtype):
                if direction == -1:
                    # backward: only use prompts from the nearest prompt frame (<= start_frame_idx)
                    prompt_frame = self._find_nearest_prompt_frame(start_frame_idx)
                    if prompt_frame is None:
                        return {}
                    for label in self.labels:
                        obj_id = self._get_object_id(label.group_id)
                        pts = []
                        lbls = []
                        for pt in label.pts.get(self.current_block, []):
                            if pt.idx == prompt_frame:
                                pts.append([pt.x, pt.y])
                                lbls.append(pt.pt_type)
                        if pts:
                            self.predictor.add_new_points_or_box(
                                inference_state=self.inference_state,
                                frame_idx=prompt_frame,
                                obj_id=obj_id,
                                points=np.array(pts),
                                labels=np.array(lbls),
                            )
                        for box in label.boxes.get(self.current_block, []):
                            if box.idx == prompt_frame:
                                self.predictor.add_new_points_or_box(
                                    inference_state=self.inference_state,
                                    frame_idx=prompt_frame,
                                    obj_id=obj_id,
                                    box=np.array([box.fx, box.fy, box.x, box.y]),
                                )
                                break
                else:
                    # forward: compute propagation range first, then only add prompts within range
                    prop_end = len(self.media_files) if end_frame_idx is None else end_frame_idx
                    max_efi = 1000000
                    prop_extra_frame = True
                    for k,v in self.propagation_blocks[self.current_block].items():
                        if k > start_frame_idx and k < max_efi:
                            max_efi = k
                            prop_extra_frame = False
                    prop_end = int(np.amin([prop_end, max_efi]))

                    for label in self.labels:
                        obj_id = self._get_object_id(label.group_id)
                        prompt_frames = set()
                        for pt in label.pts.get(self.current_block, []):
                            if start_frame_idx <= pt.idx <= prop_end:
                                prompt_frames.add(pt.idx)
                        for box in label.boxes.get(self.current_block, []):
                            if start_frame_idx <= box.idx <= prop_end:
                                prompt_frames.add(box.idx)
                        for f_idx in sorted(prompt_frames):
                            pts = []
                            lbls = []
                            for pt in label.pts.get(self.current_block, []):
                                if pt.idx == f_idx:
                                    pts.append([pt.x, pt.y])
                                    lbls.append(pt.pt_type)
                            if pts:
                                self.predictor.add_new_points_or_box(
                                    inference_state=self.inference_state,
                                    frame_idx=f_idx,
                                    obj_id=obj_id,
                                    points=np.array(pts),
                                    labels=np.array(lbls),
                                )
                            for box in label.boxes.get(self.current_block, []):
                                if box.idx == f_idx:
                                    self.predictor.add_new_points_or_box(
                                        inference_state=self.inference_state,
                                        frame_idx=f_idx,
                                        obj_id=obj_id,
                                        box=np.array([box.fx, box.fy, box.x, box.y]),
                                    )
                                    break
            results = {}
            if direction == 1:
                if end_frame_idx is None:
                    end_frame_idx = len(self.media_files)
                if not prop_extra_frame:
                    max_efi_adj = max_efi
                else:
                    max_efi_adj = 1000000
                end_frame_idx = int(np.amin([end_frame_idx, max_efi_adj]))
                if not prop_extra_frame:
                    end_frame_idx += 1  # include the checkpoint frame itself
                else:
                    end_frame_idx += 1  # include last frame when no checkpoint
                total_prop_frames = end_frame_idx - start_frame_idx - 1
            else:
                if end_frame_idx is None:
                    end_frame_idx = 0
                max_efi = -1000000
                for k,v in self.propagation_blocks[self.current_block].items():
                    if k < start_frame_idx and k > max_efi:
                        max_efi = k
                end_frame_idx = np.amax([end_frame_idx,max_efi])  # include checkpoint frame
                if end_frame_idx is not None:
                    total_prop_frames = start_frame_idx - end_frame_idx
                else:
                    total_prop_frames = start_frame_idx
            with torch.autocast(self.device.type, dtype=self.autocast_dtype):
                for i, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(
                        self.predictor.propagate_in_video(self.inference_state,start_frame_idx=start_frame_idx,max_frame_num_to_track=total_prop_frames,reverse=(direction != 1))):
                    masks = (out_mask_logits > 0.0).cpu().numpy()
                    frame_results = {}
                    for j, obj_id in enumerate(out_obj_ids):
                        frame_results[self.object_id_to_group_id[obj_id]] = masks[j]
                    results[out_frame_idx] = frame_results
                    if progress_callback:
                        progress = (i / total_prop_frames) * 100
                        progress_callback(f"Processing data...", progress)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.predictor.reset_state(self.inference_state)
            return results
        except Exception as e:
            print(f"Error propagating masks: {str(e)}")
            if progress_callback:
                progress_callback(f"Error propagating masks: {str(e)}", 0)
            return {}
    @torch.inference_mode()
    def propagate_all_prompts(self, progress_callback=None):
        """Per-label independent propagation: each label gets its own clean inference state,
        propagated only within its own prompt range, then merged."""
        if not self.tracking_init:
            print("Tracking not initialized!")
            return {}
        try:
            total_frames = len(self.media_files)
            # Collect prompts per label
            label_prompts = {}
            gid_to_name = {}
            for label in self.labels:
                frames_data = {}
                for pt in label.pts.get(self.current_block, []):
                    if pt.idx not in frames_data:
                        frames_data[pt.idx] = {"pts": [], "labels": [], "boxes": []}
                    frames_data[pt.idx]["pts"].append([pt.x, pt.y])
                    frames_data[pt.idx]["labels"].append(pt.pt_type)
                for box in label.boxes.get(self.current_block, []):
                    if box.idx not in frames_data:
                        frames_data[box.idx] = {"pts": [], "labels": [], "boxes": []}
                    frames_data[box.idx]["boxes"].append([box.fx, box.fy, box.x, box.y])
                if frames_data:
                    label_prompts[label.group_id] = frames_data
                    gid_to_name[label.group_id] = label.name
            if not label_prompts:
                return {}
            # Per-label propagation, store binary masks + confidence score (saves memory)
            label_masks = {}   # {group_id: {frame_idx: bool_ndarray}}
            label_scores = {}  # {group_id: {frame_idx: float}} for merge priority
            num_labels = len(label_prompts)
            for li, (group_id, frames_data) in enumerate(label_prompts.items()):
                self.predictor.reset_state(self.inference_state)
                obj_id = self._get_object_id(group_id)
                label_name = gid_to_name[group_id]
                # Add prompts — single call per frame to avoid clear_old_points overwrite
                with torch.autocast(self.device.type, dtype=self.autocast_dtype):
                    for frame_idx in sorted(frames_data.keys()):
                        data = frames_data[frame_idx]
                        pts = np.array(data["pts"]) if data["pts"] else None
                        lbls = np.array(data["labels"]) if data["labels"] else None
                        bx = np.array(data["boxes"][0]) if data["boxes"] else None
                        self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            points=pts, labels=lbls, box=bx
                        )
                # Propagate
                min_frame = min(frames_data.keys())
                this_masks = {}
                this_scores = {}
                with torch.autocast(self.device.type, dtype=self.autocast_dtype):
                    for i, (out_frame_idx, _, out_mask_logits) in enumerate(
                            self.predictor.propagate_in_video(self.inference_state, start_frame_idx=min_frame, max_frame_num_to_track=total_frames - min_frame, reverse=False)):
                        logit = out_mask_logits[0]
                        this_masks[out_frame_idx] = (logit > 0.0).cpu().numpy()
                        this_scores[out_frame_idx] = logit.max().item()
                        if progress_callback:
                            progress = ((li * 2 * total_frames + i) / (num_labels * 2 * total_frames)) * 100
                            progress_callback(f"Propagating {label_name} (forward)...", progress)
                if min_frame > 0:
                    with torch.autocast(self.device.type, dtype=self.autocast_dtype):
                        for i, (out_frame_idx, _, out_mask_logits) in enumerate(
                                self.predictor.propagate_in_video(self.inference_state, start_frame_idx=min_frame, max_frame_num_to_track=min_frame, reverse=True)):
                            if out_frame_idx not in this_masks:
                                logit = out_mask_logits[0]
                                this_masks[out_frame_idx] = (logit > 0.0).cpu().numpy()
                                this_scores[out_frame_idx] = logit.max().item()
                            if progress_callback:
                                progress = ((li * 2 * total_frames + total_frames + i) / (num_labels * 2 * total_frames)) * 100
                                progress_callback(f"Propagating {label_name} (backward)...", progress)
                label_masks[group_id] = this_masks
                label_scores[group_id] = this_scores
            # Merge: higher confidence label gets priority at overlapping pixels
            all_frames = set()
            for lm in label_masks.values():
                all_frames.update(lm.keys())
            results = {}
            group_ids = list(label_masks.keys())
            for frame_idx in sorted(all_frames):
                # Sort labels by confidence at this frame (highest first)
                scored = [(label_scores[gid].get(frame_idx, -10.0), gid) for gid in group_ids if frame_idx in label_masks[gid]]
                scored.sort(reverse=True)
                frame_results = {}
                occupied = None
                for _, gid in scored:
                    mask = label_masks[gid][frame_idx]
                    if occupied is not None:
                        mask = mask & ~occupied
                    frame_results[gid] = mask
                    if occupied is None:
                        occupied = mask.copy()
                    else:
                        occupied = occupied | mask
                results[frame_idx] = frame_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.predictor.reset_state(self.inference_state)
            return results
        except Exception as e:
            print(f"Error in propagate_all_prompts: {str(e)}")
            import traceback
            traceback.print_exc()
            if progress_callback:
                progress_callback(f"Error: {str(e)}", 0)
            return {}
        finally:
            self.predictor.reset_state(self.inference_state)
    def propagate_to_all(self, current_frame_idx, start_frame_idx = None, end_frame_idx = None, progress_callback = None, flag = 1):
        forward_prop_results = self.propagate(1,current_frame_idx,end_frame_idx,progress_callback,flag)
        backward_prop_results = self.propagate(-1,current_frame_idx,start_frame_idx,progress_callback,flag)
        return forward_prop_results | backward_prop_results