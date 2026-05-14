import os
from PIL import Image, ImageDraw, ImageFont
import cv2  
import time
import numpy as np
from label import Label_Handler
from sam_annotator import SAM_Annotator
# from openpyxl import Workbook  # 不再需要（export_*_data 方法已删除）
import shutil
import re
import json
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
import gc
class Annotator:
    def __init__(self):
        self.base_checkpoint_path = "./checkpoints/"
        self.init_sam2()

        self.media_files = []
        self.curr_img_idx = -1
        self.current_block = 0
        self.idx_to_path = {}
        self.video_name = ""
        self.media_path = "."
        
        self.overlay_img = None
        self.composite_mask = None
        self.masks = {}
        self.overlay_imgs = {}
        self.combined_masks = {}
        self.cache_images = True
        self.current_img_width = 0
        self.current_img_height = 0

        self.label_handler = Label_Handler()
        self.curr_label_idx = -1
        self.image_file_types=["jpg","JPG","jpeg","JPEG","png"]
        self.video_file_types=["mp4","avi","mov","mkv"]

        self.mode = "prompts"
        self.fn_modes = ["prompts","tracking","correction","export"]
        self.view_mode = "prompts"
        self.view_modes = ["original","prompts","overlay","masks"]
        self.session_name = "Session"
        
        self.tracking_results = {}
        self.extra_frame_path = {}
        self.extra_frame = {}
        self.extra_frame_masks = {}
        self.extracted_frames = []
        self.num_blocks = 0
        self.block_size = 0
        self.curr_img_shape = (0, 0, 0)
        self.cap = None
        self.extract_dir = ""

        self.read_frames = -1
        self.cap = None
    # SESSION HANDLING
    
    def reset(self):
        self.media_files = []
        self.curr_img_idx = -1
        self.overlay_img = None
        self.composite_mask = None
        self.original_img = None
        self.overlay_imgs = {}
        self.combined_masks = {}
        self.masks = {}
        self.tracking_results = {}
        self.label_handler = Label_Handler()
        self.curr_label_idx = -1
        self.mode = "prompts"
        self.view_mode = "prompts"
        self.session_name = "Session"
        self.video_name = ""
        self.media_path = "."
        self.tracking_results = {}
        self.sam_handler.labels = []
        self.sam_handler.object_id_to_group_id = {}
        self.sam_handler.object_ids = {}
        self.sam_handler.media_files = []
        self.sam_handler.propagation_blocks = {}
        self.current_block = 0
        self.extra_frame_path = {}
        self.extra_frame = {}
        self.extra_frame_masks = {}
        if self.cap is not None:
                self.cap.release()
        self.read_frames = -1
        if self.sam_handler.inference_state:
            self.sam_handler.predictor.reset_state(self.sam_handler.inference_state)
    def load_from_dict(self,dict_representation):
        self.session_name = dict_representation["session_name"]
        self.view_mode = dict_representation["view_mode"]
        self.mode = dict_representation["mode"]
        self.media_files = dict_representation["media_files"]
        self.curr_img_idx = dict_representation["curr_img_idx"]
        self.overlay_img = dict_representation["overlay_img"]
        # 旧版 session.pkl 用文件路径当 key，新版改成绝对帧号 (int)。
        # 加载时迁移：从路径 basename 解析帧号；不能解析的项丢弃。
        self.overlay_imgs = self._migrate_path_keyed_dict(dict_representation["overlay_imgs"])
        self.combined_masks = self._migrate_path_keyed_dict(dict_representation["combined_masks"])
        self.original_img = dict_representation["original_img"]
        self.composite_mask = dict_representation["composite_mask"]
        self.masks = self._migrate_path_keyed_dict(dict_representation["masks"])
        # idx_to_path 是 cv2.imread fallback 用的路径提示。从旧 .pkl 加载时直接清空，
        # _path_for_abs_idx 会回退到 temp_dir/{abs_idx:06d}.jpg 的约定路径。
        loaded_i2p = dict_representation.get("idx_to_path", {}) or {}
        self.idx_to_path = loaded_i2p if all(isinstance(k, int) for k in loaded_i2p.keys()) else {}
        # temp_dir 已废弃（改用 self.frames_dir property），忽略旧 pkl 的值
        # self.temp_dir = dict_representation["temp_dir"]
        self.current_img_width = dict_representation["current_img_width"]
        self.current_img_height = dict_representation["current_img_height"]
        self.label_handler = dict_representation["label_handler"]
        self.curr_label_idx = dict_representation["curr_label_idx"]
        self.tracking_results = dict_representation["tracking_results"]
        self.cache_images = dict_representation["cache_images"]
        self.media_path = dict_representation["media_path"]
        self.video_name = dict_representation["video_name"]
        self.sam_handler.labels = dict_representation["sam_handler_labels"]
        # Minimal old-pkl compatibility. Reconcile Labels is responsible for
        # resolving None to the JSON-derived group_id or a newly allocated one.
        for lbl in self.sam_handler.labels:
            if not hasattr(lbl, 'group_id'):
                lbl.group_id = None
        max_gid = max(
            (lbl.group_id for lbl in self.sam_handler.labels
             if isinstance(lbl.group_id, int)),
            default=0)
        self.label_handler._next_group_id = max_gid + 1
        # rebuild object_id_to_group_id (old pkl had object_id_to_label_name)
        old_mapping = dict_representation.get("sam_handler_object_id_to_label_name",
                       dict_representation.get("sam_handler_object_id_to_group_id", {}))
        self.sam_handler.object_id_to_group_id = {}
        # if old mapping values are strings (label names), convert to group_ids
        if old_mapping:
            sample_val = next(iter(old_mapping.values()))
            if isinstance(sample_val, str):
                name_to_gid = {}
                for lbl in self.sam_handler.labels:
                    if lbl.name not in name_to_gid:
                        name_to_gid[lbl.name] = lbl.group_id
                for obj_id, name in old_mapping.items():
                    if name in name_to_gid:
                        self.sam_handler.object_id_to_group_id[obj_id] = name_to_gid[name]
            else:
                self.sam_handler.object_id_to_group_id = old_mapping
        self.sam_handler.media_files = dict_representation["sam_handler_media_files"]
        self.sam_handler.curr_img_idx = dict_representation["sam_handler_curr_img_idx"]
        self.sam_handler.current_stage = dict_representation["sam_handler_current_stage"]
        self.sam_handler.model_loaded  = dict_representation["sam_handler_model_loaded"]
        self.sam_handler.model_loading = dict_representation["sam_handler_model_loading"]
        self.sam_handler.propagation_blocks = dict_representation["sam_handler_propagation_blocks"]
        self.block_size = dict_representation["block_size"]
        self.current_block = dict_representation["current_block"]
        self.num_blocks = dict_representation["num_blocks"]
        self.curr_img_shape = dict_representation["curr_img_shape"]
        self.extra_frame_path = dict_representation["extra_frame_path"]
        self.extra_frame = dict_representation["extra_frame"]
        self.extra_frame_masks = dict_representation["extra_frame_masks"]
    def compress_to_dict(self):
        dict_representation = {
            "session_name": self.session_name,
            "view_mode": self.view_mode,
            "mode": self.mode,
            "media_files": self.media_files,
            "curr_img_idx": self.curr_img_idx,
            "overlay_img": None,           # 不存渲染缓存（可从 masks 重建）
            "overlay_imgs": {},            # 不存（省 1.5 GB）
            "combined_masks": {},          # 不存（省 1.5 GB）
            "original_img": None,          # 不存（从磁盘帧文件重新读）
            "composite_mask": None,        # 不存（可从 masks 重建）
            "masks": {},  # masks exported to JSON, not stored in pkl
            "idx_to_path": self.idx_to_path,
            "temp_dir": None,  # 废弃字段，保留 key 兼容旧版 load_from_dict
            "current_img_width": self.current_img_width,
            "current_img_height": self.current_img_height,
            "label_handler": self.label_handler,
            "curr_label_idx": self.curr_label_idx,
            "tracking_results": self.tracking_results,
            "cache_images": self.cache_images,
            "media_path": self.media_path,
            "video_name": self.video_name,
            "block_size": self.block_size,
            "current_block": self.current_block,
            "num_blocks": self.num_blocks,
            "curr_img_shape": self.curr_img_shape,
            "extra_frame_path": self.extra_frame_path,
            "extra_frame": self.extra_frame,
            "extra_frame_masks": self.extra_frame_masks,
            "sam_handler_labels": self.sam_handler.labels,
            "sam_handler_object_id_to_group_id": self.sam_handler.object_id_to_group_id,
            "sam_handler_media_files": self.sam_handler.media_files,
            "sam_handler_curr_img_idx": self.sam_handler.curr_img_idx,
            "sam_handler_current_stage": self.sam_handler.current_stage,
            "sam_handler_model_loaded": self.sam_handler.model_loaded,
            "sam_handler_model_loading": self.sam_handler.model_loading,
            "sam_handler_propagation_blocks": self.sam_handler.propagation_blocks,
        }
        return dict_representation

    # OTHERS

    def set_session_name(self,session_name):
        self.session_name = session_name
    def get_session_name(self):
        return self.session_name
    def set_cache_images(self,value):
        if value == False:
            self.overlay_imgs = {}
            self.combined_masks = {}
            gc.collect()
        self.cache_images = value
    def check_label_existence(self, label_name):
        for label in self.sam_handler.labels:
            if label.name == label_name:
                return True
        return False
    def get_current_block(self):
        return self.current_block
    def set_num_blocks(self,num_blocks):
        self.num_blocks = num_blocks
        for i in range(self.num_blocks):
            self.sam_handler.propagation_blocks[i] = {}
    def set_current_block(self,block):
        if block < 0:
            block = 0
        self.current_block = block
        self.sam_handler.current_block = self.current_block
    def has_labels(self):
        return len(self.sam_handler.labels) > 0
    def has_frames(self):
        return len(self.media_files) > 0

    # ===== 项目目录属性 =====
    # 所有产物统一在 projects/{video_name}/ 下
    @property
    def project_dir(self):
        if self.video_name:
            name = os.path.splitext(os.path.basename(self.video_name))[0]
        else:
            name = self.session_name or "default"
        return os.path.join("projects", name)

    @property
    def frames_dir(self):
        return os.path.join(self.project_dir, "frames")

    @property
    def sam_extract_dir(self):
        return os.path.join(self.project_dir, ".sam_temp")

    def _migrate_path_keyed_dict(self, d):
        """兼容旧 session.pkl：把字符串路径键转成 int 帧号 (basename 解析)。
        已经是 int 键的直接返回。无法解析的项丢弃。"""
        if not d:
            return d if d is not None else {}
        sample = next(iter(d.keys()))
        if isinstance(sample, int):
            return d
        out = {}
        for k, v in d.items():
            try:
                stem = os.path.splitext(os.path.basename(str(k)))[0]
                out[int(stem)] = v
            except (ValueError, TypeError):
                pass
        return out

    # ===== 帧号 helpers =====
    # 重构后 self.masks / self.combined_masks / self.overlay_imgs 的 key 都是
    # "源视频绝对帧号 (int)"，不再是文件路径。这样切 block 后 mask 数据依然
    # 通过帧号正确索引，跟磁盘上的 temp_dir 文件解耦。
    def _abs_idx(self, local_idx):
        """当前 block 的局部帧号 → 视频绝对帧号"""
        return self.current_block * self.block_size + local_idx
    def _current_abs_idx(self):
        """当前正在显示的帧的绝对帧号"""
        return self._abs_idx(self.curr_img_idx)
    def _local_idx(self, abs_idx):
        """绝对帧号 → 当前 block 的局部帧号；若帧不在当前 block 内，结果会越界"""
        return abs_idx - self.current_block * self.block_size
    def _path_for_abs_idx(self, abs_idx):
        """根据绝对帧号查文件路径。优先用 idx_to_path 的记录（apply_masks 写入），
        fallback 到 temp_dir/{abs_idx:06d}.jpg 的约定路径。返回的路径不保证存在。"""
        if abs_idx in self.idx_to_path:
            return self.idx_to_path[abs_idx]
        return os.path.join(self.frames_dir, f"{abs_idx:06d}.jpg")
    def _frame_has_prompts(self, local_frame_idx):
        """Check if a specific frame has any prompts."""
        for label in self.sam_handler.labels:
            for pt in label.pts.get(self.current_block, []):
                if pt.idx == local_frame_idx:
                    return True
            for box in label.boxes.get(self.current_block, []):
                if box.idx == local_frame_idx:
                    return True
        return False
    def has_prompts(self, block_idx):
        for label in self.get_labels():
            if len(label.pts.get(block_idx, [])) > 0 or len(label.boxes.get(block_idx, [])):
                return True
        return False
    def get_label_idx(self, label_name):
        for idx in range(len(self.sam_handler.labels)):
            if self.sam_handler.labels[idx].name == label_name:
                return idx
        return -1
    def get_label_idx_by_group_id(self, group_id):
        for idx in range(len(self.sam_handler.labels)):
            if self.sam_handler.labels[idx].group_id == group_id:
                return idx
        return -1
    def get_current_label(self):
        if self.curr_label_idx < 0:
            return None
        return self.sam_handler.labels[self.curr_label_idx]
    def set_current_label(self,idx):
        self.curr_label_idx = idx
    def set_label_name(self, idx, name):
        self.sam_handler.labels[idx].name = name
    def get_labels(self):
        return self.sam_handler.labels
    def get_current_label_idx(self):
        return self.curr_label_idx
    def get_current_img_idx(self):
        return self.curr_img_idx
    def get_number_of_images(self):
        return len(self.media_files)
    
    # LABEL AND PROMPT HANDLING
    
    def add_label(self, label_name, group_id=None, color=None):
        new_label = self.label_handler.create_new_label(label_name, group_id=group_id, color=color)
        for i in range(self.num_blocks):
            new_label.pts[i] = []
        for i in range(self.num_blocks):
            new_label.boxes[i] = []
        for i in range(self.num_blocks):
            new_label.prop_frames[i] = set()
        self.sam_handler.labels.append(new_label)
        self.curr_label_idx = len(self.sam_handler.labels) - 1
        return new_label.col
    def remove_label(self, idx):
        if idx < 0 or idx >= len(self.sam_handler.labels):
            return False
        removed = self.sam_handler.labels.pop(idx)
        # also wipe any masks that referenced this label's group_id
        for path_masks in self.masks.values():
            if removed.group_id in path_masks:
                del path_masks[removed.group_id]
        # adjust current selection
        if len(self.sam_handler.labels) == 0:
            self.curr_label_idx = -1
        else:
            self.curr_label_idx = min(self.curr_label_idx, len(self.sam_handler.labels) - 1)
        return True
    def reassign_label_in_range(self, src_group_id, dst_group_id, start_abs, end_abs):
        """Move masks and prompts from src label to dst label within [start_abs, end_abs]."""
        print(f"[reassign] src_gid={src_group_id} (type={type(src_group_id).__name__}), "
              f"dst_gid={dst_group_id} (type={type(dst_group_id).__name__}), "
              f"range=[{start_abs}, {end_abs}]")
        src_label = dst_label = None
        for label in self.sam_handler.labels:
            if label.group_id == src_group_id:
                src_label = label
            if label.group_id == dst_group_id:
                dst_label = label
        if not src_label or not dst_label:
            print(f"[reassign] label not found! src_label={src_label}, dst_label={dst_label}")
            return 0

        dst_name = dst_label.name
        modified_frames = set()

        # 1. Update JSON files
        json_dir = os.path.join(self.project_dir, "jsons")
        print(f"[reassign] json_dir={json_dir}, exists={os.path.isdir(json_dir)}")
        if os.path.isdir(json_dir):
            n_json_files = 0
            for abs_idx in range(start_abs, end_abs + 1):
                json_path = os.path.join(json_dir, f"{abs_idx:06d}.json")
                if not os.path.exists(json_path):
                    continue
                n_json_files += 1
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                changed = False
                for shape in data.get("shapes", []):
                    shape_gid = shape.get("group_id")
                    if shape_gid == src_group_id:
                        shape["group_id"] = dst_group_id
                        shape["label"] = dst_name
                        changed = True
                    elif abs_idx == start_abs and not changed:
                        # print first non-matching shape for debug
                        print(f"[reassign] frame {abs_idx}: shape group_id={shape_gid} "
                              f"(type={type(shape_gid).__name__}) != src {src_group_id} "
                              f"(type={type(src_group_id).__name__})")
                if changed:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    modified_frames.add(abs_idx)

            print(f"[reassign] json: found {n_json_files} json files in range, matched {len(modified_frames)} frames")

        # 2. Move in-memory masks from src to dst
        print(f"[reassign] in-memory masks count: {len(self.masks)}")
        for abs_idx in range(start_abs, end_abs + 1):
            if abs_idx not in self.masks:
                continue
            frame_masks = self.masks[abs_idx]
            if src_group_id not in frame_masks:
                continue
            src_mask = frame_masks.pop(src_group_id)
            if dst_group_id in frame_masks:
                dst_arr = frame_masks[dst_group_id].astype(bool)
                src_arr = src_mask.astype(bool)
                frame_masks[dst_group_id] = PackedMasks(dst_arr | src_arr)
            else:
                frame_masks[dst_group_id] = src_mask
            modified_frames.add(abs_idx)

        # 3. Move prompts (pts and boxes) between labels
        for block_idx in range(self.num_blocks):
            block_start = block_idx * self.block_size
            block_end = block_start + self.block_size - 1
            if block_end < start_abs or block_start > end_abs:
                continue
            local_start = max(0, start_abs - block_start)
            local_end = min(self.block_size - 1, end_abs - block_start)

            # move pts
            to_move = []
            remaining = []
            for pt in src_label.pts.get(block_idx, []):
                if local_start <= pt.idx <= local_end:
                    to_move.append(pt)
                else:
                    remaining.append(pt)
            if to_move:
                src_label.pts[block_idx] = remaining
                if block_idx not in dst_label.pts:
                    dst_label.pts[block_idx] = []
                dst_label.pts[block_idx].extend(to_move)

            # move boxes
            to_move_b = []
            remaining_b = []
            for box in src_label.boxes.get(block_idx, []):
                if local_start <= box.idx <= local_end:
                    to_move_b.append(box)
                else:
                    remaining_b.append(box)
            if to_move_b:
                src_label.boxes[block_idx] = remaining_b
                if block_idx not in dst_label.boxes:
                    dst_label.boxes[block_idx] = []
                dst_label.boxes[block_idx].extend(to_move_b)

            # move prop_frames
            src_pf = src_label.prop_frames.get(block_idx, set())
            dst_pf = dst_label.prop_frames.get(block_idx, set())
            frames_to_move = {f for f in src_pf if local_start <= f <= local_end}
            if frames_to_move:
                src_label.prop_frames[block_idx] = src_pf - frames_to_move
                dst_label.prop_frames[block_idx] = dst_pf | frames_to_move

        return len(modified_frames)

    def delete_label_in_range(self, group_id, start_abs, end_abs):
        """Delete masks and prompts for a label within [start_abs, end_abs]."""
        target_label = None
        for label in self.sam_handler.labels:
            if label.group_id == group_id:
                target_label = label
                break
        if not target_label:
            return 0

        n_modified = 0

        # 1. Remove matching shapes from JSON files
        json_dir = os.path.join(self.project_dir, "jsons")
        if os.path.isdir(json_dir):
            for abs_idx in range(start_abs, end_abs + 1):
                json_path = os.path.join(json_dir, f"{abs_idx:06d}.json")
                if not os.path.exists(json_path):
                    continue
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                original_count = len(data.get("shapes", []))
                data["shapes"] = [s for s in data.get("shapes", [])
                                  if s.get("group_id") != group_id]
                if len(data["shapes"]) < original_count:
                    if len(data["shapes"]) == 0:
                        os.remove(json_path)
                    else:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    n_modified += 1

        # 2. Remove in-memory masks
        for abs_idx in range(start_abs, end_abs + 1):
            if abs_idx in self.masks and group_id in self.masks[abs_idx]:
                del self.masks[abs_idx][group_id]

        # 3. Remove prompts (pts and boxes) within the frame range
        for block_idx in range(self.num_blocks):
            block_start = block_idx * self.block_size
            block_end = block_start + self.block_size - 1
            if block_end < start_abs or block_start > end_abs:
                continue
            local_start = max(0, start_abs - block_start)
            local_end = min(self.block_size - 1, end_abs - block_start)

            # remove pts in range
            pts = target_label.pts.get(block_idx, [])
            if pts:
                target_label.pts[block_idx] = [
                    pt for pt in pts if not (local_start <= pt.idx <= local_end)]

            # remove boxes in range
            boxes = target_label.boxes.get(block_idx, [])
            if boxes:
                target_label.boxes[block_idx] = [
                    box for box in boxes if not (local_start <= box.idx <= local_end)]

            # 4. Remove prop_frames in range
            pf = target_label.prop_frames.get(block_idx, set())
            if pf:
                target_label.prop_frames[block_idx] = {
                    f for f in pf if not (local_start <= f <= local_end)}

        return n_modified

    def add_feature_to_current_label(self,feature_name):
        current_label = self.sam_handler.labels[self.curr_label_idx]
        current_label.features.append([(feature_name, self.curr_img_idx + self.current_block * self.block_size)])
    def delete_feature_from_current_label(self,feature_idx):
        current_label = self.sam_handler.labels[self.curr_label_idx]
        current_label.features.pop(feature_idx)
    def delete_selected_point(self,point_idx):
        current_label = self.sam_handler.labels[self.curr_label_idx]
        if point_idx < len(current_label.pts[self.current_block]):
            current_label.remove_pt(point_idx,self.current_block)
    def delete_selected_box(self,box_idx):
        current_label = self.sam_handler.labels[self.curr_label_idx]
        if box_idx < len(current_label.boxes[self.current_block]):
            current_label.remove_box(box_idx,self.current_block)

    def clear_label_mask_on_frame(self, abs_idx, group_id):
        """Remove a single label's mask for one frame: in-memory + JSON.
        Used to undo auto-generated masks when the user deletes all prompts.
        Returns True if anything was removed."""
        changed = False
        # 1. in-memory
        if abs_idx in self.masks and group_id in self.masks[abs_idx]:
            del self.masks[abs_idx][group_id]
            if not self.masks[abs_idx]:
                del self.masks[abs_idx]
            changed = True
        # 2. JSON file on disk
        json_path = os.path.join(self.project_dir, "jsons", f"{abs_idx:06d}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                old_n = len(data.get("shapes", []))
                data["shapes"] = [s for s in data.get("shapes", [])
                                  if s.get("group_id") != group_id]
                if len(data["shapes"]) != old_n:
                    changed = True
                    if data["shapes"]:
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    else:
                        # no shapes left → drop the file
                        os.remove(json_path)
                        if hasattr(self, "_json_frames"):
                            self._json_frames.discard(abs_idx)
            except Exception as e:
                print(f"clear_label_mask_on_frame JSON error: {e}")
        # 3. invalidate render caches for this frame
        if changed:
            self.overlay_imgs.pop(abs_idx, None)
            self.combined_masks.pop(abs_idx, None)
        return changed
    
    # MODEL HANDLING
    
    def init_sam2(self):
        self.sam_handler = SAM_Annotator(model_type="vit_h",model_cfg_path="sam2.1_hiera_l.yaml",ckpt_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), self.base_checkpoint_path+"sam2.1_hiera_large.pt"))
    def can_initialize_model(self):
        return os.path.exists(self.sam_handler.ckpt_path)
    def get_model_type(self):
        return self.sam_handler.model_type
    def model_status(self):
        return self.sam_handler.is_model_loaded()
    def load_model(self,status_callback):
        if not self.can_initialize_model():
            return False
        return self.sam_handler.load_model(status_callback)
    
    # PROPAGATION

    def initialize_tracking(self,update_progress_callback):
        self.sam_handler.media_files = self.media_files
        self.sam_handler.curr_img_idx = self.curr_img_idx
        self.sam_handler.current_block = self.current_block
        self.sam_handler.sam_extract_dir = self.sam_extract_dir
        self.mode = "tracking"
        success = self.sam_handler.init_inference_state(self.sam_extract_dir, status_callback=update_progress_callback)
        if not success:
            self.mode = "prompts"
        return success
    def propagate_batch(self, update_progress_callback):
        """Propagate using all prompts in the current block (multi-frame conditioning)."""
        self.sam_handler.media_files = self.media_files
        self.sam_handler.current_block = self.current_block
        self.tracking_results = self.sam_handler.propagate_all_prompts(progress_callback=update_progress_callback)
        if not self.tracking_results:
            return False, []
        for k, v in self.tracking_results.items():
            for group_id, _ in v.items():
                li = self.get_label_idx_by_group_id(group_id)
                if li >= 0:
                    self.sam_handler.labels[li].prop_frames[self.current_block].add(k)
        return True, list(self.tracking_results.keys())
    def apply_masks(self, update_progress_callback, is_single=False, merge=False):
        total_steps = len(list(self.tracking_results.keys()))
        current_step = 0
        n_added = 0
        n_skipped = 0
        json_dir = os.path.join(self.project_dir, "jsons")
        merged_idxs = set()
        for local_frame_idx, frame_masks in self.tracking_results.items():
            if local_frame_idx >= len(self.media_files):
                continue
            abs_idx = self._abs_idx(local_frame_idx)
            frame_path = self.media_files[local_frame_idx]
            if merge:
                # Merge mode: write each label's mask into existing JSON
                try:
                    for group_id, mask in frame_masks.items():
                        mask_bool = mask.squeeze().astype(bool)
                        self._merge_mask_to_json(abs_idx, group_id, mask_bool, frame_path)
                    n_added += len(frame_masks)
                    merged_idxs.add(abs_idx)
                except Exception as e:
                    print(f"Error merging mask for frame {local_frame_idx}: {e}")
                current_step += 1
                if current_step % 10 == 0 or current_step == total_steps:
                    update_progress_callback(f"Merging masks: {current_step}/{total_steps}", (current_step / max(1, total_steps)) * 100)
                continue
            existing_json = os.path.join(json_dir, f"{abs_idx:06d}.json")
            if os.path.exists(existing_json) and not is_single:
                if self._frame_has_prompts(local_frame_idx):
                    # propagation skips frames that have manual prompts
                    n_skipped += 1
                    current_step += 1
                    continue
            self.idx_to_path[abs_idx] = frame_path
            self.masks[abs_idx] = {}
            try:
                for group_id, mask in frame_masks.items():
                    self.masks[abs_idx][group_id] = PackedMasks(mask.squeeze().astype(bool))
                n_added += len(self.masks[abs_idx])
                progress_percent = (current_step / max(1, total_steps)) * 100
                update_progress_callback(f"Applying masks: {current_step + 1}/{total_steps}", progress_percent)
                current_step += 1
            except Exception as e:
                print(f"Error applying mask for frame {local_frame_idx}: {e}")
        self.tracking_results = {}
        if merge and n_added > 0:
            # Merge mode: JSONs already written, just update state
            self._json_frames = getattr(self, '_json_frames', set()) | merged_idxs
            self.overlay_imgs = {}
            self.combined_masks = {}
            self.mode = "correction"
            self.view_mode = "overlay"
            import gc; gc.collect()
        elif n_added > 0:
            # auto-export JSON and free memory
            update_progress_callback("Exporting JSON...", 90)
            os.makedirs(os.path.join(self.project_dir, "jsons"), exist_ok=True)
            self.export_jsons(lambda tp, step, total: None)
            # record which abs_idxs have JSON (for overlay rendering)
            exported_idxs = set(self.masks.keys())
            self._json_frames = getattr(self, '_json_frames', set()) | exported_idxs
            # free masks from memory
            self.masks = {}
            self.overlay_imgs = {}
            self.combined_masks = {}
            self.mode = "correction"
            self.view_mode = "overlay"
            import gc; gc.collect()
        else:
            self.mode = "prompts"
    def has_propagation_block(self,idx):
        return (idx in self.sam_handler.propagation_blocks[self.current_block])
    def add_propagation_block(self,idx):
        self.sam_handler.propagation_blocks[self.current_block][idx] = 1
    def remove_propagation_block(self,idx):
        del self.sam_handler.propagation_blocks[self.current_block][idx]
    def generate_mask(self):
        if not self.sam_handler.is_model_loaded():
            return False, []
        if self.curr_label_idx < 0 or len(self.media_files) < 0:
            return False, []
        labels_with_points = [label for label in self.sam_handler.labels if len(label.pts.get(self.current_block, [])) > 0]
        labels_with_boxes = [label for label in self.sam_handler.labels if len(label.boxes.get(self.current_block, [])) > 0]
        if not labels_with_points and not labels_with_boxes:
            return False, []
        self.sam_handler.media_files = self.media_files
        self.sam_handler.curr_img_idx = self.curr_img_idx
        self.sam_handler.current_block = self.current_block
        self.sam_handler.sam_extract_dir = self.sam_extract_dir
        self.tracking_results = self.sam_handler.generate_mask_for_frame(self.curr_img_idx,0)
        if not self.tracking_results:  # {} 和 None 都当失败
            self.mode = "prompts"
            return False, []
        for k, v in self.tracking_results.items():
            for group_id, _ in v.items():
                li = self.get_label_idx_by_group_id(group_id)
                if li >= 0:
                    self.sam_handler.labels[li].prop_frames[self.current_block].add(k)
        return True, list(self.tracking_results.keys())
    def propagate(self,flag,update_progress_callback):
        if flag == 0:
            self.tracking_results = self.sam_handler.propagate_to_all(
                current_frame_idx=self.curr_img_idx,
                start_frame_idx=None,
                end_frame_idx=None,
                progress_callback=update_progress_callback)
        elif flag == 1:
            self.tracking_results = self.sam_handler.propagate(1,
                start_frame_idx=self.curr_img_idx,
                end_frame_idx=None,
                progress_callback=update_progress_callback)
        else:
            self.tracking_results = self.sam_handler.propagate(-1,
                start_frame_idx=self.curr_img_idx,
                end_frame_idx=None,
                progress_callback=update_progress_callback)
        if not self.tracking_results:
            self.mode = "prompts"
            return False, []
        else:
            if self.block_size in self.tracking_results:
                self.extra_frame[self.current_block] = self.tracking_results[self.block_size]
                self.tracking_results.pop(self.block_size)
                self.extra_frame_masks[self.current_block] = {}
                for group_id, mask in self.extra_frame[self.current_block].items():
                    self.extra_frame_masks[self.current_block][group_id] = PackedMasks(mask.squeeze().astype(bool))
            for k, v in self.tracking_results.items():
                for group_id, _ in v.items():
                    li = self.get_label_idx_by_group_id(group_id)
                    if li >= 0:
                        self.sam_handler.labels[li].prop_frames[self.current_block].add(k)
        return True, list(self.tracking_results.keys())

    def propagate_single_label(self, group_id, direction, update_progress_callback):
        """Propagate only one label. direction: 1=forward, -1=backward, 0=both."""
        self.tracking_results = self.sam_handler.propagate_single_label(
            group_id, self.curr_img_idx, direction, update_progress_callback)
        if not self.tracking_results:
            self.mode = "prompts"
            return False, []
        # Update prop_frames for this label
        li = self.get_label_idx_by_group_id(group_id)
        if li >= 0:
            pf = self.sam_handler.labels[li].prop_frames.setdefault(self.current_block, set())
            for k in self.tracking_results:
                pf.add(k)
        return True, list(self.tracking_results.keys())

    def _merge_mask_to_json(self, abs_idx, group_id, mask_bin, frame_path):
        """Merge a single label's mask into the existing JSON for this frame.
        Other labels' shapes are preserved."""
        import json as _json
        json_dir = os.path.join(self.project_dir, "jsons")
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, f"{abs_idx:06d}.json")
        img_h, img_w = mask_bin.shape[:2]
        # Read existing JSON or create template
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = _json.load(f)
        else:
            data = {
                "version": "5.0.1", "flags": {},
                "shapes": [],
                "imagePath": f"{abs_idx:06d}.jpg",
                "imageData": None,
                "imageHeight": int(img_h), "imageWidth": int(img_w)
            }
        # Remove old shapes for this group_id
        data["shapes"] = [s for s in data.get("shapes", [])
                          if s.get("group_id") != group_id]
        # Convert mask to contour polygons
        mask_u8 = mask_bin.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        label_name = None
        for lbl in self.sam_handler.labels:
            if lbl.group_id == group_id:
                label_name = lbl.name
                break
        if label_name is None:
            label_name = f"label_{group_id}"
        for c in cnts:
            if cv2.contourArea(c) <= 50:
                continue
            points = c.reshape(-1, 2).tolist()
            data["shapes"].append({
                "label": label_name,
                "points": points,
                "group_id": group_id,
                "shape_type": "polygon",
                "flags": {}
            })
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)

    # AUTO-PROMPTING ~ DONE
    
    def keep_largest_connected_region(self,mask):
        labeled = label(mask > 0)
        if labeled.max() == 0:
            return np.zeros_like(mask)
        largest_region = max(regionprops(labeled), key=lambda r: r.area)
        return (labeled == largest_region.label).astype(np.uint8) * 255
    def skeletonize_single_mask(self,mask):
        if mask is None:
            return None
        mask = self.keep_largest_connected_region(mask)
        mask = mask.astype(np.uint8)
        skeleton = skeletonize(mask).astype(np.uint8)
        return skeleton
    def generate_point_prompts(self,mask):
        skeleton = self.skeletonize_single_mask(mask)
        neighbourhood_kernel = np.array([[1, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 1]], dtype=np.uint8)
        neighbor_count = ndi.convolve(skeleton.astype(np.uint8),neighbourhood_kernel,mode='constant', cval=0)
        endpoints = skeleton & (neighbor_count == 1)
        endpoints = np.vstack(np.nonzero(endpoints)).T
        junctions = skeleton & (neighbor_count >= 3)
        structure = np.ones((3, 3), dtype=np.uint8)
        labels, n_labels = ndi.label(junctions, structure=structure)
        if n_labels > 0:
            junctions = np.asarray(ndi.center_of_mass(junctions, labels, index=range(1, n_labels + 1)),dtype=np.int32)
            return np.vstack([endpoints,junctions])
        else:
            junctions = np.asarray([])
            return endpoints
    
    # VIEW MODE
    
    def get_next_view_mode(self):
        cur_abs = self._current_abs_idx()
        if self.view_mode == "original":
            return "prompts"
        elif self.view_mode == "prompts":
            if hasattr(self, 'overlay_img') and cur_abs in self.masks:
                return "overlay"
            elif hasattr(self, 'masks') and cur_abs in self.masks:
                return "masks"
            else:
                return "original"
        elif self.view_mode == "overlay":
            if hasattr(self, 'masks') and cur_abs in self.masks:
                return "masks"
            else:
                return "original"
        else:
            return "original"
    def allowed_view_mode(self,new_view_mode):
        return True  # always allow switching; frames without masks show original as fallback
    def set_view_mode(self, vm_idx):
        if vm_idx < 0 or vm_idx > 3:
            return 
        new_view_mode = self.view_modes[vm_idx]
        if self.allowed_view_mode(new_view_mode):
            self.view_mode = new_view_mode
    def toggle_view_mode(self):
        self.view_mode = self.get_next_view_mode()
    def reset_view_mode(self):
        self.mode = "prompts"
        self.view_mode = "original"
    def get_view_mode(self):
        return self.view_mode
    def get_mode(self):
        return self.mode
    
    # ANNOTATION EXPORT
    
    def bbox_from_mask(self,mask):
        if not np.any(mask):
            return None, None
        r = np.any(mask, axis=1)
        c = np.any(mask, axis=0)
        rmin,rmax = np.where(r)[0][[0,-1]]
        cmin,cmax = np.where(c)[0][[0,-1]]
        center = np.argwhere(mask).mean(axis=0)
        return [rmin,rmax,cmin,cmax],tuple(center)
    # export_manual_data: removed (dead code, was not called by export_all)
    # export_interpolated_data: removed (dead code, was not called by export_all)
    # export_overlays: removed (dead code, was not called by export_all)
    # export_masks: removed (dead code, was not called by export_all)
    # export_label_data: removed (dead code, was not called by export_all)
    def export_jsons(self, update_progress_callback):
        """Export per-frame JSON files in LabelMe format.
        File name uses the absolute source frame index (e.g. 000045.json for frame 45),
        so sparse annotations stay aligned with the original video."""
        import json
        sorted_keys = sorted(self.masks.keys())
        total_steps = len(sorted_keys)
        export_dir = os.path.join(self.project_dir, "jsons")
        img_h, img_w = self.curr_img_shape[:2] if len(self.curr_img_shape) >= 2 else (0, 0)
        for step, abs_idx in enumerate(sorted_keys):
            shapes = []
            for label in self.sam_handler.labels:
                if label.group_id not in self.masks[abs_idx]:
                    continue
                mask_bin = self.masks[abs_idx][label.group_id].astype(np.uint8) * 255
                if img_h == 0:
                    img_h, img_w = mask_bin.shape[:2]
                cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) <= 50:
                        continue
                    points = c.reshape(-1, 2).tolist()
                    shapes.append({
                        "label": label.name,
                        "points": points,
                        "group_id": label.group_id,
                        "shape_type": "polygon",
                        "flags": {}
                    })
            image_name = f"{abs_idx:06d}.jpg"
            labelme_json = {
                "version": "5.0.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": int(img_h),
                "imageWidth": int(img_w)
            }
            out_path = os.path.join(export_dir, f"{abs_idx:06d}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_json, f, ensure_ascii=False, indent=2)
            if step % 50 == 0 or step == total_steps - 1:
                update_progress_callback("json", step, total_steps)
    # export_video: removed (dead code, was not called by export_all)
    def init_export(self):
        # 只创建实际用得到的目录：jsons/ 主交付物 + overlays/ 验证样本。
        # 重构后 self.masks 用 abs_idx 当 key，不依赖磁盘文件，所以这里
        # 不再需要前置 _ensure_all_frames_on_disk() —— 验证 overlay 那 10 张
        # 才需要原图，由 export_verification_overlays 内部按需重抽。
        # 只清理导出子目录，不删整个 project_dir（里面还有 frames/ 和 session.pkl）
        for subdir in ("jsons", "overlays"):
            p = os.path.join(self.project_dir, subdir)
            shutil.rmtree(p, ignore_errors=True)
            os.makedirs(p, exist_ok=True)

    # export_manifest: removed (dead code, was not called by export_all)
    def _render_overlay_from_json(self, abs_idx, json_dir):
        """Render a single overlay image from JSON + original frame. Returns BGR image or None."""
        import json as _json
        json_path = os.path.join(json_dir, f"{abs_idx:06d}.json")
        frame_path = self._path_for_abs_idx(abs_idx)
        if not os.path.exists(json_path) or not os.path.exists(frame_path):
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            frame_json = _json.load(f)
        img = cv2.imread(frame_path)
        if img is None:
            return None
        overlay = img.copy()
        shape_labels = []  # [(pts, label_name, rgb_tuple)]
        for shape in frame_json.get("shapes", []):
            label_name = shape.get("label", "")
            gid = shape.get("group_id", None)
            points = shape.get("points", [])
            if not points:
                continue
            hex_col = "#ffffff"
            if gid is not None:
                for lbl in self.sam_handler.labels:
                    if lbl.group_id == gid:
                        hex_col = lbl.col
                        break
            else:
                for lbl in self.sam_handler.labels:
                    if lbl.name == label_name:
                        hex_col = lbl.col
                        break
            r, g, b = self.hex_to_rgb(hex_col)
            colour = (b, g, r)
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], colour)
            cv2.polylines(overlay, [pts], isClosed=True, color=colour, thickness=2)
            shape_labels.append((pts, label_name, (r, g, b)))
        blended = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
        # label name near the largest polygon of each label (one text per label)
        best = {}  # label_name -> (area, x, y, bw, bh, rgb)
        for pts, label_name, rgb in shape_labels:
            area = cv2.contourArea(pts)
            if label_name not in best or area > best[label_name][0]:
                x, y, bw, bh = cv2.boundingRect(pts)
                best[label_name] = (area, x, y, bw, bh, rgb)
        texts = []
        for label_name, (_, x, y, bw, bh, rgb) in best.items():
            tx = max(0, x + bw // 2 - len(label_name) * 5)
            ty = max(0, y - 22)
            texts.append((label_name, (tx, ty), rgb))
        self._put_texts_pil(blended, texts, font_size=18)
        # frame number (large, bottom-left)
        cv2.putText(blended, f"Frame {abs_idx}", (10, blended.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(blended, f"Frame {abs_idx}", (10, blended.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
        return blended

    def export_verification_overlays(self, num_samples=9):
        """导出 9 张均匀抽样的验证 overlay + 1 张 3x3 拼图总览。"""
        json_dir = os.path.join(self.project_dir, "jsons")
        out_dir = os.path.join(self.project_dir, "overlays")

        if not os.path.exists(json_dir):
            return
        sorted_keys = sorted(
            int(os.path.splitext(f)[0]) for f in os.listdir(json_dir) if f.endswith(".json")
        )
        if not sorted_keys:
            return

        n = min(num_samples, len(sorted_keys))
        if n <= 1:
            indices = [0]
        else:
            indices = [int(round(i * (len(sorted_keys) - 1) / (n - 1))) for i in range(n)]
        sample_abs_idxs = [sorted_keys[i] for i in indices]

        self._ensure_frames_on_disk(sample_abs_idxs)

        # render individual overlays
        rendered = []
        rendered_idxs = []
        for step, abs_idx in zip(indices, sample_abs_idxs):
            blended = self._render_overlay_from_json(abs_idx, json_dir)
            if blended is None:
                continue
            out_name = f"verify_{abs_idx:06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, out_name), blended)
            rendered.append(blended)
            rendered_idxs.append(abs_idx)

        # create 3x3 grid summary with title
        if len(rendered) >= 1:
            grid_cols, grid_rows = 3, 3
            cell_h, cell_w = rendered[0].shape[:2]
            thumb_w, thumb_h = cell_w // 2, cell_h // 2
            title_h = 40  # space for video name at top

            # pad to 9
            while len(rendered) < grid_cols * grid_rows:
                rendered.append(np.zeros_like(rendered[0]))
                rendered_idxs.append(-1)

            grid = np.zeros((title_h + thumb_h * grid_rows, thumb_w * grid_cols, 3),
                            dtype=np.uint8)

            # title: video/session name
            video_name = self.session_name or os.path.basename(self.project_dir)
            self._put_text_pil(grid, video_name, (10, 5), font_size=22)

            for i in range(grid_cols * grid_rows):
                r, c = i // grid_cols, i % grid_cols
                thumb = cv2.resize(rendered[i], (thumb_w, thumb_h),
                                   interpolation=cv2.INTER_AREA)
                y0 = title_h + r * thumb_h
                grid[y0:y0 + thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb
                # frame number on each cell
                if rendered_idxs[i] >= 0:
                    cv2.putText(grid, f"F{rendered_idxs[i]}", (c * thumb_w + 5, y0 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            cv2.imwrite(os.path.join(out_dir, "verify_grid.jpg"), grid)

    def export_verify_video(self, video_path=None, progress_callback=None):
        """从原始视频逐帧读取，叠加已导出的 JSON polygon，生成一个带 overlay 的验证视频。
        有 mask 的帧画彩色多边形 + label 名，没有 mask 的帧原样保留。"""
        if video_path is None:
            video_path = self.video_name
        if not video_path or not os.path.exists(video_path):
            return False, "Source video not found"
        json_dir = os.path.join(self.project_dir, "jsons")
        if not os.path.exists(json_dir):
            return False, "jsons/ directory not found. Export first."
        # 输出文件名：原视频名_verify.mp4
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(self.project_dir, f"{video_basename}_verify.mp4")

        # 收集所有 JSON 文件，按帧号索引
        json_files = {}
        for f in os.listdir(json_dir):
            if f.endswith(".json"):
                try:
                    abs_idx = int(os.path.splitext(f)[0])
                    json_files[abs_idx] = os.path.join(json_dir, f)
                except ValueError:
                    continue
        if not json_files:
            return False, "No JSON files found"

        # label id → (name, color)
        id_to_label = {i: (lbl.name, lbl.col) for i, lbl in enumerate(self.sam_handler.labels)}

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        if not writer.isOpened():
            cap.release()
            return False, "Cannot create output video"

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # 如果这一帧有 JSON 标注，叠加 polygon
            if frame_idx in json_files:
                with open(json_files[frame_idx], 'r', encoding='utf-8') as f:
                    frame_json = json.load(f)
                overlay = frame.copy()
                shape_labels = []  # [(pts, label_name, rgb_tuple)]
                for shape in frame_json.get("shapes", []):
                    label_name = shape.get("label", "")
                    gid = shape.get("group_id", None)
                    points = shape.get("points", [])
                    if not points:
                        continue
                    hex_col = "#ffffff"
                    if gid is not None:
                        for lbl in self.sam_handler.labels:
                            if lbl.group_id == gid:
                                hex_col = lbl.col
                                break
                    else:
                        for lbl in self.sam_handler.labels:
                            if lbl.name == label_name:
                                hex_col = lbl.col
                                break
                    r, g, b = self.hex_to_rgb(hex_col)
                    colour = (b, g, r)
                    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(overlay, [pts], colour)
                    cv2.polylines(overlay, [pts], True, colour, 2)
                    shape_labels.append((pts, label_name, (r, g, b)))
                frame = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)
                # label name near the largest polygon of each label
                best = {}
                for pts, label_name, rgb in shape_labels:
                    area = cv2.contourArea(pts)
                    if label_name not in best or area > best[label_name][0]:
                        x, y, bw, bh = cv2.boundingRect(pts)
                        best[label_name] = (area, x, y, bw, bh, rgb)
                texts = []
                for label_name, (_, x, y, bw, bh, rgb) in best.items():
                    tx = max(0, x + bw // 2 - len(label_name) * 5)
                    ty = max(0, y - 22)
                    texts.append((label_name, (tx, ty), rgb))
                self._put_texts_pil(frame, texts, font_size=18)
            writer.write(frame)
            if progress_callback and frame_idx % 100 == 0:
                progress_callback(frame_idx, total_frames)

        writer.release()
        cap.release()
        return True, out_path

    def _ensure_frames_on_disk(self, abs_idxs):
        """按需从源视频重抽指定 abs_idx 的帧到约定路径。
        重构后只用于 export_verification_overlays 的那 10 张样本（不再前置全量重抽）。"""
        if not self.video_name or not os.path.exists(self.video_name):
            return
        missing = []
        for abs_idx in abs_idxs:
            target = self._path_for_abs_idx(abs_idx)
            if not os.path.exists(target):
                missing.append((abs_idx, target))
        if not missing:
            return
        cap = cv2.VideoCapture(self.video_name)
        try:
            for abs_idx, target in missing:
                cap.set(cv2.CAP_PROP_POS_FRAMES, abs_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                cv2.imwrite(target, frame)
        finally:
            cap.release()
        
    # MEDIA HANDLING
    
    def reset_media(self):
        self.media_files = []
        self.curr_img_idx = -1
        self.overlay_img = None
        self.composite_mask = None
        self.original_img = None
        self.overlay_imgs = {}
        self.combined_masks = {}
        self.mode = "prompts"
        self.view_mode = "prompts"
        self.tracking_results = {}
        self.sam_handler.object_id_to_group_id = {}
        self.sam_handler.object_ids = {}
        self.sam_handler.media_files = []
        self.curr_img_shape = (0,0,0)
        if self.sam_handler.inference_state:
            self.sam_handler.predictor.reset_state(self.sam_handler.inference_state)
        self.sam_handler.tracking_init = False
        self.sam_handler.inference_state = None
    def clear_temp_dir(self):
        # 只删不建：目录会在后续 extract_frames / process_image_folder 里按需创建。
        # 这里不建是因为调用时 session_name 可能还是默认值 "Session"，
        # makedirs 会创建一个 projects/Session/ 空壳目录。
        if hasattr(self, 'video_name') and self.video_name:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
            shutil.rmtree(self.sam_extract_dir, ignore_errors=True)
    def load_main_folder_unified(self, file_path, block_size):
        self.block_size = block_size
        if file_path is not None and len(file_path) > 0:
            extension = file_path.split(".")[-1]
            if extension in self.image_file_types:
                self.process_image_folder(os.path.dirname(file_path),self.current_block * block_size,(self.current_block + 1) * block_size)
                return 0
            elif os.path.isdir(file_path):
                self.process_image_folder(file_path,self.current_block * block_size,(self.current_block + 1) * block_size)
                return 0
            elif extension in self.video_file_types:
                self.process_video_file(file_path)
                return 1
            else:
                print(f"Invalid file format!")
                return -1
    def process_image_folder(self, folder_path, start_frame, end_frame):
        self.media_path = folder_path
        self.video_name = ""
        try:
            self.extract_dir = folder_path  # 帧就在源目录里，不复制
            all_files = os.listdir(folder_path)
            self.media_files = [os.path.join(folder_path, file) for file in all_files if file.split(".")[-1] in self.image_file_types]
            def numeric_key(p):
                name = os.path.splitext(os.path.basename(p))[0]
                match = re.search(r'\d+', name)
                if match:
                    return int(match.group())
                return float('inf')
            self.media_files.sort(key=numeric_key)
            end_frame = int(np.amin([end_frame, len(self.media_files)]))

            # 边界帧（block 末尾的下一帧，SAM2 传播用，用户不可见）
            if len(self.media_files) > end_frame:
                self.extra_frame_path[self.current_block] = self.media_files[end_frame]
            else:
                self.extra_frame_path[self.current_block] = None

            # media_files 只保留当前 block 的帧
            self.media_files = self.media_files[start_frame:end_frame]

            # SAM2 需要独立的干净目录，把当前 block 的帧复制过去
            shutil.rmtree(self.sam_extract_dir, ignore_errors=True)
            os.makedirs(self.sam_extract_dir, exist_ok=True)
            for file_name in self.media_files:
                shutil.copy(file_name, os.path.join(self.sam_extract_dir, os.path.basename(file_name)))
            # 边界帧也复制
            if self.extra_frame_path.get(self.current_block) and os.path.exists(self.extra_frame_path[self.current_block]):
                shutil.copy(self.extra_frame_path[self.current_block],
                            os.path.join(self.sam_extract_dir, os.path.basename(self.extra_frame_path[self.current_block])))
            self.curr_img_idx = -1
        except Exception as e:
            print(f"Error loading folders: {str(e)}")
    def process_video_file(self, video_path):
        try:
            self.media_path = os.path.dirname(video_path)
            self.video_name = video_path
            self.extract_dir = self.frames_dir
            self.extracted_frames = []
        except Exception as e:
            print(f"Error processing video: {str(e)}")
    def get_loaded_frame_count(self):
        return len(self.media_files)
    def get_frame_count(self,video_path):
        total_frames = 0
        try:
            if self.cap is not None:
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
        except Exception as e:
            print(f"Error: {str(e)}")
        return total_frames
    def get_frame_count_dir(self,folder_path):
        temp_media_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.split(".")[-1] in self.image_file_types]
        return len(temp_media_files)
    def reset_read_frames(self):
        self.read_frames = -1
    def _get_ffmpeg_exe(self):
        """返回 ffmpeg 可执行路径。优先 imageio_ffmpeg（pip 自带），其次系统 PATH。"""
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        ff = shutil.which("ffmpeg")
        return ff  # None if not found

    # _extract_frames_ffmpeg: removed (dead code, not called by extract_frames)
    def extract_frames(self, start_frame, end_frame, interval, progress_var):
        frame_count = 0
        saved_count = 0
        self.extracted_frames = []
        # sam_extract_dir 必须每次重建（SAM2 init_state 要求干净目录）
        shutil.rmtree(self.sam_extract_dir, ignore_errors=True)
        os.makedirs(self.sam_extract_dir, exist_ok=True)
        shutil.rmtree(self.extract_dir, ignore_errors=True)
        os.makedirs(self.extract_dir, exist_ok=True)

        # cv2 逐帧循环：精确 + 速度好（比 ffmpeg select= 快 5 倍，比 ffmpeg -ss 精确）
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_name)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                frame_path = os.path.join(self.extract_dir, f"{i:06d}.jpg")
                self.extracted_frames.append(frame_path)
                cv2.imwrite(frame_path, frame)
                cv2.imwrite(os.path.join(self.sam_extract_dir, f"{i:06d}.jpg"), frame)
                saved_count += 1
            frame_count += 1
            progress = (frame_count / (end_frame - start_frame)) * 100
            progress_var.set(progress)
        # extra frame（block 边界的下一帧，给 SAM2 传播用）
        ret, frame = self.cap.read()
        if ret:
            frame_path = os.path.join(self.extract_dir, f"{(i+1):06d}.jpg")
            self.extra_frame_path[self.current_block] = frame_path
            cv2.imwrite(frame_path, frame)  # 写到 extract_dir
            cv2.imwrite(os.path.join(self.sam_extract_dir, f"{(i+1):06d}.jpg"), frame)  # 也写到 sam_extract_dir
        else:
            self.extra_frame_path[self.current_block] = None

        self.read_frames = end_frame + 1
        self.media_files = self.extracted_frames
        self.curr_img_idx = -1

    # _extract_extra_frame_ffmpeg: removed (dead code, not called by extract_frames)
    # RENDER FRAME
    
    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    _pil_font_cache = {}

    def _get_pil_font(self, size):
        if size in self._pil_font_cache:
            return self._pil_font_cache[size]
        font = None
        for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc"):
            try:
                font = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        self._pil_font_cache[size] = font
        return font

    def _put_text_pil(self, img_bgr, text, pos, font_size=16, fg_color=(0, 0, 0), outline_color=(255, 255, 255)):
        """Draw Unicode text on a BGR numpy image using PIL. Draws outline + foreground for readability."""
        self._put_texts_pil(img_bgr, [(text, pos, fg_color)], font_size=font_size, outline_color=outline_color)

    def _put_texts_pil(self, img_bgr, texts, font_size=16, outline_color=(255, 255, 255)):
        """Draw multiple Unicode texts on a BGR image with a single PIL conversion.
        texts: list of (text, (x, y), fg_color_rgb)"""
        if not texts:
            return
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = self._get_pil_font(font_size)
        for text, pos, fg_color in texts:
            x, y = pos
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
            draw.text((x, y), text, font=font, fill=fg_color)
        img_bgr[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _has_json(self, abs_idx):
        """Check if a JSON annotation file exists for this frame."""
        json_path = os.path.join(self.project_dir, "jsons", f"{abs_idx:06d}.json")
        return os.path.exists(json_path)

    def _load_shapes_from_json(self, abs_idx):
        """Load LabelMe shapes from JSON file. Returns list of shapes or None."""
        json_path = os.path.join(self.project_dir, "jsons", f"{abs_idx:06d}.json")
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("shapes", [])
        except Exception:
            return None

    def create_overlay_img(self, abs_idx=None, overwrite=False):
        """渲染 overlay：原图 + 半透明 mask。从 JSON 或内存 masks 渲染。"""
        alpha = 0.5
        if abs_idx is None:
            abs_idx = self._current_abs_idx()
        file_path = self._path_for_abs_idx(abs_idx)
        if not os.path.exists(file_path):
            return None
        img_rgb = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        self.curr_img_shape = img_rgb.shape
        combined = self.create_combined_mask(abs_idx, overwrite=overwrite)
        if combined is None:
            return None
        blended = cv2.addWeighted(img_rgb, 1, combined, alpha, 0)
        return blended

    def create_combined_mask(self, abs_idx=None, overwrite=False):
        """渲染合成 mask。优先从内存 masks 读，其次从 JSON 读。"""
        if abs_idx is None:
            abs_idx = self._current_abs_idx()

        # try memory first (during apply_masks, before export)
        if abs_idx in self.masks and self.masks[abs_idx]:
            mask_overlay = np.zeros(self.curr_img_shape, dtype=np.uint8)
            for group_id, mask in self.masks[abs_idx].items():
                label_color = self.hex_to_rgb(
                    next((l.col for l in self.sam_handler.labels if l.group_id == group_id), "#ffffff"))
                mask_overlay[np.asarray(mask, dtype=bool)] = label_color
            return mask_overlay

        # read from JSON
        shapes = self._load_shapes_from_json(abs_idx)
        if not shapes:
            return None

        h, w = self.curr_img_shape[:2] if len(self.curr_img_shape) >= 2 else (0, 0)
        if h == 0:
            return None
        mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for shape in shapes:
            label_name = shape.get("label", "")
            gid = shape.get("group_id", None)
            points = shape.get("points", [])
            if not points:
                continue
            if gid is not None:
                hex_col = next((l.col for l in self.sam_handler.labels if l.group_id == gid), "#ffffff")
            else:
                hex_col = next((l.col for l in self.sam_handler.labels if l.name == label_name), "#ffffff")
            color = self.hex_to_rgb(hex_col)
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask_overlay, [pts], color)
        return mask_overlay
    def get_img_to_resize(self):
        if self.view_mode == "overlay":
            if not hasattr(self, 'masks'):
                return None
            return self.create_overlay_img()
        elif self.view_mode == "masks":
            if not hasattr(self, 'masks'):
                return None
            return self.create_combined_mask()
        else:
            if not hasattr(self, 'original_img') or self.original_img is None:
                return None
            if self.curr_img_idx == -1:
                return None
            original_img = np.array(self.original_img)
            self.curr_img_shape = original_img.shape
            return original_img
        
    # ADD PROMPT
    
    def add_point_prompt_to_label(self,x, y, pt_type,frame_idx, label_idx):
        self.sam_handler.labels[label_idx].add_pt(x, y, pt_type,frame_idx,self.current_block)
    def add_box_prompt_to_label(self,fx,fy, x, y, pt_type,frame_idx, label_idx):
        self.sam_handler.labels[label_idx].add_box(fx, fy, x, y, pt_type,frame_idx,self.current_block)
    def add_point_prompt_to_current_label(self,x, y, pt_type,frame_idx):
        self.sam_handler.labels[self.curr_label_idx].add_pt(x, y, pt_type,frame_idx,self.current_block)
    def add_box_prompt_to_current_label(self,fx,fy, x, y, pt_type,frame_idx):
        self.sam_handler.labels[self.curr_label_idx].add_box(fx, fy, x, y, pt_type,frame_idx,self.current_block)
        
    # SWITCH IMG
    
    def load_img(self, image_path):
        self.original_img = Image.open(image_path)
    def set_img(self,idx):
        if not self.media_files:
            return
        self.curr_img_idx = idx
        self.refresh_img_data()
    def next_img(self):
        if not self.media_files:
            return
        if self.curr_img_idx < len(self.media_files) - 1:
            self.curr_img_idx += 1
            self.refresh_img_data()
    def prev_img(self):
        if not self.media_files:
            return
        if self.curr_img_idx > 0:
            self.curr_img_idx -= 1
            self.refresh_img_data()
    def refresh_img_data(self):
        cur_abs = self._current_abs_idx()
        if self.view_mode in ("original", "prompts"):
            self.load_img(self.media_files[self.curr_img_idx])
        elif self.view_mode == "overlay":
            if cur_abs in self.masks:
                overlay_img = self.create_overlay_img()
                if overlay_img is not None:
                    if not isinstance(overlay_img, Image.Image):
                        self.overlay_img = Image.fromarray(overlay_img)
                    else:
                        self.overlay_img = overlay_img
                    return
            # no mask on this frame: show original, do NOT change view_mode
            self.load_img(self.media_files[self.curr_img_idx])
        else:  # masks
            if cur_abs in self.masks:
                composite_mask = self.create_combined_mask()
                if composite_mask is not None:
                    if not isinstance(composite_mask, Image.Image):
                        self.composite_mask = Image.fromarray(composite_mask)
                    else:
                        self.composite_mask = composite_mask
                    return
            # no mask on this frame: show original, do NOT change view_mode
            self.load_img(self.media_files[self.curr_img_idx])
class PackedMasks:
    def __init__(self, arr):
        arr = np.asarray(arr, dtype=bool)
        self.data = np.packbits(arr, axis=-1)
        self.h, self.w = arr.shape
    def _unpack(self):
        return np.unpackbits(self.data, axis=-1)[:, :self.w].astype(bool)
    def get(self):
        return self._unpack()
    def set(self, arr):
        arr = np.asarray(arr, dtype=bool)
        self.data = np.packbits(arr, axis=-1)
        self.h, self.w = arr.shape
    def __array__(self, dtype=None):
        arr = self._unpack()
        return arr.astype(dtype) if dtype is not None else arr
    def astype(self, dtype):
        # 让 export 函数里 packed.astype(np.uint8) 不再 AttributeError
        return self._unpack().astype(dtype)
    @property
    def shape(self):
        return (self.h, self.w)
    def __getitem__(self, key):
        return self._unpack()[key]
    def __setitem__(self, key, value):
        arr = self._unpack()
        arr[key] = value
        self.set(arr)
    def _binary_op(self, other, op):
        return op(self._unpack(), other)
    def astype(self, dtype, **kwargs):
        return self._unpack().astype(dtype, **kwargs)
    def __gt__(self, other):
        return self._binary_op(other, np.greater)

    def __ge__(self, other):
        return self._binary_op(other, np.greater_equal)

    def __lt__(self, other):
        return self._binary_op(other, np.less)

    def __le__(self, other):
        return self._binary_op(other, np.less_equal)

    def __eq__(self, other):
        return self._binary_op(other, np.equal)

    def __ne__(self, other):
        return self._binary_op(other, np.not_equal)
