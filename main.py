"""SAMannot-DPG — DearPyGui video annotation tool."""

import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import os
import threading
import math
import json
import pickle
import subprocess

from annotator import Annotator

# ── globals ──────────────────────────────────────────────────────────────────
ann = Annotator()

TEX_TAG = "frame_texture"
IMG_TAG = "frame_image"
DRAW_TAG = "frame_drawlist"

DISPLAY_W = 1100
DISPLAY_H = 619

tex_w = DISPLAY_W
tex_h = DISPLAY_H
tex_data = np.zeros(DISPLAY_W * DISPLAY_H * 4, dtype=np.float32)

# coordinate mapping state (set when rendering frame)
img_scale = 1.0
img_x_off = 0
img_y_off = 0
img_orig_w = 1
img_orig_h = 1

# interaction state
dragging = False
drag_start = None
_right_click_start = None  # display coords at right-click down

# inference state
inference_busy = False
inference_lock = threading.Lock()

# DPG context ready flag
_dpg_ready = False
_session_load_done = False
_media_load_done = False

LABEL_PRESETS_FILE = "label_presets.json"

# ── coordinate helpers ───────────────────────────────────────────────────────

def display_to_img(dx, dy):
    ix = (dx - img_x_off) / img_scale
    iy = (dy - img_y_off) / img_scale
    return ix, iy


def img_to_display(ix, iy):
    dx = ix * img_scale + img_x_off
    dy = iy * img_scale + img_y_off
    return dx, dy


def is_in_image(dx, dy):
    ix, iy = display_to_img(dx, dy)
    return 0 <= ix < img_orig_w and 0 <= iy < img_orig_h


# ── helpers ──────────────────────────────────────────────────────────────────

def np_rgb_to_texture(img_rgb, target_w, target_h):
    global img_scale, img_x_off, img_y_off, img_orig_w, img_orig_h
    h, w = img_rgb.shape[:2]
    img_orig_w, img_orig_h = w, h
    scale = min(target_w / w, target_h / h)
    img_scale = scale
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    img_x_off, img_y_off = x_off, y_off
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    rgba = np.ones((target_h, target_w, 4), dtype=np.float32)
    rgba[:, :, :3] = canvas.astype(np.float32) / 255.0
    return rgba.flatten()


def _get_display_image():
    """Get the image to display based on current view_mode. Returns RGB numpy or None.
    Does NOT silently change view_mode."""
    if ann.curr_img_idx < 0 or not ann.media_files:
        return None

    # always read original to keep shape/size updated
    path = ann.media_files[ann.curr_img_idx]
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ann.current_img_width = img_rgb.shape[1]
    ann.current_img_height = img_rgb.shape[0]
    ann.curr_img_shape = img_rgb.shape

    vm = ann.view_mode
    if vm == "overlay":
        img = ann.create_overlay_img()
        if img is not None:
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            return img
        # no mask on this frame: show original
        return img_rgb
    elif vm == "masks":
        img = ann.create_combined_mask()
        if img is not None:
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            return img
        # no mask on this frame: show black
        return np.zeros_like(img_rgb)

    # original / prompts
    return img_rgb


def load_and_show_frame():
    global tex_data
    img_rgb = _get_display_image()
    if img_rgb is None:
        return
    ann.curr_img_shape = img_rgb.shape
    tex_data = np_rgb_to_texture(img_rgb, tex_w, tex_h)
    dpg.set_value(TEX_TAG, tex_data)
    draw_overlays()
    update_status_bar()
    update_prompt_list()
    _update_frame_slider()
    _refresh_checkpoint_button()


def update_status_bar():
    if not _dpg_ready:
        return
    try:
        if ann.media_files:
            total = len(ann.media_files)
            cur = ann.curr_img_idx
            abs_idx = ann._current_abs_idx()
            label_name = ""
            if 0 <= ann.curr_label_idx < len(ann.sam_handler.labels):
                label_name = ann.sam_handler.labels[ann.curr_label_idx].name
            mode_str = "LMB=FG RMB=BG"
            vm = ann.view_mode.capitalize()
            busy = " [BUSY]" if inference_busy else ""
            # top info bar: frame/block
            dpg.set_value("info_text",
                          f"  Frame: {cur}/{total}  |  AbsFrame: {abs_idx}  |  "
                          f"Block: {ann.current_block + 1}/{ann.num_blocks}  |  "
                          f"Block Size: {ann.block_size}")
            # bottom status: label/mode
            dpg.set_value("status_text",
                          f"  Label: {label_name}  |  {mode_str}  |  "
                          f"View: {vm}{busy}")
        else:
            dpg.set_value("info_text", "  No media loaded")
            dpg.set_value("status_text", "")
    except Exception:
        pass


def _update_frame_slider():
    if not _dpg_ready:
        return
    if ann.media_files:
        try:
            dpg.configure_item("frame_slider", max_value=len(ann.media_files) - 1)
            dpg.set_value("frame_slider", ann.curr_img_idx)
        except Exception:
            pass
    _draw_timeline()


def _draw_timeline():
    """Draw timeline: mask=blue, propagated=cyan, prompt=green, checkpoint=red, current=white."""
    if not _dpg_ready or not dpg.does_item_exist("timeline_drawlist"):
        return
    try:
        dpg.delete_item("timeline_drawlist", children_only=True)
    except Exception:
        return

    n = len(ann.media_files) if ann.media_files else 1
    tl_w = dpg.get_item_width("timeline_drawlist")
    tl_h = 20
    block = ann.current_block
    TL = "timeline_drawlist"

    # background
    dpg.draw_rectangle((0, 0), (tl_w, tl_h), color=(30, 30, 30, 255),
                       fill=(30, 30, 30, 255), parent=TL)

    if n <= 1 or not ann.media_files:
        return

    def frame_x(idx):
        return int(idx / max(n - 1, 1) * (tl_w - 1))

    # layer 1: mask frames (blue fill) — from JSON files on disk
    json_dir = os.path.join(ann.project_dir, "jsons")
    if os.path.exists(json_dir):
        for fname in os.listdir(json_dir):
            if fname.endswith(".json"):
                try:
                    abs_idx = int(os.path.splitext(fname)[0])
                    local = ann._local_idx(abs_idx)
                    if 0 <= local < n:
                        x = frame_x(local)
                        dpg.draw_line((x, 0), (x, tl_h), color=(50, 70, 150, 255),
                                      parent=TL)
                except ValueError:
                    continue

    # layer 2: propagated frames per label (cyan ticks) — from label.prop_frames
    prop_frames = set()
    for label in ann.sam_handler.labels:
        for idx in label.prop_frames.get(block, set()):
            prop_frames.add(idx)
    for idx in prop_frames:
        if 0 <= idx < n:
            x = frame_x(idx)
            dpg.draw_line((x, 2), (x, tl_h - 2), color=(0, 180, 220, 255),
                          parent=TL)

    # layer 3: prompt frames (green, thicker)
    prompt_frames = set()
    for label in ann.sam_handler.labels:
        for pt in label.pts.get(block, []):
            prompt_frames.add(pt.idx)
        for box in label.boxes.get(block, []):
            prompt_frames.add(box.idx)
    for idx in prompt_frames:
        if 0 <= idx < n:
            x = frame_x(idx)
            dpg.draw_line((x, 0), (x, tl_h), color=(0, 220, 0, 255),
                          thickness=2, parent=TL)

    # layer 4: checkpoint frames (red, thicker)
    try:
        for idx in ann.sam_handler.propagation_blocks.get(block, {}):
            if 0 <= idx < n:
                x = frame_x(idx)
                dpg.draw_line((x, 0), (x, tl_h), color=(255, 50, 50, 255),
                              thickness=3, parent=TL)
    except Exception:
        pass

    # layer 5: current frame (white, prominent)
    if ann.curr_img_idx >= 0:
        x = frame_x(ann.curr_img_idx)
        dpg.draw_line((x, 0), (x, tl_h), color=(255, 255, 255, 255),
                      thickness=2, parent=TL)

    # legend (compact, right-aligned)
    lx = tl_w - 280
    ly = 3
    for color, text in [((0,220,0,255), "Prompt"),
                        ((0,180,220,255), "Propagated"),
                        ((50,70,150,255), "Mask"),
                        ((255,50,50,255), "Ckpt"),
                        ((255,255,255,255), "Current")]:
        dpg.draw_rectangle((lx, ly), (lx+8, ly+8), fill=color, color=color, parent=TL)
        dpg.draw_text((lx+10, ly-2), text, color=(200,200,200,255), size=12, parent=TL)
        lx += 56


# ── draw overlays (points, boxes) ────────────────────────────────────────────

OVERLAY_NODE = "overlay_node"


def draw_overlays():
    if not _dpg_ready:
        return
    # only clear the overlay node, never touch the base draw_image
    try:
        if dpg.does_item_exist(OVERLAY_NODE):
            dpg.delete_item(OVERLAY_NODE)
        dpg.add_draw_node(parent=DRAW_TAG, tag=OVERLAY_NODE)
    except Exception as e:
        print(f"draw_overlays error: {e}")
        return

    if ann.curr_img_idx < 0 or not ann.media_files:
        return

    # draw checkpoint indicator (always visible, regardless of view mode)
    try:
        if ann.has_propagation_block(ann.curr_img_idx):
            dpg.draw_text((10, 10), "CHECKPOINT", color=(255, 80, 80, 255),
                          size=20, parent=OVERLAY_NODE)
    except Exception:
        pass

    # only draw prompts in prompts view mode
    if ann.view_mode not in ("prompts", "original"):
        return
    block = ann.current_block
    frame_idx = ann.curr_img_idx

    for li, label in enumerate(ann.sam_handler.labels):
        hex_col = label.col
        r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
        label_color = (r, g, b, 255)

        for pt in label.pts.get(block, []):
            if pt.idx != frame_idx:
                continue
            dx, dy = img_to_display(pt.x, pt.y)
            col = (0, 255, 0, 255) if pt.pt_type == 1 else (255, 0, 0, 255)
            dpg.draw_circle((dx, dy), 6, color=col, fill=col, parent=OVERLAY_NODE)
            dpg.draw_text((dx + 8, dy - 8), str(li), color=label_color, size=14,
                          parent=OVERLAY_NODE)

        for box in label.boxes.get(block, []):
            if box.idx != frame_idx:
                continue
            dx1, dy1 = img_to_display(box.fx, box.fy)
            dx2, dy2 = img_to_display(box.x, box.y)
            dpg.draw_rectangle((dx1, dy1), (dx2, dy2), color=label_color, thickness=2,
                               parent=OVERLAY_NODE)
            dpg.draw_text((dx1, dy1 - 16), str(li), color=label_color, size=14,
                          parent=OVERLAY_NODE)


# ── label management ─────────────────────────────────────────────────────────

def load_label_presets():
    if os.path.exists(LABEL_PRESETS_FILE):
        with open(LABEL_PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_label_presets(presets):
    with open(LABEL_PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)


def refresh_label_listbox():
    items = []
    for i, label in enumerate(ann.sam_handler.labels):
        items.append(f"{i}: {label.name}")
    dpg.configure_item("label_listbox", items=items)
    if 0 <= ann.curr_label_idx < len(items):
        dpg.set_value("label_listbox", items[ann.curr_label_idx])


def update_prompt_list():
    items = []
    if ann.curr_label_idx < 0 or ann.curr_label_idx >= len(ann.sam_handler.labels):
        dpg.configure_item("prompt_listbox", items=items)
        return
    label = ann.sam_handler.labels[ann.curr_label_idx]
    block = ann.current_block
    frame_idx = ann.curr_img_idx
    for i, pt in enumerate(label.pts.get(block, [])):
        if pt.idx == frame_idx:
            tp = "FG" if pt.pt_type == 1 else "BG"
            items.append(f"Pt{i} [{tp}] ({int(pt.x)},{int(pt.y)})")
    for i, box in enumerate(label.boxes.get(block, [])):
        if box.idx == frame_idx:
            items.append(f"Box{i} ({int(box.fx)},{int(box.fy)})->({int(box.x)},{int(box.y)})")
    dpg.configure_item("prompt_listbox", items=items)


def cb_label_listbox(sender, app_data):
    if not app_data:
        return
    try:
        idx = int(app_data.split(":")[0])
        ann.set_current_label(idx)
        update_status_bar()
        update_prompt_list()
        draw_overlays()
    except (ValueError, IndexError):
        pass


def cb_add_label(sender, app_data):
    name = dpg.get_value("label_name_input").strip()
    if not name:
        return
    ann.add_label(name)
    presets = load_label_presets()
    if name not in presets:
        presets.append(name)
        save_label_presets(presets)
    dpg.set_value("label_name_input", "")
    refresh_label_listbox()
    update_status_bar()
    draw_overlays()


def cb_remove_label(sender, app_data):
    if ann.curr_label_idx < 0:
        return
    ann.remove_label(ann.curr_label_idx)
    refresh_label_listbox()
    update_status_bar()
    update_prompt_list()
    draw_overlays()


def cb_label_prev():
    if not ann.sam_handler.labels:
        return
    new_idx = (ann.curr_label_idx - 1) % len(ann.sam_handler.labels)
    ann.set_current_label(new_idx)
    refresh_label_listbox()
    update_status_bar()
    update_prompt_list()
    draw_overlays()


def cb_label_next():
    if not ann.sam_handler.labels:
        return
    new_idx = (ann.curr_label_idx + 1) % len(ann.sam_handler.labels)
    ann.set_current_label(new_idx)
    refresh_label_listbox()
    update_status_bar()
    update_prompt_list()
    draw_overlays()


def cb_label_library(sender, app_data):
    _rebuild_lib_window()


def _rebuild_lib_window():
    presets = load_label_presets()
    if dpg.does_item_exist("lib_window"):
        dpg.delete_item("lib_window")
    with dpg.window(label="Label Library", tag="lib_window", modal=True,
                    width=350, height=450, pos=(400, 150)):
        dpg.add_text("Check = add to session, X = delete from library")
        dpg.add_separator()
        if not presets:
            dpg.add_text("Library is empty.")
        for name in presets:
            already = ann.check_label_existence(name)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label=name, default_value=already,
                                 tag=f"lib_cb_{name}")
                dpg.add_button(label="X", width=25,
                               callback=_cb_lib_delete, user_data=name)
        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply", callback=_cb_lib_apply,
                           user_data=presets)
            dpg.add_button(label="Cancel",
                           callback=lambda: dpg.delete_item("lib_window"))


def _cb_lib_delete(sender, app_data, user_data):
    """Delete a label from the library (presets file), then rebuild the dialog."""
    name = user_data
    presets = load_label_presets()
    if name in presets:
        presets.remove(name)
        save_label_presets(presets)
    # also remove from current session if present
    if ann.check_label_existence(name):
        idx = ann.get_label_idx(name)
        if idx >= 0:
            ann.remove_label(idx)
    refresh_label_listbox()
    update_status_bar()
    update_prompt_list()
    draw_overlays()
    # rebuild the dialog to reflect changes
    _rebuild_lib_window()


def _cb_lib_apply(sender, app_data, user_data):
    for name in user_data:
        tag = f"lib_cb_{name}"
        if not dpg.does_item_exist(tag):
            continue
        checked = dpg.get_value(tag)
        exists = ann.check_label_existence(name)
        if checked and not exists:
            ann.add_label(name)
        elif not checked and exists:
            idx = ann.get_label_idx(name)
            if idx >= 0:
                ann.remove_label(idx)
    dpg.delete_item("lib_window")
    refresh_label_listbox()
    update_status_bar()
    update_prompt_list()
    draw_overlays()


# ── mouse interaction ────────────────────────────────────────────────────────

def _get_local_mouse_in_drawlist():
    """Get mouse position in drawlist-local coordinates."""
    try:
        # get_mouse_pos(local=False) = global screen coords
        gx, gy = dpg.get_mouse_pos(local=False)
        dl_min = dpg.get_item_rect_min(DRAW_TAG)
        return gx - dl_min[0], gy - dl_min[1]
    except Exception:
        return -1, -1


def cb_mouse_click(sender, app_data):
    """Left click = foreground point (pt_type=1)."""
    if ann.curr_img_idx < 0:
        return
    if ann.curr_label_idx < 0:
        _show_progress("Select a label first.", 0)
        return
    lx, ly = _get_local_mouse_in_drawlist()
    if not is_in_image(lx, ly):
        return
    ix, iy = display_to_img(lx, ly)
    ix = max(0, min(ix, img_orig_w - 1))
    iy = max(0, min(iy, img_orig_h - 1))
    ann.add_point_prompt_to_current_label(ix, iy, 1, ann.curr_img_idx)
    draw_overlays()
    update_prompt_list()


def _draw_temp_box():
    """Called every frame. Detects right-button drag for box drawing, or click for bg point."""
    global dragging, drag_start, _right_click_start
    if drag_start is None:
        return
    # right button released?
    if not dpg.is_mouse_button_down(dpg.mvMouseButton_Right):
        if dragging:
            _finish_drag()
        else:
            # no drag happened → it was a right click → add background point
            if ann.curr_img_idx >= 0 and ann.curr_label_idx >= 0:
                ix, iy = drag_start
                ix = max(0, min(ix, img_orig_w - 1))
                iy = max(0, min(iy, img_orig_h - 1))
                ann.add_point_prompt_to_current_label(ix, iy, 0, ann.curr_img_idx)  # 0 = background
                draw_overlays()
                update_prompt_list()
            drag_start = None
            _right_click_start = None
        return
    # button still held — check if moved enough to start drag
    if not dragging and _right_click_start is not None:
        lx, ly = _get_local_mouse_in_drawlist()
        dx = abs(lx - _right_click_start[0])
        dy = abs(ly - _right_click_start[1])
        if dx > 5 or dy > 5:
            dragging = True  # start drag mode
    if not dragging:
        return
    if ann.curr_img_idx < 0 or ann.curr_label_idx < 0:
        return
    lx, ly = _get_local_mouse_in_drawlist()
    ix = max(0, min(display_to_img(lx, ly)[0], img_orig_w - 1))
    iy = max(0, min(display_to_img(lx, ly)[1], img_orig_h - 1))
    draw_overlays()
    dx1, dy1 = img_to_display(drag_start[0], drag_start[1])
    dx2, dy2 = img_to_display(ix, iy)
    dpg.draw_rectangle((dx1, dy1), (dx2, dy2), color=(255, 255, 0, 255),
                       thickness=2, parent=OVERLAY_NODE)


def cb_mouse_down_right(sender, app_data):
    """Right mouse button pressed — record start position for drag or bg point."""
    global dragging, drag_start, _right_click_start
    if ann.curr_img_idx < 0 or ann.curr_label_idx < 0:
        return
    lx, ly = _get_local_mouse_in_drawlist()
    if is_in_image(lx, ly):
        drag_start = display_to_img(lx, ly)
        _right_click_start = (lx, ly)  # save display coords to detect drag vs click
        dragging = False  # don't start dragging yet, wait for movement


def _finish_drag():
    """Finalize the bounding box on right button release."""
    global dragging, drag_start, _right_click_start
    if not dragging or drag_start is None:
        dragging = False
        drag_start = None
        _right_click_start = None
        return
    if ann.curr_img_idx < 0 or ann.curr_label_idx < 0:
        dragging = False
        drag_start = None
        return
    lx, ly = _get_local_mouse_in_drawlist()
    ix, iy = display_to_img(lx, ly)
    ix = max(0, min(ix, img_orig_w - 1))
    iy = max(0, min(iy, img_orig_h - 1))
    fx, fy = drag_start
    if abs(ix - fx) > 5 and abs(iy - fy) > 5:
        x1, x2 = min(fx, ix), max(fx, ix)
        y1, y2 = min(fy, iy), max(fy, iy)
        ann.add_box_prompt_to_current_label(x1, y1, x2, y2, 1, ann.curr_img_idx)
    dragging = False
    drag_start = None
    _right_click_start = None
    draw_overlays()
    update_prompt_list()


# ── SAM2 inference (Step 4) ──────────────────────────────────────────────────

def _set_busy(busy):
    global inference_busy
    inference_busy = busy
    _update_inference_buttons()
    update_status_bar()


def _update_inference_buttons():
    for tag in ("btn_single", "btn_forward", "btn_backward", "btn_all"):
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, enabled=not inference_busy)


def _show_progress(msg, pct=0):
    if not _dpg_ready:
        return
    try:
        dpg.set_value("progress_bar", pct / 100.0)
        dpg.set_value("progress_text", msg)
    except Exception:
        pass


def _progress_callback(msg, pct=0):
    _show_progress(msg, pct)


def _run_inference(task_fn):
    """Run an inference task in a background thread with GPU lock."""
    def wrapper():
        with inference_lock:
            _set_busy(True)
            try:
                task_fn()
            except Exception as e:
                print(f"Inference error: {e}")
                import traceback; traceback.print_exc()
                ann.reset_view_mode()
                _show_progress(f"Error: {e}", 0)
            finally:
                _set_busy(False)
    threading.Thread(target=wrapper, daemon=True).start()


def cb_single(sender, app_data):
    """Single frame mask generation."""
    if inference_busy:
        _show_progress("Inference busy, please wait.", 0)
        return
    if not ann.model_status():
        _show_progress("Model not loaded! Click 'Load Model' first.", 0)
        return
    if ann.curr_label_idx < 0:
        _show_progress("No label selected. Add a label first.", 0)
        return
    if not ann.has_prompts(ann.current_block):
        _show_progress("No prompts on this block. Click on the image to add points.", 0)
        return

    def task():
        _show_progress("Generating mask...", 10)
        success, prop_frames = ann.generate_mask()
        if success:
            _show_progress("Applying masks...", 50)
            ann.apply_masks(_progress_callback, is_single=True)
            ann.view_mode = "overlay"
            load_and_show_frame()
            _show_progress("Single done.", 100)
        else:
            # check if current frame has prompts
            cur_pts = sum(1 for l in ann.sam_handler.labels
                         for p in l.pts.get(ann.current_block, []) if p.idx == ann.curr_img_idx)
            cur_boxes = sum(1 for l in ann.sam_handler.labels
                          for b in l.boxes.get(ann.current_block, []) if b.idx == ann.curr_img_idx)
            if cur_pts == 0 and cur_boxes == 0:
                _show_progress(f"Frame {ann.curr_img_idx} has no prompts. "
                               f"Navigate to a frame with prompts (green on timeline).", 0)
            else:
                _show_progress(f"Generation failed on frame {ann.curr_img_idx}.", 0)
    _run_inference(task)


def _check_inference_ready(need_prompts=True):
    """Common pre-checks for all inference operations. Returns True if ready."""
    if inference_busy:
        _show_progress("Inference busy, please wait.", 0)
        return False
    if not ann.model_status():
        _show_progress("Model not loaded! Click 'Load Model' first.", 0)
        return False
    if not ann.has_frames():
        _show_progress("No frames loaded.", 0)
        return False
    if need_prompts and not ann.has_prompts(ann.current_block):
        _show_progress("No prompts on this block.", 0)
        return False
    return True


def _init_tracking_if_needed():
    """Initialize SAM2 tracking if not already done. Returns True on success."""
    if not ann.sam_handler.tracking_init:
        _show_progress("Initializing tracking...", 5)
        ok = ann.initialize_tracking(_progress_callback)
        if not ok:
            _show_progress("Tracking init failed.", 0)
            return False
    return True


def cb_forward(sender, app_data):
    if not _check_inference_ready():
        return
    def task():
        if not _init_tracking_if_needed():
            return
        _show_progress("Propagating forward...", 10)
        success, frames = ann.propagate(1, _progress_callback)
        if success:
            ann.apply_masks(_progress_callback)
            ann.view_mode = "overlay"
            load_and_show_frame()
            _show_progress("Forward done.", 100)
        else:
            _show_progress("Forward propagation failed.", 0)
    _run_inference(task)


def cb_backward(sender, app_data):
    if not _check_inference_ready():
        return
    def task():
        if not _init_tracking_if_needed():
            return
        _show_progress("Propagating backward...", 10)
        success, frames = ann.propagate(-1, _progress_callback)
        if success:
            ann.apply_masks(_progress_callback)
            ann.view_mode = "overlay"
            load_and_show_frame()
            _show_progress("Backward done.", 100)
        else:
            _show_progress("Backward propagation failed.", 0)
    _run_inference(task)


def cb_all(sender, app_data):
    if not _check_inference_ready():
        return
    def task():
        if not _init_tracking_if_needed():
            return
        _show_progress("Propagating all...", 10)
        success, frames = ann.propagate(0, _progress_callback)
        if success:
            ann.apply_masks(_progress_callback)
            ann.view_mode = "overlay"
            load_and_show_frame()
            _show_progress("All done.", 100)
        else:
            _show_progress("Propagation failed.", 0)
    _run_inference(task)


def cb_propagate_range(sender, app_data):
    if not _check_inference_ready():
        return
    def task():
        if not _init_tracking_if_needed():
            return
        _show_progress("Propagate Range (batch)...", 10)
        success, frames = ann.propagate_batch(_progress_callback)
        if success:
            ann.apply_masks(_progress_callback)
            ann.view_mode = "overlay"
            load_and_show_frame()
            _show_progress("Propagate Range done.", 100)
        else:
            _show_progress("Propagate Range failed.", 0)
    _run_inference(task)


# ── checkpoint (propagation block) ───────────────────────────────────────────

def cb_toggle_checkpoint(sender, app_data):
    """Add or remove a propagation checkpoint at the current frame."""
    if ann.curr_img_idx < 0 or not ann.media_files:
        return
    idx = ann.curr_img_idx
    if ann.has_propagation_block(idx):
        ann.remove_propagation_block(idx)
    else:
        ann.add_propagation_block(idx)
    _refresh_checkpoint_button()
    draw_overlays()


def _refresh_checkpoint_button():
    """Update the checkpoint button text based on current frame."""
    if not _dpg_ready or not dpg.does_item_exist("btn_checkpoint"):
        return
    try:
        if ann.curr_img_idx >= 0 and ann.has_propagation_block(ann.curr_img_idx):
            dpg.configure_item("btn_checkpoint", label="Remove Checkpoint")
        else:
            dpg.configure_item("btn_checkpoint", label="Add Checkpoint")
    except Exception:
        pass


MODEL_OPTIONS = {
    "Large":  ("sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt"),
    "Base+":  ("sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "Small":  ("sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt"),
    "Tiny":   ("sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt"),
}


def cb_load_model(sender, app_data):
    """Load SAM2 model in background thread."""
    if inference_busy:
        _show_progress("Inference busy.", 0)
        return

    # get selected model size
    model_name = "Large"
    if _dpg_ready and dpg.does_item_exist("model_combo"):
        model_name = dpg.get_value("model_combo")
    cfg, ckpt = MODEL_OPTIONS.get(model_name, MODEL_OPTIONS["Large"])

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    ckpt_path = os.path.join(base, ckpt)
    if not os.path.exists(ckpt_path):
        _show_progress(f"Checkpoint not found: {ckpt}. "
                       f"Run: python download_checkpoints.py --only {model_name.lower().rstrip('+')}", 0)
        return

    def task():
        import torch, gc
        # unload previous model if loaded
        if ann.model_status():
            _show_progress("Unloading previous model...", 5)
            if ann.sam_handler.tracking_init and ann.sam_handler.inference_state:
                ann.sam_handler.predictor.reset_state(ann.sam_handler.inference_state)
            ann.sam_handler.inference_state = None
            ann.sam_handler.tracking_init = False
            del ann.sam_handler.predictor
            ann.sam_handler.model_loaded = False
            ann.sam_handler.model_loading = False
            ann.sam_handler.current_stage = 0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ann.sam_handler.model_cfg_path = cfg
        ann.sam_handler.ckpt_path = ckpt_path
        _show_progress(f"Loading SAM2 {model_name}...", 10)
        ok = ann.load_model(lambda msg: _show_progress(msg, 50))
        if ok:
            _show_progress(f"SAM2 {model_name} loaded.", 100)
        else:
            _show_progress(f"Failed to load {model_name}.", 0)
    _run_inference(task)


# ── view mode (Step 5) ───────────────────────────────────────────────────────

def cb_set_view_mode(mode_idx):
    ann.set_view_mode(mode_idx)
    load_and_show_frame()


# ── navigation (Step 6) ─────────────────────────────────────────────────────

def cb_frame_slider(sender, app_data):
    idx = int(app_data)
    if 0 <= idx < len(ann.media_files):
        ann.set_img(idx)
        load_and_show_frame()




def _get_prompt_frames():
    """Get set of local frame indices that have manual prompts (points/boxes)."""
    block = ann.current_block
    frames = set()
    for label in ann.sam_handler.labels:
        for pt in label.pts.get(block, []):
            frames.add(pt.idx)
        for box in label.boxes.get(block, []):
            frames.add(box.idx)
    return frames


def cb_jump_prev_prompt():
    """Q: jump to previous frame with manual prompts."""
    if ann.curr_img_idx <= 0:
        return
    prompt_frames = _get_prompt_frames()
    for i in range(ann.curr_img_idx - 1, -1, -1):
        if i in prompt_frames:
            ann.set_img(i)
            load_and_show_frame()
            return


def cb_jump_next_prompt():
    """E: jump to next frame with manual prompts."""
    if not ann.media_files:
        return
    prompt_frames = _get_prompt_frames()
    for i in range(ann.curr_img_idx + 1, len(ann.media_files)):
        if i in prompt_frames:
            ann.set_img(i)
            load_and_show_frame()
            return


# ── keyboard ─────────────────────────────────────────────────────────────────

def _is_input_focused():
    for tag in ("label_name_input", "block_size_input", "session_name_input"):
        if dpg.does_item_exist(tag) and dpg.is_item_focused(tag):
            return True
    return False


def cb_key_handler(sender, app_data):
    key = app_data
    pass  # pt_mode removed

    if _is_input_focused():
        return

    shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)

    if shift:
        # Shift+1/2/3/4 → view mode
        if key == dpg.mvKey_1:
            cb_set_view_mode(0)  # original
            return
        elif key == dpg.mvKey_2:
            cb_set_view_mode(1)  # prompts
            return
        elif key == dpg.mvKey_3:
            cb_set_view_mode(2)  # overlay
            return
        elif key == dpg.mvKey_4:
            cb_set_view_mode(3)  # masks
            return

    if key == dpg.mvKey_A:
        cb_prev_frame()
    elif key == dpg.mvKey_D:
        cb_next_frame()
    elif key == dpg.mvKey_W:
        cb_label_prev()
    elif key == dpg.mvKey_S:
        cb_label_next()
    elif key == dpg.mvKey_P:
        pass  # P key no longer needed: LMB=FG, RMB=BG
    elif key == dpg.mvKey_C:
        cb_clear_prompts()
    elif key == dpg.mvKey_Q:
        cb_jump_prev_prompt()
    elif key == dpg.mvKey_E:
        cb_jump_next_prompt()
    elif key == dpg.mvKey_Delete:
        cb_delete_selected_prompt()
    elif key == dpg.mvKey_F11:
        cb_toggle_fullscreen()
    elif key == dpg.mvKey_Escape:
        dpg.focus_item("primary_window")


def cb_prev_frame():
    ann.prev_img()
    load_and_show_frame()


def cb_next_frame():
    ann.next_img()
    load_and_show_frame()


def cb_clear_prompts():
    if ann.curr_label_idx < 0:
        return
    label = ann.sam_handler.labels[ann.curr_label_idx]
    block = ann.current_block
    frame_idx = ann.curr_img_idx
    if block in label.pts:
        label.pts[block] = [p for p in label.pts[block] if p.idx != frame_idx]
    if block in label.boxes:
        label.boxes[block] = [b for b in label.boxes[block] if b.idx != frame_idx]
    draw_overlays()
    update_prompt_list()


def cb_delete_selected_prompt():
    if ann.curr_label_idx < 0:
        return
    val = dpg.get_value("prompt_listbox")
    if not val:
        return
    if val.startswith("Pt"):
        try:
            idx = int(val.split("[")[0].replace("Pt", ""))
            ann.delete_selected_point(idx)
        except (ValueError, IndexError):
            pass
    elif val.startswith("Box"):
        try:
            idx = int(val.split(" ")[0].replace("Box", ""))
            ann.delete_selected_box(idx)
        except (ValueError, IndexError):
            pass
    draw_overlays()
    update_prompt_list()


# ── media loading ────────────────────────────────────────────────────────────

# store file_path for block switching
_loaded_file_path = ""
_max_frame = 0


class _ProgressVar:
    """Mimic tkinter DoubleVar for annotator.extract_frames compatibility."""
    def __init__(self):
        self.val = 0
    def set(self, v):
        self.val = v
        _show_progress(f"Extracting frames... {int(v)}%", v)


def _do_load_media(file_path, block_size):
    global _loaded_file_path, _max_frame
    _loaded_file_path = file_path
    ann.block_size = block_size

    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    is_video = ext in ("mp4", "avi", "mov", "mkv")
    is_dir = os.path.isdir(file_path)

    # set session name from media
    if is_video:
        media_basename = os.path.splitext(os.path.basename(file_path))[0]
    elif is_dir:
        # folder selected: if it's named "frames", use parent dir name (= video name)
        folder_name = os.path.basename(file_path)
        if folder_name == "frames":
            media_basename = os.path.basename(os.path.dirname(file_path))
        else:
            media_basename = folder_name
    else:
        parent = os.path.dirname(file_path)
        folder_name = os.path.basename(parent)
        if folder_name == "frames":
            media_basename = os.path.basename(os.path.dirname(parent))
        else:
            media_basename = folder_name or os.path.splitext(os.path.basename(file_path))[0]
    if media_basename:
        ann.set_session_name(media_basename)
        if _dpg_ready:
            try:
                dpg.set_value("session_name_input", media_basename)
            except Exception:
                pass

    # get total frames
    if is_video:
        total = ann.get_frame_count(file_path)
    elif is_dir:
        total = ann.get_frame_count_dir(file_path)
    else:
        total = ann.get_frame_count_dir(os.path.dirname(file_path))
    _max_frame = total

    num_blocks = max(1, math.ceil(total / block_size))
    ann.set_num_blocks(num_blocks)
    ann.set_current_block(0)

    # only clear sam_extract_dir (SAM2 needs clean dir), preserve frames/ (pre-extract reuse)
    import shutil as _shutil
    _shutil.rmtree(ann.sam_extract_dir, ignore_errors=True)

    result = ann.load_main_folder_unified(file_path, block_size)

    if result == 1:
        # video mode: extract frames for current block
        _extract_block_frames(0, block_size, total)
    elif result == 0:
        # image folder mode: frames already loaded
        pass
    else:
        _show_progress("Failed to load media.", 0)
        return

    if ann.media_files:
        ann.set_img(0)


def _extract_block_frames(block_idx, block_size, total):
    """Extract frames for a given block, with progress updates."""
    start = block_idx * block_size
    end = min(start + block_size, total)
    _show_progress(f"Extracting frames {start}-{end}...", 0)

    # check if frames already exist on disk (smart reuse)
    frames_dir = ann.frames_dir
    expected = os.path.join(frames_dir, f"{start:06d}.jpg")
    if os.path.exists(expected):
        # frames already exist, reuse them
        _show_progress("Frames found on disk, loading...", 50)
        ann.extract_dir = frames_dir
        import re
        all_files = sorted(
            [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
             if f.endswith(".jpg")],
            key=lambda p: int(re.search(r'\d+', os.path.basename(p)).group())
            if re.search(r'\d+', os.path.basename(p)) else 0
        )
        # filter to current block range
        block_files = []
        for f in all_files:
            try:
                idx = int(os.path.splitext(os.path.basename(f))[0])
                if start <= idx < end:
                    block_files.append(f)
            except ValueError:
                continue
        if block_files:
            ann.media_files = block_files
            ann.curr_img_idx = -1
            # also prepare sam_extract_dir
            import shutil
            shutil.rmtree(ann.sam_extract_dir, ignore_errors=True)
            os.makedirs(ann.sam_extract_dir, exist_ok=True)
            for f in block_files:
                shutil.copy(f, os.path.join(ann.sam_extract_dir, os.path.basename(f)))
            # extra frame
            extra_idx = end
            extra_file = os.path.join(frames_dir, f"{extra_idx:06d}.jpg")
            if os.path.exists(extra_file):
                ann.extra_frame_path[ann.current_block] = extra_file
                shutil.copy(extra_file, os.path.join(ann.sam_extract_dir, os.path.basename(extra_file)))
            _show_progress(f"Loaded {len(block_files)} frames from disk.", 100)
            return

    # no cached frames, extract from video
    pv = _ProgressVar()
    ann.extract_frames(start, end, 1, pv)
    n = len(ann.media_files)
    _show_progress(f"Extracted {n} frames.", 100)


def _on_file_selected(sender, app_data):
    """DPG file dialog callback for Load Video/Image."""
    selections = app_data.get("selections", {})
    if not selections:
        return
    file_path = list(selections.values())[0]
    block_size = int(dpg.get_value("block_size_input"))

    def do_load():
        global _media_load_done
        _do_load_media(file_path, block_size)
        _media_load_done = True
    threading.Thread(target=do_load, daemon=True).start()


def _on_folder_selected(sender, app_data):
    """DPG file dialog callback for Load Folder."""
    folder = app_data.get("file_path_name", "")
    if not folder or not os.path.isdir(folder):
        return
    block_size = int(dpg.get_value("block_size_input"))

    def do_load():
        global _media_load_done
        _do_load_media(folder, block_size)
        _media_load_done = True
    threading.Thread(target=do_load, daemon=True).start()


def cb_load_media(sender, app_data):
    tag = "file_dialog_media"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.file_dialog(label="Select video or image file",
                         callback=_on_file_selected,
                         width=800, height=500, tag=tag,
                         default_path=os.getcwd()):
        dpg.add_file_extension(".mp4", color=(0, 255, 0))
        dpg.add_file_extension(".avi", color=(0, 255, 0))
        dpg.add_file_extension(".mov", color=(0, 255, 0))
        dpg.add_file_extension(".mkv", color=(0, 255, 0))
        dpg.add_file_extension(".jpg", color=(0, 200, 255))
        dpg.add_file_extension(".jpeg", color=(0, 200, 255))
        dpg.add_file_extension(".png", color=(0, 200, 255))
        dpg.add_file_extension(".*")


def cb_load_folder(sender, app_data):
    tag = "file_dialog_folder"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.file_dialog(label="Select image folder",
                         callback=_on_folder_selected,
                         directory_selector=True,
                         width=800, height=500, tag=tag,
                         default_path=os.getcwd()):
        pass


def cb_block_prev(sender, app_data):
    if ann.current_block <= 0:
        return
    ann.set_current_block(ann.current_block - 1)
    _reload_block()


def cb_block_next(sender, app_data):
    if ann.current_block >= ann.num_blocks - 1:
        return
    ann.set_current_block(ann.current_block + 1)
    _reload_block()


def _reload_block():
    global _max_frame
    if not ann.video_name and not ann.media_path:
        return
    block_size = ann.block_size

    # reset tracking state when switching blocks (same as source project)
    target_block = ann.current_block
    ann.reset_media()
    ann.set_current_block(target_block)

    if ann.video_name and os.path.exists(ann.video_name):
        if _max_frame <= 0:
            _max_frame = ann.get_frame_count(ann.video_name)
        ann.load_main_folder_unified(ann.video_name, block_size)
        _extract_block_frames(ann.current_block, block_size, _max_frame)
    else:
        folder = ann.media_path
        if folder and os.path.isdir(folder):
            ann.process_image_folder(folder, ann.current_block * block_size,
                                     (ann.current_block + 1) * block_size)
    if ann.media_files:
        ann.set_img(0)
    load_and_show_frame()


# ── session management (Step 7) ──────────────────────────────────────────────

def cb_save_session(sender, app_data):
    if not ann.media_files:
        _show_progress("No media loaded.", 0)
        return
    os.makedirs(ann.project_dir, exist_ok=True)
    session_name = ann.session_name or "session"
    path = os.path.join(ann.project_dir, f"{session_name}.pkl")
    data = ann.compress_to_dict()
    with open(path, "wb") as f:
        pickle.dump(data, f)
    _show_progress(f"Saved: {path}", 100)


def _on_session_selected(sender, app_data):
    """DPG file dialog callback for Load Session."""
    selections = app_data.get("selections", {})
    if not selections:
        return
    path = list(selections.values())[0]

    def do_load():
        global _max_frame
        import shutil as _shutil
        import re

        _show_progress("Loading session...", 10)
        with open(path, "rb") as f:
            data = pickle.load(f)
        ann.load_from_dict(data)

        _show_progress("Restoring frames...", 30)
        temp_img_idx = ann.curr_img_idx
        ann.read_frames = -1
        block_size = ann.block_size
        start_frame = ann.current_block * block_size

        # check if frames exist on disk (same logic as source project)
        frames_dir = ann.frames_dir
        expected_frame = os.path.join(frames_dir, f"{start_frame:06d}.jpg")
        frames_exist = os.path.exists(expected_frame)

        if frames_exist:
            # frames on disk: rebuild media_files + sam_extract_dir
            _show_progress("Frames found, rebuilding SAM temp...", 50)
            ann.extract_dir = frames_dir

            # scan disk for current block frames
            end_frame = start_frame + block_size
            if ann.video_name and os.path.exists(ann.video_name):
                _max_frame = ann.get_frame_count(ann.video_name)
                end_frame = min(end_frame, _max_frame)
            else:
                total_on_disk = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
                _max_frame = total_on_disk
                end_frame = min(end_frame, total_on_disk)

            extracted = []
            for i in range(start_frame, end_frame):
                fp = os.path.join(frames_dir, f"{i:06d}.jpg")
                if os.path.exists(fp):
                    extracted.append(fp)
            ann.media_files = extracted

            # rebuild sam_extract_dir (SAM2 needs clean dir with current block frames)
            _shutil.rmtree(ann.sam_extract_dir, ignore_errors=True)
            os.makedirs(ann.sam_extract_dir, exist_ok=True)
            for fp in extracted:
                dst = os.path.join(ann.sam_extract_dir, os.path.basename(fp))
                if not os.path.exists(dst):
                    _shutil.copy2(fp, dst)
            # extra frame for propagation
            extra_fp = os.path.join(frames_dir, f"{end_frame:06d}.jpg")
            if os.path.exists(extra_fp):
                ann.extra_frame_path[ann.current_block] = extra_fp
                dst = os.path.join(ann.sam_extract_dir, os.path.basename(extra_fp))
                if not os.path.exists(dst):
                    _shutil.copy2(extra_fp, dst)

        elif ann.video_name and os.path.exists(ann.video_name):
            # frames missing but video exists: re-extract
            _show_progress("Re-extracting frames from video...", 50)
            _max_frame = ann.get_frame_count(ann.video_name)
            ann.load_main_folder_unified(ann.video_name, block_size)
            end_frame = min(start_frame + block_size, _max_frame)
            ann.extract_frames(start_frame, end_frame, 1, _ProgressVar())

        elif ann.media_path and os.path.isdir(ann.media_path):
            # image folder mode: reload from source folder
            _show_progress("Reloading image folder...", 50)
            ann.load_main_folder_unified(ann.media_path, block_size)
            _max_frame = ann.get_frame_count_dir(ann.media_path)

        else:
            _show_progress("Cannot find frames or source video.", 0)
            return

        # restore frame position
        if ann.media_files:
            ann.set_img(min(temp_img_idx, len(ann.media_files) - 1))

        # reset SAM2 state (predictor can't be pickled, must reload)
        ann.sam_handler.tracking_init = False
        ann.sam_handler.inference_state = None
        ann.sam_handler.model_loaded = False
        ann.sam_handler.model_loading = False
        ann.sam_handler.current_stage = 0

        # ensure a label is selected
        if ann.curr_label_idx < 0 and len(ann.sam_handler.labels) > 0:
            ann.set_current_label(0)

        # force prompts view so annotations are visible
        ann.view_mode = "prompts"
        ann.mode = "prompts"

        # signal main thread to refresh UI
        global _session_load_done
        _session_load_done = True

    threading.Thread(target=do_load, daemon=True).start()


def cb_load_session(sender, app_data):
    tag = "file_dialog_session"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.file_dialog(label="Load Session",
                         callback=_on_session_selected,
                         width=800, height=500, tag=tag,
                         default_path=os.path.join(os.getcwd(), "projects")):
        dpg.add_file_extension(".pkl", color=(255, 200, 0))


def cb_reset_session(sender, app_data):
    ann.reset()
    refresh_label_listbox()
    update_prompt_list()
    dpg.set_value(TEX_TAG, np.zeros(tex_w * tex_h * 4, dtype=np.float32))
    draw_overlays()
    update_status_bar()
    _show_progress("Session reset.", 0)


def cb_session_name(sender, app_data):
    name = app_data.strip()
    if name:
        ann.set_session_name(name)


# ── export (Step 8) ──────────────────────────────────────────────────────────

def cb_export_annotations(sender, app_data):
    json_dir = os.path.join(ann.project_dir, "jsons")
    has_jsons = os.path.exists(json_dir) and any(f.endswith(".json") for f in os.listdir(json_dir))
    if not has_jsons and not ann.masks:
        _show_progress("No annotations to export. Run inference first.", 0)
        return

    def task():
        _show_progress("Exporting...", 5)
        # if there are still masks in memory (shouldn't normally happen), export them
        if ann.masks:
            os.makedirs(os.path.join(ann.project_dir, "jsons"), exist_ok=True)
            ann.export_jsons(lambda tp, step, total: _show_progress(
                f"Exporting {tp}: {step+1}/{total}", (step / max(1, total)) * 80 + 10))
            ann.masks = {}
        _show_progress("Generating verification overlays...", 90)
        os.makedirs(os.path.join(ann.project_dir, "overlays"), exist_ok=True)
        ann.export_verification_overlays(10)
        _show_progress(f"Export done: {ann.project_dir}", 100)
    _run_inference(task)


def _do_export_verify_video(video_path):
    """Actually run the verify video export with a known video path."""
    def task():
        _show_progress("Generating verify video...", 5)
        ok, result = ann.export_verify_video(
            video_path=video_path,
            progress_callback=lambda cur, total: _show_progress(
                f"Verify video: {cur}/{total}", (cur / max(1, total)) * 100))
        if ok:
            _show_progress(f"Verify video: {result}", 100)
        else:
            _show_progress(f"Failed: {result}", 0)
    _run_inference(task)


def _on_verify_video_selected(sender, app_data):
    """User picked the source video for verify video export."""
    selections = app_data.get("selections", {})
    if not selections:
        return
    video_path = list(selections.values())[0]
    _do_export_verify_video(video_path)


def cb_export_verify_video(sender, app_data):
    json_dir = os.path.join(ann.project_dir, "jsons")
    if not os.path.exists(json_dir) or not any(f.endswith(".json") for f in os.listdir(json_dir)):
        _show_progress("No JSON annotations found. Run inference first.", 0)
        return
    # find source video: video_name (video mode) or _preextract_source_video (pre-extract mode)
    video_path = ann.video_name
    if not video_path or not os.path.exists(video_path):
        video_path = getattr(ann, '_preextract_source_video', '')
    if not video_path or not os.path.exists(video_path):
        # ask user to select the source video
        tag = "file_dialog_verify_video"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        with dpg.file_dialog(label="Select source video for verify video",
                             callback=_on_verify_video_selected,
                             width=800, height=500, tag=tag,
                             default_path=os.getcwd()):
            dpg.add_file_extension(".mp4", color=(0, 255, 0))
            dpg.add_file_extension(".avi", color=(0, 255, 0))
            dpg.add_file_extension(".mov", color=(0, 255, 0))
            dpg.add_file_extension(".mkv", color=(0, 255, 0))
        return

    _do_export_verify_video(video_path)


def _on_preextract_video_selected(sender, app_data):
    """DPG callback: user picked a video for pre-extraction."""
    selections = app_data.get("selections", {})
    if not selections:
        return
    video_path = list(selections.values())[0]

    def task():
        # validate
        if not os.path.exists(video_path):
            _show_progress("Video not found.", 0)
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _show_progress("Cannot open video.", 0)
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            _show_progress("Video has 0 frames.", 0)
            return

        # output dir: projects/{video_name}/frames/
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join("projects", video_name, "frames")

        # check if already extracted
        if os.path.exists(out_dir):
            existing = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
            if len(existing) > 0:
                _show_progress(f"Frames already exist ({len(existing)} files). Use Load Folder to open.", 100)
                return

        os.makedirs(out_dir, exist_ok=True)

        # disk space check
        try:
            import shutil as _shutil
            est_gb = total_frames * 400 / 1024 / 1024
            free_gb = _shutil.disk_usage(os.path.dirname(os.path.abspath(video_path))).free / (1024**3)
            if est_gb > free_gb * 0.9:
                _show_progress(f"Warning: need ~{est_gb:.1f}GB, only {free_gb:.1f}GB free.", 0)
        except Exception:
            pass

        ffmpeg = ann._get_ffmpeg_exe()
        if ffmpeg:
            _show_progress(f"Pre-extracting {total_frames} frames (ffmpeg)...", 0)
            cmd = [
                ffmpeg, "-y",
                "-i", video_path,
                "-vsync", "0",
                "-frames:v", str(total_frames),
                "-qscale:v", "2",
                "-start_number", "0",
                os.path.join(out_dir, "%06d.jpg"),
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            n_done = 0
            while proc.poll() is None:
                time.sleep(0.3)
                # count extracted frames incrementally
                while n_done < total_frames and os.path.exists(
                        os.path.join(out_dir, f"{n_done:06d}.jpg")):
                    n_done += 1
                pct = min(99, (n_done / total_frames) * 100)
                _show_progress(f"Extracting: {n_done}/{total_frames} ({pct:.0f}%)", pct)
            # final count
            while n_done < total_frames and os.path.exists(
                    os.path.join(out_dir, f"{n_done:06d}.jpg")):
                n_done += 1
            if proc.returncode == 0 and n_done > 0:
                _show_progress(f"Done: {n_done} frames in {out_dir}. Use Load Folder to open.", 100)
            else:
                _show_progress(f"ffmpeg failed (rc={proc.returncode}), got {n_done} frames.", 0)
        else:
            _show_progress(f"Pre-extracting {total_frames} frames (cv2)...", 0)
            cap = cv2.VideoCapture(video_path)
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), frame)
                if i % 50 == 0:
                    _show_progress(f"Extracting: {i}/{total_frames}",
                                   (i / total_frames) * 100)
            cap.release()
            n_done = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
            _show_progress(f"Done: {n_done} frames. Use Load Folder to open.", 100)

        # store source video path for verify-video export
        ann._preextract_source_video = os.path.abspath(video_path)
    _run_inference(task)


def cb_pre_extract(sender, app_data):
    """Open file dialog to select video for pre-extraction."""
    tag = "file_dialog_preextract"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.file_dialog(label="Select video to pre-extract",
                         callback=_on_preextract_video_selected,
                         width=800, height=500, tag=tag,
                         default_path=os.getcwd()):
        dpg.add_file_extension(".mp4", color=(0, 255, 0))
        dpg.add_file_extension(".avi", color=(0, 255, 0))
        dpg.add_file_extension(".mov", color=(0, 255, 0))
        dpg.add_file_extension(".mkv", color=(0, 255, 0))


# ── fullscreen toggle (Step 9) ───────────────────────────────────────────────

_fullscreen = False

def cb_toggle_fullscreen():
    global _fullscreen
    _fullscreen = not _fullscreen
    dpg.toggle_viewport_fullscreen()


# ── build UI ─────────────────────────────────────────────────────────────────

def _setup_font():
    """Register a larger default font with Chinese support."""
    with dpg.font_registry():
        # Try to find a Chinese-capable font on Windows
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
            "C:/Windows/Fonts/simhei.ttf",     # SimHei
            "C:/Windows/Fonts/simsun.ttc",     # SimSun
        ]
        font_path = None
        for fp in font_paths:
            if os.path.exists(fp):
                font_path = fp
                break

        if font_path:
            with dpg.font(font_path, 18) as default_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            dpg.bind_font(default_font)
        else:
            # fallback: just use default font at larger size
            pass


PANEL_W = 250  # left panel width


def build_ui():
    global tex_w, tex_h, tex_data

    dpg.create_context()
    dpg.create_viewport(title="SAMannot-DPG", width=1536, height=860)

    _setup_font()

    tex_w = DISPLAY_W
    tex_h = DISPLAY_H
    tex_data = np.zeros(tex_w * tex_h * 4, dtype=np.float32)

    with dpg.texture_registry():
        dpg.add_dynamic_texture(width=tex_w, height=tex_h, default_value=tex_data,
                                tag=TEX_TAG)

    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=cb_key_handler)
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=cb_mouse_click)
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=cb_mouse_down_right)

    with dpg.window(tag="primary_window"):
        # ── toolbar ──
        with dpg.group(horizontal=True):
            dpg.add_button(label="Pre-extract", callback=cb_pre_extract)
            dpg.add_button(label="Load Folder", callback=cb_load_folder)
            dpg.add_spacer(width=10)
            dpg.add_text("Block Size:")
            dpg.add_input_int(tag="block_size_input", default_value=200, width=80,
                              min_value=10, max_value=10000, min_clamped=True,
                              max_clamped=True)
            dpg.add_spacer(width=10)
            dpg.add_button(label="<<", callback=cb_block_prev)
            dpg.add_button(label=">>", callback=cb_block_next)
            dpg.add_spacer(width=20)
            dpg.add_combo(items=list(MODEL_OPTIONS.keys()),
                          default_value="Large", tag="model_combo", width=80)
            dpg.add_button(label="Load Model", callback=cb_load_model)
        # ── toolbar row 2: session + info ──
        with dpg.group(horizontal=True):
            dpg.add_text("Session:")
            dpg.add_input_text(tag="session_name_input", default_value="Session",
                               width=120, callback=cb_session_name)
            dpg.add_button(label="Save", callback=cb_save_session)
            dpg.add_button(label="Load Session", callback=cb_load_session)
            dpg.add_button(label="Reset", callback=cb_reset_session)
            dpg.add_spacer(width=20)
            dpg.add_text("  No media loaded", tag="info_text", color=(200, 220, 255))

        # ── main area ──
        with dpg.group(horizontal=True):
            # ── left panel ──
            with dpg.child_window(width=PANEL_W, height=tex_h, border=True, tag="left_panel"):
                # ── Labels ──
                with dpg.collapsing_header(label="Labels (W/S)", default_open=True):
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="label_name_input", hint="New label",
                                           width=130, on_enter=True, callback=cb_add_label)
                        dpg.add_button(label="+", callback=cb_add_label, width=28)
                        dpg.add_button(label="-", callback=cb_remove_label, width=28)
                    dpg.add_listbox(tag="label_listbox", items=[], num_items=5,
                                    callback=cb_label_listbox, width=-1)
                    dpg.add_button(label="Library", callback=cb_label_library, width=-1)

                # ── Prompts ──
                with dpg.collapsing_header(label="Prompts", default_open=True):
                    dpg.add_listbox(tag="prompt_listbox", items=[], num_items=3, width=-1)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Delete", callback=cb_delete_selected_prompt,
                                       width=105)
                        dpg.add_button(label="Clear", callback=cb_clear_prompts, width=105)

                # ── SAM2 ──
                with dpg.collapsing_header(label="SAM2 Inference", default_open=True):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Single", callback=cb_single,
                                       tag="btn_single", width=105)
                        dpg.add_button(label="Forward", callback=cb_forward,
                                       tag="btn_forward", width=105)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Backward", callback=cb_backward,
                                       tag="btn_backward", width=105)
                        dpg.add_button(label="All", callback=cb_all,
                                       tag="btn_all", width=105)
                    dpg.add_button(label="Checkpoint", callback=cb_toggle_checkpoint,
                                   tag="btn_checkpoint", width=-1)
                    dpg.add_text("", tag="progress_text", wrap=PANEL_W - 30)
                    dpg.add_progress_bar(tag="progress_bar", default_value=0, width=-1)

                # ── View ──
                with dpg.collapsing_header(label="View (Shift+1/2/3/4)", default_open=True):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Original",
                                       callback=lambda: cb_set_view_mode(0), width=50)
                        dpg.add_button(label="Prompt",
                                       callback=lambda: cb_set_view_mode(1), width=50)
                        dpg.add_button(label="Overlay",
                                       callback=lambda: cb_set_view_mode(2), width=50)
                        dpg.add_button(label="Mask",
                                       callback=lambda: cb_set_view_mode(3), width=50)

                # ── Export ──
                with dpg.collapsing_header(label="Export", default_open=False):
                    dpg.add_button(label="Export Annotations",
                                   callback=cb_export_annotations, width=-1)
                    dpg.add_button(label="Export Verify Video",
                                   callback=cb_export_verify_video, width=-1)

                # ── Shortcuts ──
                with dpg.collapsing_header(label="Shortcuts", default_open=False):
                    dpg.add_text("A/D  prev/next frame", color=(150,150,150))
                    dpg.add_text("W/S  prev/next label", color=(150,150,150))
                    dpg.add_text("Q/E  jump to prompt", color=(150,150,150))
                    dpg.add_text("P    toggle FG/BG", color=(150,150,150))
                    dpg.add_text("C    clear prompts", color=(150,150,150))
                    dpg.add_text("Del  delete prompt", color=(150,150,150))
                    dpg.add_text("F11  fullscreen", color=(150,150,150))

            # ── frame display ──
            with dpg.group():
                with dpg.drawlist(width=tex_w, height=tex_h, tag=DRAW_TAG):
                    dpg.draw_image(TEX_TAG, pmin=(0, 0), pmax=(tex_w, tex_h), tag=IMG_TAG)

        # ── timeline + status (compact) ──
        TIMELINE_H = 24
        with dpg.drawlist(width=tex_w + PANEL_W + 16, height=TIMELINE_H,
                          tag="timeline_drawlist"):
            pass
        with dpg.group(horizontal=True):
            dpg.add_slider_int(tag="frame_slider", default_value=0, min_value=0,
                               max_value=1, width=tex_w + PANEL_W - 200,
                               callback=cb_frame_slider, format="Frame %d")
            dpg.add_text("", tag="status_text")

    dpg.set_primary_window("primary_window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    global _dpg_ready
    _dpg_ready = True

    while dpg.is_dearpygui_running():
        global _session_load_done, _media_load_done
        if _media_load_done:
            _media_load_done = False
            refresh_label_listbox()
            load_and_show_frame()
        if _session_load_done:
            _session_load_done = False
            try:
                dpg.set_value("session_name_input", ann.session_name)
            except Exception:
                pass
            refresh_label_listbox()
            load_and_show_frame()
            _show_progress(f"Session loaded: {ann.session_name} "
                           f"({len(ann.media_files)} frames)", 100)
        _draw_temp_box()
        dpg.render_dearpygui_frame()

    _dpg_ready = False
    dpg.destroy_context()


if __name__ == "__main__":
    build_ui()
