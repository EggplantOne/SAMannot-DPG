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
_auto_single_timer = None
_auto_single_enabled = True

# DPG context ready flag
_dpg_ready = False
_session_load_done = False
_media_load_done = False

# Thread-safe UI update queue (DPG is NOT thread-safe)
_main_thread_id = threading.get_ident()
_pending_progress = []            # queued (msg, pct) updates from worker threads
_pending_session_name = None      # str or None
_pending_busy_update = False

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
TOAST_NODE = "label_toast_node"
_toast_timer = None


def show_label_toast():
    """Briefly show the current label name as a centered toast on the canvas."""
    global _toast_timer
    if not _dpg_ready:
        return
    if ann.curr_label_idx < 0 or ann.curr_label_idx >= len(ann.sam_handler.labels):
        return
    label = ann.sam_handler.labels[ann.curr_label_idx]
    try:
        hex_col = label.col or "#C8C8C8"
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
    except (ValueError, IndexError, TypeError):
        r, g, b = 200, 200, 200

    try:
        if dpg.does_item_exist(TOAST_NODE):
            dpg.delete_item(TOAST_NODE)
        dpg.add_draw_node(parent=DRAW_TAG, tag=TOAST_NODE)

        idx_text = f"#{ann.curr_label_idx:02d}"
        name_text = label.name
        size = 26
        idx_size = 18
        # rough text width estimates (DPG has no measure API)
        # ASCII ≈ 0.55, CJK ≈ 1.05 of size
        def _est_w(s, sz):
            w = 0.0
            for ch in s:
                w += sz * (1.05 if ord(ch) > 0x2E80 else 0.55)
            return int(w)
        name_w = _est_w(name_text, size)
        pad_x = 22
        pad_y = 14
        bar_w = 5            # left accent bar
        bar_gap = 14
        box_h = size + pad_y * 2
        # content: [bar] [name] -- idx tag pinned to right
        idx_tag_w = _est_w(idx_text, idx_size) + 14
        idx_gap = 14
        content_w = bar_w + bar_gap + name_w + idx_gap + idx_tag_w
        box_w = content_w + pad_x * 2

        cx = tex_w // 2
        y_top = 32
        x1 = cx - box_w // 2
        y1 = y_top
        x2 = x1 + box_w
        y2 = y1 + box_h
        radius = box_h // 2  # pill shape

        # drop shadow (offset, very transparent)
        dpg.draw_rectangle((x1 + 2, y1 + 4), (x2 + 2, y2 + 4),
                           color=(0, 0, 0, 60), fill=(0, 0, 0, 60),
                           rounding=radius, parent=TOAST_NODE)
        # accent glow behind pill (uses label color, very low alpha)
        dpg.draw_rectangle((x1 - 1, y1 - 1), (x2 + 1, y2 + 1),
                           color=(r, g, b, 90), fill=(0, 0, 0, 0),
                           rounding=radius + 1, thickness=2, parent=TOAST_NODE)
        # main pill background
        dpg.draw_rectangle((x1, y1), (x2, y2),
                           color=(60, 60, 70, 230), fill=(28, 28, 34, 225),
                           rounding=radius, thickness=1, parent=TOAST_NODE)

        # left accent bar (rounded)
        bar_x1 = x1 + pad_x
        bar_y1 = y1 + pad_y - 2
        bar_x2 = bar_x1 + bar_w
        bar_y2 = y2 - pad_y + 2
        dpg.draw_rectangle((bar_x1, bar_y1), (bar_x2, bar_y2),
                           color=(r, g, b, 255), fill=(r, g, b, 255),
                           rounding=bar_w // 2, parent=TOAST_NODE)

        # label name (bright white)
        name_x = bar_x2 + bar_gap
        name_y = y1 + (box_h - size) // 2 - 3
        dpg.draw_text((name_x, name_y), name_text,
                      color=(245, 245, 250, 255), size=size, parent=TOAST_NODE)

        # index tag pill on the right (dimmer)
        tag_x2 = x2 - pad_x
        tag_x1 = tag_x2 - idx_tag_w
        tag_y1 = y1 + pad_y - 2
        tag_y2 = y2 - pad_y + 2
        tag_h = tag_y2 - tag_y1
        dpg.draw_rectangle((tag_x1, tag_y1), (tag_x2, tag_y2),
                           color=(0, 0, 0, 0), fill=(r, g, b, 55),
                           rounding=tag_h // 2, parent=TOAST_NODE)
        idx_x = tag_x1 + (idx_tag_w - _est_w(idx_text, idx_size)) // 2
        idx_y = tag_y1 + (tag_h - idx_size) // 2 - 2
        dpg.draw_text((idx_x, idx_y), idx_text,
                      color=(r, g, b, 255), size=idx_size, parent=TOAST_NODE)
    except Exception as e:
        print(f"show_label_toast error: {e}")
        return

    def _clear():
        try:
            if dpg.does_item_exist(TOAST_NODE):
                dpg.delete_item(TOAST_NODE)
        except Exception:
            pass

    if _toast_timer is not None:
        try:
            _toast_timer.cancel()
        except Exception:
            pass
    _toast_timer = threading.Timer(1.2, _clear)
    _toast_timer.daemon = True
    _toast_timer.start()


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
            col = label_color if pt.pt_type == 1 else (255, 0, 0, 255)
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
    if not dpg.does_item_exist("label_list_container"):
        return
    dpg.delete_item("label_list_container", children_only=True)
    for i, label in enumerate(ann.sam_handler.labels):
        hex_col = label.col or "#C8C8C8"
        try:
            r = int(hex_col[1:3], 16)
            g = int(hex_col[3:5], 16)
            b = int(hex_col[5:7], 16)
        except (ValueError, IndexError):
            r, g, b = 200, 200, 200
        with dpg.group(horizontal=True, parent="label_list_container"):
            dpg.add_text("■", color=(r, g, b, 255))
            dpg.add_selectable(label=f"{i}: {label.name}",
                               tag=f"label_sel_{i}",
                               user_data=i,
                               callback=cb_label_listbox,
                               default_value=(i == ann.curr_label_idx))
    _refresh_reassign_combos()


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


def cb_label_listbox(sender, app_data, user_data):
    idx = user_data
    if idx is None or idx < 0 or idx >= len(ann.sam_handler.labels):
        if dpg.does_item_exist(sender):
            dpg.set_value(sender, False)
        return
    for j in range(len(ann.sam_handler.labels)):
        tag = f"label_sel_{j}"
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, j == idx)
    ann.set_current_label(idx)
    update_status_bar()
    update_prompt_list()
    draw_overlays()


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
    show_label_toast()


def cb_label_next():
    if not ann.sam_handler.labels:
        return
    new_idx = (ann.curr_label_idx + 1) % len(ann.sam_handler.labels)
    ann.set_current_label(new_idx)
    refresh_label_listbox()
    update_status_bar()
    update_prompt_list()
    draw_overlays()
    show_label_toast()


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
            # remove all instances of this class name
            while True:
                idx = ann.get_label_idx(name)
                if idx < 0:
                    break
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


def _is_mouse_over_canvas():
    """Check if the mouse is hovering over the canvas drawlist, not over UI panels."""
    try:
        return dpg.is_item_hovered(DRAW_TAG)
    except Exception:
        return False


def cb_mouse_click(sender, app_data):
    """Left click = foreground point (pt_type=1)."""
    if ann.curr_img_idx < 0:
        return
    if not _is_mouse_over_canvas():
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
    schedule_auto_single()


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
                schedule_auto_single()
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
    if not _is_mouse_over_canvas():
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
    schedule_auto_single()


# ── SAM2 inference (Step 4) ──────────────────────────────────────────────────

def _set_busy(busy):
    global inference_busy, _pending_busy_update
    inference_busy = busy
    if not _dpg_ready or threading.get_ident() != _main_thread_id:
        _pending_busy_update = True
        return
    _update_inference_buttons()
    update_status_bar()


def _update_inference_buttons():
    for tag in ("btn_single", "btn_forward", "btn_backward", "btn_all"):
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, enabled=not inference_busy)


def _show_progress(msg, pct=0):
    global _pending_progress
    if not _dpg_ready:
        return
    if threading.get_ident() != _main_thread_id:
        # Off main thread: defer to render loop
        _pending_progress.append((msg, pct))
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


def _frame_has_prompts_for_current_label():
    if ann.curr_label_idx < 0 or ann.curr_label_idx >= len(ann.sam_handler.labels):
        return False
    label = ann.sam_handler.labels[ann.curr_label_idx]
    block = ann.current_block
    frame = ann.curr_img_idx
    if any(p.idx == frame for p in label.pts.get(block, [])):
        return True
    if any(b.idx == frame for b in label.boxes.get(block, [])):
        return True
    return False


def _any_label_has_prompts_on_current_frame():
    block = ann.current_block
    frame = ann.curr_img_idx
    for lbl in ann.sam_handler.labels:
        if any(p.idx == frame for p in lbl.pts.get(block, [])):
            return True
        if any(b.idx == frame for b in lbl.boxes.get(block, [])):
            return True
    return False


def _cancel_auto_single_timer():
    """Cancel a pending auto-single timer. Called on every frame-switch entry
    so a deferred inference scheduled for the previous frame doesn't fire
    after the user has moved on (which would race with curr_img_idx)."""
    global _auto_single_timer
    if _auto_single_timer is not None:
        try:
            _auto_single_timer.cancel()
        except Exception:
            pass
        _auto_single_timer = None


def schedule_auto_single(delay=0.12):
    """Debounced auto single-frame inference after prompt edits."""
    global _auto_single_timer
    if not _auto_single_enabled:
        return
    if not _dpg_ready:
        return
    if not ann.model_status():
        return
    if _auto_single_timer is not None:
        try:
            _auto_single_timer.cancel()
        except Exception:
            pass
    _auto_single_timer = threading.Timer(delay, _auto_single_fire)
    _auto_single_timer.daemon = True
    _auto_single_timer.start()


def _auto_single_fire():
    if inference_busy:
        # try again shortly
        schedule_auto_single(0.2)
        return
    if not ann.model_status():
        return
    if not _any_label_has_prompts_on_current_frame():
        return
    cb_single(None, None)


def cb_single(sender, app_data):
    """Single frame mask generation."""
    if inference_busy:
        if sender is not None:
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
            if _is_single_label_mode():
                # Only keep the selected label's result, merge into existing JSON
                selected_gid = ann.sam_handler.labels[ann.curr_label_idx].group_id
                for fidx in list(ann.tracking_results.keys()):
                    ann.tracking_results[fidx] = {
                        gid: m for gid, m in ann.tracking_results[fidx].items()
                        if gid == selected_gid
                    }
                ann.apply_masks(_progress_callback, merge=True)
            else:
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


def _is_single_label_mode():
    """Check if the 'Single Label' checkbox is enabled."""
    try:
        return dpg.get_value("chk_single_label")
    except Exception:
        return False


def cb_forward(sender, app_data):
    if not _check_inference_ready():
        return
    def task():
        if not _init_tracking_if_needed():
            return
        if _is_single_label_mode():
            if ann.curr_label_idx < 0:
                _show_progress("No label selected.", 0); return
            label = ann.sam_handler.labels[ann.curr_label_idx]
            _show_progress(f"Forward '{label.name}'...", 10)
            success, frames = ann.propagate_single_label(label.group_id, 1, _progress_callback)
            merge = True
        else:
            _show_progress("Propagating forward...", 10)
            success, frames = ann.propagate(1, _progress_callback)
            merge = False
        if success:
            ann.apply_masks(_progress_callback, merge=merge)
            ann.view_mode = "overlay"
            load_and_show_frame()
            _draw_timeline()
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
        if _is_single_label_mode():
            if ann.curr_label_idx < 0:
                _show_progress("No label selected.", 0); return
            label = ann.sam_handler.labels[ann.curr_label_idx]
            _show_progress(f"Backward '{label.name}'...", 10)
            success, frames = ann.propagate_single_label(label.group_id, -1, _progress_callback)
            merge = True
        else:
            _show_progress("Propagating backward...", 10)
            success, frames = ann.propagate(-1, _progress_callback)
            merge = False
        if success:
            ann.apply_masks(_progress_callback, merge=merge)
            ann.view_mode = "overlay"
            load_and_show_frame()
            _draw_timeline()
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
        if _is_single_label_mode():
            if ann.curr_label_idx < 0:
                _show_progress("No label selected.", 0); return
            label = ann.sam_handler.labels[ann.curr_label_idx]
            _show_progress(f"All '{label.name}'...", 10)
            success, frames = ann.propagate_single_label(label.group_id, 0, _progress_callback)
            merge = True
        else:
            _show_progress("Propagating all...", 10)
            success, frames = ann.propagate(0, _progress_callback)
            merge = False
        if success:
            ann.apply_masks(_progress_callback, merge=merge)
            ann.view_mode = "overlay"
            load_and_show_frame()
            _draw_timeline()
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


_selected_model_name = "Large"

def _cb_load_model_menu(sender, app_data, user_data):
    """DearPyGui menu callback for loading a named SAM2 model."""
    _load_model_by_name(user_data)


def _load_model_by_name(name):
    """Set model name and trigger load."""
    global _selected_model_name
    if not name or name not in MODEL_OPTIONS:
        _show_progress(f"Unknown SAM2 model: {name}", 0)
        print(f"[SAM2] Unknown model selection: {name}", flush=True)
        return
    _selected_model_name = name
    _show_progress(f"Load requested: SAM2 {name}", 1)
    print(f"[SAM2] Load requested: {name}", flush=True)
    cb_load_model(None, None)


def cb_load_model(sender, app_data):
    """Load SAM2 model in background thread."""
    if inference_busy:
        _show_progress("Inference busy.", 0)
        return

    # get selected model size
    model_name = _selected_model_name
    cfg, ckpt = MODEL_OPTIONS.get(model_name, MODEL_OPTIONS["Large"])
    print(f"[SAM2] Preparing {model_name}: cfg={cfg}, ckpt={ckpt}", flush=True)

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    ckpt_path = os.path.join(base, ckpt)
    if not os.path.exists(ckpt_path):
        _show_progress(f"Checkpoint not found: {ckpt}. "
                       f"Run: python download_checkpoints.py --only {model_name.lower().rstrip('+')}", 0)
        print(f"[SAM2] Checkpoint not found: {ckpt_path}", flush=True)
        return

    def task():
        import torch, gc
        print(f"[SAM2] Loading {model_name} from {ckpt_path}", flush=True)
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
            print(f"[SAM2] {model_name} loaded", flush=True)
        else:
            _show_progress(f"Failed to load {model_name}.", 0)
            print(f"[SAM2] Failed to load {model_name}", flush=True)
    _run_inference(task)


# ── view mode (Step 5) ───────────────────────────────────────────────────────

def cb_set_view_mode(mode_idx):
    ann.set_view_mode(mode_idx)
    load_and_show_frame()


# ── navigation (Step 6) ─────────────────────────────────────────────────────

def cb_frame_slider(sender, app_data):
    idx = int(app_data)
    if 0 <= idx < len(ann.media_files):
        _cancel_auto_single_timer()
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
            _cancel_auto_single_timer()
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
            _cancel_auto_single_timer()
            ann.set_img(i)
            load_and_show_frame()
            return


# ── keyboard ─────────────────────────────────────────────────────────────────

def _is_input_focused():
    for tag in ("label_name_input", "block_size_input", "session_name_input",
                 "preextract_path_input", "reassign_start", "reassign_end",
                 "goto_frame_input"):
        if dpg.does_item_exist(tag) and dpg.is_item_focused(tag):
            return True
    return False


def cb_key_handler(sender, app_data):
    key = app_data
    pass  # pt_mode removed

    if _is_input_focused():
        return

    ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

    if ctrl:
        if key == dpg.mvKey_S:
            cb_save_session(None, None)
            return

    # 1/2/3/4 → view mode (input fields are already filtered by _is_input_focused above)
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
    elif key == dpg.mvKey_Z:
        cb_undo_last_prompt()
    elif key == dpg.mvKey_Delete:
        cb_delete_selected_prompt()
    elif key == dpg.mvKey_G:
        cb_goto_frame_dialog()
    elif key == dpg.mvKey_F11:
        cb_toggle_fullscreen()
    elif key == dpg.mvKey_Escape:
        dpg.focus_item("primary_window")


def cb_goto_frame_dialog():
    """G: open dialog to jump to an absolute frame number."""
    if not ann.media_files:
        return
    cur_abs = ann._current_abs_idx()
    dpg.set_value("goto_frame_input", cur_abs)
    dpg.configure_item("goto_frame_modal", show=True)
    dpg.focus_item("goto_frame_input")


def _cb_goto_frame_apply(sender=None, app_data=None):
    """Jump to the absolute frame entered in the goto dialog."""
    target = dpg.get_value("goto_frame_input")
    _hide_modal("goto_frame_modal")
    dpg.focus_item("primary_window")
    if target < 0 or target >= _max_frame:
        _show_progress(f"Frame {target} out of range (0-{_max_frame - 1}).", 0)
        return
    target_block = target // ann.block_size
    target_local = target % ann.block_size
    if target_block != ann.current_block:
        _cancel_auto_single_timer()
        ann.set_current_block(target_block)
        _reload_block()
    if 0 <= target_local < len(ann.media_files):
        _cancel_auto_single_timer()
        ann.set_img(target_local)
        load_and_show_frame()


def cb_prev_frame():
    _cancel_auto_single_timer()
    ann.prev_img()
    load_and_show_frame()


def cb_next_frame():
    _cancel_auto_single_timer()
    ann.next_img()
    load_and_show_frame()


def cb_clear_prompts():
    """Clear all labels' prompts on the current frame + delete JSON."""
    block = ann.current_block
    frame_idx = ann.curr_img_idx
    for label in ann.sam_handler.labels:
        if block in label.pts:
            label.pts[block] = [p for p in label.pts[block] if p.idx != frame_idx]
        if block in label.boxes:
            label.boxes[block] = [b for b in label.boxes[block] if b.idx != frame_idx]
    # also delete the JSON for this frame
    abs_idx = ann._abs_idx(frame_idx)
    json_path = os.path.join(ann.project_dir, "jsons", f"{abs_idx:06d}.json")
    if os.path.exists(json_path):
        os.remove(json_path)
    # drop in-memory masks for this frame so overlay/masks views refresh immediately
    if abs_idx in ann.masks:
        del ann.masks[abs_idx]
    if abs_idx in ann.overlay_imgs:
        del ann.overlay_imgs[abs_idx]
    if abs_idx in ann.combined_masks:
        del ann.combined_masks[abs_idx]
    # also clear prop_frames for this frame
    for label in ann.sam_handler.labels:
        pf = label.prop_frames.get(block, set())
        pf.discard(frame_idx)
    load_and_show_frame()


def _post_prompt_edit():
    """After any prompt edit on the current frame, refresh UI and either
    schedule auto-inference, or — if the current label has no prompts left
    on this frame — wipe its previously-auto-generated mask so the screen
    matches what the user sees."""
    draw_overlays()
    update_prompt_list()
    if ann.curr_label_idx < 0 or ann.curr_img_idx < 0:
        return
    label = ann.sam_handler.labels[ann.curr_label_idx]
    block = ann.current_block
    frame = ann.curr_img_idx
    has_pt = any(p.idx == frame for p in label.pts.get(block, []))
    has_box = any(b.idx == frame for b in label.boxes.get(block, []))
    if has_pt or has_box:
        schedule_auto_single()
        return
    # no prompts left for this label on this frame → drop its mask
    abs_idx = ann._abs_idx(frame) if hasattr(ann, "_abs_idx") else (
        ann.current_block * ann.block_size + frame)
    if ann.clear_label_mask_on_frame(abs_idx, label.group_id):
        load_and_show_frame()


def cb_undo_last_prompt():
    """Undo the last added point or box for the current label on the current frame."""
    if ann.curr_label_idx < 0 or ann.curr_img_idx < 0:
        return
    label = ann.sam_handler.labels[ann.curr_label_idx]
    block = ann.current_block
    frame = ann.curr_img_idx
    # find the last point on this frame
    pts = label.pts.get(block, [])
    for i in range(len(pts) - 1, -1, -1):
        if pts[i].idx == frame:
            pts.pop(i)
            _post_prompt_edit()
            return
    # if no point found, try last box on this frame
    boxes = label.boxes.get(block, [])
    for i in range(len(boxes) - 1, -1, -1):
        if boxes[i].idx == frame:
            boxes.pop(i)
            _post_prompt_edit()
            return


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
    _post_prompt_edit()


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
    # 切到新项目（没有 pkl 的路径）：先清空上一个项目残留的 labels / masks /
    # tracking_results 等状态，否则旧 label 会留在 UI 里。session pkl 加载走
    # _load_session_from_path → load_from_dict，自带状态替换，不需要这一步。
    ann.reset()
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
        global _pending_session_name
        ann.set_session_name(media_basename)
        if _dpg_ready:
            if threading.get_ident() != _main_thread_id:
                _pending_session_name = media_basename
            else:
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
    """DPG file dialog callback for Load Folder.
    If a pkl session exists in the project dir, load it fully (= Load Session).
    Otherwise, just load media frames."""
    folder = app_data.get("file_path_name", "")
    if not folder or not os.path.isdir(folder):
        return

    # 推导 project_dir，检查是否有 pkl
    folder_name = os.path.basename(folder)
    if folder_name == "frames":
        media_basename = os.path.basename(os.path.dirname(folder))
    else:
        media_basename = folder_name
    project_dir = os.path.join("projects", media_basename)

    pkl_path = None
    if os.path.isdir(project_dir):
        pkl_files = [f for f in os.listdir(project_dir) if f.endswith(".pkl")]
        if len(pkl_files) == 1:
            pkl_path = os.path.join(project_dir, pkl_files[0])

    if pkl_path:
        # 有 pkl → 走完整的 Load Session 流程
        _load_session_from_path(pkl_path)
    else:
        # 没有 pkl → 正常加载 media
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
                         default_path=os.getcwd() if os.getcwd().isascii() else os.path.expanduser("~")):
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
                         default_path=os.getcwd() if os.getcwd().isascii() else os.path.expanduser("~")):
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

    # reset tracking state when switching blocks, preserve view_mode
    target_block = ann.current_block
    saved_view_mode = ann.view_mode
    saved_mode = ann.mode
    ann.reset_media()
    ann.set_current_block(target_block)
    ann.view_mode = saved_view_mode
    ann.mode = saved_mode

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


def _autosave_pkl():
    """Silent pkl save used after operations that already wrote to JSON on disk
    (reassign, delete-in-range), to keep pkl in sync. Returns True on success."""
    if not ann.media_files:
        return False
    try:
        os.makedirs(ann.project_dir, exist_ok=True)
        session_name = ann.session_name or "session"
        pkl_path = os.path.join(ann.project_dir, f"{session_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(ann.compress_to_dict(), f)
        return True
    except Exception as e:
        print(f"[autosave] failed: {e}")
        return False


def _reconcile_session_group_ids():
    """Load Session 后：扫描 JSON 文件，确保 pkl 的 group_id 和 JSON 一致。
    如果不一致，pkl 适配 JSON（JSON 是落盘数据，pkl 迁就它）。"""
    from collections import defaultdict

    json_dir = os.path.join(ann.project_dir, "jsons")
    if not os.path.isdir(json_dir):
        return

    # Step 1: 扫描 JSON，收集每个 name 的 group_id 集合 + 最后标注帧号
    #         同时记录 group_id=None 的 name（旧版 JSON 兼���）
    name_to_gids = defaultdict(set)
    names_with_none_gid = set()  # name 出现过 group_id=None
    last_annotated_frame = -1
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    if not json_files:
        return
    total_jsons = len(json_files)
    _show_progress(f"Checking {total_jsons} JSON files...", 0)
    for i, fname in enumerate(json_files):
        fpath = os.path.join(json_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            if shapes:
                try:
                    frame_idx = int(os.path.splitext(fname)[0])
                    last_annotated_frame = max(last_annotated_frame, frame_idx)
                except ValueError:
                    pass
            for shape in shapes:
                name = shape.get("label", "")
                gid = shape.get("group_id")
                if not name:
                    continue
                if gid is not None:
                    name_to_gids[name].add(gid)
                else:
                    names_with_none_gid.add(name)
        except Exception:
            continue
        if (i + 1) % 100 == 0 or i + 1 == total_jsons:
            _show_progress(
                f"Checking JSONs... {i+1}/{total_jsons}",
                int((i + 1) / total_jsons * 100))

    # Step 1.5: pkl 内部 group_id 去重
    # 不同 name 的 pkl label 撞同一 gid 时，按 gid 查 label 的 "first match wins" 路径
    # （annotator.py:313, 1428, 1448）会把多个 label 的 mask 全染成第一个 label 的颜色。
    # 优先保留 name 与 JSON 该 gid 对应的 label；其余分配新 gid。
    gid_to_pkl_labels = defaultdict(list)
    for label in ann.sam_handler.labels:
        if isinstance(label.group_id, int):
            gid_to_pkl_labels[label.group_id].append(label)
    pkl_dups = {gid: lbls for gid, lbls in gid_to_pkl_labels.items() if len(lbls) > 1}

    dedup_moves = []  # [(name, old_gid, new_gid)]
    if pkl_dups:
        all_used_gids = set()
        for gids in name_to_gids.values():
            all_used_gids.update(gids)
        for label in ann.sam_handler.labels:
            if isinstance(label.group_id, int):
                all_used_gids.add(label.group_id)
        fresh_gid = max(all_used_gids) + 1 if all_used_gids else 1

        for gid, dup_labels in pkl_dups.items():
            keeper = None
            for lbl in dup_labels:
                if lbl.name in name_to_gids and gid in name_to_gids[lbl.name]:
                    keeper = lbl
                    break
            if keeper is None:
                keeper = dup_labels[0]
            for lbl in dup_labels:
                if lbl is keeper:
                    continue
                old = lbl.group_id
                lbl.group_id = fresh_gid
                dedup_moves.append((lbl.name, old, fresh_gid))
                fresh_gid += 1

        if dedup_moves:
            # 撞 gid 时 mask 数据归属本就是模糊的（写入会互相覆盖），保留在 keeper 名下；
            # 被腾出的 label 从空 mask 重新开始。
            ann.label_handler._next_group_id = max(
                ann.label_handler._next_group_id, fresh_gid)
            _show_progress(
                "Dedup pkl group_ids: " + ", ".join(
                    f"'{n}' #{o}->#{nn}" for n, o, nn in dedup_moves),
                0)

    # 为 group_id=None 的 JSON / pkl label 分配 gid：
    # 1) 同名 JSON 后续已有数字 gid，则沿用该 gid；
    # 2) 否则尽量复用 pkl 里同名单实例的数字 gid；
    # 3) 都没有时才分配新 gid。
    # 关键：考虑 pkl 中已有 label 的 gid，避免与"pkl 有但 JSON 没引用"的 label 撞车
    json_gids_used = set()
    for gids in name_to_gids.values():
        json_gids_used.update(gids)

    # pkl 单实例 name → gid 的映射（仅当 gid 不与 JSON 已用 gid 冲突时记录）
    pkl_name_count = defaultdict(int)
    for label in ann.sam_handler.labels:
        pkl_name_count[label.name] += 1
    pkl_unique_gid = {}
    for label in ann.sam_handler.labels:
        if (pkl_name_count[label.name] == 1 and
                isinstance(label.group_id, int) and
                label.group_id not in json_gids_used):
            pkl_unique_gid[label.name] = label.group_id

    # 全部已用 gid = JSON 中的 + pkl 中的，next_gid 从最大值后跳
    all_existing_gids = set(json_gids_used)
    for label in ann.sam_handler.labels:
        if isinstance(label.group_id, int):
            all_existing_gids.add(label.group_id)
    next_gid = max(all_existing_gids) + 1 if all_existing_gids else 1

    pkl_names_with_none_gid = {
        label.name for label in ann.sam_handler.labels
        if label.group_id is None
    }
    names_to_resolve = set(names_with_none_gid) | pkl_names_with_none_gid

    none_gid_assigned = {}  # name → resolved gid for old None group_id
    ordered_names_to_resolve = []
    seen_names_to_resolve = set()
    for label in ann.sam_handler.labels:
        if label.name in names_to_resolve and label.name not in seen_names_to_resolve:
            ordered_names_to_resolve.append(label.name)
            seen_names_to_resolve.add(label.name)
    for name in sorted(names_to_resolve - seen_names_to_resolve):
        ordered_names_to_resolve.append(name)

    for name in ordered_names_to_resolve:
        if name in name_to_gids and name_to_gids[name]:
            # 该 name 已有 gid，取最小的
            none_gid_assigned[name] = min(name_to_gids[name])
        elif name not in name_to_gids:
            # 该 name 只出现过 group_id=None
            if name in pkl_unique_gid:
                # 优先复用 pkl 同名 label 的 gid（避免后续 remap 和潜在的额外帧 mask 错位）
                gid = pkl_unique_gid[name]
            else:
                # 没有可复用的 → 分配新 gid（已跳过所有 pkl 占用的 gid）
                gid = next_gid
                next_gid += 1
            none_gid_assigned[name] = gid
            name_to_gids[name].add(gid)

    pkl_fixed_count = 0
    if none_gid_assigned:
        for label in ann.sam_handler.labels:
            if label.group_id is None and label.name in none_gid_assigned:
                label.group_id = none_gid_assigned[label.name]
                pkl_fixed_count += 1
        if pkl_fixed_count:
            max_gid = max(
                (lbl.group_id for lbl in ann.sam_handler.labels
                 if isinstance(lbl.group_id, int)),
                default=0)
            ann.label_handler._next_group_id = max(
                ann.label_handler._next_group_id, max_gid + 1)

    # 回写 JSON：把 group_id=None 修正为正确的数字
    fixed_count = 0
    if none_gid_assigned:
        _show_progress(f"Fixing {len(names_with_none_gid)} label(s) with missing group_id...", 0)
        for i, fname in enumerate(json_files):
            fpath = os.path.join(json_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                modified = False
                for shape in data.get("shapes", []):
                    if shape.get("group_id") is None:
                        name = shape.get("label", "")
                        if name in none_gid_assigned:
                            shape["group_id"] = none_gid_assigned[name]
                            modified = True
                if modified:
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    fixed_count += 1
            except Exception:
                continue
            if (i + 1) % 100 == 0 or i + 1 == total_jsons:
                _show_progress(
                    f"Fixing JSONs... {i+1}/{total_jsons}",
                    int((i + 1) / total_jsons * 100))
        _show_progress(f"Fixed group_id in {fixed_count} JSON files.", 100)

    if not name_to_gids:
        return

    # Step 2: pkl 适配 JSON — 按 name 分组，排序后一一对应
    pkl_name_to_labels = defaultdict(list)
    for label in ann.sam_handler.labels:
        pkl_name_to_labels[label.name].append(label)

    remap = {}  # old_pkl_gid → new_json_gid
    errors = []

    for name, labels in pkl_name_to_labels.items():
        if name not in name_to_gids:
            continue
        raw_pkl_gids = [l.group_id for l in labels]
        if any(not isinstance(gid, int) for gid in raw_pkl_gids):
            errors.append(f"'{name}': pkl has unresolved group_id=None")
            continue
        pkl_gids = sorted(raw_pkl_gids)
        json_gids = sorted(name_to_gids[name])
        if len(pkl_gids) != len(json_gids):
            errors.append(
                f"'{name}': pkl has {len(pkl_gids)} instance(s), "
                f"JSON has {len(json_gids)} — please check")
            continue
        for old_gid, new_gid in zip(pkl_gids, json_gids):
            if old_gid != new_gid:
                remap[old_gid] = new_gid

    # 记录需要添加的缺失 labels（remap 之后再添加，避免 gid 冲突）
    missing = []  # [(name, gid), ...]
    for name, gids in name_to_gids.items():
        if name not in pkl_name_to_labels:
            for gid in sorted(gids):
                missing.append((name, gid))

    if errors:
        _show_progress("group_id mismatch: " + "; ".join(errors), 0)

    if not remap and not missing and not dedup_moves and fixed_count == 0 and pkl_fixed_count == 0:
        if not errors:
            _show_progress("Reconcile: group_ids already consistent.", 100)
        return

    # Step 3: 先执行 remap
    if remap:
        # 用临时负数 key 避免链式覆盖（如 1→3, 3→5）
        tmp_remap = {old: -(i + 1) for i, old in enumerate(remap.keys())}
        tmp_to_final = {-(i + 1): new for i, (old, new) in enumerate(remap.items())}

        for label in ann.sam_handler.labels:
            if label.group_id in remap:
                label.group_id = remap[label.group_id]

        # ann.masks / ann.tracking_results: {abs_idx: {gid: data}}
        # ann.extra_frame / ann.extra_frame_masks: {block: {gid: data}}
        # 内层 dict 都是 gid → data，结构一致，统一处理
        for d in [ann.masks, ann.tracking_results,
                  ann.extra_frame, ann.extra_frame_masks]:
            for frame_data in d.values():
                if not isinstance(frame_data, dict):
                    continue
                for old_gid, tmp_gid in tmp_remap.items():
                    if old_gid in frame_data:
                        frame_data[tmp_gid] = frame_data.pop(old_gid)
                for tmp_gid, final_gid in tmp_to_final.items():
                    if tmp_gid in frame_data:
                        frame_data[final_gid] = frame_data.pop(tmp_gid)

        for obj_id in list(ann.sam_handler.object_id_to_group_id.keys()):
            gid = ann.sam_handler.object_id_to_group_id[obj_id]
            if gid in remap:
                ann.sam_handler.object_id_to_group_id[obj_id] = remap[gid]

    # Step 4: remap 完成后，添加缺失的 labels（此时不会有 gid 冲突）
    added = []
    for name, gid in missing:
        ann.add_label(name, group_id=gid)
        added.append(f"{name} #{gid}")

    # 更新计数器
    all_gids = set()
    for gids in name_to_gids.values():
        all_gids.update(gid for gid in gids if isinstance(gid, int))
    if all_gids:
        ann.label_handler._next_group_id = max(
            ann.label_handler._next_group_id,
            max(all_gids) + 1)

    # 定位到最后一个有标注的帧（仅当 pkl 没有有效位置时）
    if last_annotated_frame >= 0 and ann.curr_img_idx <= 0:
        block_size = ann.block_size
        if block_size > 0:
            target_block = last_annotated_frame // block_size
            frame_in_block = last_annotated_frame % block_size
            ann.set_current_block(target_block)
            ann.curr_img_idx = frame_in_block

    # 保存修正后的 pkl（先备份现有 pkl 到带时间戳的 .bak，保留历史）
    session_name = ann.session_name or "session"
    pkl_path = os.path.join(ann.project_dir, f"{session_name}.pkl")
    bak_name = None
    if os.path.exists(pkl_path):
        import shutil as _shutil
        from datetime import datetime as _dt
        bak_name = f"{session_name}.pkl.bak.{_dt.now().strftime('%Y%m%d-%H%M%S')}"
        bak_path = os.path.join(ann.project_dir, bak_name)
        try:
            _shutil.copy2(pkl_path, bak_path)
        except Exception as e:
            print(f"[reconcile] backup failed: {e}")
            bak_name = None
    data = ann.compress_to_dict()
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    parts = []
    if dedup_moves:
        parts.append(f"deduped {len(dedup_moves)} pkl group_id(s)")
    if fixed_count:
        parts.append(f"fixed group_id in {fixed_count} JSON file(s)")
    if pkl_fixed_count:
        parts.append(f"fixed {pkl_fixed_count} pkl label group_id(s)")
    if remap:
        parts.append(f"remapped {len(remap)} group_id(s)")
    if added:
        parts.append(f"added {len(added)} label(s): {', '.join(added)}")
    suffix = f" (backup: {bak_name})" if bak_name else ""
    _show_progress(f"Reconcile done: {'; '.join(parts)}. Session saved{suffix}.", 100)


def _load_session_from_path(path):
    """Load a session pkl file. Used by both Load Session and Load Folder (when pkl found)."""
    def do_load():
        global _max_frame, _session_load_done
        import shutil as _shutil

        _show_progress("Loading session...", 10)
        with open(path, "rb") as f:
            data = pickle.load(f)
        ann.load_from_dict(data)

        _show_progress("Restoring frames...", 30)
        temp_img_idx = ann.curr_img_idx
        ann.read_frames = -1
        block_size = ann.block_size
        start_frame = ann.current_block * block_size

        # Always reset media_path to frames_dir so that block switching
        # works even when the session was saved on a different machine.
        ann.media_path = os.path.abspath(ann.frames_dir)

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

        # clear stale path mappings (pkl may contain paths from another machine)
        ann.idx_to_path = {}

        # ensure a label is selected
        if ann.curr_label_idx < 0 and len(ann.sam_handler.labels) > 0:
            ann.set_current_label(0)

        # force prompts view so annotations are visible
        ann.view_mode = "prompts"
        ann.mode = "prompts"

        # signal main thread to refresh UI
        _session_load_done = True

    threading.Thread(target=do_load, daemon=True).start()


def _on_session_selected(sender, app_data):
    """DPG file dialog callback for Load Session."""
    selections = app_data.get("selections", {})
    if not selections:
        return
    path = list(selections.values())[0]
    _load_session_from_path(path)


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
                             default_path=os.getcwd() if os.getcwd().isascii() else os.path.expanduser("~")):
            dpg.add_file_extension(".mp4", color=(0, 255, 0))
            dpg.add_file_extension(".avi", color=(0, 255, 0))
            dpg.add_file_extension(".mov", color=(0, 255, 0))
            dpg.add_file_extension(".mkv", color=(0, 255, 0))
        return

    _do_export_verify_video(video_path)


def _on_preextract_video_selected(sender, app_data):
    """DPG callback: user picked a video for pre-extraction."""
    try:
        selections = app_data.get("selections", {})
        if selections:
            video_path = list(selections.values())[0]
        else:
            # fallback: user typed path directly in the dialog bar
            video_path = app_data.get("file_path_name", "")
        if not video_path:
            return
    except Exception as e:
        print(f"Error parsing file dialog result: {e}")
        _show_progress(f"Error: cannot parse selected path.", 0)
        return

    # validate video first
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

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join("projects", video_name, "frames")

    if os.path.exists(out_dir):
        existing = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
        if len(existing) > 0:
            _show_progress(f"Frames already exist ({len(existing)} files). Use Load Folder to open.", 100)
            return

    # show quality selection dialog
    _show_extract_quality_dialog(video_path, total_frames, out_dir)


def _show_extract_quality_dialog(video_path, total_frames, out_dir):
    """Show a dialog for the user to choose extraction quality before extracting."""
    tag = "extract_quality_modal"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)

    est_standard_gb = total_frames * 200 / 1024 / 1024  # ~200KB per frame at qscale=2
    est_high_gb = total_frames * 300 / 1024 / 1024       # ~300KB per frame at qscale=1

    def on_quality_selected(quality):
        _hide_modal(tag)
        _do_pre_extract(video_path, total_frames, out_dir, quality == "high")

    with dpg.window(label="Pre-extract Quality", tag=tag,
                    modal=True, show=True, width=350, height=160, no_resize=True):
        dpg.add_text(f"{total_frames} frames to extract")
        dpg.add_spacer(height=5)
        with dpg.group(horizontal=True):
            dpg.add_button(label=f"Standard (~{est_standard_gb:.1f} GB)",
                           callback=lambda: on_quality_selected("standard"), width=-1)
        with dpg.group(horizontal=True):
            dpg.add_button(label=f"High Quality (~{est_high_gb:.1f} GB)",
                           callback=lambda: on_quality_selected("high"), width=-1)
        dpg.add_button(label="Cancel", callback=lambda: _hide_modal(tag), width=-1)


def _do_pre_extract(video_path, total_frames, out_dir, high_quality):
    """Actually run the pre-extraction after quality is chosen."""

    def task():

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

        qscale = "1" if high_quality else "2"

        ffmpeg = ann._get_ffmpeg_exe()
        if ffmpeg:
            mode_str = "high quality" if high_quality else "standard"
            _show_progress(f"Pre-extracting {total_frames} frames ({mode_str}, ffmpeg)...", 0)
            cmd = [
                ffmpeg, "-y",
                "-i", video_path,
                "-vsync", "0",
                "-frames:v", str(total_frames),
                "-qscale:v", qscale,
                "-start_number", "0",
                os.path.join(out_dir, "%06d.jpg"),
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            n_done = 0
            while proc.poll() is None:
                time.sleep(0.3)
                while n_done < total_frames and os.path.exists(
                        os.path.join(out_dir, f"{n_done:06d}.jpg")):
                    n_done += 1
                pct = min(99, (n_done / total_frames) * 100)
                _show_progress(f"Extracting: {n_done}/{total_frames} ({pct:.0f}%)", pct)
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
            jpg_quality = [cv2.IMWRITE_JPEG_QUALITY, 100 if high_quality else 95]
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), frame, jpg_quality)
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
                         default_path=os.getcwd() if os.getcwd().isascii() else os.path.expanduser("~")):
        dpg.add_file_extension(".mp4", color=(0, 255, 0))
        dpg.add_file_extension(".avi", color=(0, 255, 0))
        dpg.add_file_extension(".mov", color=(0, 255, 0))
        dpg.add_file_extension(".mkv", color=(0, 255, 0))
        dpg.add_file_extension(".*")




# ── reassign label ───────────────────────────────────────────────────────────

def cb_reassign_label(sender, app_data):
    """Reassign masks/prompts from one label to another within a frame range."""
    if not ann.sam_handler.labels or len(ann.sam_handler.labels) < 2:
        _show_progress("Need at least 2 labels to reassign.", 0)
        return
    src_name = dpg.get_value("reassign_src")
    dst_name = dpg.get_value("reassign_dst")
    if not src_name or not dst_name or src_name == dst_name:
        _show_progress("Source and target must be different labels.", 0)
        return
    start_abs = dpg.get_value("reassign_start")
    end_abs = dpg.get_value("reassign_end")
    if start_abs > end_abs:
        _show_progress("Start frame must be <= end frame.", 0)
        return
    # find group_ids from display names like "0: 抓钳"
    try:
        src_idx = int(src_name.split(":")[0])
        dst_idx = int(dst_name.split(":")[0])
    except (ValueError, IndexError):
        _show_progress("Invalid label selection. Please re-select.", 0)
        return
    if src_idx < 0 or src_idx >= len(ann.sam_handler.labels) or \
       dst_idx < 0 or dst_idx >= len(ann.sam_handler.labels):
        _show_progress("Label index out of range. Please re-select.", 0)
        return
    src_gid = ann.sam_handler.labels[src_idx].group_id
    dst_gid = ann.sam_handler.labels[dst_idx].group_id

    n = ann.reassign_label_in_range(src_gid, dst_gid, start_abs, end_abs)
    saved = _autosave_pkl()
    suffix = " (pkl saved)" if saved else " (pkl save failed)"
    _show_progress(f"Reassigned {n} frames: {ann.sam_handler.labels[src_idx].name} -> {ann.sam_handler.labels[dst_idx].name} [{start_abs}-{end_abs}]{suffix}", 100)
    load_and_show_frame()
    _draw_timeline()


def _refresh_reassign_combos():
    """Update the reassign source/target combo boxes with current labels."""
    if not dpg.does_item_exist("reassign_src"):
        return
    items = [f"{i}: {l.name}" for i, l in enumerate(ann.sam_handler.labels)]
    dpg.configure_item("reassign_src", items=items)
    dpg.configure_item("reassign_dst", items=items)
    # Reset selected values to avoid stale references after label changes
    dpg.set_value("reassign_src", items[0] if items else "")
    dpg.set_value("reassign_dst", items[1] if len(items) > 1 else "")


# ── delete label in range ────────────────────────────────────────────────────

def cb_delete_label_in_range(sender, app_data):
    """Delete masks/prompts for a label within a frame range."""
    if not ann.sam_handler.labels:
        _show_progress("No labels to delete.", 0)
        return
    label_name = dpg.get_value("delete_range_label")
    if not label_name:
        _show_progress("Please select a label.", 0)
        return
    start_abs = dpg.get_value("delete_range_start")
    end_abs = dpg.get_value("delete_range_end")
    if start_abs > end_abs:
        _show_progress("Start frame must be <= end frame.", 0)
        return
    try:
        label_idx = int(label_name.split(":")[0])
    except (ValueError, IndexError):
        _show_progress("Invalid label selection. Please re-select.", 0)
        return
    if label_idx < 0 or label_idx >= len(ann.sam_handler.labels):
        _show_progress("Label index out of range. Please re-select.", 0)
        return
    gid = ann.sam_handler.labels[label_idx].group_id
    n = ann.delete_label_in_range(gid, start_abs, end_abs)
    saved = _autosave_pkl()
    suffix = " (pkl saved)" if saved else " (pkl save failed)"
    _show_progress(f"Deleted {n} frames of '{ann.sam_handler.labels[label_idx].name}' in [{start_abs}-{end_abs}]{suffix}", 100)
    load_and_show_frame()
    _draw_timeline()


def _cb_delete_range_apply(sender, app_data):
    """Apply delete-in-range from modal."""
    cb_delete_label_in_range(sender, app_data)
    _hide_modal("delete_range_modal")


def _refresh_delete_range_combo():
    """Update the delete-range combo box with current labels."""
    if not dpg.does_item_exist("delete_range_label"):
        return
    items = [f"{i}: {l.name}" for i, l in enumerate(ann.sam_handler.labels)]
    dpg.configure_item("delete_range_label", items=items)
    dpg.set_value("delete_range_label", items[0] if items else "")


# ── modal dialogs ────────────────────────────────────────────────────────────

def _show_modal(tag):
    """Show a modal dialog window."""
    if dpg.does_item_exist(tag):
        _refresh_reassign_combos()
        _refresh_delete_range_combo()
        dpg.configure_item(tag, show=True)


def _hide_modal(tag):
    """Hide a modal dialog window."""
    if dpg.does_item_exist(tag):
        dpg.configure_item(tag, show=False)


def _cb_preextract_path_apply(sender, app_data):
    """Apply pre-extract from pasted path in modal."""
    video_path = dpg.get_value("preextract_path_input").strip().strip('"').strip("'")
    _hide_modal("preextract_modal")
    if not video_path:
        _show_progress("Please paste a video path first.", 0)
        return
    _on_preextract_video_selected(None, {"file_path_name": video_path, "selections": {}})


def _cb_reassign_apply(sender, app_data):
    """Apply reassign from modal."""
    cb_reassign_label(sender, app_data)
    _hide_modal("reassign_modal")


def _cb_reconcile_apply(sender, app_data):
    """Run reconcile in a background thread."""
    _hide_modal("reconcile_modal")
    if not ann.media_files:
        _show_progress("No media loaded. Load a folder or session first.", 0)
        return
    new_block_size = int(dpg.get_value("reconcile_block_size"))
    def task():
        global _session_load_done, _max_frame
        import shutil as _shutil

        # 应用新的 block_size
        ann.block_size = new_block_size
        frames_dir = ann.frames_dir
        if os.path.isdir(frames_dir):
            total = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        else:
            total = _max_frame if _max_frame > 0 else 0
        if total > 0:
            ann.num_blocks = max(1, math.ceil(total / new_block_size))

        _reconcile_session_group_ids()

        # 重新加载刚保存的 pkl，确保帧、block 位置都正确
        session_name = ann.session_name or "session"
        pkl_path = os.path.join(ann.project_dir, f"{session_name}.pkl")
        if not os.path.exists(pkl_path):
            _session_load_done = True
            return

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        ann.load_from_dict(data)

        # 覆盖 pkl 里的 block_size（用户可能修改了）
        ann.block_size = new_block_size
        if total > 0:
            ann.num_blocks = max(1, math.ceil(total / new_block_size))
        if ann.current_block >= ann.num_blocks:
            ann.current_block = max(0, ann.num_blocks - 1)

        # 加载对应 block 的帧
        block_size = ann.block_size
        start_frame = ann.current_block * block_size
        frames_dir = ann.frames_dir
        expected_frame = os.path.join(frames_dir, f"{start_frame:06d}.jpg")

        if os.path.exists(expected_frame):
            end_frame = start_frame + block_size
            total_on_disk = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
            _max_frame = total_on_disk
            end_frame = min(end_frame, total_on_disk)
            extracted = []
            for i in range(start_frame, end_frame):
                fp = os.path.join(frames_dir, f"{i:06d}.jpg")
                if os.path.exists(fp):
                    extracted.append(fp)
            ann.media_files = extracted
            _shutil.rmtree(ann.sam_extract_dir, ignore_errors=True)
            os.makedirs(ann.sam_extract_dir, exist_ok=True)
            for fp in extracted:
                dst = os.path.join(ann.sam_extract_dir, os.path.basename(fp))
                if not os.path.exists(dst):
                    _shutil.copy2(fp, dst)

        if ann.media_files:
            ann.set_img(min(ann.curr_img_idx, len(ann.media_files) - 1))

        ann.sam_handler.tracking_init = False
        ann.sam_handler.inference_state = None
        ann.sam_handler.model_loaded = False
        ann.sam_handler.model_loading = False
        ann.sam_handler.current_stage = 0
        ann.idx_to_path = {}

        _session_load_done = True
    threading.Thread(target=task, daemon=True).start()


def _build_modal_dialogs():
    """Create modal dialog windows (hidden by default)."""
    # Pre-extract path dialog
    with dpg.window(label="Pre-extract from path", tag="preextract_modal",
                    modal=True, show=False, width=500, height=100, no_resize=True):
        dpg.add_input_text(tag="preextract_path_input", hint="Paste video path here",
                           width=-1, on_enter=True, callback=_cb_preextract_path_apply)
        with dpg.group(horizontal=True):
            dpg.add_button(label="OK", callback=_cb_preextract_path_apply, width=100)
            dpg.add_button(label="Cancel", callback=lambda: _hide_modal("preextract_modal"), width=100)

    # Go-to-frame dialog
    with dpg.window(label="Go to Frame", tag="goto_frame_modal",
                    modal=True, show=False, width=300, height=90, no_resize=True):
        dpg.add_input_int(tag="goto_frame_input", default_value=0, width=-1,
                          min_value=0, min_clamped=True,
                          on_enter=True, callback=_cb_goto_frame_apply)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Go", callback=_cb_goto_frame_apply, width=100)
            dpg.add_button(label="Cancel", callback=lambda: (_hide_modal("goto_frame_modal"), dpg.focus_item("primary_window")), width=100)

    # Reassign label dialog
    with dpg.window(label="Reassign Label", tag="reassign_modal",
                    modal=True, show=False, width=350, height=200, no_resize=True):
        dpg.add_text("Source label:")
        dpg.add_combo(tag="reassign_src", items=[], width=-1)
        dpg.add_text("Target label:")
        dpg.add_combo(tag="reassign_dst", items=[], width=-1)
        with dpg.group(horizontal=True):
            dpg.add_text("Frames:")
            dpg.add_input_int(tag="reassign_start", default_value=0, width=100,
                              min_value=0, min_clamped=True)
            dpg.add_text("-")
            dpg.add_input_int(tag="reassign_end", default_value=0, width=100,
                              min_value=0, min_clamped=True)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply", callback=_cb_reassign_apply, width=100)
            dpg.add_button(label="Cancel", callback=lambda: _hide_modal("reassign_modal"), width=100)

    # Delete Label in Range dialog
    with dpg.window(label="Delete Label in Range", tag="delete_range_modal",
                    modal=True, show=False, width=350, height=160, no_resize=True):
        dpg.add_text("Label to delete:")
        dpg.add_combo(tag="delete_range_label", items=[], width=-1)
        with dpg.group(horizontal=True):
            dpg.add_text("Frames:")
            dpg.add_input_int(tag="delete_range_start", default_value=0, width=100,
                              min_value=0, min_clamped=True)
            dpg.add_text("-")
            dpg.add_input_int(tag="delete_range_end", default_value=0, width=100,
                              min_value=0, min_clamped=True)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply", callback=_cb_delete_range_apply, width=100)
            dpg.add_button(label="Cancel", callback=lambda: _hide_modal("delete_range_modal"), width=100)

    # Reconcile Labels dialog
    with dpg.window(label="Reconcile Labels", tag="reconcile_modal",
                    modal=True, show=False, width=450, height=250, no_resize=True):
        dpg.add_text("Scan all JSON annotation files and fix group_id\n"
                     "mismatches between pkl session and JSON data.\n\n"
                     "Use this when masks appear white after loading\n"
                     "a session (group_id inconsistency).\n\n"
                     "IMPORTANT: Delete any incorrectly annotated JSON\n"
                     "files BEFORE running this (e.g. blocks annotated\n"
                     "after a wrong Load Folder), otherwise the tool\n"
                     "cannot distinguish errors from multi-instance labels.",
                     wrap=430)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text("Block Size:")
            dpg.add_input_int(tag="reconcile_block_size", default_value=200, width=120,
                              min_value=10, max_value=10000, min_clamped=True, max_clamped=True)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Run Reconcile", callback=_cb_reconcile_apply, width=130)
            dpg.add_button(label="Cancel", callback=lambda: _hide_modal("reconcile_modal"), width=100)

    # Settings dialog
    with dpg.window(label="Settings", tag="settings_modal",
                    modal=True, show=False, width=300, height=180, no_resize=True):
        dpg.add_text("Session Name:")
        dpg.add_input_text(tag="session_name_input", default_value="Session",
                           width=-1, callback=cb_session_name)
        dpg.add_spacer(height=5)
        dpg.add_text("Block Size:")
        dpg.add_input_int(tag="block_size_input", default_value=200, width=-1,
                          min_value=10, max_value=10000, min_clamped=True,
                          max_clamped=True)
        dpg.add_spacer(height=5)
        dpg.add_button(label="Close", callback=lambda: _hide_modal("settings_modal"), width=-1)


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
                dpg.add_font_range(0x2500, 0x25FF)  # box drawing + geometric shapes (incl. ■)
            dpg.bind_font(default_font)
        else:
            # fallback: just use default font at larger size
            pass


PANEL_W = 250  # left panel width


def build_ui():
    global tex_w, tex_h, tex_data, _main_thread_id
    _main_thread_id = threading.get_ident()

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
        # ── menu bar ──
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Pre-extract (browse)", callback=cb_pre_extract)
                dpg.add_menu_item(label="Pre-extract (paste path)", callback=lambda: _show_modal("preextract_modal"))
                dpg.add_menu_item(label="Load Folder", callback=cb_load_folder)
                dpg.add_separator()
                dpg.add_menu_item(label="Save", callback=cb_save_session)
                dpg.add_menu_item(label="Load Session", callback=cb_load_session)
                dpg.add_separator()
                dpg.add_menu_item(label="Reset", callback=cb_reset_session)
            with dpg.menu(label="Edit"):
                dpg.add_menu_item(label="Reassign Label", callback=lambda: _show_modal("reassign_modal"))
                dpg.add_menu_item(label="Delete Label in Range", callback=lambda: _show_modal("delete_range_modal"))
                dpg.add_menu_item(label="Reconcile Labels", callback=lambda: _show_modal("reconcile_modal"))
            dpg.add_menu_item(label="Settings", callback=lambda: _show_modal("settings_modal"))
            dpg.add_menu_item(label="<<", callback=cb_block_prev)
            dpg.add_menu_item(label=">>", callback=cb_block_next)
            with dpg.menu(label="Load Model"):
                for name in MODEL_OPTIONS.keys():
                    dpg.add_menu_item(label=name, callback=_cb_load_model_menu, user_data=name)
            dpg.add_text("  No media loaded", tag="info_text", color=(200, 220, 255))

        _build_modal_dialogs()

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
                    dpg.add_child_window(tag="label_list_container",
                                         width=-1, height=120, border=True)
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
                    dpg.add_checkbox(label="Single Label Mode", tag="chk_single_label",
                                     default_value=False)
                    dpg.add_button(label="Checkpoint", callback=cb_toggle_checkpoint,
                                   tag="btn_checkpoint", width=-1)
                    dpg.add_text("", tag="progress_text", wrap=PANEL_W - 30)
                    dpg.add_progress_bar(tag="progress_bar", default_value=0, width=-1)

                # ── View ──
                with dpg.collapsing_header(label="View (1/2/3/4)", default_open=True):
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
        global _pending_progress, _pending_session_name, _pending_busy_update
        # flush thread-safe UI updates
        if _pending_busy_update:
            _pending_busy_update = False
            _update_inference_buttons()
            update_status_bar()
        if _pending_progress:
            pp = _pending_progress[-1]
            _pending_progress.clear()
            try:
                dpg.set_value("progress_bar", pp[1] / 100.0)
                dpg.set_value("progress_text", pp[0])
            except Exception:
                pass
        psn = _pending_session_name
        if psn is not None:
            _pending_session_name = None
            try:
                dpg.set_value("session_name_input", psn)
            except Exception:
                pass
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
