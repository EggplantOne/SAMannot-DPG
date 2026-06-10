# Changelog

## [Unreleased]
### Added
- **Overlap mask inspection**: UI overlay/mask views now make overlapping masks inspectable without changing annotation data. Overlap areas are highlighted, hovering the cursor over an overlap shows the involved mask labels, and the View panel adds `Check Overlaps` to scan a frame range or the whole project for overlapping masks. Scan results list frame number, overlap pixels, region count, and involved labels, and selecting a result jumps to that frame in Overlay view.
- **Overlap review navigator**: overlap scan results now remain available while reviewing frames, with Previous/Next Pending navigation, Accepted/Pending review states, and per-frame rechecks that mark cleared overlaps as Resolved.
- **Cross-drive file browser**: media, folder, session, pre-extract, and verify-video selection now use an in-app file browser that can navigate Windows drives and non-ASCII paths.
- **Crash-safe session persistence**: session `.pkl` saves and LabelMe JSON writes now use same-directory temp files followed by atomic replace. Session saves also write a `.session_state.json` sidecar with label metadata, prompt points/boxes, propagation frames, cursor position, block size, media path, and extra-frame path so a damaged pickle can be rebuilt from recent annotation state.
- **Overlay mask alpha 滑块**：左侧 View 折叠面板下方新增 `Mask alpha` 滑块（0.0~1.0，默认 0.5），实时调整 overlay 视图中 mask 的浓度。alpha 值写入 session.pkl（`compress_to_dict` / `load_from_dict` 加 `mask_alpha` 字段，旧 pkl 无此字段时 fallback 0.5），重开项目记得上次的浓度。混合方式仍为加法叠加（`cv2.addWeighted(img, 1, mask, alpha, 0)`），不改观感。`create_overlay_img` 内新增 per-frame `_overlay_blend_cache`，缓存当前帧的 `img_rgb` 和 `combined` mask，拖动滑块时只重做混合那一步（跳过 `cv2.imread` 和 `cv2.fillPoly`），实时响应不卡顿；缓存在切帧时按 abs_idx 自动失效，并在 `apply_masks` / `reset` / `clear_label_mask_on_frame` 等 mask 变更点显式失效
- **自动单帧推理**：添加 prompt（点 / 框）后自动触发 Single 推理，无需手动点 Single 按钮。带 ~120ms 去抖，连点多个点不会重复触发；推理正忙时排队等空闲后再跑；自动触发的推理失败/忙状态不再弹"Inference busy"。手动点 Single 行为保持不变
- **删光 prompt 自动清 mask**：在当前帧通过 `Z`（撤回）或 `Delete`（删选中）把当前 label 在该帧的最后一个 point/box 删掉时，自动同步清掉该 label 在该帧之前自动生成的 mask（内存 + JSON 同步；JSON 内 shapes 清空后整个 JSON 文件一并删除，时间线蓝色标记消失），画面立即刷新与 prompts 一致。`annotator.py` 新增 `clear_label_mask_on_frame(abs_idx, group_id)` 工具方法
- **W/S 切换 label 时弹窗显示当前 label**：画面顶部居中显示一个带 label 颜色色条 + 索引徽章的 pill toast（1.2s 自动消失），切 label 时不用再低头看左侧 Labels 列表就能确认当前选中的是哪个

### Fixed
- **Faster overlap scans**: `Check Overlaps` now uses an exact scan-only mask compositor with bounding-box rejection and up to four frame workers. It avoids display-only color buffers and repeated full-frame copies while preserving full-resolution overlap results.
- **External project/session paths**: `Load Session` and `Load Folder` now keep frames, JSON, overlays, SAM temp data, and saves in the selected project's real directory instead of forcing `projects/{name}`. Project roots and `frames` subfolders outside the repository are recognized, and image reads/writes support non-ASCII Windows paths.
- **Stale overlap overlay after prompt edits**: prompt edits now invalidate the current frame render/hover caches and immediately refresh Overlay/Mask views. Current-frame rendering also merges JSON-backed masks with in-memory masks by `group_id`, so editing one propagated label no longer hides other labels' JSON masks or leaves an orphaned overlap cue behind.
- **Reconcile Labels block-size changes**: Reconcile now saves back to the currently loaded `.pkl` path, remaps block-indexed prompts/masks/propagation state when `block_size` changes, preserves the absolute current frame, reloads the same file after saving, and still keeps the existing group-id reconciliation behavior.
- 2026-05-26: Fixed prompt points/boxes being visible in both Original (1) and Prompts (2) view modes. Original view now keeps the frame clean while Prompts view remains the only mode that draws prompt overlays.
- **快速切帧导致 mask 错位写盘**：修复推理过程中（或自动单帧推理的 120ms 防抖窗口内）用户切帧/切 block 时，结果会被写到错误绝对帧号 JSON 的偶发竞态。该 bug 表现为"某些帧上突然出现不属于附近帧的 mask、关掉项目重开仍在"。
  - `annotator.py` `generate_mask` 函数入口一次性快照 `curr_img_idx` 和 `current_block` 为本地变量，后续读取全部走快照，避免 719/722 两行先后读出不同值。
  - `apply_masks` 新增 `_inference_block_snap` 字段读取（由 `generate_mask` / `propagate` 入口写入），按推理开始时的 block 计算 `abs_idx`，即使用户在 `apply_masks` 执行期间切了 block，结果仍写到正确的绝对帧号；并在写盘前校验源帧文件存在，不一致则 silently 丢弃，避免错位 JSON 永久残留。
  - `main.py` 新增 `_cancel_auto_single_timer()`，在所有切帧入口（`cb_frame_slider`、Q/E 跳转、`cb_jump_to_frame`、`cb_prev_frame`/`cb_next_frame`、跨 block）调用一次，关闭"用户点完点立刻拖滑条"那段 120ms 防抖窗口的 race。
- **Reconcile Labels**: 修复旧版 pkl 中 label 已有 `group_id` 属性但值为 `None` 时无法被正确迁移的问题。现在 Reconcile 会同时解析 JSON 里的 `group_id=null` 和 pkl label 的 `group_id=None`：同名 JSON 后续已有数字 `group_id` 时优先沿用该值；同名器械全为 `None` 时才按 pkl label 顺序分配新实例 ID；已有数字 `group_id` 的正常项目保持不变。
- **Reconcile Labels**: `group_id=None` 不再进入 gid 去重、`max()`、`sorted()` 和 remap 计算，避免旧 pkl 在 reconcile 过程中触发类型错误或把多个 label 误判成同一个实例。
- Fixed SAM2 mask over-expansion on Intel XPU by disabling autocast for XPU inference paths. XPU now runs SAM2 in float32 while CUDA keeps AMP enabled, avoiding the positive-logit drift that made masks cover most of the frame.
- Fixed `Load Model` menu callbacks by passing the model name through DearPyGui `user_data` instead of relying on a lambda closure.
- Moved inference busy/progress UI updates back onto the DearPyGui main thread to avoid missing progress updates and native UI crashes from worker-thread DPG calls.

### Changed
- **View 模式切换改为单键 1/2/3/4**：之前需要 Shift+1/2/3/4，现在直接按数字键即可切换 Original / Prompts / Overlay / Mask。输入框焦点内的数字输入不受影响（`_is_input_focused()` 已在键盘 handler 顶部拦截）。README、左侧 View 区域 collapsing header 标题同步更新。
- Added concise terminal logs for SAM2 model loading, checkpoint selection, and load failures.

### Verified
- On Intel Arc 130T with `torch 2.11.0+xpu`, Tiny single-frame mask area matched CPU output on a real annotated frame: `0.0201` mask ratio after the fix versus `0.5503` before the fix.
- Warmed-up Tiny inference on the same project was faster on XPU than CPU: single-frame prediction about 2.0x faster and 6-frame propagation about 3.3x faster.

## [1.5.4] - 2026-05-05
### Added
- Label 列表每行名字前显示 mask 颜色的小色块,UI 与画面 overlay 颜色对应,扫一眼就能知道哪个 label 是哪个色：将原 `add_listbox` 替换为 `child_window` 内逐行 `add_text("■", color=label.col) + add_selectable`,选中态由 selectable 自带高亮维护,W/S 切换、Library 增删等所有调 `refresh_label_listbox()` 的入口自动同步

### Fixed
- **Reconcile Labels**: 新增 pkl 内部 `group_id` 去重(step 1.5)。多个不同 name 的 pkl label 撞同一个 gid 时,按 gid 查 label 的 first-match-wins 路径(`get_label_idx_by_group_id`、`create_combined_mask`、`get_mask_color` 等)会把多个 label 的 mask 全染成第一个 label 的颜色,reassign 也会找错 src_label 导致 prompts 不被搬运。现在优先保留 name 与 JSON 该 gid 对应的 label,其余自动分配新 gid;reconcile 末尾保存 pkl 落盘
- **切换项目 label 没清空**: `_do_load_media` 开头补 `ann.reset()`,清掉上一个项目残留的 `sam_handler.labels`、`masks`、`tracking_results`、`propagation_blocks` 等状态。之前 Load Folder 选了没有 pkl 的新目录时只走 `_do_load_media`,完全不动 labels,旧 label 列表会一直残留在 UI 里(Load Session 走 `load_from_dict` 自带状态替换,不受影响)
- 字体加载补充 `add_font_range(0x2500, 0x25FF)`(Box Drawing + Geometric Shapes 块,包含 ■ U+25A0)。`mvFontRangeHint_Chinese_Full` 默认不覆盖几何符号块,导致色块字符显示为 tofu/问号

### Changed
- **Reassign Label / Delete Label in Range 完成后自动保存 pkl**: 这两个 op 在过程中已经把 JSON 修改写入磁盘,如果 pkl 不一起落盘,中途崩溃或忘 Ctrl+S 退出会导致 JSON 是新的、pkl 是旧的"半保存"脏状态(下次打开 prompts 与 JSON 对不上)。新增 `_autosave_pkl()` 静默保存,失败时打印警告但不抛异常;进度条后缀显示 `(pkl saved)` / `(pkl save failed)`。原则:碰 JSON 的 op 自动同步 pkl,只动内存的小操作(单点 prompt、`-` 删 label、改名)仍由用户手动 Save

## [1.5.3] - 2026-05-05
### Fixed
- 修复按 `C` 键清空当前帧标注后画面不立即刷新的问题：原本只调 `draw_overlays()` 重画提示点/框那一层，未重新渲染底图纹理；overlay/masks 视图下 `create_combined_mask` 又优先从 `self.masks[abs_idx]` 内存读，导致即使 JSON 已删，画面仍显示旧 mask，要切到相邻帧再切回才消失。现在同时清掉 `self.masks` / `self.overlay_imgs` / `self.combined_masks` 中该帧的条目，并改调 `load_and_show_frame()` 立即刷新底图

## [1.5.2] - 2026-05-05
### Added
- **Go to Frame** 跳帧功能：按 `G` 键弹窗输入绝对帧号，自动切换到对应 block 和帧（输入框 Enter 直接确认）
- Reconcile Labels 写 pkl 前自动备份原 pkl 到 `{session}.pkl.bak.YYYYMMDD-HHMMSS`，保留多份历史，覆写崩溃可手动恢复；完成消息显示具体备份文件名

### Fixed
- 修复后台线程直接调用 DPG 引发的偶发崩溃：`_show_progress` 和 session_name_input 写入引入主线程延迟刷新机制（`_pending_progress` / `_pending_session_name` 在 build_ui 主循环 flush）
- 修复加载其他机器保存的 session 后切换 block 失效的问题：`_load_session_from_path` 总是把 `media_path` 重置为本地 `frames_dir` 的绝对路径
- 修复 Reassign Label / Delete Label in Range 弹窗在 label 列表变化后选项显示陈旧值导致索引越界崩溃：解析索引加 try/except + 范围校验，刷新 combo 后主动重置 selected 值到首项
- **Reconcile Labels** 给 `group_id=None` 分配新 gid 时考虑 pkl 中已有 gid，避免与"pkl 创建过但 JSON 没引用"的 label 撞车；优先复用 pkl 同名 label 的 gid，消除 set 迭代顺序带来的非确定性，并大幅减少不必要的 remap
- **Reconcile Labels** remap 时同步迁移 `extra_frame` / `extra_frame_masks`（之前只迁 `masks` / `tracking_results`），修复 remap 触发后 block 末尾额外帧 mask 与 label 错位的问题
- 修复 `reassign_label_in_range` 返回值只统计 JSON 改动数的问题：现在按"实际被修改的帧"去重计数（包含仅内存 mask 迁移而 JSON 未变的帧）

### Changed
- `_is_input_focused` 把 `goto_frame_input` 加入白名单，避免 goto 弹窗内 Shift+1/2/3/4 误触发 view mode 切换、A/D 误切帧等冲突
- `requirements.txt`：dearpygui 从 `>=2.0` 锁定到 `==2.2`，避免不同次版本间 API 差异导致的偶发兼容问题

## [1.5.1] - 2026-05-01
### Changed
- 导出验证视频和 overlay 图片时，label 名称直接标注在对应 mask 区域上方（使用 label 自身颜色 + 白色描边），取代之前左上角的小图例，更直观清晰

### Fixed
- 修复导出验证视频和 overlay 图片时中文 label 名称显示为 `????` 的问题（OpenCV putText 不支持非 ASCII 字符，改用 PIL 渲染文字，使用系统中文字体）

## [1.5.0] - 2026-04-22
### Added
- **Delete Label in Range** 功能（Edit 菜单）：删除指定帧区间内某个 label 的 mask 和标注（JSON shapes、内存 mask、prompts、prop_frames），其他 label 不受影响
- **Reconcile Labels** 功能（Edit 菜单）：扫描项目 JSON 文件，自动修复 pkl 与 JSON 之间的 group_id 不一致问题（白色 mask），支持自定义 Block Size
- **Load Folder 自动恢复 session**：Load Folder 时如果项目目录已有 pkl，自动走完整的 Load Session 流程，恢复 labels、帧位置和 block
- group_id 语义修正：`group_id` 从类别 ID 改为实例 ID，对齐 LabelMe 标准（`label` = 类别名，`group_id` = 实例 ID）
- 支持同名 label 多实例：可添加多个同名 label（如两把"抓钳"），UI 显示 `#group_id` 区分
- `create_new_label()` 和 `add_label()` 支持传入 `group_id` 和 `color` 参数

### Fixed
- 修复跨 block 切换时 overlay 模式被自动重置为 prompts 的问题
- Reconcile 兼容旧版 JSON（`group_id=None` 的 shape 自动分配正确的 group_id 并回写）
- Reconcile 修复链式 remap 覆盖问题（如 1→3, 3→5 不再互相干扰）

### Changed
- Label Library 取消勾选时删除该类别的所有实例（之前只删第一个）
- Label 列表显示格式从 `0: 抓钳` 改为 `0: 抓钳 #1`

## [1.4.3] - 2026-04-20
### Changed
- Prompts 视图下前景点颜色改为跟随 label 自身颜色（与 mask 颜色一致），背景点保持红色

### Fixed
- 修复 Reassign Label 后删除旧标签导致 mask 变白的问题（reassign 只更新了磁盘 JSON，未同步迁移内存中的 mask 数据）

## [1.4.2] - 2026-04-19
### Fixed
- 修复 Single Label Mode 下 Single 按钮覆盖其他 label mask 的问题（勾选 Single Label Mode 时，Single 按钮现在只更新选中 label 的结果，合并写入已有 JSON，不再冲掉其他 label 的标注）

## [1.4.1] - 2026-04-18
### Fixed
- 修复 Reassign Label 后删除旧标签导致 mask 变白的问题（reassign 只更新了磁盘 JSON 的 group_id，未同步迁移内存中的 mask 数据，删除旧标签时内存 mask 被一并清除）

## [1.4.0] - 2026-04-15
### Added
- 新增 **Single Label Mode** 开关（SAM2 Inference 区域的复选框）
  - 开启后，Forward / Backward / All 只传播当前选中的 label，推理结果合并写入已有 JSON，不影响其他 label 的标注
  - 适用于"先标注器械 A 再追加器械 B"的增量标注场景，无需重跑所有 label
- **Ctrl+S** 快捷键一键保存 session

### Changed
- 标签调色板红色后移：前几个 label 不再分配红色（手术视频中红色 mask 易与组织混淆），红色系移至调色板末尾


### Fixed
- 修复 Single Label Mode 下传播不受 Checkpoint 限制的问题（现在与正常模式一致，前向停在下一个 checkpoint，反向停在上一个）

## [1.3.0] - 2026-04-14
### Added
- UI 重构：顶部 toolbar 改为菜单栏（File / Edit / Settings），界面更简洁
- 新增 Reassign Label 功能（Edit 菜单），可在指定帧范围内将标注从一个标签批量转移到另一个标签
- Load Model 改为子菜单，直接选择模型大小即加载，无需先选再点
- Pre-extract 抽帧前新增质量选择弹窗（Standard / High Quality），显示预估磁盘占用

## [1.2.0] - 2026-04-13
### Fixed
- 修复前向传播时添加了传播范围外的 prompt 导致 mask 漂移的问题（如远处 checkpoint 的旧标注染色到其他物体）
- 修复 Intel XPU 上推理精度严重下降的问题（autocast 从 float16 改为 bfloat16，Intel 核显对 FP16 硬件支持差）
- 修复点击 UI 控件（模型选择、标签库等）时误触画布添加标注点的问题
- 修复 Pre-extract 文件对话框在中文路径下显示空目录的问题（DPG 已知 bug，中文路径导致 C++ 层解析失败）
- 修复 Pre-extract 手动输入路径时程序崩溃的问题

### Added
- Pre-extract 旁新增路径输入框 + Go 按钮，可直接粘贴视频路径抽帧，绕过 DPG 文件对话框
- Z 键撤回当前标签在当前帧的最后一个标注点

## [1.1.1] - 2026-04-12
### Fixed
- 修复 Load Session 后 overlay 模式看不到 mask 的问题（清空旧机器的 idx_to_path 路径映射）
- 修复 Load Session 后切 block 失效的问题（修正跨机器的 media_path 路径）

### Changed
- C 键改为清除当前帧**所有 label** 的标注（之前只清当前选中的 label），同时删除该帧的 JSON 文件

## [1.1.0] - 2026-04-11
### Added
- 支持同一标签多实例标注（如两把抓钳可以用同名标签独立标注和传播）
- 引入 `group_id` 机制，导出 JSON 兼容 LabelMe `group_id` 字段
- 旧 pkl 自动迁移：加载后 save 即可转为新格式
- 支持 Intel Arc GPU (XPU)，包括代码适配和安装指南

### Fixed
- 修复 Single 无法覆盖已传播帧的 mask 的问题
- 修复重新传播无法覆盖旧传播结果的问题（传播时自动保护有 prompt 标注的帧不被覆盖）
- 修复 Backward 传播逻辑，改为只使用当前帧或最近的 prompt 帧往回传播

### Changed
- 所有推理函数添加 `torch.inference_mode` 以减少显存占用

## [1.0.0] - 2026-04-11
### Added
- 初始版本发布
- SAM2.1 视频标注工具，支持 Large/Base+/Small/Tiny 四种模型
- 支持单帧预测 (Single)、前向/后向/全量传播
- 支持多标签、点提示和框提示
- 支持 Block 分段标注
- LabelMe 格式 JSON 导出
- 跨平台 checkpoint 下载脚本
