# Changelog

## [1.5.1] - 2026-05-01
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
