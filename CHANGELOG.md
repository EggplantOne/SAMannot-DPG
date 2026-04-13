# Changelog

## [1.2.0] - 2026-04-13
### Fixed
- 修复前向传播时添加了传播范围外的 prompt 导致 mask 漂移的问题（如远处 checkpoint 的旧标注染色到其他物体）
- 修复 Pre-extract 文件对话框在中文路径下显示空目录的问题（DPG 已知 bug，中文路径导致 C++ 层解析失败）
- Pre-extract 回调增加 `file_path_name` fallback，防止手动输入路径时程序崩溃

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
