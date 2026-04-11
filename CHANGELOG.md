# Changelog

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
