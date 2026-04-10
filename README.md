# SAMannot-DPG

基于 [SAM2](https://github.com/facebookresearch/sam2)（Meta 的视频分割模型）的 **视频实例分割标注工具**，使用 [DearPyGui](https://github.com/hoffstadt/DearPyGui) 构建高性能 GPU 渲染界面。

你可以用这个工具在视频中标注物体，SAM2 会自动把标注传播到其他帧，最终导出 LabelMe 格式的 JSON 分割结果。

---

## 你需要准备什么

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Windows 10/11 | Linux 也可以，但本教程以 Windows 为例 |
| 显卡 | NVIDIA GPU，显存 >= 6GB | RTX 2060 及以上均可。**没有 N 卡无法运行 SAM2 推理** |
| 显卡驱动 | 支持 CUDA 11.7+ | 打开命令行输入 `nvidia-smi`，能看到版本号就行 |
| 磁盘空间 | >= 10 GB | 模型权重约 850MB，全量抽帧约 4GB（1080p 视频 15000 帧） |
| Anaconda | 已安装 | 没装的话去 [Anaconda 官网](https://www.anaconda.com/download) 下载安装 |

### 怎么检查显卡驱动？

打开命令行（按 `Win+R`，输入 `cmd`，回车），输入：

```
nvidia-smi
```

如果能看到 GPU 名称和 CUDA Version，说明驱动没问题。如果提示"不是内部命令"，说明没装驱动。

---

## 安装步骤（一步一步来）

打开命令行（cmd 或 PowerShell 都可以），按顺序执行以下命令。

### 第 1 步：进入项目目录

```bash
cd 你的项目路径\SAMannot-DPG
```

> 替换成你实际解压/克隆项目的路径，比如 `cd D:\projects\SAMannot-DPG`。

### 第 2 步：创建虚拟环境

```bash
conda create -n samannot python=3.10 -y
```

这会创建一个叫 `samannot` 的独立 Python 环境，不会影响你电脑上其他 Python 程序。

### 第 3 步：激活环境

```bash
conda activate samannot
```

激活后，命令行前面会出现 `(samannot)`，表示你现在在这个环境里了。

> **以后每次打开命令行，都要先运行这条命令。**

### 第 4 步：安装 PyTorch（CUDA 版）

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

> **这一步必须在第 5 步之前执行！** 如果顺序反了，会装成 CPU 版，SAM2 跑不了。
>
> 如果你的 CUDA 版本不是 12.8，把 `cu128` 换成对应版本，比如 `cu124`（CUDA 12.4）。不确定的话用 `nvidia-smi` 查看右上角的 CUDA Version。

### 第 5 步：验证 PyTorch 安装

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**正确输出**（版本号可能不同）：

```
2.7.0+cu128 True
```

如果看到 `+cpu` 或者 `False`，说明装错了，用这条命令重装：

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps torch torchvision torchaudio
```

### 第 6 步：安装其他依赖

```bash
pip install -r requirements.txt
```

### 第 7 步：安装 SAM2 模块

```bash
cd sam2
pip install -e . --no-deps
cd ..
```

`--no-deps` 是为了防止 SAM2 的安装脚本把你刚装好的 CUDA 版 PyTorch 覆盖成 CPU 版。

### 第 8 步：配置 SAM2 的导入路径

```bash
python -c "import sysconfig, pathlib; p = pathlib.Path(sysconfig.get_paths()['purelib']) / 'samannot_sam2.pth'; p.write_text(str(pathlib.Path('.').resolve()) + '\n'); print('OK:', p)"
```

这条命令会在 Python 的包目录下创建一个 `.pth` 文件，让 Python 能找到 SAM2 的代码。

### 第 9 步：下载模型权重

```bash
python download_checkpoints.py --only large
```

这会下载 `sam2.1_hiera_large.pt`（约 850 MB）到 `checkpoints/` 目录。如果网速慢，可以手动从浏览器下载后放到 `checkpoints/` 文件夹里：

```
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 第 10 步：最终验证

```bash
python -c "import torch, sam2; from sam2.build_sam import build_sam2_video_predictor; print('OK', torch.__version__, torch.cuda.is_available())"
```

如果输出 `OK 2.x.x+cu128 True` 且没有报错，恭喜你，环境配好了！

---

## 启动

```bash
conda activate samannot
cd 你的项目路径\SAMannot-DPG
python main.py
```

---

## 使用流程

### 1. 抽帧

- 点击工具栏的 **Pre-extract** 按钮，选择视频文件（.mp4 等）
- 等待抽帧完成（进度条会显示进度）
- 抽好的帧保存在 `projects/{视频名}/frames/` 目录下
- 如果之前已经抽过，会自动跳过

> 全量抽帧一个 1080p、15000 帧的视频大约需要 4GB 磁盘空间。

### 2. 加载帧

- 点击 **Load Folder** 选择刚才抽好的帧目录（`projects/{视频名}/frames/`）
- **Block Size** 设置每次处理的帧数，默认 200，显存小可以调到 50~100
- 用 **<<** / **>>** 按钮切换 Block

### 3. 加载模型

- 点击工具栏的 **Load Model** 按钮
- 等待左侧进度条显示加载完成（通常 5~15 秒）
- 模型只需加载一次，切 Block 不需要重新加载

### 4. 添加标签

- 在左侧 Labels 区输入标签名（比如 "剪刀"），点 **+** 添加
- 用 **W/S** 键切换当前标签
- 点 **Library** 可以从预设标签库里批量选择
- 点标签旁的 **X** 可以从库中删除

### 5. 打标注

- **鼠标左键** 点击画面 = 前景点（绿色，表示"这里是目标"）
- **鼠标右键** 点击画面 = 背景点（红色，表示"这里不是目标"）
- **鼠标右键拖拽** = 画一个矩形框
- 按 **C** 清空当前帧当前标签的标注
- 按 **Delete** 删除选中的标注

### 6. 生成分割 & 传播

- **Single**：只处理当前帧（必须在有标注的帧上点）
- **Forward**：从当前帧往后传播到 checkpoint 或 block 末尾
- **Backward**：从当前帧往前传播到 checkpoint 或 block 开头
- **All**：向前向后都传播
- **Prop Range**：用当前 block 所有帧的标注做批量传播（效果最好）
- **Checkpoint**：在当前帧设置/取消传播断点，Forward/Backward 会停在 checkpoint 处

> 推理时按钮会变灰，等进度条跑完再操作。

### 7. 查看结果

用 **Shift+1/2/3/4** 或左侧 View 区按钮切换视图：

| 视图 | 说明 |
|------|------|
| **Original** | 原图 |
| **Prompt** | 原图 + 标注点/框 |
| **Overlay** | 原图 + 半透明彩色 mask 叠加（无 mask 的帧显示原图） |
| **Mask** | 纯彩色 mask（无 mask 的帧显示黑色） |

> 切换视图模式后，在有 mask 和没有 mask 的帧之间切换时，视图模式会保持不变。

### 8. 保存 & 加载

- **Save**：保存当前标注状态到 `projects/{视频名}/{视频名}.pkl`
- **Load Session**：加载之前保存的 `.pkl` 文件，继续标注（加载后需要重新点 Load Model）
- **Reset**：清空所有标注，重新开始

### 9. 导出

- 展开左侧 **Export** 区域
- **Export Annotations**：导出 LabelMe 格式 JSON + 10 张验证图片
- **Export Verify Video**：生成叠加标注的验证视频（如果是帧文件夹模式，会弹窗让你选源视频）

---

## 快捷键一览

| 按键 | 功能 |
|------|------|
| `A` / `D` | 上一帧 / 下一帧 |
| `W` / `S` | 上一个标签 / 下一个标签 |
| `Q` / `E` | 跳到上/下一个有标注的帧 |
| `C` | 清空当前帧当前标签的标注 |
| `Delete` | 删除选中的标注 |
| `Shift+1/2/3/4` | 切换视图模式（Original/Prompt/Overlay/Mask） |
| `F11` | 全屏 / 退出全屏 |
| `Esc` | 清除输入框焦点（恢复快捷键） |
| 鼠标左键 | 前景点（绿色） |
| 鼠标右键 | 背景点（红色） |
| 鼠标右键拖拽 | 画矩形框 |

> 如果按 A/D 没反应，可能是光标在输入框里。按 `Esc` 或点一下画面就好了。

---

## 时间线（底部进度条）

底部的时间线用颜色标记不同类型的帧：

| 颜色 | 含义 |
|------|------|
| 绿色 | 有手动标注（prompt）的帧 |
| 青色 | SAM2 传播过的帧 |
| 蓝色 | 有 mask 的帧 |
| 红色 | Checkpoint（传播断点） |
| 白色 | 当前帧位置 |

---

## 项目目录结构

每个视频的所有产出文件都在 `projects/{视频名}/` 下：

```
projects/
└── 我的手术视频/
    ├── frames/                       <- 从视频抽出的帧图片
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   └── ...
    ├── .sam_temp/                     <- SAM2 推理用的临时目录（自动管理）
    ├── 我的手术视频.pkl               <- Save 产出的标注存档
    ├── jsons/                         <- Export 产出的 LabelMe JSON
    │   ├── 000002.json                <- 文件名 = 视频帧号
    │   └── ...
    ├── overlays/                      <- Export 产出的验证图片
    └── 我的手术视频_verify.mp4        <- Export Verify Video 产出
```

### JSON 格式（LabelMe）

每个 JSON 文件对应一帧，格式为标准 LabelMe：

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "剪刀",
      "points": [[x1, y1], [x2, y2], ...],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "000045.jpg",
  "imageData": null,
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

---

## 显存不足怎么办？

1. **减小 Block Size**：比如改成 50 或 100
2. **换小模型**：编辑 `annotator.py` 的 `init_sam2` 函数，把 `large` 改成 `tiny` 或 `small`：

   ```python
   # 改成 tiny（最省显存）
   model_cfg_path="sam2.1_hiera_t.yaml"
   ckpt_path=...+"sam2.1_hiera_tiny.pt"
   ```

   同时下载对应权重：`python download_checkpoints.py --only tiny`

| 模型 | 配置文件 | 权重文件 | 显存占用 |
|------|----------|----------|----------|
| tiny | `sam2.1_hiera_t.yaml` | `sam2.1_hiera_tiny.pt` | ~2 GB |
| small | `sam2.1_hiera_s.yaml` | `sam2.1_hiera_small.pt` | ~3 GB |
| base+ | `sam2.1_hiera_b+.yaml` | `sam2.1_hiera_base_plus.pt` | ~4 GB |
| large | `sam2.1_hiera_l.yaml` | `sam2.1_hiera_large.pt` | ~6 GB |

---

## 常见问题

### Q: `nvidia-smi` 提示"不是内部命令"

你没有 NVIDIA 显卡，或者没装驱动。去 [NVIDIA 驱动下载页](https://www.nvidia.com/drivers) 安装。

### Q: PyTorch 装成了 CPU 版

重新执行第 4 步，用 `--force-reinstall --no-deps` 强制重装。

### Q: `import sam2` 报错 ModuleNotFoundError

重新执行第 7 步和第 8 步。

### Q: 点 Load Model 后报错 "Failed to load model"

检查 `checkpoints/` 目录下有没有 `sam2.1_hiera_large.pt` 文件。没有就重新执行第 9 步。

### Q: Single 显示 "Frame X has no prompts"

Single 只处理**当前帧**上的标注。确认你在有绿色点/框的帧上点 Single。可以按 Q/E 跳到有标注的帧。

### Q: Load Session 后推理报错

Load Session 后需要重新点 **Load Model**（模型不能保存到 pkl 文件，每次启动都要重新加载）。

### Q: 中文路径报错

项目目录和视频文件建议放在纯英文路径下，比如 `D:\code\` 而不是 `D:\我的项目\`。

### Q: 20 系（RTX 2060/2070/2080）显卡能用吗？

可以。SAM2 支持 CUDA 11.7+，20 系完全兼容。只是没有 30 系以上的 tf32 加速，推理会稍慢一些。

---

## 致谢

本工具基于 [SAMannot](https://github.com/gergelydinya/SAMannot) 和 [SAM2](https://github.com/facebookresearch/sam2) 开发。
