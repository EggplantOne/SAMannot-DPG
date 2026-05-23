# SAMannot-DPG

基于 [SAM2](https://github.com/facebookresearch/sam2)（Meta 的视频分割模型）的 **视频实例分割标注工具**，使用 [DearPyGui](https://github.com/hoffstadt/DearPyGui) 构建高性能 GPU 渲染界面。

你可以用这个工具在视频中标注物体，SAM2 会自动把标注传播到其他帧，最终导出 LabelMe 格式的 JSON 分割结果。

### v1.5.0 更新 (2026-04-22)

- **Reconcile Labels**（Edit 菜单）：修复 mask 变白问题。扫描 JSON 自动修正 group_id 不一致，支持旧版 JSON 兼容（`group_id=None` 自动分配），可自定义 Block Size
- **Load Folder 自动恢复**：Load Folder 时自动检测已有 session，直接恢复到上次工作位置
- **group_id 语义修正**：对齐 LabelMe 标准，`group_id` 改为实例 ID，支持同名 label 多实例标注
- **跨 block 保持 overlay 模式**：切换 block 不再重置为 prompts 视图

---

## 你需要准备什么

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Windows 10/11 | Linux 也可以，但本教程以 Windows 为例 |
| 显卡 | NVIDIA GPU 或 Intel Arc GPU | 见下方说明 |
| 磁盘空间 | >= 10 GB | 模型权重约 850MB，全量抽帧约 4GB（1080p 视频 15000 帧） |
| Anaconda | 已安装 | 没装的话去 [Anaconda 官网](https://www.anaconda.com/download) 下载安装 |

### 支持的显卡

| 显卡类型 | 示例 | 推荐模型 | 说明 |
|----------|------|----------|------|
| NVIDIA 独显 | RTX 2060/3060/4060 等 | Large | 最佳体验，所有功能正常 |
| Intel Arc 独显/核显 | Arc A770, Arc 130T 等 | Tiny | 速度较慢但可用，见 Intel 安装说明 |
| 无 GPU / 不支持的 GPU | — | Tiny | 回退到 CPU，非常慢 |

### 怎么确认你的显卡类型？

打开命令行（`Win+R` → 输入 `cmd` → 回车）：

```bash
# 检查是否有 NVIDIA GPU
nvidia-smi
# 如果能看到 GPU 名称和 CUDA Version，说明你有 NVIDIA 显卡

# 如果上面报错，检查是否有 Intel GPU
# 打开 设备管理器 → 显示适配器，看是否有 "Intel Arc" 或 "Intel Iris" 字样
```

---

## 安装步骤 — NVIDIA GPU（推荐）

适用于：RTX 2060/3060/4060 等 NVIDIA 独立显卡。

打开命令行（cmd 或 PowerShell），按顺序执行：

### 第 1 步：进入项目目录

```bash
cd 你的项目路径\SAMannot-DPG
```

### 第 2 步：创建虚拟环境

```bash
conda create -n samannot python=3.10 -y
```

### 第 3 步：激活环境

```bash
conda activate samannot
```

> 以后每次打开命令行都要先运行这条。

### 第 4 步：安装 PyTorch（CUDA 版）

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

> **必须在第 5 步之前执行！** 顺序反了会装成 CPU 版。
>
> 如果你的 CUDA 版本不是 12.8，把 `cu128` 换成对应版本（如 `cu124`）。用 `nvidia-smi` 查看右上角的 CUDA Version。

### 第 5 步：验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

应输出 `2.x.x+cu128 True`。如果看到 `+cpu` 或 `False`：

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps torch torchvision torchaudio
```

### 第 6-10 步：通用步骤

跳到下方 [通用安装步骤](#通用安装步骤) 继续。

---

## 安装步骤 — Intel Arc GPU

适用于：Intel Arc A770/A750/A380 独显，或 Intel Core Ultra 处理器的集成 Arc 核显（如 Arc 130T）。

> **为什么 Intel 能跑？** 本工具的代码已做了设备适配——自动检测 `cuda`（NVIDIA）、`xpu`（Intel）、`cpu` 三种设备并选择最佳的。SAM2 的核心推理（卷积、注意力等）都是标准 PyTorch 操作，Intel 的 XPU 后端完全支持。
>
> **性能预期：** Intel Arc 核显（如 130T）比 NVIDIA 独显慢约 10-15 倍，但比纯 CPU 快 3-5 倍。推荐使用 **Tiny 模型** + **小 Block Size**（100-150），单帧推理约 1-3 秒，传播 100 帧约 2-5 分钟。
>
> **会不会损坏电脑？** 不会。核显有硬件级温度保护，过热时自动降频。最差情况是推理时电脑变卡（CPU 和 GPU 共享内存带宽），推理结束就恢复。建议推理时关闭其他大型程序（浏览器、游戏）。

### 第 1 步：进入项目目录

```bash
cd 你的项目路径\SAMannot-DPG
```

### 第 2 步：创建虚拟环境

```bash
conda create -n samannot python=3.10 -y
```

### 第 3 步：激活环境

```bash
conda activate samannot
```

### 第 4 步：安装 PyTorch（Intel XPU 版）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

> XPU stability note: SAM2 inference on Intel XPU should run in float32. This project disables autocast on XPU because BF16 autocast can make SAM2 mask logits drift positive and produce masks that cover most of the frame. CUDA still uses autocast.

> **为什么用 `xpu` 而不是 `cu128`？** Intel GPU 不支持 CUDA（那是 NVIDIA 专属），而是用 Intel 自己的 XPU 后端。PyTorch 2.5+ 已原生支持 Intel XPU，不需要额外装插件。
>
> pip 会自动选择跟你 Python 版本兼容的最新 PyTorch 版本。如果最新版有问题，可以指定版本：
> ```bash
> pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu
> ```

### 第 5 步：验证

```bash
python -c "import torch; print(torch.__version__, hasattr(torch, 'xpu') and torch.xpu.is_available())"
```

You should see a `+xpu` PyTorch build and `True`, for example:

```text
2.11.0+xpu True
```

应输出 `2.x.x+xpu True`。如果输出 `False`：
- 确认你的 Intel GPU 驱动是最新的（[Intel 驱动下载](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)）
- 确认你的 Intel GPU 支持 XPU（Arc 系列和部分 Iris Xe 支持，老款核显不支持）

### 第 6-10 步：通用步骤

跳到下方 [通用安装步骤](#通用安装步骤) 继续。**注意第 7 步和第 9 步有 Intel 专用说明。**

---

## 通用安装步骤

不管你是 NVIDIA 还是 Intel，从这里继续：

### 第 6 步：安装其他依赖

```bash
pip install -r requirements.txt
```

### 第 7 步：安装 SAM2 模块

**NVIDIA 用户：**
```bash
cd sam2
pip install -e . --no-deps
cd ..
```

**Intel 用户：**
```bash
cd sam2
set SAM2_BUILD_CUDA=0
pip install -e . --no-deps
cd ..
```

> `SAM2_BUILD_CUDA=0` 跳过编译 CUDA 自定义算子（Intel GPU 不支持 CUDA kernel）。这个算子只用于 mask 后处理的小优化，跳过不影响分割效果。
>
> `--no-deps` 防止安装脚本把你刚装好的 GPU 版 PyTorch 覆盖成 CPU 版。

### 第 8 步：配置 SAM2 的导入路径

```bash
python -c "import sysconfig, pathlib; p = pathlib.Path(sysconfig.get_paths()['purelib']) / 'samannot_sam2.pth'; p.write_text(str(pathlib.Path('.').resolve()) + '\n'); print('OK:', p)"
```

### 第 9 步：下载模型权重

**NVIDIA 独显推荐 Large（精度最高，减少人工复查）：**
```bash
python download_checkpoints.py --only large
```

**显存不足可选 Base+：**
```bash
python download_checkpoints.py --only base
```

**Intel GPU 或 NVIDIA 显存不足推荐 Tiny（149MB，更快更省显存）：**
```bash
python download_checkpoints.py --only tiny
```

> 也可以手动下载放到 `checkpoints/` 目录：
> - Base+: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt`
> - Large: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`
> - Tiny: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`

### 第 10 步：最终验证

```bash
python -c "import torch, sam2; from sam2.build_sam import build_sam2_video_predictor; print('OK', torch.__version__)"
```

输出 `OK 2.x.x...` 且没有报错，环境就配好了！

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

- 点击菜单栏 **File → Pre-extract (browse)** 选择视频文件，或 **File → Pre-extract (paste path)** 直接粘贴视频路径
- 弹出质量选择窗口，显示帧数和预估磁盘占用：**Standard**（约 200KB/帧）或 **High Quality**（约 300KB/帧，接近无损）
  - 例：1080p 视频 15000 帧，Standard 约 3GB，High Quality 约 4.5GB
- 等待抽帧完成（进度条会显示进度）
- 抽好的帧保存在 `projects/{视频名}/frames/` 目录下
- 如果之前已经抽过，会自动跳过

### 2. 加载帧

- 点击菜单栏 **File → Load Folder** 选择刚才抽好的帧目录（`projects/{视频名}/frames/`）
- **Block Size** 在菜单栏 **Settings** 中设置，默认 200，显存小可以调到 50~100
- 用菜单栏的 **<<** / **>>** 按钮切换 Block

### 3. 加载模型

- 点击菜单栏 **Load Model**，选择模型大小（Large/Base+/Small/Tiny）即开始加载
- 等待左侧进度条显示加载完成（通常 5~15 秒）
- 模型只需加载一次，切 Block 不需要重新加载

### 4. 添加标签

- 在左侧 Labels 区输入标签名（比如 "剪刀"），点 **+** 添加
- 支持同名标签：如果一帧里有两个同类物体（如两把抓钳），可以添加两个 "抓钳" 标签，它们各自独立标注和传播
- 用 **W/S** 键切换当前标签
- 点 **Library** 可以从预设标签库里批量选择
- 点标签旁的 **X** 可以从库中删除

### 5. 打标注

- **鼠标左键** 点击画面 = 前景点（绿色，表示"这里是目标"）
- **鼠标右键** 点击画面 = 背景点（红色，表示"这里不是目标"）
- **鼠标右键拖拽** = 画一个矩形框
- 按 **C** 清空当前帧**所有标签**的标注（同时删除该帧导出过的 JSON）
- 按 **Delete** 删除选中的标注
- 按 **Z** 撤回当前 label 在当前帧的最后一个 prompt；如果撤回后该 label 在本帧已无任何 prompt，会同步清掉之前自动生成的 mask（画面与 prompts 保持一致）

> **自动单帧推理**：加载模型后，每加一个点/框会自动触发一次 Single 推理（约 120ms 去抖），无需手动点 **Single** 按钮。Forward / Backward / All 仍需手动触发。

### 6. 生成分割 & 传播

- **Single**：只处理当前帧（必须在有标注的帧上点）
- **Forward**：从当前帧往后传播到 checkpoint 或 block 末尾
- **Backward**：从当前帧往前传播到 checkpoint 或 block 开头
- **All**：向前向后都传播
- **Single Label Mode**（复选框）：勾选后，Forward/Backward/All 只传播当前选中的 label，结果合并到已有 JSON 中，不覆盖其他 label 的标注。适用于先标注 A 再追加 B 的场景
- **Checkpoint**：在当前帧设置/取消传播断点，Forward/Backward 会停在 checkpoint 处

> 推理时按钮会变灰，等进度条跑完再操作。

### 7. 查看结果

用 **1/2/3/4** 或左侧 View 区按钮切换视图：

| 视图 | 说明 |
|------|------|
| **Original** | 原图 |
| **Prompt** | 原图 + 标注点/框 |
| **Overlay** | 原图 + 半透明彩色 mask 叠加（无 mask 的帧显示原图） |
| **Mask** | 纯彩色 mask（无 mask 的帧显示黑色） |

> 切换视图模式后，在有 mask 和没有 mask 的帧之间切换时，视图模式会保持不变。

### 8. 保存 & 加载

- **File → Save**：保存当前标注状态到 `projects/{视频名}/{视频名}.pkl`
- **File → Load Session**：加载之前保存的 `.pkl` 文件，继续标注（加载后需要重新点 Load Model）
- **File → Reset**：清空内存中的标注状态（不删除已导出的 JSON 文件）

> Edit 菜单提供 **Reassign Label** / **Delete Label in Range** / **Reconcile Labels** 三种批量编辑功能，详见下方 [Edit 菜单 — 批量编辑](#edit-菜单--批量编辑)。

### 9. 导出

- 展开左侧 **Export** 区域
- **Export Annotations**：导出 LabelMe 格式 JSON + 10 张验证图片
- **Export Verify Video**：生成叠加标注的验证视频（如果是帧文件夹模式，会弹窗让你选源视频）

---

## Edit 菜单 — 批量编辑

菜单栏 **Edit** 下提供三种跨帧批量操作。每项都同时改写 **磁盘上的 JSON** 和 **内存中的 prompts / masks / prop_frames**，无需重新传播。

> 帧号都是**绝对帧号**（视频原始帧号，时间线上看到的那个），不是 block 内的局部下标。范围都是**闭区间** `[start, end]`。

### Reassign Label — 把一个标签的标注转移到另一个标签

把指定帧范围内属于"源标签"的所有 mask 和 prompts 转移给"目标标签"。

**典型场景：**
- 标到一半才发现前 200 帧把 "抓钳" 错标成了 "剪刀"，想批量改回去
- 同名多实例下不小心把一个实例的标注混进了另一个实例

**怎么用：**
1. Edit → Reassign Label
2. **Source label**：要从哪个标签搬走（下拉框显示 `序号: 标签名`）
3. **Target label**：搬到哪个标签
4. **Frames**：起止绝对帧号
5. Apply

如果目标标签在某帧已经存在 mask，源 mask 会和目标 mask **按位或合并**（不会覆盖丢失）。

### Delete Label in Range — 在帧范围内删除某个标签的标注

把指定帧范围内属于该标签的所有 mask 和 prompts 全部删掉。

**典型场景：**
- 某个标签在 500-800 帧之间传播错了，想整段删掉重新打
- 物体已经离开画面但 mask 还在拖尾

**怎么用：**
1. Edit → Delete Label in Range
2. **Label to delete**：要清掉的标签
3. **Frames**：起止绝对帧号
4. Apply

如果某帧 JSON 中删完后再没有任何 shape，整个 JSON 文件会被自动删除（时间线上的蓝色标记会消失）。

### Reconcile Labels — 修复 group_id 不一致

扫描所有 JSON，把 JSON 里的 `group_id` 与 pkl 里的标签对齐，修复以下情况：
- 加载 session 后 mask 显示成白色（pkl 和 JSON 的 group_id 对不上）
- 旧版 JSON 没有 `group_id` 字段（自动按 label 名分配）
- 多实例标签被错误合并

**怎么用：**
1. Edit → Reconcile Labels
2. **Block Size**：可以改 block size（用旧 block size 标注后想换粒度时用得上）
3. Run Reconcile

> ⚠️ **重要：** 跑之前先手动删掉**明显标错的 JSON**（例如选错文件夹后误标的整段 block）。否则工具无法区分"错误标注"和"同名多实例"，会把它们当成合法实例保留。

---

## 快捷键一览

| 按键 | 功能 |
|------|------|
| `A` / `D` | 上一帧 / 下一帧 |
| `W` / `S` | 上一个标签 / 下一个标签（画面顶部短暂弹出当前标签名 + 颜色提示） |
| `Q` / `E` | 跳到上/下一个有标注的帧 |
| `G` | 弹窗输入绝对帧号跳转（跨 block 也可以，会自动切到对应 block） |
| `C` | 清空当前帧所有标签的标注（同时删除该帧 JSON） |
| `Z` | 撤回当前标签在当前帧的最后一个标注点 |
| `Delete` | 删除选中的标注 |
| `1/2/3/4` | 切换视图模式（Original/Prompt/Overlay/Mask） |
| `F11` | 全屏 / 退出全屏 |
| `Ctrl+S` | 保存 session |
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
      "label": "抓钳",
      "points": [[x1, y1], [x2, y2], ...],
      "group_id": 1,
      "shape_type": "polygon",
      "flags": {}
    },
    {
      "label": "抓钳",
      "points": [[x3, y3], [x4, y4], ...],
      "group_id": 2,
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

> `group_id` 用于区分同类物体的不同实例（如两把抓钳）。不同类别的物体通过 `label` 字段区分。

---

## 模型选择与性能

工具栏下拉框可选模型大小，不需要改代码：

| 模型 | 权重大小 | 显存占用 | 单帧速度(NVIDIA) | 推荐场景 |
|------|----------|----------|-------------------|----------|
| Tiny | 149 MB | ~500 MB | ~73ms | Intel GPU / 显存不足 / 快速预览 |
| Small | 184 MB | ~800 MB | ~100ms | 轻量需求 |
| Base+ | 323 MB | ~1.2 GB | ~140ms | 显存不足时的替代选择 |
| **Large** | **856 MB** | **~1.6 GB** | **~198ms** | **推荐，精度最高，减少人工复查** |

**显存不足？** 减小 Block Size（100-150）或换 Tiny 模型。

**Intel 核显用户提速建议：**
- 选 **Tiny** 模型
- Block Size 设 **100-150**
- 用 **Checkpoint** 控制传播范围（不用把 Block Size 调小也能减少单次传播帧数）
- 推理时关闭浏览器等占内存的程序

**Intel XPU performance note:**
- First inference can be slower because PyTorch XPU initializes and compiles kernels.
- After warm-up on Intel Arc 130T with Tiny, this project measured about 2.0x faster single-frame prediction and about 3.3x faster 6-frame propagation compared with CPU.
- If masks cover most of the frame only on XPU, ensure this version is used so XPU autocast is disabled.

---

## 常见问题

### Q: `nvidia-smi` 提示"不是内部命令"

你可能没有 NVIDIA 显卡。检查设备管理器 → 显示适配器：
- 看到 "Intel Arc" → 按 [Intel 安装步骤](#安装步骤--intel-arc-gpu) 操作
- 看到 "Intel UHD/Iris"（非 Arc）→ 只能用 CPU 模式，会很慢
- 看到 "NVIDIA GeForce/RTX" → 去 [NVIDIA 驱动下载页](https://www.nvidia.com/drivers) 安装驱动

### Q: Intel GPU 验证时 `torch.xpu.is_available()` 返回 False

- 更新 Intel GPU 驱动到最新版：[Intel 驱动下载](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)
- 确认你装的是 XPU 版 PyTorch（`--index-url https://download.pytorch.org/whl/xpu`）
- 部分老款 Intel 核显（非 Arc 架构）不支持 XPU

### Q: Intel GPU 上推理很慢

正常。Intel 核显比 NVIDIA 独显慢 10-15 倍。建议用 Tiny 模型 + 小 Block Size + Checkpoint 控制范围。

### Q: PyTorch 装成了 CPU 版

重新执行第 4 步，用 `--force-reinstall --no-deps` 强制重装。

### Q: `import sam2` 报错 ModuleNotFoundError

重新执行第 7 步和第 8 步。

### Q: 点 Load Model 后报错 "Failed to load model"

检查 `checkpoints/` 下有没有对应的 `.pt` 文件（下拉框选了 Tiny 就要有 `sam2.1_hiera_tiny.pt`）。没有就重新执行第 9 步。

### Q: Single 显示 "Frame X has no prompts"

Single 只处理**当前帧**上的标注。确认你在有绿色点/框的帧上点 Single。按 Q/E 跳到有标注的帧。

### Q: Load Session 后推理报错

Load Session 后需要重新点 **Load Model**（模型不能保存到 pkl 文件，每次启动都要重新加载）。

### Q: 中文路径报错

项目目录和视频文件建议放在纯英文路径下。

### Q: 推理时电脑会坏吗？

不会。GPU 有硬件级温度保护，过热自动降频。核显功耗上限约 15-30W，远低于独显。

---

## 致谢

本工具基于 [SAMannot](https://github.com/gergelydinya/SAMannot) 和 [SAM2](https://github.com/facebookresearch/sam2) 开发。
