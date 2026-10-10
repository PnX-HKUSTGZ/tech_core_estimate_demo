# 独立 YOLO 数据集离线增强

`dataset_augment` 从完整 YOLO Detection / Pose 数据集生成独立的新数据集。只依赖标准库、NumPy、OpenCV 和 PyYAML，使用仓库根目录 `.venv`；核心包不导入 `nn`、`blender`、`perception`，无需其他模块的中间结果。所有处理在 CPU 上运行。

算法来源固定为 `tech_core_hkust@cb5ef2f3dea1223c0fedbd358c4129b2c586cb33`：成像和遮挡来自 `automation/augment_photometric.py`，随机和目标框裁剪来自 YOLO 在线增强实现；通用标签合同、仿射、翻转、独立数据集读写及执行器在本仓库实现。授权见根目录 `THIRD_PARTY_NOTICES.md`。

## 快速使用

从仓库根目录运行：

```bash
uv run --no-sync python -m dataset_augment \
  --source /path/to/dataset --output /path/to/augmented

# 同一套参数通过 rm-nn 转发：
uv run --no-sync rm-nn yolo augment \
  --source /path/to/dataset --output /path/to/augmented \
  --set generation.copies_per_image=3 \
  --set 'transforms.affine.degrees=[-15,15]'

# 检查全部输入和预计数量，不写输出数据集：
uv run --no-sync python -m dataset_augment \
  --source /path/to/dataset --output /path/to/augmented --dry-run

# 查看所有最终配置，包括每个划分解析后的预设：
uv run --no-sync python -m dataset_augment --print-config
```

默认仅增强 train，每张原图保留并生成一个变体，训练集成为原来的两倍；val/test 原样复制。输出目录必须不存在。YAML 示例见 [configs/default.yaml](configs/default.yaml)。

```yaml
dataset:
  source: /path/to/dataset
  output: /path/to/augmented
generation:
  copies_per_image: 2
splits:
  train:
    overrides:
      transforms:
        affine:
          degrees: [-15, 15]
        occlusion:
          p: 0.3
```

```bash
uv run --no-sync python -m dataset_augment --config augment.yaml
```

Python API：

```python
from dataset_augment import augment_dataset, load_config

if __name__ == "__main__":
    config = load_config("augment.yaml", overrides=["generation.copies_per_image=3"])
    summary = augment_dataset(config)
```

`load_config(path=None, *, overrides=(), source=None, output=None, workers=None, seed=None)` 返回 `AugmentConfig`；`augment_dataset(config, dry_run=False)` 返回 JSON 可序列化的摘要字典。`dataset_augment.config.config_to_dict(config)` 返回配置副本，包含 `resolved_splits`。API 调用应放在 `if __name__ == "__main__"` 下，保证进程池可安全启动。完整最小脚本见 [examples/augment_dataset.py](../../examples/augment_dataset.py)。

## 输入、输出与标签合同

- `source` 接受数据集目录（读取其中 `dataset.yaml`）或 YAML 文件。支持标准 `images/<split>/`、`labels/<split>/` 布局及其中的嵌套子目录。划分键为 `train`、`val`、`test`；已声明的划分必须非空，未声明的划分不生成。数据集 YAML 的 `path` 相对其所在目录解析（省略则为该目录）；train/val/test 路径相对 path 或使用绝对路径，必须指向 `images/<split>`。对应标签从同级 `labels/<split>` 读取。
- Detection 每行 `class cx cy w h`；Pose 每行在其后接 `K` 个 `x y v`，YAML 声明 `kpt_shape: [K,3]`。框、点使用归一化坐标；`v=0/1/2` 分别为缺失、遮挡、可见。`names` 为非空列表或从 0 起连续的 ID 映射，类别 ID 必须在其中。任意类别数、K 点数、多目标、空标签负样本均受支持。
- 输入 YAML 可含 `names`、`kpt_names`、`flip_idx` 等有效类别和关键点元数据。无需固定双类别、12 点或特定图片名称。缺失标签默认报错；合法空文件直接作为负样本。
- 图片支持常见 JPEG、PNG、BMP、WebP、TIFF，要求 8 位三通道或灰度；灰度变体转为 BGR。带 alpha 的图片明确拒绝。`encoding.format=source` 需要源格式可被 OpenCV 编码。
- 第一版不支持分割、OBB、`[K,2]` 关键点、网络下载、TXT 图片清单或划分路径列表。非法标签、不可解码图片、无效映射及输出重叠均在生成前报错。
- 输入只读，保留原图和未增强划分按字节复制；输出为独立普通文件。输出不能与任何输入目录重叠。有效元数据保留，路径重写；不复制训练缓存和失效 COCO 标注。
- 正样本若几何增强后失去所有目标，默认最多尝试 10 次，随后恢复原始几何继续遮挡和成像；原始负样本不触发此重试。待增强的原始正样本连过滤条件（包括指定 resize 后的回退条件）也无法满足时，预检拒绝。

对每个启用的划分，`N` 为输入图数，`f=fraction`，`C=copies_per_image`，按稳定哈希顺序精确选取 `M=floor(N*f+0.5)` 张：保留原图时输出 `N+M*C` 张；不保留原图时输出 `N-M+M*C` 张。关闭增强的划分输出 `N` 张；不重新划分、跨划分采样或修改类别。

### 几何、可见性和翻转语义

固定顺序为 **裁剪 → 仿射 → 翻转 → letterbox → 遮挡 → 白平衡/曝光 → 对比度 → Gamma → 暗角 → 模糊 → 噪声 → JPEG 退化 → 编码**。

几何变换同步应用于图片、框四角和有效关键点。坐标采用连续像素边缘：合法范围为闭区间 `[0,W]`、`[0,H]`，归一化点允许恰为 0 或 1；因此边界点翻转两次后仍可恢复。仅容忍浮点误差量级的越界（`1e-12 * max(W,H,1)`）并夹紧到边缘。图片采样映射至 OpenCV 像素中心。框多边形与每一步视口求交，面积比例按多边形计算，最终取轴对齐包围框。关键点首次出界立即设为 `(0,0,0)`，之后填边不会恢复它；`v=0` 始终保持零填充。删除框时删除对应整行点。

Pose 水平翻转优先使用 `transforms.horizontal_flip.keypoint_indices`，否则用输入 YAML 的 `flip_idx`；垂直翻转必须显式配置 `vertical_flip.keypoint_indices`。映射是长度 K 的完整置换，且执行两次恢复原序；无需换名时也必须明确给出恒等映射。不根据点名猜测映射，不做按类别的不同映射。显式水平映射写入输出 YAML 的 `flip_idx`。Detection 无需点映射。

## 完整配置参考

下表列出全部可配置项。除 resize、geometry 和互斥 blur 外，各增强都有 `enabled`（布尔）及 `p`（0–1），按每个生成变体独立抽取。浮点范围均匀采样，整数范围包含两端，相同端点表示常数。设置 `enabled: false` 可关闭；允许所有增强都不触发，并在逐图记录中体现。

优先级：**预设 → YAML 全局配置 → YAML 划分覆盖 → CLI 全局覆盖 → CLI 划分覆盖**。列表整体替换。所有字段均可通过重复 `--set 完整键=YAML值` 设置，未知字段/错误类型/非法范围直接报错。多个 `--set` 不得重复或使用重叠的父子键。`--source/--output/--workers/--seed` 是对应字段别名，与同键 `--set` 同时出现时报错。YAML 相对路径相对配置文件，CLI 相对路径相对当前目录。`--print-config` 只解析配置然后退出，不读取数据集；输入 YAML 中的 flip_idx 等元数据在执行预检后补齐到输出审计配置。`--dry-run` 完整读取、解码、检查输入并输出预计数量，不创建输出或临时目录。两种模式互斥。

### 数据集与生成

| 完整配置键 | 默认值／合法值 | 含义 |
|---|---|---|
| `dataset.source` | 执行时必填 | 输入目录或数据集 YAML；`--print-config` 可省略 |
| `dataset.output` | 执行时必填 | 不存在且不与输入重叠的新目录 |
| `dataset.task` | `auto`；`detect`、`pose` | auto 根据 `kpt_shape` 决定任务 |
| `dataset.allow_missing_labels` | `false` | true 将缺失标签视为空，并计数 |
| `generation.copies_per_image` | `1`，整数 ≥1 | 每张选中源图的变体数 |
| `generation.keep_originals` | `true` | false 用变体替换选中原图；未选中仍复制 |
| `generation.seed` | `20261003`，0…2⁶⁴−1 | 全局种子 |
| `splits.<name>.enabled` | train=true，val/test=false | 是否增强该划分，`<name>` 仅 train/val/test |
| `splits.<name>.fraction` | `1.0`，0–1 | 精确选择比例 |
| `splits.<name>.preset` | train=`moderate`，其余=`mild` | 使用的增强预设 |
| `splits.<name>.seed` | `null` 或 64 位非负整数 | null 由总种子和划分名派生 |
| `splits.<name>.overrides` | `{}` | 仅包含 `transforms`、`labels` 的划分专用覆盖 |

`transforms` 和 `labels` 是全局配置组，显式设置会作用于所有划分的相应字段。只想修改训练集时使用 `splits.train.overrides`。验证集使用 mild 时，几何及遮挡默认关闭；显式启用才改变该行为。

### 几何增强（以下键加前缀 `transforms.`）

| 配置键 | moderate 默认 | 含义和范围 |
|---|---|---|
| `crop.enabled/p` | `true / 0.30` | mild 的 enabled=false；每次最多一个裁剪 |
| `crop.mode` | `choice` | `random`、`box`、`choice` |
| `crop.choice_weights` | `{random: 0.60, box: 0.40}` | 相对权重，非负且和大于零；选中后不再抽概率 |
| `crop.random.width_ratio` | `[0.65,1.0]` | 裁剪宽度占原宽比例，0 < 下限 ≤ 上限 ≤ 1 |
| `crop.random.height_ratio` | `[0.65,1.0]` | 裁剪高度比例，独立于宽度 |
| `crop.box.target_classes` | `null` | null 为全部类别，或不重复的非负类别 ID 列表 |
| `crop.box.scale` | `[1.4,3.2]` | 随机目标框宽高的放大倍数，正数 |
| `crop.box.center_jitter` | `0.15` | 中心偏移幅度，占目标框宽/高比例，非负 |
| `crop.box.min_crop_size_ratio` | `0.25` | 相对原图的最小裁剪宽、高，(0,1]；无合格框时跳过 |
| `affine.enabled/p` | `true / 0.50` | mild 的 enabled=false；以当前画布中心一次变换 |
| `affine.degrees` | `[-10,10]` | 平面旋转角度，正值为视觉逆时针 |
| `affine.scale` | `[0.90,1.10]` | 等比例缩放，正数；画布固定，放大可裁边，缩小填边 |
| `affine.translate_x` | `[-0.05,0.05]` | 水平位移占当前宽度比例 |
| `affine.translate_y` | `[-0.05,0.05]` | 垂直位移占当前高度比例 |
| `horizontal_flip.enabled/p` | `false / 0.30` | 水平镜像 |
| `horizontal_flip.keypoint_indices` | `null` | 显式点映射，否则读取 YAML flip_idx |
| `vertical_flip.enabled/p` | `false / 0.10` | 垂直镜像 |
| `vertical_flip.keypoint_indices` | `null` | Pose 启用时必须给出 |
| `resize.size` | `source` | 恢复当前源图大小，或 `[width,height]` 正整数；始终居中等比 letterbox |
| `geometry.interpolation` | `linear` | `nearest`、`linear`、`cubic` |
| `geometry.fill_rgb` | `[114,114,114]` | 仿射及 letterbox 的填边 RGB 整数，0–255 |

resize 仅作用于变体；保留的原图及复制划分保持原尺寸、原字节。

### 成像增强（以下键加前缀 `transforms.`）

| 配置键 | moderate 默认 | mild 默认 | 作用 |
|---|---|---|---|
| `exposure.enabled/p` | `true / 1` | 相同 | 曝光 |
| `exposure.ev` | `[-0.35,0.35]` | `[-0.15,0.15]` | RGB × 2^EV |
| `white_balance.enabled/p` | `true / 1` | 相同 | 通道增益 |
| `white_balance.r_gain` | `[0.90,1.12]` | `[0.96,1.04]` | 红通道独立增益，正数 |
| `white_balance.g_gain` | `[0.90,1.12]` | `[0.96,1.04]` | 绿通道独立增益，正数 |
| `white_balance.b_gain` | `[0.90,1.12]` | `[0.96,1.04]` | 蓝通道独立增益，正数 |
| `contrast.enabled/p` | `true / 1` | 相同 | 对比度 |
| `contrast.factor` | `[0.86,1.18]` | `[0.95,1.06]` | (x−center)×factor+center，正数 |
| `contrast.center` | `0.5` | 相同 | 对比中心，0–1 |
| `gamma.enabled/p` | `true / 1` | 相同 | Gamma |
| `gamma.gamma` | `[0.85,1.18]` | `[0.94,1.06]` | clip(x,0,1)^gamma，正数，大于 1 变暗 |
| `vignette.enabled/p` | `true / 0.65` | 相同 | 边缘变暗 |
| `vignette.strength` | `[0,0.28]` | `[0,0.10]` | 暗角强度，0–1 |
| `vignette.center_x` | `[0.42,0.58]` | `[0.47,0.53]` | 暗角中心归一化 x，0–1 |
| `vignette.center_y` | `[0.42,0.58]` | `[0.47,0.53]` | 暗角中心归一化 y，0–1 |
| `blur.enabled` | `true` | 相同 | 互斥选择，无额外 p |
| `blur.weights` | `{none: 0.40, gaussian: 0.35, motion: 0.25}` | `{none: 0.70, gaussian: 0.30, motion: 0}` | 不模糊、高斯、短线运动三选一，非负且和大于零 |
| `blur.kernel_sizes` | `[3,5,7]` | `[3]` | 正奇数像素核尺寸 |
| `blur.kernel_weights` | `[0.55,0.35,0.10]` | `[1]` | 与核尺寸等长、非负且和大于零 |
| `blur.gaussian_sigma` | `[0.5,1.6]` | `[0.35,0.75]` | 高斯标准差，像素，正数 |
| `blur.motion_angle` | `[0,180]` | 相同但 motion 默认关闭 | 方向角均匀采样，上界不含 |
| `noise.enabled/p` | `true / 0.65` | 相同 | 每像素每通道独立高斯噪声 |
| `noise.sigma_uint8` | `[0,7]` | `[0,3]` | 零均值噪声标准差，0–255 数值尺度，非负 |
| `jpeg.enabled/p` | `true / 1` | 相同 | JPEG 编解码退化 |
| `jpeg.quality` | `[70,100]` | `[85,100]` | 整数范围，1–100 |

以上为显示 RGB 上的成像变换，不是标定后的物理相机模型。本模块不含雾、锐化、色差、Mosaic、MixUp、CutMix 或其他在线 Albumentations 项。

### 矩形遮挡（以下键加前缀 `transforms.occlusion.`）

| 配置键 | 默认值 | 含义 |
|---|---|---|
| `enabled/p` | `true / 0.20` | mild 默认关闭；概率分母是生成变体数 |
| `count` | `[1,2]` | 正整数矩形数量范围 |
| `width_ratio` | `[0.04,0.14]` | 当前画布宽度比例，(0,1] |
| `height_ratio` | `[0.04,0.14]` | 当前画布高度比例，(0,1] |
| `keypoint_anchor_p` | `0.80` | 有 v=2 点时以其为中心的概率 |
| `anchor_jitter` | `0.25` | 中心随机偏移，矩形宽/高倍数，非负 |
| `fill_mode` | `random_color` | `random_color`、`constant`、`noise` |
| `fill_rgb` | `[0,0,0]` | constant 颜色，0–255 RGB 整数 |
| `color_base` | `[20,230]` | 随机纯色整数基值，0–255 |
| `color_jitter` | `[-20,20]` | RGB 每通道整数偏移，−255…255；最终裁剪至 0–255 |
| `max_bbox_overlap` | `1.0` | 每个矩形覆盖任意框的面积比例上限，0–1 |
| `max_attempts` | `20` | 每个矩形最大采样次数，正整数；失败跳过该矩形 |

无可见关键点时（包括 Detection、负样本）随机选中心。遮挡把覆盖的 `v=2` 改为 `v=1`，坐标不变，已有 `v=0/1` 不变，框不缩小。遮挡矩形使用半开范围 `[x0,x1) × [y0,y1)`，与图像切片一致，右/下边界上的点不算被此矩形遮挡；这与几何视口的闭区间合法性分别定义。矩形可重叠，单矩形覆盖约束不等于累计约束；累计覆盖率按矩形并集记录。源脚本按整个训练集精确抽取 20% 遮挡，本模块按每个生成变体概率 0.20 抽取。

### 标签过滤

| 完整配置键 | 默认值 | 含义 |
|---|---|---|
| `labels.min_bbox_area_ratio` | `0.10` | 每次视口裁剪保留的最小框多边形面积比例，0–1 |
| `labels.min_bbox_width_px` | `2` | 最终框最小宽度，正数像素 |
| `labels.min_bbox_height_px` | `2` | 最终框最小高度，正数像素 |
| `labels.min_keypoints` | `0` | Pose 每目标最少图内点（v>0），非负整数 |
| `labels.empty_positive_policy` | `retry` | `retry` 重试/回退；`allow` 允许正样本成为负样本 |
| `labels.max_geometry_attempts` | `10` | 含首次的最多几何尝试次数，正整数 |
| `labels.precision` | `8` | 新标签归一化数值的小数位数，1–17 |

### 编码、并发、日志与预览

| 完整配置键 | 默认值 | 含义 |
|---|---|---|
| `encoding.format` | `png` | 变体格式：`png`、`jpg`、`source` |
| `encoding.jpeg_quality` | `95` | 未触发 JPEG 退化时的 JPG 保存质量，整数 1–100 |
| `encoding.png_compression` | `3` | PNG 压缩，整数 0–9 |
| `runtime.workers` | `4` | CPU 进程数，整数 1–64 |
| `runtime.opencv_threads` | `1` | 仅工作进程内 OpenCV 线程数，正整数 |
| `runtime.max_in_flight` | `null` → `2 × workers` | 有界在途任务数，正整数 |
| `runtime.progress_every` | `100` | 每处理多少输出样本记录进度，正整数 |
| `logging.level` | `INFO` | DEBUG、INFO、WARNING、ERROR、CRITICAL |
| `logging.max_bytes` | `10485760` | 每个日志文件的轮转大小，正整数字节 |
| `logging.backup_count` | `3` | 轮转备份数，正整数 |
| `preview.count_per_split` | `8` | 每个增强划分的前后对照预览数，非负整数，0 关闭 |
| `preview.overlay_labels` | `true` | 预览叠加框和关键点可见性 |
| `preview.tile_width` | `640` | 每个预览图块的宽度，正整数 |

PNG 变体触发 JPEG 退化后解码再存 PNG；JPG 变体直接复用该次 JPEG 字节，不进行第二次有损编码。未触发退化而保存 JPG 时，仍有 JPG 格式本身的压缩。`source` 格式行为依照源扩展名。

## 可复现与验收记录

每个变体的种子由全局/划分种子、源相对路径、变体编号、操作名和几何重试编号派生；不同操作不共用随调用顺序漂移的随机流。相同输入和配置在 1/4 个工作进程下得到相同图像、标签与确定性逐图记录。不同 OpenCV/NumPy 版本可能影响编码或数值，依赖版本写入记录。manifest 另外记录源与输出 dataset.yaml 的 SHA-256，以及 Python、NumPy、OpenCV、PyYAML 和项目版本；发布前核对源 YAML 未被改动。模型权重和 GPU 不参与本模块。

输出包含：

```text
output/
├── images/{train,val,test}/...
├── labels/{train,val,test}/...
├── dataset.yaml
└── augmentation/
    ├── config.json            # 最终配置及 resolved_splits
    ├── images.jsonl           # 每图路径、SHA-256、种子、变换、可见性/过滤/回退
    ├── previews/              # 前后对比，按配置决定数量
    ├── run.log                # 轮转日志
    ├── status.json
    └── manifest.json          # 仅全量成功写入
```

完整输出在临时目录生成，逐文件原子写入，全量成功后发布最终目录。失败状态、日志和已有逐图审计保留于输出父目录的 `.<output-name>.augmentation-*` 临时目录，其位置会写到 stderr；清理其中图像、标签和预览，不会出现完成标记；修复原因后重新运行。第一版不支持断点续跑。

根目录运行测试与类型检查：

```bash
uv run --no-sync python -m unittest discover -s tests -p 'test_augment*.py' -v
uvx --from mypy==1.18.2 mypy --cache-dir=/dev/null src/dataset_augment
```

输出 dataset.yaml 仅保留 `names`、`nc`、`kpt_shape`、`kpt_names`、`flip_idx`，划分重写为 `images/<split>`，不写 `path`；整份输出移动到其他目录后仍可独立加载。预览按照稳定顺序等间距抽取变体，不根据视觉效果筛选。

验收记录位于根目录 `plan/yolo_offline_augment/`。
