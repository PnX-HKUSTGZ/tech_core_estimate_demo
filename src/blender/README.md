# Blender 合成数据生成

本目录负责从 Blender 场景生成带几何标注的目标素材，并将素材合成为 YOLO Pose 数据集。当前公开场景文件为 `exchange.blend`。
渲染脚本运行在 Blender 内置 Python 中；素材合成和标注可视化运行在仓库根目录的统一 uv 环境中。

## 目录结构

```text
blender/
├── configs/
│   ├── example.yaml       # 渲染配置示例
│   └── description.md     # 配置字段说明
├── pipeline/
│   ├── render_dataset.py      # Blender 渲染入口
│   ├── compose_dataset.py     # 素材合成与 YOLO 标签生成
│   ├── visualize_keypoints.py # 标注可视化
│   ├── context.py / ops.py / utils.py
│   └── __init__.py
├── setup_blender_env.sh        # Blender 4.5.0 与渲染依赖安装
├── requirements-blender.txt    # Blender 内置 Python 的固定依赖
└── exchange.blend              # 公开 Blender 场景
```

## 数据生成流程

```text
Blender 场景 + configs/*.yaml
        |
        v
render_dataset.py
        |
        |  RGBA 图片、目标框、关键点、可见性、相机参数
        v
compose_dataset.py
        |
        v
YOLO Pose 数据集
```

渲染阶段固定相机，并对目标物体、灯光和场景参数进行采样。每一帧会执行关键点投影和可见性判断，再输出目标图片及其标注。合成阶段可以使用真实背景图片池，将目标素材与背景进行亮度匹配，并写出标准的 `images/{train,val}`、`labels/{train,val}` 和 `dataset.yaml`。

## 环境

渲染流水线固定使用 Blender 4.5.0。安装脚本默认将 Blender 下载到仓库根目录被忽略的 `.tools/`，用 uv 将固定版本的 `numpy`、`opencv-python-headless` 和 `PyYAML` 安装到 Blender 自带的 Python 3.11。脚本不会修改 shell 配置文件。公开场景位于当前目录的 `exchange.blend`：

```bash
bash src/blender/setup_blender_env.sh
BLENDER_BIN="$PWD/.tools/blender-4.5.0-linux-x64/blender"
```

## 渲染

渲染入口使用当前目录的公开 Blender 场景文件：

```bash
"$BLENDER_BIN" -b src/blender/exchange.blend --python-exit-code 1 \
  -P src/blender/pipeline/render_dataset.py -- \
  --config src/blender/configs/example.yaml \
  --output_dir /absolute/path/to/rendered_assets \
  --n_images 1000 \
  --seed 40
```

示例配置默认使用 CPU。可通过命令行覆盖渲染分辨率、采样数、灯光类型、灯条颜色以及分片参数。完整字段见 [configs/description.md](configs/description.md)。

单卡 GPU 渲染时，先从 `nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used --format=csv` 选择空闲的物理 GPU UUID，再在启动 Blender 的同一条命令中限制 CUDA 可见性：

```bash
GPU_UUID='GPU-替换为实际物理UUID'
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$BLENDER_BIN" -b src/blender/exchange.blend --python-exit-code 1 \
  -P src/blender/pipeline/render_dataset.py -- \
  --config src/blender/configs/example.yaml \
  --output_dir /absolute/path/to/rendered_assets \
  --n_images 1000 --device GPU --gpu_uuid "$GPU_UUID"
```

渲染脚本会核对 `CUDA_VISIBLE_DEVICES`、物理 UUID 与 OptiX PCI 地址，只启用匹配的一块卡；任何缺失或不一致都会在渲染前报错。`--python-exit-code 1` 使 Blender 在脚本报错时返回非零退出码。CUDA 可见性无法单独约束图形渲染，GPU 作业仍需在启动、运行和退出时检查完整 `nvidia-smi` 进程表，包括图形进程。

渲染输出结构为：

```text
rendered_assets/
├── images/
├── annotations.json
├── config.yaml
└── meta.json
```

使用分片参数时，渲染器分别写入 `annotations_shardN.json`；合成工具只读取 `annotations.json`，因此须先合并各分片的 `images` 记录并保留一致的目标与关键点元数据，再运行合成。

## 合成 YOLO 数据集

将一个或多个 Blender 输出目录合成为训练数据集：

```bash
uv run --no-sync python src/blender/pipeline/compose_dataset.py \
  --blender_dir /absolute/path/to/rendered_assets \
  --output_dir /absolute/path/to/exchange_pose
```

如果有多个素材目录，可以使用 `--blender_dirs`。背景图片、验证集比例和随机种子等参数通过命令行设置，具体参数可以查看：

使用单个素材目录时，合成结果位于 `<output_dir>/<素材目录名>_compose/`；使用多个目录时直接位于 `<output_dir>/`。

```bash
uv run --no-sync python src/blender/pipeline/compose_dataset.py --help
```

## 标注可视化

```bash
uv run --no-sync python src/blender/pipeline/visualize_keypoints.py \
  --dataset_dir /absolute/path/to/rendered_assets \
  --n 20 \
  --out_dir /absolute/path/to/visualized_samples
```

可视化工具读取渲染阶段的 `annotations.json`，用于检查目标框、关键点和可见性。合成后的 YOLO 数据集可使用 `src/nn/hrnet/data/visualize_dataset.py` 进一步抽查。

## 下游接口

合成脚本输出的目录可以直接交给 `src/nn/yolo/configs/exchange_pose_dataset.yaml` 所描述的 YOLO Pose 训练流程。检测数据集由 YOLO Pose 数据集转换得到：

```bash
uv run --no-sync python src/nn/yolo/pose2detect.py \
  --source /absolute/path/to/exchange_pose \
  --output /absolute/path/to/exchange_detect
```
