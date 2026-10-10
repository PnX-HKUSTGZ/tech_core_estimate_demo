# tech_core_estimate_demo

基于 YOLO、LiteHRNet 和 Blender 的 Robomaster 交换目标位姿识别与合成数据实验仓库。

- [`src/nn`](src/nn/README.md)：YOLO 与 LiteHRNet 的训练、推理和导出。
- [`src/blender`](src/blender/README.md)：Blender 渲染、数据合成和标注可视化。
- [`src/perception`](src/perception/README.md)：可供其他模块调用的 OpenVINO 识别与外壳/装配柱独立位姿解算。
- [`src/dataset_augment`](src/dataset_augment/README.md)：独立 YOLO 检测/关键点数据集离线增强，支持 YAML 和命令行配置。

## Python 环境

普通 Python 工具统一使用仓库根目录的 Python 3.10 uv 项目及 `.venv`。先安装
[uv](https://docs.astral.sh/uv/getting-started/installation/)，然后从仓库根目录执行：

```bash
bash setup_env.sh
uv run --no-sync rm-nn check
uv run --no-sync rm-nn --help
```

`setup_env.sh` 按根目录 `uv.lock` 分阶段安装依赖，编译 MMCV CPU 算子并针对 NumPy 2.x
重编 `xtcocotools`；首次安装或重建 `.venv` 时必须运行此脚本。后续普通命令都从仓库根目录
使用 `uv run --no-sync`，避免触发 uv 自动同步并替换本地编译的扩展。例如：

```bash
uv run --no-sync rm-nn yolo train --config src/nn/yolo/configs/train_pose.yaml
uv run --no-sync python src/blender/pipeline/compose_dataset.py --help
uv run --no-sync python src/blender/pipeline/visualize_keypoints.py --help
```

Blender 渲染使用 Blender 4.5.0 自带的 Python，由 uv 对该解释器单独安装依赖。安装与渲染命令见
[`src/blender/README.md`](src/blender/README.md)。Blender 的解释器与仓库 `.venv` 相互独立。

训练数据、模型权重和运行输出不随仓库分发，请按各工具说明提供路径。
