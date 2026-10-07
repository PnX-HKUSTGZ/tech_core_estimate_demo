# HKUST NN 与 Blender 移植报告

2026-10-07，源提交：`cb5ef2f3dea1223c0fedbd358c4129b2c586cb33`。

## 完成内容

- 从源提交移植 `src/nn` 的 46 个及 `src/blender` 的 13 个受追踪文件，包括 14,158,883 字节的 `exchange.blend`；没有搬运虚拟环境、缓存、数据、权重或训练产物。后续为环境、文档、渲染设备逻辑及空白字符校验修改 10 个源文件，删去 `src/nn` 的 3 个独立 uv 项目文件；其余 46 个源文件保持字节相同。
- 根目录 `pyproject.toml`、`uv.lock`、`.python-version` 和 `setup_env.sh` 构成唯一普通 Python 环境。Python 3.10.19，109 个锁定包；`rm26-nn` 保留原命令名和 `nn` 导入名。日常命令使用 `uv run --no-sync`，避免重同步覆盖本地编译的 MMCV 和 `xtcocotools` 扩展。合成所需 `tqdm`、导出依赖和 cu130 PyTorch 版本均纳入根项目。
- Blender 4.5.0 使用自带 Python 3.11.11。`setup_blender_env.sh` 默认安装在被忽略的 `.tools/`，通过 uv 安装固定的 NumPy 1.26.4、OpenCV headless 4.11.0.86、PyYAML 6.0.2；不修改 shell 配置，下载临时文件会清理。示例渲染配置默认 CPU。
- 单卡 GPU 模式要求 `--device GPU --gpu_uuid <物理 UUID>` 与启动前的 `CUDA_VISIBLE_DEVICES` 一致。渲染脚本用 `nvidia-smi` 将 UUID 映射为 PCI 地址，关闭所有 Cycles 设备后只启用唯一匹配的 OptiX 设备；缺失、失配或不唯一时在创建输出前报错。

## 实际验收

| 验收项 | 结果 |
| --- | --- |
| `uv lock --check --offline`、根目录 `bash setup_env.sh` | 通过；根 `.venv` 的 Python 为 3.10.19，锁 109 包 |
| `uv pip check --python .venv/bin/python` | 106 个已装包兼容 |
| `CUDA_VISIBLE_DEVICES='' uv run --no-sync rm26-nn check` | 通过；两份 LiteHRNet 模型构建和 MMCV CPU NMS 成功 |
| `uv run --no-sync rm26-nn --help` 及合成、可视化 CLI 帮助 | 通过 |
| Blender Python 与根 `.venv` | 分别为 3.11.11 和 3.10.19；Blender 内依赖版本符合固定清单 |
| CPU 与单卡 GPU 渲染 | 各得到一张 640×360 RGBA 图片、一个 `annotations.json` 记录，均含 `pillar` 和 `exchange`；图像与关键点叠加已目视检查 |
| GPU 选卡 | GPU6，UUID `GPU-4de9d6c5-5e95-6c64-86ac-0c8874ce6597`，PCI `00000001:23:00.0`；本次 Blender PID 3027461 仅在 GPU6 出现，约 3040 MiB，退出后回到 21 MiB。GPU0 同时有既有的其他 Blender PID，不属于本次任务 |
| GPU UUID 失配 | Blender 非零退出，未创建渲染输出目录 |
| 数据合成及可视化 | 两张渲染素材合成 train/val 各一张、各一个标签；`dataset.yaml` 有两类和 `kpt_shape: [12, 3]`；关键点可视化输出一张图 |

CPU/GPU 最小渲染使用 1 sample 和显式超时；GPU 启动、运行及退出检查包含完整 `nvidia-smi` 计算与图形进程表。最终再次检查，本次 PID 已退出，无遗留 GPU6 上下文。验收用的渲染、合成和可视化输出已从项目 `tmp/` 清理；保留可复现的根 `.venv` 和 Blender 安装。根 `.venv` 约 5.8 GiB、Blender 安装约 1.3 GiB，均为当前运行所需环境。

## 边界

数据集与模型权重未随移植复制，故未执行训练或基于实际权重的推理。`src/nn/hrnet/experimental` 保留原始历史代码，其中既有硬编码数据路径和缺失模块，不纳入主线验收。现有合成脚本读取 `annotations.json`；分片渲染产物需先合并为此文件再合成。
