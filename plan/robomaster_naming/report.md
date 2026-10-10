# Robomaster 说明与 NN 命令命名报告

2026-10-10，已完成。

## 修改

- 根 README 与 `pyproject.toml` 的项目说明使用 Robomaster。
- 控制台入口为 `rm-nn = "nn.__main__:main"`，同步 CLI 帮助、安装脚本、NN/HRNet/YOLO 文档和导出示例。
- 历史移植报告的命令示例更新为当前名称，并注明更新日期和历史验收边界。
- 使用 `uv pip install --python .venv/bin/python --no-deps --no-build-isolation --editable .` 刷新根环境，已安装的新命令和包描述与配置一致，旧控制台入口已移除。
- 外部 Hugging Face 仓库、上游许可来源、实验数据路径与训练环境变量保留实际标识。

## 验证

| 检查 | 结果 |
| --- | --- |
| `uv run --no-sync rm-nn --help` | 通过，显示新命令名 |
| `rm-nn hrnet --help`、`rm-nn yolo --help` | 经根 uv 环境运行通过 |
| `uv run --no-sync python -m nn --help` | 通过 |
| `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 uv run --no-sync rm-nn check` | 退出码 0，两份 LiteHRNet 构建、MMCV CPU NMS、COCO 扩展检查通过 |
| `uv lock --check --offline` | 通过，109 个锁定包；无需修改锁文件 |
| `uv pip check --python .venv/bin/python` | 通过，106 个已安装包兼容 |
| `bash -n setup_env.sh`、`git diff --check` | 通过 |
| 全部相关源码、配置及 Markdown 中旧命令残留检查 | 无残留 |

环境自检输出了依赖的 `pkg_resources` 弃用提示，以及 Ultralytics 将用户设置重置为默认值的提示（`~/.config/Ultralytics/settings.json`）；检查本身完成且退出码为 0。此次不新增测试文件，使用实际安装入口验证命令改名。

无数据或权重变更。

## 提交整理

2026-10-10 按用户要求整理为本地提交，主题为 `refactor: rename NN command to rm-nn and adopt Robomaster branding`，共 11 个文件。独立核对说明文档、包配置、安装脚本与 CLI 命名一致；提交前复核新命令帮助、锁文件、Shell 语法和暂存格式检查，全部通过。此前通过的环境自检结果见上表。

`AGENTS.md` 的独立类型约束编辑保留在工作区，本次不纳入提交。未推送远端。
