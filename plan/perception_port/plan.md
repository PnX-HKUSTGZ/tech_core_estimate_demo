# 可调用识别与双刚体位姿解算模块

## 状态

2026-10-07：实施与验收完成。目标仓库起始提交 `ab448fc4015de792eb78935035eebbe25f705932`，起始工作树干净。源算法固定为 `cb5ef2f3dea1223c0fedbd358c4129b2c586cb33`。本次修改随功能提交归档。

2026-10-07：按用户要求整理为一个本地功能 commit，主题为 `feat: add callable OpenVINO perception and dual-body pose estimation`。提交范围共 19 个文件，24 项测试、暂存格式检查、锁文件和依赖兼容检查通过；模型与小样本留在 Git 忽略目录。

## 待办

- [x] 实现 `perception` 公共类型、OpenVINO YOLO + LiteHRNet 识别和独立 station7/pillar5 位姿解算。
- [x] 注册根 uv 包、编写调用示例和模块文档。
- [x] 验证无目标/多目标、模型合同、坐标恢复、双刚体独立性、退化与拒解：24 项自动测试通过。
- [x] 取得公开 YOLO v10.12 / LiteHRNet30 v9.1 OpenVINO 权重并记录版本与校验值：5 文件共 26,682,611 字节，固定远端 revision 并逐文件验证。
- [x] 使用 Blender 小样本完成真实 CPU 推理与两分支解算：3/3 图像、6/6 分支成功，记录完整矩阵与误差并撰写报告。
- [x] 清理本次临时资源并核对最终工作树：删除 `tmp/perception_port/`；模型和小样本保留在忽略目录，锁文件有效。
- [x] 将本次迁移整理为本地功能 commit，核对暂存范围、测试和提交后工作树。

## 实施记录

- 真实权重访问问题已通过现有本机代理与 TLS 1.2 解决；保持证书验证和固定提交校验。库调用不会自动下载。
- 保留模型原始热图峰值（可大于 1）；关键点门限只由位姿模块应用。
- 修复 OpenCV 4.11 对部分平面点集的 SQPnP RANSAC 数值失败：仅在平面 RANSAC 失败时使用 AP3P 选内点，最终仍为 SQPnP + LM。
- 加入装配柱同时旋转/平移的显式独立性测试，确认外壳矩阵保持一致。共面解明确标记，未声明姿态唯一。
- 根 uv editable 安装、锁文件和依赖兼容检查通过；最小示例用真实模型与相机参数正常输出两套位姿。
- 结果入口：[report.md](report.md)、[model_manifest.json](model_manifest.json)、[real_model_validation.json](real_model_validation.json)。

## 验收约定

普通代码复用根目录 `.venv`，Blender 继续使用独立 Python。模型存于被忽略的 `models/`，库不自动下载。测试默认 CPU；真实模型验证未完成时不得用测试替身声称完成。长任务日志轮转，下载使用 `.part`，成功后清理中间文件。
