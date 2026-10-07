# 可调用识别与双刚体位姿解算

`perception` 提供内存图像到命名关键点、命名关键点到相机系位姿的两个独立接口。识别使用公开 OpenVINO YOLO + LiteHRNet 模型；位姿分别拟合固定外壳 station7 和移动装配柱 pillar5。

## 调用

环境使用仓库根目录 `.venv`。模型由调用方提供，初始化加载一次，后续图像复用识别器：

```python
from perception import OpenVINORecognizer, PoseEstimator

recognizer = OpenVINORecognizer(
    yolo_model="models/perception/yolo/v10.12/openvino_best/best.xml",
    hrnet_model="models/perception/litehr/litehrnet30_v9.1/openvino_best/hrnet_heatmap.xml",
)
solver = PoseEstimator()

# image_bgr: uint8 H×W×3；K、dist 对应这张原图。
for detection in recognizer.predict(image_bgr):
    result = solver.estimate(detection, K, dist)
    if result.station.success:
        consume_station(result.T_camera_from_station)
    if result.pillar.success:
        consume_pillar(result.T_camera_from_pillar)
```

两个 `consume_*` 是调用方自己的处理函数。相机畸变参数可省略，表示零畸变。可运行示例在 `examples/estimate_image.py`，用 `uv run --no-sync python examples/estimate_image.py --help` 查看参数；相机内参必须对应原图尺寸。

## 模型准备

从仓库根目录显式下载公开 YOLO v10.12 和 LiteHRNet-30 v9.1：

```bash
uv run --no-sync python scripts/download_perception_models.py
```

脚本固定远端仓库 `hkustenterprize/RM26_engineer_exchange_model` 的提交 `afab41a8b61c399bf71480f0fd52352ed660b994`，逐文件核对大小、远端 Git blob/LFS 校验值并记录本地 SHA-256。默认输出在被忽略的 `models/perception/`，下载记录为其中的 `manifest.json`。中断下载保留可续传的 `.part`，校验通过后才发布正式文件。脚本支持 `HTTPS_PROXY`、`HF_ENDPOINT`，必要时可用 `--tls12` 兼容代理；证书验证始终开启。

本次使用的权重与实际 OpenVINO 输入输出记录在 [model_manifest.json](../../plan/perception_port/model_manifest.json)。库初始化只读取调用方指定的 XML/BIN，不执行下载。

## 数据合同

`predict()` 返回按检测分数降序排列的 `list[KeypointDetection]`，只包含 class1 目标；无目标是空列表，不使用全图关键点回退。输入必须是非空 `uint8 H×W×3` BGR 图像。

| 字段 | 含义 |
| --- | --- |
| `bbox_xyxy` | 原图像素中的目标框，shape `(4,)` |
| `detection_score` | YOLO 检测分数 |
| `keypoints_xy` | 原图中的 12 个关键点，shape `(12, 2)` |
| `keypoint_scores` | 12 个原始热图峰值，不解释为可见性概率 |
| `keypoint_names` | 固定物体语义顺序，不随屏幕朝向重新排序 |
| `image_shape_hw` | 原图 `(height, width)` |

点序为 `TL, TR, BL, BR, ring, light_BR, light_TR, shell_R, shell_M, shell_L, light_TL, light_BL`。前五点属于柱体，后七点属于外壳。识别保留所有点，位姿模块分别筛选各刚体的可用点。

`PoseResult.station` 和 `.pillar` 各自是 `BodyPose`，包含 `success`、`transform`、`failure_reason`、`used_names`、`inlier_names`、`inlier_mean_error_px`、`all_points_mean_error_px`、`reprojection_errors_px`、`fit_rank`、`planar_geometry`、`candidate_count` 和 `pose_uniqueness_verified`。逐点残差与 `used_names` 同序；失败分支的 `transform` 为 `None`。

模型路径、模型张量、图像、相机和配置的合同错误抛异常；关键点不足、退化、PnP 失败或残差超限返回该刚体的失败结果。一个刚体失败不影响另一个刚体。实例按同步顺序复用；并行调用方应为每个工作线程创建独立识别器。

## 坐标与默认设置

两套矩阵都是物体系到相机系：`p_camera = T_camera_from_object @ p_object`，shape `(4, 4)`，平移单位米。相机轴为 OpenCV 的右、下、前；外壳原点为零滑移 ring，柱体原点为当前 ring。两个刚体可独立运动；默认模板来自固定源提交中的几何配置，可通过 `station_points`、`pillar_points` 显式传入同名的实测三维点。

识别默认两模型均为 `CPU`，YOLO 阈值 `0.30`、margin `0`、bbox padding `1.10`、HRNet 输入 `256×256`，RGB mean/std 沿用公开链路。仅接受 YOLO 六列检测输出和 HRNet 12 通道热图。CPU/OpenVINO 设备名与 PyTorch CUDA 设备名不同，设备必须实际存在，不自动回退。

解算默认热图峰值门限 `0.60`、至少四点，开启 RANSAC（`3 px`、置信度 `0.99`、100 次），再运行 SQPnP、LM 和全模板正深度检查。最终按内点平均重投影误差 `≤5 px` 接受，同时报告所有使用点的残差。外壳本身共面；柱体缺 ring 时也可能成为平面拟合，`planar_geometry=True` 且不声明唯一解。像素残差阈值不代表毫米级物理精度。

OpenCV 4.11 的 SQPnP RANSAC 对部分无噪声平面点集也会失败；当平面拟合的 RANSAC 失败或内点不足时，使用 AP3P RANSAC 重新选择内点，最终位姿仍由 SQPnP + LM 得到。`candidate_count` 表示通过数值与正深度检查的候选数；即使只有一个候选，也不证明姿态唯一，`pose_uniqueness_verified` 保持 `False`。

算法与默认几何取自源提交 `cb5ef2f3dea1223c0fedbd358c4129b2c586cb33` 的 `runtime/core/arm_exchange_core/perception/detector.py`、`perception/pose.py` 和 `system_config.example.yaml`；保留其预处理与坐标语义，将外壳和装配柱拆为独立刚体拟合。

## 验证

从根目录运行 `uv run --no-sync python -m unittest discover -s tests -p 'test_*.py'`。测试包括真正执行的最小 OpenVINO CPU 图、识别坐标恢复，以及两个已知刚体的独立投影恢复与拒解。公开模型和 Blender 样本的验收记录在 `plan/perception_port/report.md`。
