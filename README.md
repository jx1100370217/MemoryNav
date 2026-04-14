<div align="center">

# 🧠 MemoryNav

**视觉记忆导航系统 | Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-2.2.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v2.2.0)

基于视觉位置识别（VPR）和拓扑地图的机器人记忆导航系统

[English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | **中文**

</div>

---

## 📖 简介

MemoryNav 是一个面向移动机器人的视觉记忆导航系统。系统通过 4 个环视鱼眼相机采集图像，利用 VPR 技术在预建的拓扑记忆图中定位，结合 YOLOv8n 视觉遮挡检测和 Qwen3.5-9B 视觉语言模型进行兜底打点导航，实现"记住去过的地方，再走一次"的记忆导航能力。

### 核心能力

- **🛰️ 在线主动建图** (`online_mapper/`, **v2.3.0**)：三层架构（Geometry+Topology+Semantics）流式在线建图。**VGGT-1B 几何前端**（depth + VO + 占据栅格 dense 点云 单次推理同时供给）、**结构化节点命名** `NodeName(category, organization, nearby_plates, ...)`（`前台·DEEPROUTE.AI` 取代 `DEEPROUTE.AI前台` 字符串拼接）、门牌**两阶段归属**（防 EUMANN 串扰）、ConnectionBuilder **几何方向先验**（cos 相似度 + 反向硬惩罚修复 cam→neighbor 错配）、多帧投票幻觉过滤、闭环几何验证、双语命名、空间 KNN 邻接重建；输出 `merged_labeled_data/` schema 并新增 `category/organization/nearby_plates/nearby_landmarks` 字段。完整设计见 [`docs/online_mapper.md`](docs/online_mapper.md)
- **🔍 多方案 VPR 定位**：支持 4 种 SOTA 视觉位置识别方案，统一配置文件一键切换
- **🗺️ 拓扑记忆图**：自动从标注数据构建节点-边拓扑图，支持最短路径规划
- **🔄 循环移位匹配**：4 相机循环移位算法，支持任意朝向下的定位与偏转角估计
- **🎯 DINOv3 子图匹配**：基于 DINOv3 密集 patch 特征的子图定位，三级级联匹配（small→mid→big）+ 全相机遍历，滑动窗口余弦相似度匹配
- **💾 帧间缓存复用**：基于 DINOv2 VPR 特征的帧间相似度判断（零额外推理成本），匹配失败时智能复用上一帧成功结果
- **🔭 Lookahead 双重确认**：步骤切换时同时验证 VPR 定位和下一步子图匹配，避免过早 advance
- **📤 统一输出格式**：记忆模式开关两种状态下输出格式一致，始终提供 `pixel_target`
- **🚧 YOLOv8n 遮挡检测**：子图匹配失败时自动检测注意力相机画面是否被遮挡（行人、物体等），遮挡时原地等待，消除后继续导航
- **🤖 Qwen3.5 兜底打点**：子图匹配失败且未遮挡时自动切换 Qwen3.5-9B 视觉语言模型打点定位，统一使用"通道正中间位置+景深"找路策略，避免找不到参照物时瞎指方向
- **📷 鱼眼去畸变**：自动从 `cam/params.yaml` 加载相机内参，对输入图像做柱面投影去畸变，提升 VPR 及子图匹配精度
- **🧭 像素→机器人坐标转换**：将 `pixel_target` 归一化坐标经柱面角度、相机方位角、俯仰角估距完整管线，转换为机器人运动坐标 `[x_forward, y_lateral, 0.0]`
- **🔄 侧面相机旋转处理**：camera_3/camera_4 匹配成功时自动输出原地旋转动作，引导机器人朝向目标
- **🌐 WebSocket 服务**：实时流式接收图像、返回导航指令
- **⚙️ 统一配置管理**：所有 VPR 参数集中在 `deploy/vpr_config.yaml`，一处修改全局生效

---

## 🏗️ 系统架构

```
MemoryNav/
├── memory_nav/                     # 核心记忆导航模块
│   ├── memory_navigator.py         # 导航器主接口
│   ├── memory_models.py            # 数据模型 (Node, Edge, Plan, VPRResult)
│   ├── memory_graph.py             # 拓扑图 (BFS/Dijkstra 路径规划)
│   ├── memory_vpr.py               # VPR 匹配引擎 (循环移位 + 无序匹配)
│   ├── memory_builder.py           # 记忆构建器 (从标注数据构建拓扑图)
│   ├── sub_image_matcher.py        # 子图匹配器 (DINOv3 密集特征匹配)
│   ├── occlusion_detector.py       # YOLOv8n 遮挡检测器
│   ├── fisheye_undistort.py        # 鱼眼去畸变 (柱面投影)
│   ├── coord_transform.py          # 像素→机器人坐标转换
│   ├── qwen35_point_grounder.py    # Qwen3.5 打点封装 (兜底模型)
│   ├── qwen35_grounding_server.py  # Qwen3.5 子进程推理服务
│   ├── vpr_factory.py              # VPR 提取器工厂
│   ├── vpr_config_loader.py        # 统一配置加载器
│   ├── selavpr_extractor.py        # SelaVPR++ (DINOv2 + MultiConv)
│   ├── megaloc_extractor.py        # MegaLoc (DINOv2 + OT聚合)
│   ├── effovpr_extractor.py        # EffoVPR (DINOv2 多层CLS token)
│   ├── anyloc_extractor.py         # AnyLoc (DINOv2 + VLAD)
│   └── selavpr_model/              # SelaVPR++ 模型代码
├── deploy/                         # 部署入口
│   ├── ws_proxy_with_memory.py     # WebSocket 代理服务 (主入口)
│   ├── vpr_config.yaml             # VPR 统一配置文件
│   ├── build_memory.sh             # 记忆构建脚本
│   └── start_server.sh             # 服务启动脚本
├── cam/                            # 多目鱼眼相机
│   ├── params.yaml                 # 相机内参 & 外参配置
│   └── tools/                      # 独立工具 (无 ROS2/CUDA 依赖)
├── scripts/                        # 工具脚本
│   └── memory_visualization_server.py  # 可视化服务 (子图匹配 + 打点 + 遮挡检测)
├── pretrained/                     # 预训练模型 (YOLOv8n, DINOv3 等)
├── merged_labeled_data/            # 记忆标注数据
├── online_mapper/                  # 🛰️ 在线主动建图模块 (v2.3.0, 三层架构)
│   ├── config.py                   # 全局配置 (含 depth/vo/occ_backend 开关)
│   ├── run_online_map.py           # CLI 入口 (baseline 离线回放)
│   ├── core/online_mapper_core.py  # ⭐ run() + process_frame() + finalize() 流式主循环
│   ├── geometry/                   # Geometry 层 (VGGT-1B 几何前端)
│   │   ├── vggt_backend.py         #   VGGT-1B 单例 + 滑窗
│   │   ├── depth_estimator.py      #   DA-V2 + VGGTDepthEstimator + 工厂
│   │   ├── visual_odometry.py      #   MonoVO + VGGTVisualOdometry + 工厂
│   │   ├── pose_graph.py           #   scipy LM pose graph
│   │   ├── junction_detector.py    #   4-camera depth 路口判定 (stateless)
│   │   └── occupancy.py            #   1D ray-cast + dense 点云直填
│   ├── topology/                   # Topology 层
│   │   ├── graph.py                #   TopoGraph / TopoNode
│   │   ├── keyframe_selector.py    #   多触发关键帧选择
│   │   ├── loop_closure.py         #   auto-tune + ORB 几何验证闭环
│   │   ├── frontier_nbv.py         #   frontier / NBV 候选推荐
│   │   ├── connection_builder.py   #   ⭐ next_positions: 几何方向先验
│   │   └── auto_sub_image_extractor.py  # 打点裁剪 + 走廊中间帧匹配
│   ├── semantics/                  # Semantics 层
│   │   ├── open_set_detector.py    #   Grounding-DINO 封装
│   │   ├── scene_graph.py          #   层次场景图
│   │   ├── door_plate_tracker.py   #   门牌多帧代表帧选择 + 两阶段归属
│   │   ├── node_naming.py          # ⭐ 结构化命名 NodeName
│   │   ├── node_category.py        # ⭐ 节点类别分类器 + CN/EN 映射
│   │   ├── hallucination_filter.py # ⭐ STRICT prompt + QwenVerifier + MultiFrameVoter
│   │   ├── colocation_merger.py    # ⭐ 同位置合并 (用 NodeName.merge_names)
│   │   ├── semantic_dedup.py       #   语义重名去重
│   │   └── auto_landmark_namer.py  #   Qwen3.5 场景命名
│   ├── vpr/
│   │   └── node_distance_estimator.py  # VPR 节点距离估计
│   ├── viz/visualize.py            #   finalize 末尾产出 pose_graph / occupancy / timeline
│   └── io/
│       └── merged_data_writer.py   #   输出 merged_labeled_data + 结构化字段
├── third_party/vggt_space/         # VGGT 源码 (.gitignore, 从 HF Space 下载)
├── pretrained/                     # 模型权重 (.gitignore)
│   ├── vggt-1b/                    #   facebook/VGGT-1B
│   ├── depth-anything-v2-small-hf/ #   备用 depth backend
│   ├── grounding-dino-base/        #   IDEA-Research/grounding-dino-base
│   └── dinov3_vitb16.safetensors   #   VPR backbone
├── tests/                          # 测试
│   └── test_memory_ws.py           # WebSocket 集成测试
└── docs/                           # 文档
    └── online_mapper.md            # 📘 online_mapper 完整设计文档 (v2.3.0)
```

---

## 📷 鱼眼去畸变

系统在 VPR 匹配和子图匹配前，自动对 4 路鱼眼图像进行柱面投影去畸变：

### 工作原理

1. 启动时从 `cam/params.yaml` 加载各相机内参（`xi, fx, fy, cx, cy`）和畸变系数（`k1, k2, p1, p2`）
2. 每个相机预计算一次 remap 查找表（柱面投影 + pitch_up 视角偏移）
3. 每帧推理前调用 `cv2.remap` 完成去畸变，计算开销极低
4. 若 `cam/params.yaml` 不存在，自动跳过去畸变，不影响服务启动

```python
from memory_nav.fisheye_undistort import FisheyeUndistorter

undistorter = FisheyeUndistorter.from_yaml("cam/params.yaml")
# 批量去畸变 4 路相机图像
perspective_images = undistorter.undistort_batch(camera_images)
```

---

## 🧭 像素→机器人坐标转换

将 `pixel_target: [x_norm, y_norm]` 通过完整物理管线转换为机器人运动坐标 `[x_forward, y_lateral, 0.0]`：

### 转换管线

```
x_norm → 柱面水平角 → + 相机方位角 → 全局 yaw
y_norm → 柱面垂直角 → 俯仰角 → 距离估算（相机高度 + pitch_up）
yaw + distance → (x_forward, y_lateral)
```

### 侧面相机旋转处理

当 camera_3 或 camera_4 匹配成功时（朝向后方），系统不输出前进动作，而是通过坐标转换获取实际 yaw 角度，输出原地旋转动作 `[0, 0, yaw_rad]`，引导机器人先转向目标方向。Qwen3.5 兜底打点时，侧面相机同样输出旋转动作 `[0, 0, 0.785]`（约45°）。

### 参数配置（`coord_transform.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEFAULT_FOV` | 180° | 柱面图视场角 |
| `DEFAULT_WIDTH` | 1920 | 柱面图宽度 |
| `DEFAULT_HEIGHT` | 1536 | 柱面图高度 |
| `DEFAULT_CAMERA_HEIGHT` | 1.0m | 相机距地面高度 |
| `DEFAULT_PITCH_UP` | 15° | 去畸变 pitch_up 偏移角 |
| `MIN_DISTANCE` / `MAX_DISTANCE` | 0.3m / 30.0m | 距离估算范围 |

### 相机方位角（硬编码，从 `cam/params.yaml` T_ic 计算）

| 相机 | 方位角 |
|------|--------|
| camera_1 | +39.42° |
| camera_2 | −35.84° |
| camera_3 | −142.04° |
| camera_4 | +143.52° |

### 响应新增字段

坐标转换结果附加在 `memory_info.coord_transform` 中：

```json
"memory_info": {
    "coord_transform": {
        "yaw_global_deg": -12.3,
        "depression_deg": 8.5,
        "distance": 2.4,
        "elapsed_ms": 0.3
    }
}
```

---

## 🎯 子图匹配导航

基于 **DINOv3** 密集 patch 特征的子图匹配导航方案：

### 工作原理

1. **记忆构建时**：为每条边标注 `camera_name`（目标所在相机）和三级 `crop_image`（big/mid/small 注意力子图）
2. **导航执行时**：遍历所有 4 个相机，每个相机执行 small→mid→big 三级级联匹配，选择全局最高 confidence 的结果
3. **目标定位**：DINOv3 ViT-B/16 提取密集 patch token → 滑动窗口 + unfold 加速 → 余弦相似度最大位置 → 输出为 `pixel_target`
4. **匹配阈值**：置信度 ≥ `SUB_MATCH_CONFIDENCE_THRESHOLD`（当前 **0.60**）视为匹配成功
5. **帧间缓存**：匹配失败时基于 DINOv2 VPR 特征的帧间相似度判断（阈值 **0.70**），若场景变化小则复用上一帧成功结果，步骤切换时清空缓存
6. **回退机制**：缓存也无法复用时，触发 Qwen3.5 兜底打点

### 边数据结构

```yaml
edge:
  camera_name: "camera_2"              # 目标所在相机
  landmark_name: "电梯"                 # 地标名称
  crop_image_paths:                     # 三级注意力子图
    big: "path/to/big.jpg"
    mid: "path/to/mid.jpg"
    small: "path/to/small.jpg"
```

### 输出格式

所有响应统一包含 `pixel_target: [x, y]`（归一化 0~1）及机器人运动 `action`：

| 场景 | pixel_target 来源 | action 来源 | memory_active |
|------|-------------------|-------------|---------------|
| 记忆开启 + 子图匹配成功 | `sub_image_match.match.center_pct` | coord_transform | `true` |
| 记忆开启 + 帧间缓存复用 | 延用上一帧缓存 | coord_transform | `true` |
| 记忆开启 + Qwen3.5 兜底 | Qwen3.5 打点归一化坐标 | coord_transform | `true` |
| 记忆开启 + 遮挡检测 | 无（原地等待） | `[0, 0, 0]` | `true` |
| 记忆开启 + 侧面相机匹配 | `sub_image_match.match.center_pct` | 原地旋转 `[0, 0, yaw]` | `true` |

---

## 🚧 遮挡检测

当子图匹配失败时（无论 VPR 是否成功），系统自动对注意力相机执行遮挡检测，判断是否因视觉遮挡导致：

### 工作原理

1. **遮挡检测触发**：子图匹配失败即触发，不依赖 VPR 结果
2. **遮挡检测相机选择**：使用子图匹配得分最高（但低于阈值）的 camera，而非静态的 `step.camera_name`，更准确反映注意力区域实际所在
3. **YOLOv8n 推理**：检测画面中的近距离前景物体（person、backpack、umbrella、handbag、suitcase），计算 bbox 面积占比
4. **遮挡判定**：单个遮挡物面积占比 ≥ **25%**（默认阈值）→ 判定为遮挡
5. **遮挡时行为**：输出 `action: [0, 0, 0]`（原地等待），清除子图匹配缓存，下一帧继续检测
6. **未遮挡时行为**：使用 Qwen3.5 打点（固定"通道正中间位置+景深"找路）继续导航

### 导航决策流程

```
每帧处理:
  ├─ 子图匹配 (全4相机 × 3级cascade)
  ├─ Lookahead 下一步子图匹配
  │
  ├─ 子图匹配失败时:
  │   ├─ YOLOv8n 遮挡检测 (对子图匹配得分最高的 camera)
  │   │   ├─ 遮挡 → action=[0,0,0] 原地等待，清除子图缓存
  │   │   └─ 未遮挡 → Qwen3.5 打点(找路: 通道正中间+景深) 继续导航
  │   │              └─ 打点也失败 → 重发记忆引导
  │   └─ (遮挡检测与VPR结果无关)
  │
  ├─ VPR 匹配成功:
  │   ├─ 匹配到目标节点 + sim≥0.70:
  │   │   ├─ 最后一步 → 直接 advance
  │   │   ├─ 下一步子图匹配成功 → Lookahead 确认 → advance
  │   │   └─ 下一步子图匹配未成功 → VPR HELD，暂不切换
  │   └─ 匹配到其他节点 / sim<0.70 → 继续当前步骤
  │
  └─ VPR 匹配失败:
      ├─ 子图匹配成功 → 继续用子图匹配结果导航
      ├─ 子图匹配失败 + Qwen3.5 有结果 → 用打点结果导航
      └─ 子图匹配失败 + Qwen3.5 无结果 → 重发记忆引导
```

### 响应新增字段

遮挡检测结果附加在 `memory_info` 中：

```json
"memory_info": {
    "phase": "occluded",
    "consecutive_occlusions": 3,
    "occlusion": {
        "occluded": true,
        "max_area_ratio": 0.32,
        "total_area_ratio": 0.32,
        "detections": [
            {"class_name": "person", "confidence": 0.87, "area_ratio": 0.32}
        ],
        "reason": "person 占画面 32.0% (>= 25% 阈值)"
    }
}
```

---

## ✨ VPR 方案对比

MemoryNav 支持 **4 种** VPR 方案，通过 `deploy/vpr_config.yaml` 统一切换：

| 方案 | 参数值 | 发表 | 特征维度 | Backbone | 特点 |
|------|--------|------|---------|----------|------|
| **SelaVPR++** ⭐ | `selavpr` | T-PAMI 2025 | 4096D | DINOv2-L + MultiConv | **推荐方案**，支持 hashing+rerank，官方最强配置 |
| **MegaLoc** | `megaloc` | CVPR 2025 | 8448D | DINOv2-B + OT聚合 | 综合性能最强，多数据集 SOTA |
| **EffoVPR** | `effovpr` | arXiv 2024 | 3072D | DINOv2-B 多层CLS | 轻量快速，适合实时场景 |
| **AnyLoc** | `anyloc` | RA-L 2023 | 可配置 | DINOv2-B + VLAD | 经典稳定，聚类数可调 |

---

## ⚙️ 统一配置

所有 VPR 相关参数集中在 `deploy/vpr_config.yaml` 中管理，修改后重启服务即可生效：

```yaml
# VPR 方法: selavpr | megaloc | effovpr | anyloc
vpr_method: selavpr

# GPU 设备
device: "cuda:0"

# 匹配模式 (各方案独立设置)
# false: 循环移位匹配 + 朝向估计 (SelaVPR++)
# true: 无序贪心匹配 (AnyLoc/MegaLoc/EffoVPR)
order_invariant:
  selavpr: false
  megaloc: true
  effovpr: true
  anyloc: true

# VPR 相似度阈值 (各方案独立设置)
similarity_threshold:
  selavpr: 0.60
  megaloc: 0.60
  effovpr: 0.80
  anyloc: 0.70

# SelaVPR++ 专用配置
selavpr:
  backbone: dinov2-large      # dinov2-base (2048D) 或 dinov2-large (4096D)
  aggregation: gem            # gem, boq, salad
  use_hashing: true           # 开启深度哈希
  use_rerank: true            # 开启重排 (需 use_hashing=true)

# AnyLoc 专用配置
anyloc:
  dino_model: dinov2_vitb14
  agg_mode: vlad
  num_clusters: 32
  domain: indoor
  max_img_size: 630
```

**切换方案只需修改 `vpr_method` 一行**，以下模块自动读取统一配置：
- `ws_proxy_with_memory.py` — WebSocket 导航服务
- `memory_visualization_server.py` — 可视化服务
- `memory_builder.py` / `memory_navigator.py` — 核心模块
- `build_memory.sh` — 构建脚本

> ⚠️ 切换 VPR 方案后需要重新构建记忆缓存：`bash deploy/build_memory.sh`

---

## 🛰️ 在线建图 (online_mapper)

`online_mapper/` 是 MemoryNav 的建图模块, 采用三层架构 (Geometry + Topology + Semantics) 流式在线建图, 产出 `merged_labeled_data/` schema.

- **Geometry 层**: VGGT-1B 几何前端 (depth + VO + 占据栅格 dense 点云 单次推理), scipy LM pose graph, 4-camera depth 路口检测
- **Topology 层**: 多触发关键帧 (VPR + 位移 + 旋转 + 信息增益 + 路口), auto-tune + ORB 几何验证闭环, ConnectionBuilder 几何方向先验
- **Semantics 层**: STRICT prompt + QwenVerifier + MultiFrameVoter 多帧投票, 7 大类白名单, ColocationMerger 同位置合并, 结构化 NodeName + CN/EN 双语命名
- **API**: `OnlineMapperCore.run()` / `process_frame(frame)` / `finalize()`, 支持流式输入
- **可视化**: `finalize()` 末尾自动产出 `pose_graph.png` / `occupancy.png` / `keyframe_timeline.png` / `scene_overview.txt`

完整设计文档见 **[`docs/online_mapper.md`](docs/online_mapper.md)** (v2.3.0, 12 章).
迭代历史 (v2.1.0 → v2.3.0) 见 [`docs/online_mapper.md` §10](docs/online_mapper.md). r1→r6 早期 metrics 见 **[`online_mapper/RESULTS.md`](online_mapper/RESULTS.md)**.

### 基线离线回放

```bash
python online_mapper/run_online_map.py \
    --input_dir memory_test_data \
    --output_dir online_mapper/output \
    --vpr_config deploy/vpr_config.yaml
```

### WebSocket 双模式接入

`deploy/ws_proxy_with_memory.py` 监听 **9528** 端口, 单个连接同时支持**导航**与**建图**两种模式, 由 `session_state['mode']` 管理 (默认 `nav`).

| 命令 | 说明 |
|------|------|
| `{"command": "start_mapping"}` | 切换到 mapping 模式, 初始化 `OnlineMapperCore` |
| `{"command": "stop_mapping"}` | 停止建图, 触发 `finalize()` 并切回 nav |
| `{"command": "mapping_status"}` | 查询当前建图状态 (模式 / 已处理帧数 / 产物路径) |

- **建图帧**: 每帧 `{id, pts, images: {camera_1..4}}` 走 `process_mapping_frame`, 调用 `OnlineMapperCore.process_frame` + 最终 `finalize`
- **模型共享**: 双模式共享 `MemoryNavigator.extractor` (SelaVPR), 避免重复加载
- **产物目录**: `deploy/logs/mapping_output/session_{ts}_{client_id}/` (仅 `run_online_map.py` baseline 仍沿用 `online_mapper/output/`)
- **临时帧目录**: `deploy/logs/mapping_frames/session_*/`, `finalize` 后自动清理
- **断线保护**: 客户端断开时服务端自动 `finalize` 保住数据

生成的数据可直接用于记忆构建:

```bash
bash deploy/build_memory.sh --data_dir deploy/logs/mapping_output/session_*/merged_labeled_data
```

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav
pip install -r requirements/core_requirements.txt
pip install -e .
```

### 配置相机参数（可选）

如有实体相机，将标定文件放置为 `cam/params.yaml`，服务启动时自动加载鱼眼去畸变。

### 配置 VPR 方案

编辑 `deploy/vpr_config.yaml`，选择你需要的 VPR 方案和参数。

### 构建记忆库

```bash
# 自动从 vpr_config.yaml 读取 VPR 方案
bash deploy/build_memory.sh

# 或指定参数覆盖
bash deploy/build_memory.sh --method megaloc --gpu 0
```

### 启动导航服务

```bash
# 自动从 vpr_config.yaml 读取配置
python deploy/ws_proxy_with_memory.py
```

### Python API

```python
from memory_nav import MemoryNavigator

# 自动使用 vpr_config.yaml 中的配置
navigator = MemoryNavigator(vpr_method='selavpr', device='cuda:0')
navigator.load_memory(path='memory_nav/memory_cache', data_dir='merged_labeled_data')

# VPR 定位
images = {'camera_1': img1, 'camera_2': img2, 'camera_3': img3, 'camera_4': img4}
features = {cam: navigator.extractor.extract(img) for cam, img in images.items()}
result = navigator.vpr.locate(features)
print(f"定位: {result.matched_node_name}, 相似度: {result.similarity:.4f}")

# 规划导航
plan = navigator.navigate_to("前台", camera_images=images)
for step in plan['plan']['steps']:
    print(f"  → {step['to_node']['name']}, camera={step['camera_name']}, landmark={step['landmark_name']}")

# 子图匹配（导航执行中）
match = navigator.match_current_step(images)
if match and match['match']['found']:
    center = match['match']['center_pct']
    print(f"目标定位: ({center['x']:.3f}, {center['y']:.3f})")
```

---

## 📡 WebSocket 协议

### 请求格式

```json
{
    "id": "robot_01",
    "pts": 1709558400,
    "task": "导航到前台",
    "images": {
        "front_1": "<base64>",
        "camera_1": "<base64>",
        "camera_2": "<base64>",
        "camera_3": "<base64>",
        "camera_4": "<base64>"
    }
}
```

### 响应格式

```json
{
    "status": "success",
    "id": "robot_01",
    "task_status": "executing",
    "action": [[0.5, -0.1, 0.0]],
    "pixel_target": [0.485, 0.521],
    "memory_active": true,
    "camera_name": "camera_2",
    "landmark_name": "电梯",
    "landmark_name_eng": "elevator",
    "position_name_eng": "C8 front desk",
    "crop_image_paths": {
        "big": "path/to/big.jpg",
        "mid": "path/to/mid.jpg",
        "small": "path/to/small.jpg"
    },
    "crop_image_path": "path/to/big.jpg",
    "sub_image_match": {
        "camera_name": "camera_2",
        "landmark_name": "电梯",
        "match": {
            "found": true,
            "confidence": 0.92,
            "center_pct": {"x": 0.485, "y": 0.521},
            "top_left_pct": {"x": 0.302, "y": 0.358},
            "bottom_right_pct": {"x": 0.668, "y": 0.684}
        },
        "matched_scale": "mid",
        "memory_camera": "camera_2"
    },
    "fallback_instruction": null,
    "memory_info": {
        "frame_similarity": 0.85,
        "cache_action": "accepted",
        "plan_path": ["12", "8", "4"],
        "current_step": 1,
        "total_steps": 3,
        "from_node": "大厅",
        "to_node": "前台",
        "vpr_similarity": 0.85,
        "vpr_confidence": 0.85,
        "vpr_matched_node": "node_5",
        "heading_offset": -37.5,
        "consecutive_misses": 0,
        "consecutive_occlusions": 0,
        "occlusion": null,
        "lookahead_conf": 0.68,
        "lookahead_found": true,
        "coord_transform": {
            "yaw_global_deg": -12.3,
            "depression_deg": 8.5,
            "distance": 2.4,
            "elapsed_ms": 0.3
        }
    },
    "message": "记忆导航: 大厅 → 前台 (步骤1/3)"
}
```

### 控制命令

| 命令 | 说明 |
|------|------|
| `reset` | 重置 Agent 和记忆状态 |
| `toggle_memory` | 切换记忆导航开关 |
| `memory_status` | 查看记忆导航详情（含可用目的地列表） |
| `reset_memory` | 仅重置记忆状态（Agent 历史保留） |
| `session_status` | 查看会话状态 |

---

## 📐 相机布局

系统使用 4 个鱼眼相机（等角投影，HFOV=190°）：

```
            前方 (0°)
              ↑
     cam_1 (-37.5°)  cam_2 (+37.5°)
              │
     cam_4 (-142.5°) cam_3 (+142.5°)
              ↓
            后方 (180°)
```

循环移位匹配支持 4 种朝向偏移：`0°`, `-75°`, `180°`, `+105°`

---

## 🧪 测试

```bash
# 单元测试
python -m pytest tests/unit_test/test_basic.py -v

# WebSocket 导航回放 (默认 nav 模式, 含逐帧决策日志 + 统计报告)
python tests/test_memory_ws.py
python tests/test_memory_ws.py --mode nav

# WebSocket 建图回放 (start_mapping → 全量 49 帧 → stop_mapping)
python tests/test_memory_ws.py --mode mapping
```

`test_memory_ws.py` 是双模式合并脚本 (替代原 `deploy/test_mapping_client.py`):

- **nav 模式**: 原有导航回放, 输出 VPR 匹配、子图匹配置信度、lookahead 置信度、camera、action、决策类型 + 节点/决策分布统计
- **mapping 模式**: 自动跑 start_mapping → 逐帧喂入 → stop_mapping, 打印拓扑/关键帧/门牌/runtime 分解 + 产物路径

---

## 📋 更新日志

### v2.3.0

- **🛰️ 在线主动建图模块** (`online_mapper/`): 流式在线建图, 三层架构 (Geometry + Topology + Semantics)
  - **Geometry 层**: 单目 ORB+EssentialMatrix VO, Depth-Anything-V2, scipy LM pose graph, 2D 占据栅格, 4-camera depth 路口检测
  - **Topology 层**: 多触发关键帧 (VPR + 位移 + 旋转 + 信息增益), auto-tune + ORB 几何验证的全局闭环, 空间 KNN + 时间相邻并集的邻接重建
  - **Semantics 层**: STRICT prompt + QwenVerifier 二次验证 + MultiFrameVoter 多帧投票 + substring 变体合并 + 7 大类别白名单 (NodeCategoryClassifier) + ColocationMerger 同位置节点合并 + CN/EN 双语命名 + NameDeduplicator 重名后缀
  - 输出 `merged_labeled_data/` schema, 额外产出 `scene_graph.json` / `pose_graph.json` / `online_mapping_log.jsonl` / `metrics.json`
  - **测试数据 (49 帧) 最终结果**: 5 个高质量 node (打印区 / 前台 / NEUMANN强电井 / 关爱室 / DEEPROUTE.AI前台), 0 幻觉, 0 重名, 2 loop closures
  - 完整设计文档: **[`docs/online_mapper.md`](docs/online_mapper.md)** (约 47000 字, 13 章)
  - 迭代历史 (r1→r6): [`online_mapper/RESULTS.md`](online_mapper/RESULTS.md)

### v2.2.0

- **🆕 自动建图模块**: 从图像序列全自动生成拓扑导航图, 输出与手工标注 `merged_labeled_data/` 完全兼容
  - 三阶段 Pipeline: VPR 节点创建 → 语义增补 (门牌/标识检测) → 连接生成 (打点 + crop)
  - Qwen3.5 vLLM 推理后端, 支持场景命名、文字识别、打点定位
  - 语义节点检测器自动识别会议室名、门牌号等有导航意义的标识
  - DINOv3 走廊中间帧匹配 + 匈牙利算法最优 camera→邻居分配
  - 4 cameras 并行调用 vLLM (Phase 1.5 加速 1.3x, Phase 2 加速 1.6x, 总体 315s→238s)

### v2.1.0 (文档同步)

- **📝 文档与代码对齐**：遮挡面积阈值 35% → **25%**（匹配代码默认值）
- **📝 子图匹配阈值修正**：0.65 → **0.60**（匹配 `SUB_MATCH_CONFIDENCE_THRESHOLD`）
- **📝 遮挡触发条件修正**：不再描述为"VPR失败时触发"，实际为**子图匹配失败即触发**（不依赖VPR结果）
- **📝 新增侧面相机旋转处理**：文档补充 camera_3/camera_4 匹配时的旋转动作逻辑
- **📝 新增 VPR 到达阈值**：`VPR_ARRIVE_THRESHOLD = 0.70`
- **📝 新增无序匹配模式**：`order_invariant` 配置项说明

### v2.0.0

- **🆕 YOLOv8n 遮挡检测**：新增 `memory_nav/occlusion_detector.py`，子图匹配失败时自动检测视觉遮挡
  - 使用 YOLOv8n（6MB）检测 person、backpack、umbrella、handbag、suitcase 等近距离前景物体
  - 遮挡判定基于 bbox 面积占比（默认阈值 25%），GPU 推理 ~30ms
  - 遮挡时输出 `action: [0, 0, 0]` 原地等待，遮挡消除后自动恢复导航
  - 未遮挡时使用 Qwen3.5 打点（固定"通道正中间位置+景深"找路）继续导航
- **🔄 导航决策简化**：移除旧的趋势判断方案（Case B 跳步 / Case C 重规划 / Case D 相似度趋势检测）
  - VPR 匹配到非目标节点 → 统一继续当前步骤（取代复杂的跳步/重规划逻辑）
  - VPR 丢失 → 遮挡检测 + Qwen3.5 兜底打点找路（取代不可靠的趋势判断）
- **🎯 子图匹配 best_fail_camera**：`match_current_step()` 在全部失败时记录得分最高的 camera，遮挡检测使用该 camera 而非静态 `step.camera_name`
- **🖥️ 可视化新增遮挡检测 Tab**：`memory_visualization_server.py` 新增 🚧 遮挡检测验证 Tab
  - 上传相机图片，可调面积阈值和 YOLO 置信度
  - 实时展示检测框、面积占比、遮挡判定结果

### v1.9.0

- **🔭 Lookahead 双重确认**：步骤切换条件从单一 VPR 匹配升级为 VPR + 下一步子图匹配双重确认
  - 每帧对当前步骤和下一步同时做子图匹配（lookahead 不走缓存逻辑）
  - VPR 匹配到目标节点时，需下一步子图匹配成功（`conf >= 0.60`）才 advance
  - 最后一步无需 lookahead，直接 advance
  - 新增 `VPR HELD` 状态：VPR 到了但 lookahead 未确认，暂缓切换
- **🎯 子图匹配阈值统一**：`SUB_MATCH_CONFIDENCE_THRESHOLD = 0.60` 作为唯一真相源
  - 服务端 → `MemoryNavigator` → `SubImageMatcher` 全链路传参
  - 测试端通过 `from deploy.ws_proxy_with_memory import SUB_MATCH_CONFIDENCE_THRESHOLD` 引用

### v1.8.0

- **🆕 鱼眼去畸变**：新增 `memory_nav/fisheye_undistort.py`，移植自 `cam/tools/fisheye_undist_cpu.h`
- **🆕 像素→机器人坐标转换**：新增 `memory_nav/coord_transform.py`
- **🆕 cam/ 目录**：纳入多目鱼眼相机 ROS2 节点源码及相机参数配置

### v1.7.0

- **Qwen3.5 兜底打点**：新增基于 Qwen3.5-9B VLM 的打点方案，统一使用"通道正中间位置+景深"找路策略
- **InternVLA 按需加载**：InternVLA 模型默认不加载，需要时按需启动

### v1.6.0

- **子图匹配精简**：移除 SuperPoint+LightGlue 和 Qwen3.5 方案，仅保留 **DINOv3** 密集特征匹配
- **匹配阈值统一**：置信度阈值统一为 `SUB_MATCH_CONFIDENCE_THRESHOLD`（0.60）
- **帧间相似度升级**：SSIM 替换为 DINOv2 帧间相似度，阈值 0.70
- **三级 crop 级联匹配**：small/mid/big 三种裁剪尺度级联匹配 + 全相机遍历

### v1.5.0

- **输出格式统一**：记忆关闭时输出与 `ws_proxy.py` 完全一致，始终包含 `pixel_target`
- **子图匹配缓存**：置信度低于阈值时自动延用上一帧成功结果

### v1.4.0

- **子图匹配导航**：从角度导航升级到 SuperPoint + LightGlue 子图匹配

### v1.3.0

- **统一配置管理**：所有 VPR 参数集中到 `deploy/vpr_config.yaml`

### v1.2.0

- **多 VPR 方案支持**：新增 SelaVPR++、MegaLoc、EffoVPR 三种方案

### v1.1.0

- **记忆导航服务**：WebSocket 代理 + VPR 定位 + 路径规划

### v1.0.0

- **基础框架**：拓扑记忆图、AnyLoc VPR、InternVLA 推理

---

## 📚 引用

如果本项目对您的研究有帮助，请引用相关 VPR 论文：

```bibtex
@article{selavprpp2025,
  title={SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition},
  author={Lu, Feng and Jin, Tong and others},
  journal={IEEE T-PAMI},
  year={2026},
  volume={48},
  number={3},
  pages={2731-2748}
}

@inproceedings{megaloc2025,
  title={MegaLoc: One Retrieval to Place Them All},
  author={Berton, Gabriele and Masone, Carlo},
  booktitle={CVPR Workshops},
  year={2025}
}

@article{effovpr2024,
  title={Effective Foundation Model Utilization for Visual Place Recognition},
  author={Tzachor, Issar and others},
  journal={arXiv:2405.18065},
  year={2024}
}

@article{anyloc2023,
  title={AnyLoc: Towards Universal Visual Place Recognition},
  author={Keetha, Nikhil and others},
  journal={IEEE RA-L},
  year={2023}
}
```

---

## 📄 License

本项目采用 [MIT License](LICENSE) 开源协议。
