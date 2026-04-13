# MemoryNav 项目技术文档

> **版本**: 基于最新代码，截止 2026-04-13
>
> **代码根**: `/home/ubuntu/Disk/codes/jianxiong/MemoryNav/`

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 项目结构](#2-项目结构)
- [3. 记忆导航系统](#3-记忆导航系统)
  - [3.1 系统总览](#31-系统总览)
  - [3.2 数据模型](#32-数据模型)
  - [3.3 记忆构建](#33-记忆构建)
  - [3.4 拓扑图与路径规划](#34-拓扑图与路径规划)
  - [3.5 VPR 视觉位置识别](#35-vpr-视觉位置识别)
  - [3.6 导航器](#36-导航器)
  - [3.7 子图匹配](#37-子图匹配)
  - [3.8 帧间相似度缓存](#38-帧间相似度缓存)
  - [3.9 遮挡检测](#39-遮挡检测)
  - [3.10 Qwen3.5 兜底打点](#310-qwen35-兜底打点)
  - [3.11 坐标转换](#311-坐标转换)
  - [3.12 鱼眼去畸变](#312-鱼眼去畸变)
  - [3.13 WebSocket 服务](#313-websocket-服务)
- [4. 在线建图系统](#4-在线建图系统)
  - [4.1 系统总览](#41-系统总览)
  - [4.2 几何层](#42-几何层)
  - [4.3 拓扑层](#43-拓扑层)
  - [4.4 语义层](#44-语义层)
  - [4.5 输出与终结化](#45-输出与终结化)
  - [4.6 配置项](#46-配置项)
- [5. 离线建图](#5-离线建图)
- [6. 部署与运维](#6-部署与运维)
- [附录 A: 关键阈值汇总](#附录-a-关键阈值汇总)
- [附录 B: WebSocket 协议详情](#附录-b-websocket-协议详情)

---

## 1. 项目概述

MemoryNav 是一套**纯视觉记忆导航系统**，让机器人能够：

1. **记住去过的地方**（离线/在线建图 → 语义拓扑图）
2. **认出当前位置**（VPR 视觉位置识别）
3. **规划并执行导航**（最短路径 + 子图匹配 + 兜底打点）

整个系统不依赖 GPS、激光雷达或任何外部定位硬件，**只用 4 个鱼眼相机 + 1 个前置相机**的图像完成全部感知和导航。

### 硬件假设

| 传感器 | 规格 | 用途 |
|--------|------|------|
| camera_1~4 | 鱼眼相机, HFOV=190° | VPR 定位、子图匹配、遮挡检测 |
| front_1 | 前置 RGB 相机 | 模型推理（预留） |

### GPU 分配

| GPU | 模型 | 用途 |
|-----|------|------|
| GPU 0 | SelaVPR++ (DINOv2-large) | VPR 特征提取 |
| GPU 0 | DINOv3 (ViT-B/16) | 子图匹配 |
| GPU 0 | YOLOv8n | 遮挡检测 |
| GPU 1 | Qwen3.5-9B (vLLM) | 兜底打点 + 在线建图命名 |

---

## 2. 项目结构

```
MemoryNav/
├── memory_nav/                       # 核心导航库 (v3.0.0, 19 个 Python 文件)
│   ├── __init__.py                   # 版本号 + 统一导出
│   ├── memory_models.py              # 数据模型
│   ├── memory_graph.py               # 拓扑图 + 路径规划
│   ├── memory_vpr.py                 # VPR 定位 (FAISS + 循环移位)
│   ├── memory_builder.py             # 记忆构建器
│   ├── memory_navigator.py           # 导航器主类
│   ├── sub_image_matcher.py          # DINOv3 子图匹配
│   ├── qwen35_point_grounder.py      # Qwen3.5 打点 (双后端)
│   ├── qwen35_grounding_server.py    # Qwen3.5 子进程服务端
│   ├── coord_transform.py            # 柱面像素 → 机器人坐标
│   ├── fisheye_undistort.py          # 鱼眼去畸变 (柱面投影)
│   ├── occlusion_detector.py         # YOLOv8n 遮挡检测
│   ├── vpr_factory.py                # VPR 提取器工厂
│   ├── vpr_config_loader.py          # VPR 配置加载
│   ├── selavpr_extractor.py          # SelaVPR++ 提取器
│   ├── anyloc_extractor.py           # AnyLoc 提取器
│   ├── megaloc_extractor.py          # MegaLoc 提取器
│   ├── effovpr_extractor.py          # EffoVPR 提取器
│   └── megaloc_model.py              # MegaLoc 网络定义
│
├── deploy/                           # 部署服务
│   ├── ws_proxy_with_memory.py       # WebSocket 主服务 (端口 9528, 2085 行)
│   ├── ws_client.py                  # WebSocket 客户端 (765 行)
│   ├── vpr_config.yaml               # VPR 统一配置
│   ├── build_memory.sh               # 记忆构建脚本
│   ├── start_qwen_vllm.sh            # Qwen3.5 vLLM 启动脚本
│   ├── start_server.sh               # 服务启动脚本
│   └── logs/                         # 日志 + 可视化图像
│
├── online_mapper/                    # 在线建图 (v2.3.0)
│   ├── config.py                     # OnlineMapperConfig
│   ├── run_online_map.py             # CLI 入口
│   ├── core/
│   │   ├── online_mapper_core.py     # 主编排器 (~984 行)
│   │   └── stream_loader.py          # 流式帧加载
│   ├── geometry/
│   │   ├── vggt_backend.py           # VGGT-1B 单例 + 滑窗
│   │   ├── depth_estimator.py        # 深度估计 (VGGT / DA-V2)
│   │   ├── visual_odometry.py        # 视觉里程计 (VGGT / ORB)
│   │   ├── occupancy.py              # 占据栅格
│   │   ├── pose_graph.py             # 位姿图优化
│   │   └── junction_detector.py      # 路口检测
│   ├── topology/
│   │   ├── keyframe_selector.py      # 多触发关键帧选择
│   │   ├── loop_closure.py           # 闭环检测 (auto-tune)
│   │   ├── connection_builder.py     # 邻接构建 (几何先验)
│   │   └── graph.py                  # TopoGraph / TopoNode
│   ├── semantics/
│   │   ├── open_set_detector.py      # Grounding-DINO 封装
│   │   ├── door_plate_tracker.py     # 门牌多帧追踪
│   │   ├── hallucination_filter.py   # 三层防幻觉 (Qwen验证 + 多帧投票 + 去重)
│   │   ├── node_category.py          # 节点类别分类器
│   │   ├── node_naming.py            # 结构化命名 NodeName
│   │   ├── colocation_merger.py      # 同位置节点合并
│   │   └── scene_graph.py            # 层次场景图
│   └── io/
│       └── merged_data_writer.py     # 输出写入器
│
├── offline_mapper/                   # 离线建图 (只读, 供 online_mapper import)
│   ├── auto_mapper_core.py
│   ├── auto_sub_image_extractor.py   # 子图提取 (被 connection_builder 继承)
│   ├── auto_landmark_namer.py        # 地标命名 (Qwen vLLM)
│   ├── node_distance_estimator.py    # 节点距离估计
│   └── validate_output.py            # 输出校验
│
├── merged_labeled_data/              # 记忆数据 (44 个节点目录)
├── pretrained/                       # 模型权重 (.gitignore)
│   ├── vggt-1b/                      # VGGT-1B (5.0GB)
│   ├── grounding-dino-base/          # Grounding-DINO
│   ├── dinov3_vitb16.safetensors     # DINOv3 子图匹配骨干
│   ├── yolov8n.pt                    # 遮挡检测
│   └── depth-anything-v2-small-hf/   # 备用深度后端
├── cam/
│   └── params.yaml                   # 鱼眼内参 + 畸变系数 + T_ic 外参
└── third_party/
    └── vggt_space/                   # VGGT 源码 (HF Space)
```

---

## 3. 记忆导航系统

### 3.1 系统总览

记忆导航的完整工作流程：

```
                         ┌──────────────────────────────────┐
                         │    merged_labeled_data/           │
                         │    (44个节点, 每节点4张相机图      │
                         │     + crops子图 + JSON元数据)      │
                         └───────────────┬──────────────────┘
                                         │ MemoryBuilder
                                         ▼
                         ┌──────────────────────────────────┐
                         │    MemoryGraph + MemoryVPR        │
                         │    (拓扑图 + FAISS特征索引)        │
                         └───────────────┬──────────────────┘
                                         │ pickle 缓存
                                         ▼
┌──────────┐  WebSocket   ┌──────────────────────────────────┐
│ 机器人端  │ ──请求──→   │  ws_proxy_with_memory.py          │
│ (4鱼眼+   │             │  ┌────────────────────────────┐  │
│  front_1) │  ←响应──    │  │ MemoryNavigator            │  │
└──────────┘             │  │  ├─ VPR 定位 (SelaVPR++)    │  │
                         │  │  ├─ 路径规划 (NetworkX)     │  │
                         │  │  ├─ 子图匹配 (DINOv3)      │  │
                         │  │  ├─ 遮挡检测 (YOLOv8n)     │  │
                         │  │  ├─ Qwen3.5 兜底打点        │  │
                         │  │  └─ 坐标转换 → action       │  │
                         │  └────────────────────────────┘  │
                         └──────────────────────────────────┘
```

每帧请求的处理流程：

```
1. 解码图像 (base64 → numpy)
2. 鱼眼去畸变 (4相机并行 cv2.remap)
3. VPR 定位 (SelaVPR++ batch 特征提取 → FAISS 循环移位匹配)
4. 若有活跃导航计划:
   4a. 子图匹配 (当前步 + lookahead下一步)
   4b. 帧间相似度缓存判断
   4c. 遮挡检测 (子图匹配失败时)
   4d. Qwen3.5 兜底打点 (未遮挡时)
   4e. 导航决策 (advance/hold/wait/fallback)
5. 若无活跃计划: 尝试从 task 创建新导航计划
6. 坐标转换 → 输出 action
```

### 3.2 数据模型

所有数据结构定义在 `memory_nav/memory_models.py`。

#### MemoryEdge（记忆边）

记忆边描述从一个节点看向邻居节点时的导航线索：

```python
@dataclass
class MemoryEdge:
    target_node_id: str               # 目标节点ID
    target_node_name: str             # 目标节点名称 (中文)
    camera_name: str                  # 看到目标的相机 (camera_1~4)
    landmark_name: str                # 注意力地标名称 (如"电梯门")
    crop_image_paths: Dict[str, str]  # 三级子图路径 {"big":..., "mid":..., "small":...}
    big_box: Tuple[float, ...]        # big 级别归一化 bbox
    mid_box: Tuple[float, ...]        # mid 级别归一化 bbox
    small_box: Tuple[float, ...]      # small 级别归一化 bbox
    target_node_name_eng: str         # 目标节点英文名
    landmark_name_eng: str            # 地标英文名
    crop_image_path: str              # 向 big 的兼容引用
```

三级子图是在线/离线建图时从相机原图中裁切出的注意力区域，从 small（最精确）到 big（最宽松），用于级联匹配。

#### MemoryNode（记忆节点）

```python
@dataclass
class MemoryNode:
    node_id: str                      # 节点ID
    node_name: str                    # 节点名称 (中文)
    node_name_eng: str                # 节点英文名
    camera_images: Dict[str, str]     # {camera_id: 图片路径}
    camera_features: Dict[str, np.ndarray]  # {camera_id: VPR特征向量}
    fused_feature: np.ndarray         # 4相机特征均值 (L2归一化)
    edges: List[MemoryEdge]           # 出边列表
    base_path: str                    # 节点数据目录路径
    timestamp: int                    # 时间戳
```

`fused_feature` 是 4 个相机特征向量的算术平均后 L2 归一化，用于 FAISS 快速检索。

#### NavigationStep（导航步骤）

```python
@dataclass
class NavigationStep:
    from_node_id: str
    from_node_name: str
    to_node_id: str
    to_node_name: str
    camera_name: str                  # 目标所在相机
    landmark_name: str                # 注意力地标
    crop_image_paths: Dict[str, str]  # 三级子图
    step_index: int
    # 以及对应的英文名字段
```

#### NavigationPlan（导航计划）

```python
@dataclass
class NavigationPlan:
    start_node_id: str
    start_node_name: str
    goal_node_id: str
    goal_node_name: str
    path: List[str]                   # 节点ID序列
    steps: List[NavigationStep]       # 逐步指令
    total_steps: int
    success: bool
```

#### VPRResult（VPR 定位结果）

```python
@dataclass
class VPRResult:
    matched_node_id: str
    matched_node_name: str
    similarity: float                 # 4相机平均余弦相似度
    confidence: float                 # 同 similarity
    camera_scores: Dict[str, float]   # 各相机分数
    heading_offset: float             # 朝向偏移 (度)
    best_shift: int                   # 最佳循环移位 (0-3)
```

### 3.3 记忆构建

`memory_nav/memory_builder.py` 中的 `MemoryBuilder` 负责将 `merged_labeled_data/` 目录转化为可导航的记忆图。

#### 输入格式

```
merged_labeled_data/
├── 1/                                # 节点 1
│   ├── 1770097720_camera_1.jpg       # 4路相机原图
│   ├── 1770097720_camera_2.jpg
│   ├── 1770097720_camera_3.jpg
│   ├── 1770097720_camera_4.jpg
│   ├── crops/                        # 邻居子图 (big/mid/small)
│   │   ├── ...__2__big__xxx.jpg
│   │   ├── ...__2__mid__xxx.jpg
│   │   └── ...__2__small__xxx.jpg
│   └── node_position_info.json       # 节点元数据
├── 2/ ...
└── ...
```

`node_position_info.json` 结构：

```json
{
  "self_position": {
    "position_id": "5",
    "position_name": "前台·DEEPROUTE.AI",
    "position_name_eng": "Reception · DEEPROUTE.AI",
    "camera_1": "1770097843_camera_1.jpg",
    "camera_2": "1770097843_camera_2.jpg",
    "camera_3": "1770097843_camera_3.jpg",
    "camera_4": "1770097843_camera_4.jpg"
  },
  "next_positions": [
    {
      "position_id": "7",
      "position_name": "关爱室",
      "camera_name": "camera_1",
      "landmark_name": "绿植墙",
      "crop_image_path": "crops/...__big__...jpg",
      "crop_image_paths": {"big": "...", "mid": "...", "small": "..."},
      "big_box": "0.515,0.311,0.785,0.648",
      "mid_box": "...",
      "small_box": "...",
      "position_name_eng": "Care Room",
      "landmark_name_eng": "green wall"
    }
  ]
}
```

#### 构建流程

```
MemoryBuilder.build_from_directory(data_dir)
  │
  ├─ 1. 扫描 data_dir 下所有子目录，按 node_id 排序
  │
  ├─ 2. 逐节点解析 node_position_info.json
  │     ├─ self_position → MemoryNode (ID, 名称, 相机图路径)
  │     └─ next_positions → List[MemoryEdge] (camera_name, landmark, crops)
  │
  ├─ 3. VPR 特征提取 (逐节点)
  │     ├─ 对 camera_1~4 的图片调用 extractor.extract(image)
  │     ├─ AnyLoc 特殊处理: 先收集所有描述子 → 训练 VLAD 词汇 → 重新提取
  │     └─ fused_feature = L2_normalize(mean(camera_features))
  │
  ├─ 4. 构建 MemoryGraph
  │     ├─ 添加所有 MemoryNode
  │     └─ 为每条 MemoryEdge 建立双向图边 (NetworkX 无向图)
  │
  ├─ 5. 构建 MemoryVPR
  │     ├─ 为每个节点注册 4 个相机特征
  │     └─ 构建 FAISS IndexFlatIP 索引 (fused_feature)
  │
  └─ 6. 序列化保存 (pickle)
        └─ 加载时校验特征维度与当前 VPR 方法是否一致
```

构建脚本 `deploy/build_memory.sh` 封装了上述流程，读取 `vpr_config.yaml` 获取配置。

### 3.4 拓扑图与路径规划

`memory_nav/memory_graph.py` 中的 `MemoryGraph` 管理整个记忆拓扑图。

#### 图结构

- **底层**: NetworkX 无向图 (`nx.Graph`)
- **节点**: 以 `node_id` 为键存储 `MemoryNode`
- **边**: 每条 `MemoryEdge` 产生一条无向边
- **后备**: FAISS 不可用或 NetworkX 不可用时，均有纯 Python 后备实现

#### 目的地搜索

`find_node_by_name(query)` 实现多策略模糊搜索：

```
1. 精确匹配 node_id
2. 精确匹配 node_name (中文) 或 node_name_eng (英文)
3. 子串匹配 (query 是 node_name 的子串, 或反过来)
4. 英文子串匹配
5. 全部失败 → 返回 None
```

`MemoryNavigator.find_destination()` 进一步增强：

```
1. 调用 graph.find_node_by_name(task)
2. 失败 → 正则去除中英文前缀 ("前往/去/到/go to/navigate to/bring me to/...")
3. 用清理后的目的地名重新搜索
```

#### 路径规划

```python
def plan_navigation(self, start_id, goal_id) -> NavigationPlan:
    path = nx.shortest_path(self.graph, start_id, goal_id)  # BFS后备
    steps = []
    for i in range(len(path) - 1):
        edge = self.nodes[path[i]].get_edge_to(path[i+1])
        steps.append(NavigationStep(
            from_node=path[i], to_node=path[i+1],
            camera_name=edge.camera_name,
            landmark_name=edge.landmark_name,
            crop_image_paths=edge.crop_image_paths,
            ...
        ))
    return NavigationPlan(path=path, steps=steps, ...)
```

### 3.5 VPR 视觉位置识别

`memory_nav/memory_vpr.py` 中的 `MemoryVPR` 是定位的核心。

#### 相机布局

4 个鱼眼相机围绕机器人安装，HFOV=190°，去畸变后 FOV=180°：

```
           camera_2 (+37.5°, 右前)
                 ╲
                  ╲
   camera_1 ─────── 🤖 ─────── camera_4
   (-37.5°, 左前)   │         (+143.5°, 左后)
                     │
                camera_3 (+142.5°, 右后)
```

#### 循环移位匹配算法

这是 SelaVPR++ 使用的核心匹配方法，利用 4 相机的空间排列约束：

```
定义 4 种 shift (机器人朝向偏移):
  shift=0: query[1,2,3,4] ↔ memory[1,2,3,4], heading_offset=0°
  shift=1: query[1,2,3,4] ↔ memory[2,3,4,1], heading_offset=-75°
  shift=2: query[1,2,3,4] ↔ memory[3,4,1,2], heading_offset=-180°
  shift=3: query[1,2,3,4] ↔ memory[4,1,2,3], heading_offset=+105°

对每个候选节点 n, 每种 shift s:
  similarity(n,s) = mean(cosine(query[i], memory[shift_map(i,s)])) for i=1..4

全局最优:
  (best_node, best_shift) = argmax similarity(n,s)
  heading_offset = shift_heading_offsets[best_shift]
```

heading_offset 表示机器人相对记忆时刻的朝向偏转，供后续导航决策使用。

#### FAISS 索引

- 使用 `faiss.IndexFlatIP` (内积 = 余弦相似度，因为特征已 L2 归一化)
- fused_feature 用于快速粗筛 Top-K 候选节点
- 然后对候选节点执行完整的 4 相机循环移位匹配
- numpy 后备: 当 FAISS 不可用时使用矩阵乘法

#### 无序贪心匹配

AnyLoc/MegaLoc/EffoVPR 等方法不依赖相机排列，使用贪心二分匹配：

```
对每个候选节点:
  构建 4×4 相似度矩阵: sim[i][j] = cosine(query_cam_i, memory_cam_j)
  贪心匹配: 每次取全局最大, 标记已用行列
  similarity = 已匹配对的平均相似度
```

#### VPR 方法配置

通过 `deploy/vpr_config.yaml` 统一配置：

```yaml
vpr_method: selavpr           # selavpr | anyloc | megaloc | effovpr

order_invariant:              # 是否使用无序匹配
  selavpr: false              # 循环移位
  megaloc: true               # 贪心二分
  effovpr: true
  anyloc: true

similarity_threshold:         # 各方法的匹配阈值
  selavpr: 0.60
  megaloc: 0.60
  effovpr: 0.80
  anyloc: 0.70

selavpr:                      # SelaVPR++ 专用配置
  backbone: dinov2-large      # dinov2-base (2048D) / dinov2-large (4096D)
  aggregation: gem            # gem / boq / salad
  use_hashing: true           # 二进制哈希加速
  use_rerank: true            # rerank 精排
```

`vpr_factory.py` 根据配置创建对应的提取器：

| 方法 | 模型 | 特征维度 | 特点 |
|------|------|---------|------|
| selavpr | DINOv2-large + MultiConv Adapter | 4096D | 精度最高, 支持循环移位 |
| anyloc | DINOv2-ViT-B/14 + VLAD | 32×768=24576D | 需训练词汇 |
| megaloc | DINOv2 + Optimal Transport | 8448D | 大特征维度 |
| effovpr | DINOv2 multi-layer GeM | 768D | 最轻量 |

### 3.6 导航器

`memory_nav/memory_navigator.py` 中的 `MemoryNavigator` 是整个导航系统的入口类。

#### 初始化

```python
MemoryNavigator(
    graph=MemoryGraph,          # 拓扑图
    vpr=MemoryVPR,              # VPR索引
    feature_extractor=...,       # VPR特征提取器 (SelaVPR++)
    sub_image_method="dinov3",   # 子图匹配方法
    confidence_threshold=0.65,   # 子图匹配阈值
    qwen35_gpu="1",              # Qwen3.5 GPU
    ...
)
```

内部组件延迟加载:
- `SubImageMatcher`: 首次调用 `match_current_step()` 时初始化
- `Qwen35PointGrounder`: 首次调用 `fallback_point_grounding()` 时启动

#### 核心方法

**`locate_by_images(camera_images, return_features=False)`**

```
输入: {camera_1: BGR_image, ..., camera_4: BGR_image}
流程:
  1. extractor.extract_batch([img1, img2, img3, img4])  # 批量前向 (单次forward)
  2. vpr.locate(query_features, threshold)               # 循环移位匹配
输出: (VPRResult, query_features)  # 当 return_features=True
```

**`match_current_step(camera_images, step)`**

级联子图匹配，跨全部 4 个相机寻找最佳匹配：

```
对 camera_1, camera_2, camera_3, camera_4 分别执行:
  _cascade_match_single_camera(camera_image, step):
    1. 尝试 small 子图 → 匹配成功(conf >= threshold) → 返回
    2. 尝试 mid 子图   → 匹配成功 → 返回
    3. 尝试 big 子图   → 匹配成功 → 返回
    4. 全部失败 → 返回最高 confidence 结果 (found=False)

汇总 4 个相机结果:
  选择 confidence 最高的相机结果作为最终结果
  记录 best_camera_name (即使全部失败也记录得分最高的相机)
```

**`fallback_point_grounding(camera_images, landmark_name, target_camera)`**

```
1. 优先尝试 target_camera
2. 失败 → 遍历其余相机
3. 调用 Qwen35PointGrounder.predict(camera_image, landmark_name)
4. 返回 {success, point: [x_norm, y_norm], camera_name, confidence}
```

### 3.7 子图匹配

`memory_nav/sub_image_matcher.py` 实现基于 DINOv3 的密集特征匹配。

#### 模型

- **DINOv3 ViT-B/16**: 768 维 patch token 特征
- 权重: `pretrained/dinov3_vitb16.safetensors` (本地优先, 线上回退)
- 输入分辨率: 相机图缩放到 518px, 子图按比例缩放 (最小 2×patch_size, 最大 280px)

#### 匹配算法

```
1. 提取相机图的 dense patch tokens → [H_cam, W_cam, 768] 特征图
2. 提取子图的 dense patch tokens → [H_crop, W_crop, 768] 特征图
3. 滑动窗口匹配:
   - 用 F.unfold 将相机特征图展开为所有 (H_crop × W_crop) 大小的窗口
   - 每个窗口内所有 patch 与子图 patch 做余弦相似度
   - 每个窗口的得分 = 所有 patch 对的平均相似度
4. 找到得分最高的窗口位置
5. confidence = 最高窗口得分
6. 若 confidence >= threshold(0.60) → 返回匹配区域 bbox
```

#### SubImageMatchResult

```python
@dataclass
class SubImageMatchResult:
    found: bool                   # 是否匹配成功
    confidence: float             # [0, 1]
    x_min, y_min, x_max, y_max: int  # 像素坐标 bbox
    top_left_pct: Dict            # 百分比坐标 {x, y}
    bottom_right_pct: Dict
    center_pct: Dict              # 中心点百分比
    elapsed_ms: float             # 耗时
    method: str                   # "dinov3"
```

### 3.8 帧间相似度缓存

`deploy/ws_proxy_with_memory.py` 中实现，利用 VPR 已提取的 DINOv2 特征做零开销的帧间比较。

#### 原理

VPR 流程每帧已提取 4 个相机的 DINOv2 特征，直接复用做帧间余弦相似度，无需额外推理。

#### 缓存逻辑 (`_cache_or_reuse_sub_match`)

```
输入: 当前帧子图匹配结果, 当前帧VPR特征, 匹配相机名

如果子图匹配完全无结果 (None):
  若有缓存 + 帧相似度 >= 0.70 → 复用缓存, cache_action="reused"
  否则 → 清除缓存, cache_action="cleared"

如果子图匹配成功 (confidence >= 0.60):
  采纳结果, 更新缓存, cache_action="accepted"

如果子图匹配失败 (confidence < 0.60):
  若有缓存 + 帧相似度 >= 0.70 → 复用缓存, cache_action="reused"
  若有缓存 + 帧相似度 < 0.70  → 清除缓存, cache_action="cleared"
  无缓存 → cache_action="no_cache"
```

帧相似度计算 (`_frame_similarity_dino`):

```python
def _frame_similarity_dino(feat1, feat2, camera_name=None):
    # feat1, feat2: {camera_id: ndarray} 字典
    # 若指定 camera_name, 只比较该相机; 否则比较所有共有相机
    similarities = [cosine(feat1[cam], feat2[cam]) for cam in common_cameras]
    return mean(similarities)
```

### 3.9 遮挡检测

`memory_nav/occlusion_detector.py` 中的 `OcclusionDetector`。

#### 触发条件

子图匹配失败时自动触发，不依赖 VPR 结果。使用子图匹配得分最高（但低于阈值）的那个相机的图像。

#### 检测流程

```
1. YOLOv8n 推理 (confidence_threshold=0.4)
2. 遍历检测结果, 找出遮挡类别:
   person(0), backpack(24), umbrella(25), handbag(26), suitcase(28)
3. 计算每个遮挡物 bbox 面积占画面总面积的比例
4. 任一遮挡物面积比 >= 25% → 判定为遮挡
```

#### OcclusionResult

```python
@dataclass
class OcclusionResult:
    occluded: bool                     # 是否遮挡
    max_area_ratio: float              # 最大单个遮挡物面积比
    total_area_ratio: float            # 所有遮挡物总面积比
    detections: List[Dict]             # 每个检测目标的详情
    occluding_classes: List[str]       # 触发遮挡的类别
    elapsed_ms: float
```

#### 遮挡时行为

- 输出 `action: [0, 0, 0]` (原地等待)
- 清除子图匹配缓存
- `nav_state.consecutive_occlusions += 1`

### 3.10 Qwen3.5 兜底打点

`memory_nav/qwen35_point_grounder.py` 中的 `Qwen35PointGrounder`。

#### 触发条件

子图匹配失败 + 未遮挡 + 非 VPR 快速 advance 场景。

#### 双后端架构

```
优先: vLLM HTTP API (端口 8199, 与在线建图共享同一 vLLM 服务)
      └─ 并行多相机推理 (ThreadPoolExecutor)
      └─ 延迟低 (~1-2s), 不占额外显存

回退: 子进程模式 (qwen3 conda 环境)
      └─ stdin/stdout JSON 行协议
      └─ 串行遍历相机, confidence >= 0.5 提前退出
      └─ 首次调用需加载模型 (~30s)
```

#### 两步推理法

解决 Qwen3.5 "无论目标是否存在都强行输出坐标"的问题：

```
步骤 1: 存在性检测
  Prompt: 'Is "xxx" visible in this image? Answer with ONLY one word: yes or no'
  输出: 1~2 个 token
  "no" → 直接返回 not_found, 不执行打点

步骤 2: 条件打点 (仅 yes 时执行)
  Prompt: 'Point to "xxx" in this image. Output ONLY JSON: {"point": [x, y]}'
  坐标范围: [0, 1000], 归一化到 [0, 1]
```

解析逻辑 (`_parse_coordinates`):
1. 去除 `<think>...</think>` 标签
2. 尝试 JSON 解析
3. 检查 not-found 模式 (中英文: "not found", "not visible", "未发现", "不在图中" 等)
4. 正则回退: 提取 `[数字, 数字]` 或 `(数字, 数字)` 模式

#### 固定找路策略

当前兜底打点**不再使用 step 的 landmark_name**（因为参照物经常不可见导致乱指），改为固定 prompt:

```
"通道正中间位置+景深"
```

引导机器人沿通道中轴线前进，是最安全的兜底策略。

### 3.11 坐标转换

`memory_nav/coord_transform.py` 将子图匹配或 Qwen3.5 打点输出的像素坐标转换为机器人运动指令。

#### 转换管线

```
pixel_target: [x_norm, y_norm]  (归一化到 [0, 1])
       │
       ▼
pixel_norm_to_angle(x_norm, FOV=180°)
  → horizontal_angle (弧度, 图像中心为0, 左正右负)
       │
       ▼
horizontal_angle + camera_azimuth → global_yaw (弧度)
       │
       ▼
estimate_distance_from_ynorm(y_norm):
  y_norm → 柱面垂直角 → 俯角 depression_deg
  若 depression_deg < 5° → 使用固定距离 20m (目标在水平面以上)
  否则 → distance = camera_height / tan(depression_deg)
  clamp(distance, 0.3m, 30m)
       │
       ▼
x_forward = distance × cos(global_yaw)
y_lateral = distance × sin(global_yaw)
       │
       ▼
输出: action = [[x_forward, y_lateral, 0.0]]
```

#### 相机方位角 (从 `cam/params.yaml` 的 T_ic 矩阵计算)

| 相机 | 方位角 (度, 逆时针正) |
|------|---------------------|
| camera_1 | 39.42° |
| camera_2 | -35.84° |
| camera_3 | -142.04° |
| camera_4 | 143.52° |

#### 侧面相机特殊处理

camera_3 和 camera_4 指向后方，匹配成功时不输出前进动作而是纯旋转:

```
子图匹配成功 + camera_3/camera_4:
  → action = [[0, 0, yaw_rad]]  (原地旋转)

Qwen3.5 兜底 + camera_3/camera_4:
  → action = [[0, 0, 0.785]]   (固定旋转约45°)
```

### 3.12 鱼眼去畸变

`memory_nav/fisheye_undistort.py` 实现从鱼眼原图到柱面投影图的去畸变。

#### 算法 (移植自 `cam/tools/fisheye_undist_cpu.h`)

```
输入: 鱼眼原图, 内参 [xi, fx, fy, cx, cy], 畸变 [k1, k2, p1, p2]
参数: output_fov=180°, output_size=1920×1536, pitch_up=15°

1. 为输出图每个像素 (u, v) 计算对应的 3D 射线方向
2. 应用 pitch_up 旋转
3. 通过鱼眼模型反投影到原图像素坐标
4. 生成 remap 查找表 (map_x, map_y)
5. 每帧: cv2.remap(src, map_x, map_y) → 柱面图
```

remap 表只需计算一次（初始化时），之后每帧仅做一次 `cv2.remap`。

#### 并行去畸变

`FisheyeUndistorter.undistort_batch()` 使用 ThreadPoolExecutor 并行处理 4 个相机:

```python
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {cam: pool.submit(cv2.remap, img, map_x, map_y, ...) for cam, img in images.items()}
    return {cam: f.result() for cam, f in futures.items()}
```

`cv2.remap` 底层释放 GIL，因此真正并行执行。

### 3.13 WebSocket 服务

`deploy/ws_proxy_with_memory.py` 是整个系统的运行时入口，2085 行。

#### 启动流程

```
main():
  1. 读取 vpr_config.yaml
  2. 创建 MemoryNavigator (加载 SelaVPR++ / DINOv3 / 等)
  3. 从 merged_labeled_data + pickle 缓存加载记忆图
  4. 初始化 FisheyeUndistorter (从 cam/params.yaml)
  5. 初始化 OcclusionDetector (YOLOv8n)
  6. 连接 Qwen3.5 vLLM
  7. 启动 WebSocket 服务 (0.0.0.0:9528)
```

#### 导航状态机 (MemoryNavState)

```python
@dataclass
class MemoryNavState:
    plan: NavigationPlan = None       # 当前导航计划
    current_step_idx: int = 0         # 当前步骤索引
    phase: str = "idle"               # 状态机阶段
    last_task: str = None             # 上次 task (检测变化)

    # VPR
    last_vpr_result: VPRResult = None
    last_query_features: Dict = None  # 上帧 VPR 特征 (帧间相似度用)
    consecutive_misses: int = 0       # 连续 VPR 丢失次数

    # 子图匹配缓存
    cached_sub_match: Dict = None     # 缓存的匹配结果
    cached_features: Dict = None      # 缓存帧的 VPR 特征
    last_cache_action: str = None     # accepted/reused/cleared/no_cache

    # Lookahead
    next_step_sub_match: Dict = None

    # 遮挡
    consecutive_occlusions: int = 0

    # Qwen3.5 兜底
    fallback_action: List = None
    fallback_pixel_target: List = None
    fallback_instruction: str = None
    fallback_camera_name: str = None
```

状态机流转：

```
idle ──(VPR定位成功+目的地匹配)──→ step_init
                                       │
                                  (返回记忆引导)
                                       │
                                       ▼
                                   verifying ──(VPR到达+lookahead)──→ step_init (下一步)
                                       │                                  或 completed
                                       │
                                  (子图失败+未遮挡+Qwen无结果)
                                       │
                                       ▼
                                   fallback ──(重发记忆引导)──→ step_init
                                       │
                                  (检测到遮挡)
                                       │
                                       ▼
                                   occluded ──(遮挡消除)──→ verifying
```

#### 导航决策逻辑 (每帧)

```
process_inference_with_memory():

  ── 阶段 1: 图像预处理 ──
  解码 front_1 + camera_1~4
  鱼眼去畸变 (4相机并行)

  ── 阶段 2: VPR 定位 ──
  VPR batch 特征提取
  循环移位匹配 → vpr_result

  ── 阶段 3: 活跃导航计划 ──
  if plan exists:

    3a. 当前步子图匹配 (4相机×3级cascade)
    3b. 帧间缓存判断
    3c. Lookahead 下一步子图匹配

    3d. 遮挡检测 (子图失败时):
        遮挡 → action=[0,0,0], 等待

    3e. Qwen3.5 判断 (未遮挡时):
        VPR已到目标 + (最后一步 or lookahead成功) → 跳过Qwen, 直接advance
        否则 → Qwen3.5 打点 ("通道正中间位置+景深")

    3f. 导航决策:
        Case A: VPR matched target + sim >= 0.70
                + (最后一步 or lookahead conf >= 0.60)
                → ADVANCE (切换到下一步)
        Case A': VPR matched target + lookahead 失败
                → HOLD (不切换, 继续当前步)
        Case B/C: VPR matched 其他节点 or sim < 0.70
                → 继续当前步
        Case D: VPR 丢失 + 子图成功
                → 用子图匹配结果导航
        Case D': VPR 丢失 + 子图失败 + 遮挡
                → 原地等待
        Case D'': VPR 丢失 + 子图失败 + Qwen有结果
                → 用 Qwen 打点结果导航

  ── 阶段 4: 新计划创建 ──
  if no plan + VPR 定位成功:
    尝试 find_destination(task) → plan_navigation()

  ── 阶段 5: 构建响应 ──
  坐标转换 → action
  打包 memory_info → WebSocket 响应
```

#### 性能计时

每帧记录 7 阶段耗时:

| 阶段 | 键名 | 典型耗时 |
|------|------|---------|
| 图像解码 | `1_decode_ms` | ~10ms |
| 相机解码+去畸变 | `2_cam_decode_undistort_ms` | ~50ms |
| VPR | `3_vpr_ms` | ~100ms |
| 子图匹配 | `4_sub_match_ms` | ~200ms |
| Lookahead | `5_lookahead_ms` | ~200ms |
| 遮挡检测 | `6_occlusion_ms` | ~20ms |
| Qwen3.5 兜底 | `7_qwen_fallback_ms` | ~2500ms (跳过时0) |

#### WebSocket 控制命令

| 命令 | 说明 |
|------|------|
| `reset` | 重置 Agent + 导航状态 |
| `session_status` | 查看会话状态 |
| `toggle_memory` | 切换记忆导航开关 |
| `memory_status` | 查看记忆导航详情 (含可用目的地列表) |
| `reset_memory` | 仅重置记忆状态 |

---

## 4. 在线建图系统

### 4.1 系统总览

`online_mapper/` 是**流式在线主动建图模块**，在机器人边走边拍的场景下实时构建导航用的**语义拓扑图**。

当前版本 **v2.3.0**, 架构按 **几何 / 拓扑 / 语义** 三层解耦：

```
                    OnlineMapperCore (编排器, ~984行)
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
┌─────▼──────┐         ┌──────▼──────┐         ┌──────▼──────────────┐
│ Geometry   │         │ Topology    │         │ Semantics           │
│ (几何)     │         │ (拓扑)      │         │ (语义)              │
├────────────┤         ├─────────────┤         ├─────────────────────┤
│ VGGTBackend│ ◄单例   │ KeyframeSel │         │ OpenSetDetector     │
│ ├VGGTDepth │         │ LoopCloser  │         │ DoorPlateTracker    │
│ ├VGGTVO    │ 零推理  │ TopoGraph   │         │ MultiFrameVoter     │
│ ├Occupancy │ ◄dense  │ ConnBuilder │ 几何先验│ QwenVerifier        │
│ └PoseGraph │         │ JunctionDet │         │ NodeCategoryClf     │
└────────────┘         └─────────────┘         │ ColocationMerger    │
                                               │ NodeName (结构化)   │
                                               │ NameDeduplicator    │
                                               └─────────────────────┘
```

#### 与离线建图的对比

| 维度 | offline_mapper | online_mapper v2.3.0 |
|------|---------------|---------------------|
| 时序 | 一次性看到所有帧 | 流式, 逐帧决策 |
| 几何前端 | — | VGGT-1B (depth + pose + point cloud 一次推理) |
| VO | — | 复用 VGGT pose, 零额外推理 |
| 占据栅格 | — | VGGT dense point map 直填 |
| 关键帧 | VPR + 最小帧间隔 | VPR + 累积位移 + 累积旋转 + 信息增益 |
| 闭环 | 首尾对比 | 全局 VPR + ORB 几何验证, auto-tune 阈值 |
| 节点命名 | 单帧 describe_scene | 结构化 NodeName + 多帧投票 + 二次验证 |
| cam→neighbor | 纯视觉 Hungarian | + 几何方向先验 |

### 4.2 几何层

#### 4.2.1 VGGT 后端 (`geometry/vggt_backend.py`)

VGGT-1B 是几何层的核心，**单次推理同时输出 5 种信息**：

```
输入: BGR 图像序列 (滑窗, 默认4帧)
输出:
  depth:        [H×W float] 深度图 (米, VGGT自洽尺度)
  depth_conf:   [H×W float] 深度置信度 (expp1激活, ≥1.0)
  world_points: [H×W×3 float] 稠密点云 (VGGT-world frame)
  extri:        [3×4 float] 外参 (cam-from-world, X_cam = R*X_w + T)
  intri:        [3×3 float] 内参
```

**VGGTBackend** 是进程内单例，懒加载 `pretrained/vggt-1b/model.pt`：

```python
class VGGTBackend:
    _instance = None

    @classmethod
    def get(cls, model_path, device, dtype="bf16"):
        if cls._instance is None:
            cls._instance = cls(model_path, device, dtype)
        return cls._instance

    def infer_bgr_list(self, bgr_list) -> dict:
        # 预处理: BGR→RGB, resize到518px宽, 归一化
        # VGGT forward → 解码各输出
        # 按帧拆分返回
```

**VGGTSlidingWindow** 维护一个 BGR ring buffer 提供时序上下文：

```python
class VGGTSlidingWindow:
    def __init__(self, backend, window_size=4):
        self.ring_buffer = []

    def push_and_infer(self, bgr):
        # 将 bgr 入栈, 触发推理
        # 返回: (最新帧结果, 倒数第二帧结果)  # 同坐标系

    def infer_stateless(self, bgr):
        # 单帧推理, 不入栈 (旁路用, 如 junction_detector)
```

#### 4.2.2 深度估计 (`geometry/depth_estimator.py`)

工厂 `build_depth_estimator(cfg)` 根据 `cfg.depth_backend` 返回：

| 后端 | 类 | 说明 |
|------|-----|------|
| `"vggt"` (默认) | `VGGTDepthEstimator` | 维护滑窗, 缓存 last/prev_extri, last_points_camera |
| `"da_v2"` | `DepthEstimator` | Depth-Anything-V2-Small, 旧后端, 用于回退 |

`VGGTDepthEstimator` 额外缓存：
- `last_extri`, `prev_extri`: 当前帧和上一帧外参（同坐标系）
- `last_world_points`: 当前帧点云（VGGT-world frame）
- `last_points_camera`: 当前帧点云（转到 camera frame, 给占据栅格用）

#### 4.2.3 视觉里程计 (`geometry/visual_odometry.py`)

工厂 `build_visual_odometry(cfg, depth_estimator)` 返回：

| 后端 | 类 | 耗时 |
|------|-----|------|
| `"vggt"` (默认) | `VGGTVisualOdometry` | ~0.004s/帧 (零推理, 复用外参) |
| `"orb"` | `MonoVO` | ~4.87s/49帧 (ORB+EssentialMatrix+recoverPose) |

VGGT VO 算法：

```
last_extri, prev_extri 来自同一次 VGGT 推理 (同坐标系)
R_curr = last_extri[:3,:3]; T_curr = last_extri[:3,3]
R_prev = prev_extri[:3,:3]; T_prev = prev_extri[:3,3]
C_curr = -R_curr.T @ T_curr   # 相机在世界系位置
C_prev = -R_prev.T @ T_prev
dtrans = ||C_curr - C_prev||
R_rel = R_curr @ R_prev.T     # cam_curr_from_cam_prev
dyaw = atan2(R_rel[0,2], R_rel[2,2])  # 绕 Y 轴
输出: (dtrans_m, dyaw_rad)
```

#### 4.2.4 占据栅格 (`geometry/occupancy.py`)

支持两种集成方式：

| 方式 | 后端 | 信息量 |
|------|------|--------|
| `integrate(depth_row, fov)` | DA-V2 | 1D ray-cast, ~8 free / 207 occ |
| `integrate_pointcloud(points_camera, ...)` | VGGT | dense 点云直填, ~28 free / 588 occ |

VGGT 点云直填流程：

```
1. 置信度过滤 (conf >= 1.0)
2. Z 范围过滤 ([0.05, 10] m)
3. 高度过滤 ([-1.5, 1.5] m)
4. 随机稀疏采样 (默认 6000 点)
5. Camera frame → Robot local (forward=z, left=-x)
6. 旋转 robot_theta + 平移
7. 标记 OCC 栅格
8. 沿 robot→OCC 射线等距采样标记 FREE 栅格
```

#### 4.2.5 路口检测 (`geometry/junction_detector.py`)

使用 4 个相机的深度图判断当前是否为路口：

```
对 camera_1~4 分别:
  depth.estimate_stateless(img)  # 单帧推理, 不污染滑窗
  统计各方向通行度
判断: 十字路口 / T字路口 / 直线走廊
```

### 4.3 拓扑层

#### 4.3.1 关键帧选择 (`topology/keyframe_selector.py`)

4 个触发条件 (OR 关系)，全部受 `min_keyframe_frame_interval` (默认 3) 约束：

| 触发条件 | 默认阈值 | 含义 |
|----------|---------|------|
| VPR dissimilarity | 0.50 | 当前帧与上一关键帧的 VPR 差异 |
| 累积位移 | 1.5m | VO 累积平移距离 |
| 累积旋转 | 0.6 rad | VO 累积偏航角 |
| 信息增益 | 0.05 | 占据栅格新信息比例 |

```python
def should_trigger(self, frame_idx, vpr_dissim, dtrans, dyaw, info_gain):
    if frame_idx - self.last_kf_idx < self.min_interval:
        return False
    self.acc_trans += dtrans
    self.acc_rot += abs(dyaw)
    return (vpr_dissim > self.vpr_thresh
            or self.acc_trans > self.trans_thresh
            or self.acc_rot > self.rot_thresh
            or info_gain > self.ig_thresh)
```

#### 4.3.2 闭环检测 (`topology/loop_closure.py`)

每帧执行闭环检测，特点是**自适应阈值**：

```
检测流程:
  1. 计算当前帧 VPR 特征与所有已有节点的相似度
  2. 排除时间距离 < min_gap(8) 的近邻
  3. 取 Top-K(5) 候选
  4. 对每个候选做 ORB + RANSAC 几何验证 (min_inliers=15)
  5. 验证通过 → 闭环成功

auto-tune 阈值:
  维护相似度历史 → 计算 mean + 2*sigma
  阈值 = clamp(mean + 2*sigma, 0.65, 0.92)
  连续 N 帧不命中 → 微调降低阈值
```

循环相似度计算（与导航系统一致）：

```python
def _cyclic_similarity(self, feat_a, feat_b):
    # feat_a, feat_b: {camera_1: vec, ..., camera_4: vec}
    # 4种shift, 取最佳
    best = max(mean(cosine(a[shift_map(i)], b[i])) for shift in 0..3)
    return best
```

#### 4.3.3 邻接构建 (`topology/connection_builder.py`)

`ThresholdedSubImageExtractor` 继承自 `offline_mapper.AutoSubImageExtractor`，增加了**几何方向先验**：

```
对每个节点的 4 个相机 ↔ 所有邻居节点:

1. GroundedPointer (Qwen vLLM): 在每个相机上找 "通道正中间位置" → crop bbox
2. DINOv3 CLS feature: 提取每个 crop 和每条走廊路径的视觉特征
3. 视觉相似度矩阵: sim_matrix[cam_i][nb_j] = cosine(crop_feat, corridor_feat)
4. 几何方向先验:
   cam_angles = {camera_1: 0, camera_2: -π/2, camera_3: π, camera_4: π/2}
   对每个 (cam_i, nb_j):
     nb_pose = neighbor 的 (x, y, theta)
     robot_ang = atan2(nb.y - my.y, nb.x - my.x) - my.theta
     diff = wrap(robot_ang - cam_angles[cam_i])
     angular_score = cos(diff)
     if angular_score < -0.3: hard_penalty = -1.0  # 反向惩罚

   final_sim = visual_sim + 0.6 * angular_score + hard_penalty

5. Hungarian 匹配 → cam ↔ neighbor 1-to-1 最优配对
6. 相似度阈值过滤 (connection_sim_threshold=0.40)
7. 为每个匹配对保存 big/mid/small 三级子图
```

#### 4.3.4 拓扑重建

`_rebuild_topology_neighbors_spatial(k_spatial=1, k_temporal=1)` 在 finalize 阶段执行：

```
1. 清空所有边
2. 空间 KNN: 按 pose 距离找最近的 k_spatial 个邻居
3. 时间 KNN: 按时间顺序找最近的 k_temporal 个邻居
4. 合并去重 → 重建邻接关系
```

### 4.4 语义层

#### 4.4.1 开放集检测 (`semantics/open_set_detector.py`)

基于 Grounding-DINO-Base，支持文本 query 检测：

```
默认 query:
  door plate, room number sign, printer, trash can, white chair,
  stool, elevator, fire extinguisher, potted plant, vending machine,
  sofa, table, monitor
```

#### 4.4.2 门牌追踪与三层防幻觉

**DoorPlateTracker** (`semantics/door_plate_tracker.py`):
- Grounding-DINO 每帧检测门牌
- 选取每个门牌文字的"最佳代表帧"（清晰度最高）

**三层防幻觉** (`semantics/hallucination_filter.py`):

```
第 1 层: STRICT Prompt
  Qwen 输出必须包含 confidence 字段
  不确定 → 返回 false

第 2 层: QwenVerifier 二次验证
  用整张相机图问 "图中是否真有文字 X?"
  三种 prompt 模式: strict / scene / specific-name

第 3 层: MultiFrameVoter 多帧投票
  ≥2 不同帧确认, 或同帧 ≥2 个相机确认
  白名单 fast-pass (常见功能区名称)
  子串变体合并 (如 EUMANN → NEUMANN)
```

#### 4.4.3 节点类别分类 (`semantics/node_category.py`)

决策树优先级：

```
ROOM_NUMBERED > ROOM_NAMED > FUNCTION_AREA > LANDMARK_FACILITY
  > 通用X室 > SHOP > JUNCTION (十字/T字) > REJECT
```

包含白名单（强电井、关爱室、打印区、茶水间…）、拒绝关键词（广告、品牌噪声…）、中英文映射表。

`JunctionKind` 枚举: `CROSS` (十字路口), `T_JUNCTION` (T字路口), `NONE` (直线走廊)。

#### 4.4.4 结构化命名 (`semantics/node_naming.py`)

替代旧的字符串拼接命名，解决了 "DEEPROUTE.AI前台" 这类粘连问题：

```python
@dataclass
class NodeName:
    category: str = ""              # 主类型 (中文), 如 "前台"
    category_en: str = ""           # 英文, 如 "Reception"
    organization: str = ""          # 关联实体 (品牌/门牌), 如 "DEEPROUTE.AI"
    nearby_plates: list = []        # 同节点其他门牌
    nearby_landmarks: list = []     # GD 检测到的物体
    instance_suffix: str = ""       # 重名后缀 "_2", "_3"

    def display_cn(self) -> str:
        # "前台·DEEPROUTE.AI" (中点分隔, 不再粘连)
        if self.organization and self.category and self.organization != self.category:
            base = f"{self.category}·{self.organization}"
        else:
            base = self.category or self.organization
        return f"{base}{self.instance_suffix}"
```

**organization 选择**评分：
- brand-like (Latin 大写起头) +100
- camera_1 (前向) +30
- bbox 面积 +0..20
- 投票次数 +0..20

**merge_names(anchor, other)**: 两个共位节点合并时的命名策略:
- category 取语义等级更高的一方
- organization 优先 brand-like
- nearby_plates / nearby_landmarks 取并集

**全局唯一性**: `NameDeduplicator` 按 `(category, organization)` 元组分组，VPR 高相似 → 合并；否则追加 `_2, _3...` 后缀。

#### 4.4.5 同位置节点合并 (`semantics/colocation_merger.py`)

合并条件（OR）：
- VPR 相似度 >= 0.85
- 帧间隔 <= 8 AND 空间距离 <= 0.5m

合并策略：
- 按 `_CATEGORY_RANK` 选择 anchor（功能区/房间 > SHOP）
- 调用 `NodeName.merge_names(anchor, other)` 融合命名
- SHOP 不再抢占 anchor，仅作 organization 附加

### 4.5 输出与终结化

#### 主循环 (`OnlineMapperCore.run()`)

```
StreamLoader yields frame (4 cameras + timestamp + frame_idx)
  │
  ▼
┌──────────────────────────────────────────────────────┐
│ 每帧主循环                                             │
├──────────────────────────────────────────────────────┤
│ 1. depth.estimate(camera_1)                           │
│    → VGGT 滑窗推理 (window=4)                         │
│ 2. _vo_motion(camera_1, depth)                        │
│    → VGGT VO: (dtrans_m, dyaw_rad), 零额外推理        │
│ 3. 累积 robot pose (x, y, theta)                      │
│ 4. occ.integrate_pointcloud(points, pose)             │
│    → dense 点云直填占据栅格                             │
│ 5. vpr.extract_camera_features(4 cams)                │
│ 6. loop_closer.detect(feats, node_features)           │
│    → 全局 top-k + ORB 几何验证                         │
│ 7. _scan_door_plates(frame, fidx)                     │
│    → GD 检测 → STRICT prompt → Qwen 二次验证          │
│    → voter.add(NameVote) + door_tracker.add()         │
│ 8. keyframe_selector.should_trigger?                  │
│    是 → 创建关键帧:                                    │
│      8a. junction_detector.classify(4 cams)            │
│      8b. 选 confirmed plate (functional > brand)      │
│      8c. namer.describe_scene + verifier.verify_scene  │
│      8d. category_clf.classify(...)                    │
│      8e. ACCEPTED → 创建 TopoNode + NodeName           │
│          收集 GD landmarks, 添加 spatial/loop edges    │
└──────────────────────────────────────────────────────┘
  │
  ▼ (流结束)
┌──────────────────────────────────────────────────────┐
│ _finalize() — 终结化                                    │
├──────────────────────────────────────────────────────┤
│ 1. _create_door_plate_nodes() (两阶段)                 │
│    第一遍: functional plate (强电井, 关爱室) → 独立node │
│    第二遍: brand plate (DEEPROUTE.AI) → attach 到       │
│           帧距≤12 的 functional node 作 organization    │
│ 2. ColocationMerger.merge()                            │
│    → NodeName.merge_names 融合                         │
│ 3. _rebuild_topology_neighbors_spatial(k=1,1)          │
│ 4. _generate_names()                                   │
│    → 优先从 name_struct.display_cn() 渲染              │
│ 5. NameDeduplicator.resolve()                          │
│    → (category, organization) 元组去重 + 后缀           │
│ 6. writer.write_node() → merged_labeled_data/<id>/     │
│ 7. ConnectionBuilder.build_for_node(pose_graph=...)    │
│    → 几何先验 + Hungarian 匹配 → next_positions         │
│ 8. 写 scene_graph.json / pose_graph.json / metrics     │
└──────────────────────────────────────────────────────┘
```

#### 门牌两阶段归属

解决了 v2.2.0 的 "EUMANN关爱室" 串扰 bug：

```
第一遍: functional/landmark plate (强电井, 关爱室)
        → 创建独立 door-plate node, 跳过 brand-like SHOP

第二遍: brand-like plate (DEEPROUTE.AI, NEUMANN)
        → 找帧距≤12 且 category 非空的 functional/room node
        → 若找到:
            attach 为 organization
            旧 organization 进 nearby_plates
            重定位 timestamp + cameras (display 层)
            不动 frame_idx + pose (避免 coloc 误合并)
        → 若未找到:
            创建 standalone SHOP node
```

#### 输出格式

```
output_dir/
├── 1/                                 # node_id
│   ├── 1770097720_camera_1.jpg        # 4路相机原图
│   ├── 1770097720_camera_2.jpg
│   ├── 1770097720_camera_3.jpg
│   ├── 1770097720_camera_4.jpg
│   ├── crops/                         # 三级子图
│   │   ├── ...__big__...jpg
│   │   ├── ...__mid__...jpg
│   │   └── ...__small__...jpg
│   └── node_position_info.json        # 节点元数据 (含结构化命名字段)
├── 2/ ...
│
scene_graph.json                       # 层次场景图
pose_graph.json                        # 位姿图
online_mapping_log.jsonl               # 每帧决策日志
metrics.json                           # 统计指标
```

### 4.6 配置项

```python
@dataclass
class OnlineMapperConfig:
    # IO
    input_dir: str = "memory_test_data"
    output_dir: str = "online_mapper/output/merged_labeled_data"
    vpr_config_path: str = "deploy/vpr_config.yaml"

    # 关键帧
    vpr_dissim_threshold: float = 0.50
    accumulated_translation: float = 1.5     # 米
    accumulated_rotation: float = 0.6        # rad
    info_gain_threshold: float = 0.05
    min_keyframe_frame_interval: int = 3

    # 闭环
    loop_closure_min_gap: int = 8
    loop_closure_vpr_threshold: float = 0.78
    loop_closure_top_k: int = 5
    loop_closure_geom_verify: bool = True
    loop_closure_min_inliers: int = 15

    # 几何 (VGGT)
    depth_backend: str = "vggt"              # "da_v2" | "vggt"
    vggt_model_path: str = "pretrained/vggt-1b/model.pt"
    vggt_window_size: int = 4
    vggt_dtype: str = "bf16"
    vo_backend: str = "vggt"                 # "orb" | "vggt"
    occ_backend: str = "vggt"                # "depth_row" | "vggt"

    # 语义
    enable_grounding_dino: bool = True
    grounding_dino_model_id: str = "pretrained/grounding-dino-base"
    enable_qwen_naming: bool = True
    qwen_base_url: str = "http://localhost:8199/v1"
    qwen_gpu: str = "1"
    enable_door_plate_detection: bool = True
    door_plate_min_score: float = 0.30
    enable_real_connections: bool = True
    connection_sim_threshold: float = 0.40

    # 占据栅格
    grid_resolution: float = 0.2             # 米/格
    grid_size: int = 200                     # 200×200格

    start_id: int = 1
```

#### 启动方式

```bash
# 1. 准备 (一次性)
mkdir -p third_party && huggingface-cli download facebook/vggt --repo-type space --local-dir third_party/vggt_space
huggingface-cli download facebook/VGGT-1B --local-dir pretrained/vggt-1b
# 启 Qwen vLLM (GPU1)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B --port 8199 --max-model-len 4096 &

# 2. 端到端建图
conda activate internvla
CUDA_VISIBLE_DEVICES=0 python online_mapper/run_online_map.py \
  --input memory_test_data \
  --output online_mapper/output/merged_labeled_data

# 3. 切回旧后端 (回归测试)
# 修改 config: depth_backend="da_v2", vo_backend="orb", occ_backend="depth_row"
```

---

## 5. 离线建图

`offline_mapper/` 是最早的建图模块，现在主要作为 **只读依赖** 供 online_mapper import 复用：

| 模块 | 被复用于 |
|------|---------|
| `AutoSubImageExtractor` | `connection_builder.ThresholdedSubImageExtractor` (继承) |
| `AutoLandmarkNamer` | online_mapper 的场景描述和地标命名 |
| `NodeDistanceEstimator` | 节点间距离估计 |
| `validate_output.py` | 输出格式校验 |

独立运行：`python offline_mapper/run_auto_map.py`

---

## 6. 部署与运维

### 启动顺序

```bash
# 1. 启动 Qwen3.5 vLLM (GPU1, 后台)
./deploy/start_qwen_vllm.sh

# 2. 构建/更新记忆缓存 (首次或数据变更后)
./deploy/build_memory.sh

# 3. 启动导航服务 (GPU0)
./deploy/start_server.sh
# 或直接: python deploy/ws_proxy_with_memory.py
```

### 服务端口

| 端口 | 服务 | 说明 |
|------|------|------|
| 9528 | ws_proxy_with_memory.py | 导航 WebSocket 服务 |
| 8199 | Qwen3.5 vLLM | 打点/命名 HTTP API |

### 日志

- 服务日志: `deploy/logs/ws_proxy_with_memory.log` (10MB 轮转, 5 备份)
- 可视化图像: `deploy/logs/images/` (子图匹配框、Qwen3.5 打点、遮挡检测)
- 在线建图日志: `online_mapper/output/online_mapping_log.jsonl`

---

## 附录 A: 关键阈值汇总

| 常量 | 值 | 模块 | 说明 |
|------|-----|------|------|
| SUB_MATCH_CONFIDENCE_THRESHOLD | 0.60 | ws_proxy | 子图匹配成功阈值 |
| FRAME_SIMILARITY_THRESHOLD | 0.70 | ws_proxy | 帧间缓存复用 DINOv2 相似度阈值 |
| VPR_ARRIVE_THRESHOLD | 0.70 | ws_proxy | VPR 步骤切换阈值 |
| MAX_MISSES | 8 | ws_proxy | 连续 VPR 丢失上限 |
| similarity_threshold (selavpr) | 0.60 | vpr_config | SelaVPR++ 匹配阈值 |
| similarity_threshold (effovpr) | 0.80 | vpr_config | EffoVPR 匹配阈值 |
| area_threshold | 0.25 | occlusion | 遮挡面积比阈值 (25%) |
| confidence_threshold | 0.40 | occlusion | YOLOv8n 检测置信度 |
| early_stop confidence | 0.50 | qwen35 | 子进程模式提前退出 |
| vpr_dissim_threshold | 0.50 | keyframe | 关键帧 VPR 差异阈值 |
| accumulated_translation | 1.5m | keyframe | 关键帧位移阈值 |
| accumulated_rotation | 0.6 rad | keyframe | 关键帧旋转阈值 |
| loop_closure_vpr_threshold | 0.78 | loop | 闭环 VPR 初始阈值 |
| loop_closure_min_inliers | 15 | loop | ORB 几何验证最小内点 |
| connection_sim_threshold | 0.40 | connection | 邻接视觉相似度阈值 |
| colocation VPR threshold | 0.85 | colocation | 同位置合并 VPR 阈值 |
| colocation frame_gap | 8 | colocation | 同位置合并帧间隔 |
| colocation spatial_dist | 0.5m | colocation | 同位置合并空间距离 |
| ATTACH_GAP | 12 帧 | door_plate | brand plate 归属搜索范围 |

---

## 附录 B: WebSocket 协议详情

### 请求格式

```json
{
  "id": "robot_001",
  "pts": 1770097767,
  "task": "前往C8前台",
  "images": {
    "front_1": "<base64 RGB>",
    "camera_1": "<base64 鱼眼>",
    "camera_2": "<base64 鱼眼>",
    "camera_3": "<base64 鱼眼>",
    "camera_4": "<base64 鱼眼>"
  }
}
```

特殊 task 值: `"STOP"` / `"stop"`, `"turn left"`, `"turn right"`, `"go straight"`

命令模式: `{"command": "reset"}` / `{"command": "memory_status"}` 等

### 响应格式

```json
{
  "status": "success",
  "id": "robot_001",
  "pts": 1770097767,
  "task_status": "executing | end",
  "action": [[x_forward, y_lateral, yaw]],
  "pixel_target": [x_norm, y_norm],
  "camera_name": "camera_2",
  "landmark_name": "电梯门",
  "landmark_name_eng": "elevator door",
  "position_name_eng": "C8 front desk",
  "crop_image_paths": {"big": "...", "mid": "...", "small": "..."},
  "crop_image_path": "path/to/big.jpg",
  "sub_image_match": {
    "camera_name": "camera_2",
    "landmark_name": "电梯门",
    "match": {
      "found": true,
      "confidence": 0.78,
      "top_left_pct": {"x": 0.35, "y": 0.20},
      "bottom_right_pct": {"x": 0.65, "y": 0.80},
      "center_pct": {"x": 0.50, "y": 0.50},
      "bbox_pixel": {"x_min": 224, "y_min": 128, "x_max": 416, "y_max": 512},
      "elapsed_ms": 45.2,
      "method": "dinov3"
    },
    "matched_scale": "mid",
    "memory_camera": "camera_2"
  },
  "fallback_instruction": "通道正中间位置+景深",
  "memory_active": true,
  "memory_info": {
    "frame_similarity": 0.85,
    "cache_action": "accepted",
    "plan_path": ["12", "8", "4"],
    "current_step": 0,
    "total_steps": 2,
    "from_node": "C8走廊",
    "from_node_eng": "C8 corridor",
    "to_node": "C8前台区",
    "to_node_eng": "C8 front desk",
    "from_node_id": "12",
    "to_node_id": "8",
    "heading_offset": 0.0,
    "vpr_confidence": 0.82,
    "vpr_similarity": 0.82,
    "vpr_matched_node": "12",
    "phase": "verifying",
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
  "message": "记忆导航: C8走廊 → C8前台区 (步骤1/2)"
}
```

### action 含义

| 字段 | 含义 |
|------|------|
| `action[0][0]` | x_forward: 前进距离 (米, 正=前进) |
| `action[0][1]` | y_lateral: 侧移距离 (米, 正=左移) |
| `action[0][2]` | yaw: 偏航角 (弧度, 正=逆时针) |

特殊情况:
- 原地等待 (遮挡): `[[0, 0, 0]]`
- 纯旋转 (侧面相机匹配): `[[0, 0, yaw_rad]]`
- 导航完成: `task_status="end"`
