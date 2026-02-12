# MemoryNav 记忆导航方案技术详解

> 版本：v2.1 | 文档日期：2026-02-11
> 本文档完整描述 MemoryNav 系统的所有技术细节，包括架构设计、核心算法、数据流、部署方案和 API 接口。

---

## 目录

1. [系统总览](#一系统总览)
2. [目录结构](#二目录结构)
3. [核心数据模型](#三核心数据模型)
4. [LongCLIP 特征提取](#四longclip-特征提取)
5. [多视角循环移位 VPR 匹配算法](#五多视角循环移位-vpr-匹配算法)
6. [拓扑记忆图](#六拓扑记忆图)
7. [记忆构建流程](#七记忆构建流程)
8. [导航器状态机](#八导航器状态机)
9. [WebSocket 部署服务](#九websocket-部署服务)
10. [三层导航策略](#十三层导航策略)
11. [相似度趋势检测机制](#十一相似度趋势检测机制)
12. [视觉记忆系统（扩展版）](#十二视觉记忆系统扩展版)
13. [配置参数速查](#十三配置参数速查)
14. [API 接口规范](#十四api-接口规范)

---

## 一、系统总览

MemoryNav 是一个面向移动机器人的**纯视觉记忆导航系统**，核心思路是：

1. **离线阶段**：人工采集环境中关键位置的 4 相机环视图，标注位置名称和连通关系，使用 LongCLIP 提取特征并构建 FAISS 索引 + 拓扑图。
2. **在线阶段**：机器人在导航过程中，实时采集 4 相机环视图，通过 VPR（视觉位置识别）在记忆库中定位当前位置，在拓扑图上规划最短路径，逐步引导机器人到达目标。

**端到端数据流**：

```
机器人端                       服务器端 (ws_proxy_with_memory.py, port 9528)
  │                                │
  │  WebSocket JSON               │
  │  {task, images{front_1,       │
  │   camera_1~4}, depth, pose}   │
  │ ──────────────────────────────→│
  │                                │
  │                         ┌──────┴───────┐
  │                         │ 1. 解码图像    │
  │                         │ 2. VPR 定位    │──→ LongCLIP 特征提取 (×4相机)
  │                         │              │──→ FAISS 循环移位匹配
  │                         │ 3. 路径规划    │──→ Dijkstra 最短路径
  │                         │ 4. 导航决策    │──→ 三层策略选择
  │                         │ 5. (兜底)     │──→ InternVLA 模型推理
  │                         └──────┬───────┘
  │                                │
  │  {action, pixel_target,        │
  │   angle, memory_info}          │
  │ ←──────────────────────────────│
  │                                │
  ▼ 执行动作
```

---

## 二、目录结构

```
MemoryNav/
├── deploy/                          # 部署模块（核心）
│   ├── memory_nav/                  # 记忆导航核心包
│   │   ├── __init__.py
│   │   ├── memory_models.py         # 数据模型定义
│   │   ├── memory_vpr.py            # VPR 匹配引擎 (v2.1)
│   │   ├── memory_graph.py          # 拓扑图管理
│   │   ├── memory_builder.py        # 记忆构建器 + LongCLIP 提取器
│   │   └── memory_navigator.py      # 导航器主接口
│   ├── visual_memory_system.py      # 视觉记忆系统（扩展版，含 Qwen2.5-VL）
│   ├── ws_proxy_with_memory.py      # WebSocket 部署服务（1659行）
│   ├── ws_proxy.py                  # 原始 WebSocket 代理（无记忆）
│   ├── ws_client.py                 # 测试客户端
│   ├── start_server.sh              # 启动脚本
│   └── logs/                        # 运行日志和调试图像
├── internnav/                       # InternNav 导航框架
│   ├── agent/                       # 导航智能体
│   │   ├── internvla_n1_agent_realworld.py  # InternVLA-N1 真实世界 Agent
│   │   ├── cma_agent.py             # CMA Agent
│   │   ├── seq2seq_agent.py         # Seq2Seq Agent
│   │   └── ...
│   ├── model/
│   │   ├── basemodel/
│   │   │   └── LongCLIP/            # LongCLIP 模型代码
│   │   └── encoder/
│   └── ...
├── merged_labeled_data/             # 标注数据（离线采集）
│   ├── 1/                           # 节点 1
│   │   ├── node_position_info.json  # 节点位置信息
│   │   ├── <ts>_camera_1.jpg        # 环视相机图 1
│   │   ├── <ts>_camera_2.jpg        # 环视相机图 2
│   │   ├── <ts>_camera_3.jpg        # 环视相机图 3
│   │   ├── <ts>_camera_4.jpg        # 环视相机图 4
│   │   └── <ts>_stitch.jpg          # 前视拼接图
│   ├── 2/
│   └── ...
├── checkpoints/                     # 模型权重
│   ├── longclip-B.pt                # LongCLIP ViT-B/16 权重
│   └── InternRobotics/InternVLA-N1-DualVLN/  # InternVLA 权重
├── scripts/                         # 工具和评估脚本
└── src/diffusion-policy/            # Diffusion Policy（参考实现）
```

---

## 三、核心数据模型

所有数据模型定义在 `deploy/memory_nav/memory_models.py` 中。

### 3.1 MemoryNode（记忆节点）

每个节点代表环境中一个**关键位置**（如"C8前台区"、"走廊拐角"等），包含该位置的完整视觉信息。

```python
@dataclass
class MemoryNode:
    node_id: str                  # 位置ID，对应 node_position_info.json 中的 position_id
    node_name: str                # 中文位置名称，如 "C8前台区"
    node_name_eng: str            # 英文位置名称，如 "C8 Front Desk"
    camera_images: Dict[str, str] # 4个相机图路径 {'camera_1': 'xxx_camera_1.jpg', ...}
    camera_features: Dict[str, np.ndarray]  # 4个相机的 LongCLIP 特征向量 (各768维)
    fused_feature: Optional[np.ndarray]     # 融合特征 (4个相机特征取平均，768维)
    edges: List[MemoryEdge]       # 出边列表（通往相邻节点）
    base_path: str                # 节点数据目录路径
    timestamp: str                # 采集时间戳
```

### 3.2 MemoryEdge（记忆边）

每条边代表从一个节点到相邻节点的**导航信息**。

```python
@dataclass
class MemoryEdge:
    target_node_id: str           # 目标节点ID
    target_node_name: str         # 目标节点中文名称
    target_node_name_eng: str     # 目标节点英文名称
    angle: float                  # 从当前节点前往目标节点需要的绝对地理角度（度）
    pixel_position: Tuple[float, float]  # 在前视拼接图(stitch image)上目标方向的归一化像素坐标 (x, y)∈[0,1]
    stitch_image_path: str        # 当前节点的前视拼接图路径
```

**关键说明**：

- `angle` 是**绝对地理角度**（0° 为正北，顺时针递增），不是相对于机器人当前朝向的转向角
- `pixel_position` 是在全景拼接图上标注的目标方向像素坐标，已归一化到 [0, 1] 范围

### 3.3 VPRResult（VPR匹配结果）

```python
@dataclass
class VPRResult:
    matched_node_id: str          # 匹配到的节点ID
    matched_node_name: str        # 节点中文名称
    matched_node_name_eng: str    # 节点英文名称
    similarity: float             # 4相机平均余弦相似度
    confidence: float             # 置信度（当前等于 similarity）
    camera_scores: Dict[str, float]  # 各相机的余弦相似度得分
    heading_offset: float         # 机器人朝向与记忆库朝向的偏移角度（度）
    best_shift: int               # 最佳循环移位 (0-3)
```

### 3.4 NavigationPlan（导航计划）

```python
@dataclass
class NavigationPlan:
    start_node_id: str            # 起点ID
    start_node_name: str          # 起点名称
    goal_node_id: str             # 终点ID
    goal_node_name: str           # 终点名称
    path: List[str]               # 路径节点ID序列 ['1', '3', '8', '12']
    steps: List[NavigationStep]   # 逐步导航指令列表
    total_steps: int              # 总步数（= len(steps)）
    success: bool                 # 规划是否成功
    message: str                  # 状态消息
```

### 3.5 NavigationStep（导航步骤）

路径中相邻两个节点之间的一步导航信息。

```python
@dataclass
class NavigationStep:
    from_node_id: str             # 起始节点ID
    from_node_name: str           # 起始节点名称
    to_node_id: str               # 目标节点ID
    to_node_name: str             # 目标节点名称
    angle: float                  # 绝对地理角度
    pixel_position: Tuple[float, float]  # 归一化像素目标
    stitch_image_path: str        # 前视拼接图路径
    step_index: int               # 步骤序号
```

---

## 四、LongCLIP 特征提取

### 4.1 模型架构

系统使用 **LongCLIP-B** 模型（基于 ViT-B/16），输出 768 维特征向量（部署代码中有时使用 512 维，取决于具体的 LongCLIP 变体）。

提取器定义在 `deploy/memory_nav/memory_builder.py` 中的 `LongCLIPExtractor` 类：

```python
class LongCLIPExtractor(FeatureExtractor):
    def __init__(self, model_path=None, feature_dim=768, device="cuda:0"):
        # 默认模型路径: checkpoints/longclip-B.pt
        # 加载 LongCLIP 模型的视觉编码器部分

    def extract(self, image: np.ndarray) -> np.ndarray:
        # 输入: BGR 图像 (np.ndarray)
        # 流程: BGR→RGB → PIL Image → preprocess → model.encode_image → L2 归一化
        # 输出: 768维 float32 向量，L2 范数 = 1
```

### 4.2 特征提取流程

```
BGR 图像 (H, W, 3)
    │
    ▼ cv2.cvtColor(BGR→RGB)
PIL Image (RGB)
    │
    ▼ preprocess (resize + center crop + normalize)
Tensor (1, 3, 224, 224)
    │
    ▼ model.encode_image (ViT-B/16 视觉编码器)
Tensor (1, 768)
    │
    ▼ L2 normalize: feat / ||feat||₂
np.ndarray (768,), float32, ||·||₂ = 1
```

### 4.3 备用方案

- 如果 LongCLIP 加载失败，系统会尝试 **OpenCLIP (ViT-B/32)**（512 维）
- 如果 OpenCLIP 也不可用，回退到**颜色直方图 + HOG 特征**（简化版，仅用于测试）

---

## 五、多视角循环移位 VPR 匹配算法

这是 MemoryNav 最核心的算法，定义在 `deploy/memory_nav/memory_vpr.py` 中。

### 5.1 相机布局

机器人配备 4 个环视鱼眼相机（HFOV=190°），布局如下：

```
          正前方 (0°)
             │
     cam_2   │   cam_1
    (-37.5°) │  (+37.5°)
         ╲   │   ╱
          ╲  │  ╱
           ╲ │ ╱
            [机器人]
           ╱ │ ╲
          ╱  │  ╲
         ╱   │   ╲
    cam_3    │   cam_4
   (-142.5°) │  (+142.5°)
             │
          正后方 (180°)
```

**相机中心方位角**：
| 相机 | 角度 | 方向 |
|------|------|------|
| camera_1 | +37.5° | 前右 |
| camera_2 | -37.5° | 前左 |
| camera_3 | -142.5° (217.5°) | 后左 |
| camera_4 | +142.5° | 后右 |

### 5.2 循环移位匹配原理

**核心问题**：机器人在同一位置但**朝向不同**时，4 个相机看到的画面会发生循环移位。例如：
- 原始朝向：cam_1 看到前右景象
- 顺时针旋转约 75°：cam_2 现在看到原来 cam_1 的视角

**算法步骤**：

1. **遍历 4 种循环移位** (shift = 0, 1, 2, 3)

   每种 shift 定义一种 query 相机到 memory 相机的映射：
   ```
   shift=0: query_cam[i] ↔ memory_cam[i]     (同向)
   shift=1: query_cam[(i+1)%4] ↔ memory_cam[i]  (顺时针75°)
   shift=2: query_cam[(i+2)%4] ↔ memory_cam[i]  (掉头180°)
   shift=3: query_cam[(i+3)%4] ↔ memory_cam[i]  (逆时针105°)
   ```

2. **对每种 shift，计算每个记忆节点的 4 相机平均相似度**

   ```
   对于 shift=k, 节点 N:
     sim(k, N) = (1/4) × Σᵢ cosine(query[CAM[(i+k)%4]], memory_N[CAM[i]])
   ```

3. **选择全局最优的 (节点, shift) 组合**

   ```
   (best_node, best_shift) = argmax_{N, k} sim(k, N)
   ```

4. **计算朝向偏移**

   ```python
   SHIFT_HEADING_OFFSETS = [0.0, 75.0, 180.0, -105.0]  # 度
   heading_offset = SHIFT_HEADING_OFFSETS[best_shift]
   ```

5. **阈值判断**

   - 如果 `best_avg_sim >= similarity_threshold (0.97)` → 匹配成功
   - 否则 → 匹配失败

### 5.3 FAISS 索引结构

- 每个相机维护一个独立的 `faiss.IndexFlatIP` 索引（内积，等效于 L2 归一化后的余弦相似度）
- 每个索引存储该相机方向上所有节点的 768 维特征向量
- 共 4 个独立索引 + 1 个融合特征索引
- 查询时对 4 个索引分别做 Top-K 搜索，然后在每种 shift 下聚合分数

### 5.4 不足 4 相机时的简化匹配

当查询图不满 4 个相机时，退回简单匹配模式：
- 对每个可用相机独立搜索 Top-K
- 按 (匹配数, 总分) 降序排列
- 至少 2 个相机匹配且平均相似度 >= 阈值才算成功

---

## 六、拓扑记忆图

定义在 `deploy/memory_nav/memory_graph.py` 中。

### 6.1 图结构

- 使用 **NetworkX 无向图** 表示环境拓扑
- 节点 = `MemoryNode`，存储位置的视觉特征和元数据
- 边 = `MemoryEdge`，存储节点间的导航角度和像素目标
- 边权重默认为 1.0（等权重，Dijkstra 退化为 BFS）

### 6.2 路径规划

```python
def find_shortest_path(start_id, goal_id) -> Optional[List[str]]:
    # 1. 优先使用 nx.shortest_path (Dijkstra)
    # 2. NetworkX 不可用时回退到 BFS
    # 返回: 节点 ID 序列 ['1', '3', '8', '12']
```

### 6.3 语义查询

支持按名称搜索节点（中英文双语 + 模糊匹配）：

```python
def search_nodes_by_name(query: str, limit=5) -> List[MemoryNode]:
    # 1. 完全匹配优先（中文或英文）
    # 2. 包含匹配（子串搜索）
```

### 6.4 序列化

- 使用 **pickle** 格式保存/加载图（`.pkl` 文件）
- 保存内容：节点信息、特征向量（转 list）、边信息、名称映射
- 加载时重建 NetworkX 图和 FAISS 索引

---

## 七、记忆构建流程

定义在 `deploy/memory_nav/memory_builder.py` 中。

### 7.1 数据目录结构

每个节点目录包含一个 `node_position_info.json` 文件：

```json
{
  "self_position": {
    "position_id": 8,
    "position_name": "C8前台区",
    "position_name_eng": "C8 Front Desk",
    "camera_1": "1735488000_camera_1.jpg",
    "camera_2": "1735488000_camera_2.jpg",
    "camera_3": "1735488000_camera_3.jpg",
    "camera_4": "1735488000_camera_4.jpg"
  },
  "next_positions": [
    {
      "position_id": 7,
      "position_name": "C7走廊",
      "position_name_eng": "C7 Corridor",
      "angle": 270.5,
      "pixel_position": "0.35,0.52",
      "stitch_image": "1735488000_stitch.jpg"
    },
    {
      "position_id": 9,
      "position_name": "C9会议室",
      "angle": 90.2,
      "pixel_position": "0.68,0.48",
      "stitch_image": "1735488000_stitch.jpg"
    }
  ]
}
```

### 7.2 构建步骤

```
merged_labeled_data/
    │
    ▼ MemoryBuilder.build_from_directory()
    │
    ├─ 遍历所有节点目录 (按数字排序)
    │   ├─ 读取 node_position_info.json
    │   ├─ 解析 self_position → node_id, node_name, camera_images
    │   ├─ 解析 next_positions → edges (angle, pixel_position, stitch_image)
    │   ├─ 对 4 张相机图调用 LongCLIPExtractor.extract()
    │   │   → camera_features: {cam_1: [768], cam_2: [768], cam_3: [768], cam_4: [768]}
    │   ├─ 计算融合特征 = mean(cam_1..4) → L2 normalize → fused_feature: [768]
    │   └─ 创建 MemoryNode → 添加到 MemoryGraph + MemoryVPR
    │
    ├─ MemoryGraph.add_node(node)
    │   └─ 更新 NetworkX 图、名称映射
    │
    └─ MemoryVPR.add_node_features(node_id, camera_features, fused_feature)
        └─ 更新 4 个相机的 FAISS 索引 + 融合特征索引
```

### 7.3 缓存机制

构建完成后保存到 `deploy/memory_nav/memory_cache_graph.pkl`，后续启动时直接加载缓存：

```python
navigator.load_memory(path="deploy/memory_nav/memory_cache", data_dir="merged_labeled_data")
# 1. 如果 memory_cache_graph.pkl 存在 → 直接加载
# 2. 否则从 merged_labeled_data/ 构建 → 保存缓存
```

---

## 八、导航器状态机

定义在 `deploy/memory_nav/memory_navigator.py` 中。

### 8.1 导航流程

```
  ┌──────────────┐
  │  接收目的地   │ (自然语言: "前往C8前台")
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ find_destination │ 语义匹配目的地节点
  │  支持: 直接匹配、│ 去除前缀后匹配、
  │  模糊搜索      │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ locate_by_images │ VPR 定位当前位置
  │  4相机环视图 →  │ LongCLIP → FAISS →
  │  循环移位匹配   │ → current_node_id
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ plan_navigation │ Dijkstra 规划最短路径
  │  start → goal  │ → path = [n1, n3, n8, n12]
  │  → steps       │ → 每步的 angle + pixel_position
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 逐步执行导航   │ 每步返回 angle + pixel_target
  │  VPR 持续验证  │ 到达目标节点 → advance
  │  趋势检测      │ VPR 丢失 → 模型兜底
  └──────────────┘
```

### 8.2 目的地语义匹配

`find_destination(query)` 支持多种查询格式：

1. **直接节点ID匹配**：`"8"` → 节点 8
2. **完整名称匹配**：`"C8前台区"` → 直接命中
3. **自然语言任务提取**：`"前往C8前台"` → 去除 "前往" 前缀 → 搜索 "C8前台"
4. **模糊子串搜索**：`"前台"` → 搜索包含 "前台" 的所有节点

支持的前缀关键词：`前往, 去, 到, 走到, 导航到, 带我去, go to, navigate to, take me to`

---

## 九、WebSocket 部署服务

定义在 `deploy/ws_proxy_with_memory.py`（1659 行），是完整的生产部署入口。

### 9.1 服务架构

```
                        WebSocket Server (port 9528)
                              asyncio
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
    handle_client()       handle_client()       handle_client()
    (客户端 1)             (客户端 2)             (客户端 N)
          │                     │                     │
    ┌─────┴─────┐         ┌────┴────┐
    │ session_state │     │ session_state │    (每个客户端独立)
    │ nav_state     │     │ nav_state     │
    └─────┬─────┘         └────┬────┘
          │                     │
          └──────────┬──────────┘
                     │
              全局共享实例:
              - global_agent (InternVLA-N1)
              - memory_navigator (MemoryNavigator)
              - agent_lock (asyncio.Lock)
```

### 9.2 模型初始化

服务启动时依次加载：

1. **MemoryNavigator** (LongCLIP + FAISS + 拓扑图) → `init_memory_navigator()`
2. **InternVLA-N1** (DualVLN 模型) → `init_agent()`
   - 模型路径: `checkpoints/InternRobotics/InternVLA-N1-DualVLN`
   - 输入分辨率: 384×384
   - 历史帧数: 8
   - 模型预热: 一次 dummy 推理

### 9.3 每个客户端的独立状态

```python
@dataclass
class MemoryNavState:
    plan: Optional[NavigationPlan]     # 当前活跃的导航计划
    current_step_idx: int              # 当前执行到第几步
    phase: str                         # 状态: 'idle', 'step_init', 'verifying', 'fallback', 'completed'
    consecutive_misses: int            # 连续 VPR 丢失次数
    MAX_MISSES: int = 8               # 最大连续丢失容忍次数
    source_sim_history: List[float]    # 源节点相似度历史
    target_sim_history: List[float]    # 目标节点相似度历史
    deviation_count: int               # 连续偏离次数
    TREND_WINDOW: int = 2              # 趋势检测滑动窗口
    MAX_DEVIATIONS: int = 5            # 最大偏离次数
    last_query_features: Dict          # 最近一次提取的查询特征
```

### 9.4 WebSocket 命令

| 命令 | 说明 |
|------|------|
| `reset` | 重置 InternVLA Agent + 清空记忆导航状态 |
| `session_status` | 查看会话状态 |
| `toggle_memory` | 切换记忆导航开关 |
| `memory_status` | 查看记忆导航详情 + 可用目的地列表 |
| `reset_memory` | 仅重置记忆状态（保留 Agent 历史） |

---

## 十、三层导航策略

这是 `process_inference_with_memory()` 函数（约 500 行）的核心逻辑。

### 第一层：记忆引导

当有活跃导航计划且 VPR 定位成功时，返回记忆中预存的导航信息：

```json
{
  "memory_active": true,
  "pixel_target": [0.35, 0.52],    // 归一化像素目标
  "angle": 270.5,                   // 绝对地理角度
  "action": [[0.0, 0.0, 0.0]],     // 记忆模式下不使用 action
  "memory_info": {
    "plan_path": ["1", "3", "8", "12"],
    "current_step": 1,
    "total_steps": 3,
    "from_node": "C3走廊",
    "to_node": "C8前台区",
    "phase": "step_init"
  }
}
```

### 第二层：VPR 持续验证

每次请求都用 4 相机图做 VPR 定位，判断机器人是否已到达目标节点：

| VPR 结果 | 处理策略 |
|----------|---------|
| **Case A**: 匹配到目标节点 | `advance()` → 前进到下一步 |
| **Case B**: 匹配到路径中更后面的节点 | 跳步到对应步骤 |
| **Case C**: 匹配到路径外节点（高置信度 ≥ 0.8） | 从新位置重新规划路径 |
| **Case C**: 匹配到路径外节点（低置信度） | 继续当前步骤引导 |
| **Case D**: VPR 丢失 | 进入相似度趋势检测 |

### 第三层：InternVLA 模型兜底

当记忆导航无法工作时（VPR 丢失且趋势样本不足），调用 InternVLA-N1 模型直接推理：

```python
dual_sys_output = agent.step(rgb, depth, pose, instruction, intrinsic, look_down)
# 输出:
# - output_action: 离散动作序列 [1, 1, 3, 1, 0]  (前进/左转/右转/停止)
# - output_trajectory: 33个轨迹点 [[dx, dy], ...]
# - output_pixel: 像素目标 [y, x]
```

**动作映射**：

| 动作码 | 含义 | 机器人指令 |
|--------|------|-----------|
| 0 | STOP | task_status = "end" |
| 1 | 前进 | x += 0.25m |
| 2 | 左转 | yaw += π/24 |
| 3 | 右转 | yaw -= π/24 |
| 5 | 向下看 | 触发 look_down=True 重推理 |

---

## 十一、相似度趋势检测机制

当 VPR 无法直接匹配到节点时（similarity < threshold），系统利用**相似度变化趋势**判断机器人移动方向。

### 11.1 数据采集

每次请求，无论 VPR 是否成功，都计算当前查询特征与**源节点**和**目标节点**的相似度：

```python
source_sim = vpr.get_node_similarity(query_features, source_node_id)  # 与出发节点的相似度
target_sim = vpr.get_node_similarity(query_features, target_node_id)  # 与目标节点的相似度
nav_state.source_sim_history.append(source_sim)
nav_state.target_sim_history.append(target_sim)
```

### 11.2 趋势判断

当历史长度 >= `TREND_WINDOW` (默认 2) 时，计算趋势：

```python
src_trend = source_sim_history[-1] - source_sim_history[-TREND_WINDOW]  # 源节点趋势
tgt_trend = target_sim_history[-1] - target_sim_history[-TREND_WINDOW]  # 目标节点趋势
```

### 11.3 决策逻辑

| 趋势条件 | 含义 | 动作 |
|----------|------|------|
| source↓ + target↑ | 远离源节点、接近目标 → 方向正确 | 发送 `go straight` (前进1米) |
| source↓ + target↓ | 远离两个节点 → 偏离路线 | 重发记忆引导纠偏；超限(5次)则强制 advance |
| 其他情况 | 趋势不明确 | 重发记忆引导 |
| 样本不足 | 历史长度 < TREND_WINDOW | 走 InternVLA 兜底推理 |

---

## 十二、视觉记忆系统（扩展版）

`deploy/visual_memory_system.py` 是一个更完整的视觉记忆系统实现，包含额外的组件。

### 12.1 组件架构

```
VisualMemorySystem
├── LongCLIPFeatureExtractor     # 视觉特征提取（同上）
├── SceneDescriptionGenerator    # Qwen2.5-VL 场景描述生成
├── MemoryNodeManager            # FAISS 记忆节点管理（含 SQLite 持久化）
├── GraphEdgeManager             # NetworkX 有向图边管理
├── VisualPlaceRecognizer        # VPR 位置识别
├── PathPlanner                  # 路径规划（含返回起点、语义导航）
└── SurroundCameraFusion         # 环视相机特征融合
```

### 12.2 Qwen2.5-VL 场景描述

使用 **Qwen2.5-VL-7B-Instruct** 模型为每个位置生成自然语言场景描述和语义标签：

```python
class SceneDescriptionGenerator:
    # 模型: Qwen/Qwen2.5-VL-7B-Instruct
    # 精度: bfloat16
    # 设备: cuda:0 (device_map="auto")

    def generate_description(images) -> str:
        # 输入: camera_1/camera_2 的图像
        # Prompt: "基于提供的相机视图，用2-3句话描述当前位置..."
        # 输出: "这是一个宽敞的办公区前台，左侧有接待柜台..."

    def extract_semantic_labels(images) -> List[str]:
        # Prompt: "列出这些图像中可见的关键物体和场景元素..."
        # 输出: ["走廊", "门", "标识牌", "电梯"]
```

### 12.3 SQLite 持久化

`MemoryNodeManager` 使用 SQLite 数据库持久化节点数据：

```sql
-- nodes 表
CREATE TABLE nodes (
    node_id TEXT PRIMARY KEY,
    timestamp REAL,
    scene_description TEXT,
    semantic_labels TEXT,        -- JSON 数组
    task_instruction TEXT,
    pixel_target TEXT,           -- JSON
    position_estimate TEXT,      -- JSON [x, y, yaw]
    visit_count INTEGER,
    is_landmark INTEGER
);

-- features 表
CREATE TABLE features (
    node_id TEXT PRIMARY KEY,
    global_descriptor TEXT,      -- JSON 数组 (768维)
    visual_features TEXT         -- JSON 字典 {cam_id: [768]}
);
```

### 12.4 FAISS 索引类型

支持三种索引类型，可在配置中切换：

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| `Flat` | 暴力搜索，精确但慢 | 节点数 < 1000（默认） |
| `IVF` | 倒排索引，快速近似 | 节点数 1000-100000 |
| `IVFPQ` | 倒排+乘积量化，最快 | 节点数 > 100000 |

### 12.5 语义路径搜索

`GraphEdgeManager.semantic_search_path()` 支持通过自然语言描述搜索目标位置：

```python
def semantic_search_path(start_node_id, target_description, node_manager):
    # 1. 遍历所有节点的 semantic_labels 和 scene_description
    # 2. 关键词匹配找到候选目标节点
    # 3. 对每个候选目标计算最短路径
    # 4. 返回最短的路径
```

---

## 十三、配置参数速查

### 13.1 VPR 参数 (MemoryVPR)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `feature_dim` | 768 (或 512) | LongCLIP 特征维度 |
| `similarity_threshold` | 0.97 | 循环移位匹配阈值 |

### 13.2 导航状态参数 (MemoryNavState)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_MISSES` | 8 | 连续 VPR 丢失最大次数 |
| `TREND_WINDOW` | 2 | 趋势检测滑动窗口大小 |
| `MAX_DEVIATIONS` | 5 | 最大偏离次数（超限则强制 advance） |

### 13.3 InternVLA Agent 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `device` | `cuda:0` | GPU 设备 |
| `model_path` | `checkpoints/InternRobotics/InternVLA-N1-DualVLN` | 模型路径 |
| `resize_w/h` | 384 | 输入图像分辨率 |
| `num_history` | 8 | 历史帧数量 |
| `plan_step_gap` | 8 | 规划步长间隔 |
| `STEP_SIZE` | 0.25m | 前进步长 |
| `TURN_ANGLE` | π/24 (7.5°) | 转向角度 |

### 13.4 视觉记忆系统配置 (VisualMemoryConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `longclip_model_path` | `checkpoints/longclip-B.pt` | LongCLIP 模型路径 |
| `feature_dim` | 768 | 特征维度 |
| `vlm_enabled` | True | 是否启用 Qwen2.5-VL |
| `vlm_model_path` | `Qwen/Qwen2.5-VL-7B-Instruct` | VLM 模型路径 |
| `similarity_threshold` | 0.85 | VPR 匹配阈值 |
| `recognition_top_k` | 5 | 检索返回数量 |
| `min_time_gap` | 30.0s | 回环检测最小时间间隔 |
| `max_nodes` | 1000 | 最大节点数 |
| `node_merge_threshold` | 0.90 | 节点合并阈值 |
| `faiss_index_type` | `Flat` | FAISS 索引类型 |
| `faiss_use_gpu` | True | FAISS GPU 加速 |
| `auto_save_interval` | 300s | 自动保存间隔 |

### 13.5 WebSocket 服务参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 端口 | 9528 | WebSocket 监听端口 |
| `ping_interval` | 30s | WebSocket 心跳间隔 |
| `ping_timeout` | 10s | 心跳超时 |
| `max_size` | 50MB | 最大消息大小 |
| 图像目标尺寸 | 640×480 | front_1 图像调整尺寸 |

---

## 十四、API 接口规范

### 14.1 推理请求

```json
{
  "id": "robot_001",
  "pts": 1770097720000,
  "task": "前往C8前台",
  "images": {
    "front_1": "<base64 encoded JPEG>",
    "camera_1": "<base64 encoded JPEG>",
    "camera_2": "<base64 encoded JPEG>",
    "camera_3": "<base64 encoded JPEG>",
    "camera_4": "<base64 encoded JPEG>"
  },
  "depth": "<base64 encoded depth image>",
  "pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "intrinsic": [[386.5,0,328.9,0],[0,386.5,244,0],[0,0,1,0],[0,0,0,1]],
  "look_down": false
}
```

### 14.2 记忆导航响应

```json
{
  "status": "success",
  "id": "robot_001",
  "pts": 1770097720000,
  "task_status": "executing",
  "action": [[0.0, 0.0, 0.0]],
  "pixel_target": [0.35, 0.52],
  "angle": 270.5,
  "memory_active": true,
  "memory_info": {
    "plan_path": ["1", "3", "8", "12"],
    "current_step": 1,
    "total_steps": 3,
    "from_node": "C3走廊",
    "from_node_eng": "C3 Corridor",
    "to_node": "C8前台区",
    "to_node_eng": "C8 Front Desk",
    "from_node_id": "3",
    "to_node_id": "8",
    "heading_offset": 75.0,
    "vpr_confidence": 0.982,
    "vpr_similarity": 0.982,
    "vpr_matched_node": "3",
    "phase": "step_init",
    "consecutive_misses": 0
  },
  "message": "记忆导航: C3走廊 → C8前台区 (步骤2/3)"
}
```

### 14.3 InternVLA 兜底响应

```json
{
  "status": "success",
  "id": "robot_001",
  "pts": 1770097720000,
  "task_status": "executing",
  "action": [[0.75, 0.0, -0.1309]],
  "pixel_target": [0.482, 0.315],
  "memory_active": false,
  "message": ""
}
```

### 14.4 特殊指令

| 指令 (task) | 行为 |
|-------------|------|
| `"STOP"` / `"stop"` | 立即停止，task_status = "end" |
| `"turn left"` | 左转 π/12 |
| `"turn right"` | 右转 π/12 |
| `"go straight"` | 前进 1m |
| `None` / `"None"` | 延用上次 task |

---


