# online_mapper 完整设计文档

> 面向新接手工程师. 不读源码也应能读懂整体设计 / 各模块职责 / 数据流 / 调参点 / 运行方法.
>
> 代码根: `/home/ubuntu/Disk/codes/jianxiong/MemoryNav/online_mapper/`
> 关联根: `/home/ubuntu/Disk/codes/jianxiong/MemoryNav/offline_mapper/` (只读, 不得修改)
> 本文档所有 `file:line` 引用均相对 `MemoryNav/` 仓库根.

---

## 1. 概述

### 1.1 定位

`online_mapper` 是一个**流式在线主动建图模块**, 目标是在机器人边走边拍的场景下, 实时地构建一张用于导航的**高质量语义拓扑图** (topological semantic map).

它与既有的 `offline_mapper` 是互补关系:

| 维度 | `offline_mapper` (离线) | `online_mapper` (在线) |
|---|---|---|
| 时序假设 | 一次性看到所有帧后处理 | 流式, 每次只能看到"已到达"帧 |
| 主循环 | Phase1-2 分阶段处理全部帧 | 逐帧决策: 是否建 keyframe / 是否闭环 / 是否建门牌 node |
| 关键帧策略 | VPR 余弦距离 + 最小帧间隔 | VPR + 累积位移 + 累积旋转 + 信息增益 + 路口 + 语义 白名单 |
| 闭环 | 首尾对比 (可选) | 全局 VPR + 几何验证, 每帧检测 |
| 语义命名 | Qwen describe_scene / detect_text | 同样调 Qwen, 但**叠加**多帧投票 + 二次验证 + 白名单分类器 + 幻觉过滤 |
| 节点过滤 | 无 (所有 VPR 触发点都建 node) | 类别白名单, 只保留"值得建图"的 7 大类 |
| 节点命名 | 单帧 describe_scene 结果 | 多帧一致 + 类别 canonical 名 + CN/EN 双语 |
| 输出 schema | `merged_labeled_data/` (旧格式) | **完全兼容** `merged_labeled_data/` + 额外 `scene_graph.json` / `pose_graph.json` / `online_mapping_log.jsonl` / `metrics.json` |
| 依赖 offline_mapper | — | 复用 `NodeDistanceEstimator` / `AutoSubImageExtractor` / `AutoLandmarkNamer` (子类化 + 包装, 不修改) |

**硬约束**: 不得修改 `offline_mapper/` 或导航 runtime. 可以 import 复用.

### 1.2 设计目标

1. **在线可运行**: 给一个 `StreamLoader` 逐帧喂, 每帧产出决策; 不依赖"未来帧".
2. **高质量节点**: 只保留对导航有意义的地标 (路口 / 门牌 / 功能区 / 商店 / 设施), 拒绝装饰墙面 / 绿植 / 空走廊.
3. **无幻觉**: VLM (Qwen3.5-9B vLLM) 单帧输出会幻觉, 必须用多帧投票 + 二次验证 + 白名单分类器三层防御.
4. **双语命名**: `position_name` 中文, `position_name_eng` 真英文 (而非中文复刻).
5. **空间序正确**: 拓扑邻接必须反映真实空间顺序 (通过 pose-based spatial KNN 重建, 而非 frame_idx 线性串联).
6. **验证通过**: `offline_mapper/validate_output.py` 必须 100% 通过.
7. **可消融**: 所有关键参数通过 `config.py` 可调, 每个特性 (VO/Qwen/路口/连接/门牌) 都有 enable 开关.

### 1.3 最终能力 (以 `memory_test_data/` 49 帧为例, r6 成果)

- **5 个节点**, **4 条边** (双向, 共 8 个 next_position), **2 次闭环检测**
- 节点清单 (拓扑主链):
  ```
  打印区 ─── 前台 ─── NEUMANN强电井 ─── 关爱室 ─── DEEPROUTE.AI前台
  (Printing Area) (Reception) (NEUMANN Electrical Closet) (Care Room) (DEEPROUTE.AI Reception)
  ```
- 全部节点通过多帧投票 + 二次验证 + 类别白名单确认, 零幻觉
- 全部节点双语命名
- `validator: 5/5 通过`

---

## 2. 三层架构总览

online_mapper 按**几何 / 拓扑 / 语义**三层解耦. 三层都被 `OnlineMapperCore` (core/online_mapper_core.py) 编排.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OnlineMapperCore (编排器)                     │
│   core/online_mapper_core.py:48 — 主循环 run()  :142             │
└─────┬────────────────┬────────────────┬───────────────────────┘
      │                │                │
┌─────▼──────┐   ┌─────▼──────┐   ┌─────▼──────────────────┐
│ Geometry   │   │ Topology    │   │ Semantics              │
│ (几何)     │   │ (拓扑)      │   │ (语义)                 │
├────────────┤   ├─────────────┤   ├────────────────────────┤
│ DepthEst   │   │ KeyframeSel │   │ OpenSetDetector        │
│ MonoVO     │   │ JunctionDet │   │ AutoLandmarkNamer (借) │
│ PoseGraph  │   │ LoopCloser  │   │ DoorPlateTracker       │
│ Occupancy  │   │ TopoGraph   │   │ HallucinationFilter    │
│            │   │ ConnBuilder │   │  - STRICT_PROMPT       │
│            │   │ FrontierNBV │   │  - QwenVerifier        │
│            │   │             │   │  - MultiFrameVoter     │
│            │   │             │   │  - NameDeduplicator    │
│            │   │             │   │ NodeCategoryClassifier │
│            │   │             │   │ ColocationMerger       │
│            │   │             │   │ SceneGraph             │
└────────────┘   └─────────────┘   └────────────────────────┘
```

### 2.1 文件树 (带行数)

```
online_mapper/
├── config.py                           (61 行)  全局配置
├── run_online_map.py                   (40 行)  CLI 入口
├── RESULTS.md                           — r1-r6 迭代历史与指标
├── DESIGN.md                            — 早期设计草案 (历史参考)
├── README.md                            — 快速指南
├── core/
│   ├── online_mapper_core.py          (834 行)  ⭐ 主编排器
│   └── stream_loader.py                (33 行)  流式帧加载
├── geometry/
│   ├── depth_estimator.py              (43 行)  Depth-Anything-V2 封装
│   ├── visual_odometry.py             (119 行)  ORB + EssentialMatrix VO
│   ├── pose_graph.py                   (95 行)  scipy LM pose graph
│   ├── junction_detector.py           (107 行)  4-camera depth 路口判定
│   └── occupancy.py                    (62 行)  2D occupancy grid
├── topology/
│   ├── graph.py                        (44 行)  TopoGraph / TopoNode
│   ├── keyframe_selector.py            (38 行)  多触发 keyframe 选择
│   ├── loop_closure.py                 (95 行)  闭环检测 + 几何验证
│   ├── frontier_nbv.py                 (27 行)  NBV 选点
│   └── connection_builder.py          (227 行)  next_positions 生成 (包装 offline_mapper)
├── semantics/
│   ├── open_set_detector.py            (61 行)  Grounding-DINO 封装
│   ├── door_plate_tracker.py           (61 行)  门牌多帧累积 + best frame 选择
│   ├── hallucination_filter.py        (467 行)  ⭐ 幻觉过滤三层防御
│   ├── node_category.py               (358 行)  ⭐ 节点类别分类器 + CN/EN 表
│   ├── colocation_merger.py           (206 行)  ⭐ 同位置节点合并
│   ├── scene_graph.py                  (52 行)  层次场景图
│   └── semantic_dedup.py               (37 行)  关键帧触发时的同一房间去重
├── output/
│   └── merged_data_writer.py           (94 行)  merged_labeled_data 写出
└── tests/
    └── test_loop_closure_synth.py       — 合成闭环测试
```

⭐ 标记的文件是本项目自研核心; 其余是对已有模型/工具的封装.

---

## 3. 数据流 / 主循环

### 3.1 高层 pipeline

```
┌─────────────┐
│ memory_test │
│ _data/ (49  │
│ 时间戳×4相机)│
└──────┬──────┘
       │ StreamLoader
       ▼
┌─────────────────────────────────────────────────────────────┐
│ for frame in stream_loader:                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [A] 几何预处理                                           │ │
│ │   depth_map = DepthEstimator(camera_1)                  │ │
│ │   dtrans,drot = MonoVO.estimate(camera_1, depth_map)   │ │
│ │   robot_pose  += motion                                 │ │
│ │   info_gain    = Occupancy.integrate(depth_row, pose)   │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [B] VPR 特征 (4 camera)                                  │ │
│ │   feats = SelaVPR.extract_camera_features(cameras)      │ │
│ │   sim_to_last = cyclic_cosine(feats, last_kf_features)  │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [C] 每帧全局闭环检测                                      │ │
│ │   cand = LoopCloser.detect(feats, all_node_feats, ...)  │ │
│ │   if cand: ORB+F-matrix 几何验证 → add loop edge         │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [D] 每帧 door plate 扫描 (不只 keyframe)                 │ │
│ │   GD(["door plate","room number sign"]) → bbox          │ │
│ │   crop (auto margin, <300px fallback 全图)              │ │
│ │   → Qwen STRICT_DETECT_TEXT_PROMPT                      │ │
│ │   → confidence 过滤 / 整图 verify_text                  │ │
│ │   → MultiFrameVoter.add(vote)                           │ │
│ │   → DoorPlateTracker.add(PlateObservation)              │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [E] KeyframeSelector 判断是否触发                         │ │
│ │   (VPR 距离 OR 累积位移 OR 累积旋转 OR 信息增益)           │ │
│ │   if not triggered: continue                            │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [F] keyframe 触发: 语义收集                               │ │
│ │   cam_objects = GD(4 cameras, 全量 queries)              │ │
│ │   room/landmark = 最高置信 GD label                      │ │
│ │   merge_target = semantic_dedup.find_merge_target(...)  │ │
│ │   if merge: 仅加边, 不建新 node                          │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [G] 路口检测 + describe_scene + 分类决策                  │ │
│ │   junction = JunctionDetector.classify(cameras)         │ │
│ │   plate_text = confirmed_plates[fidx] (from voter)      │ │
│ │   scene = Qwen.describe_scene + QwenVerifier            │ │
│ │   decision = NodeCategoryClassifier.classify(           │ │
│ │       plate, plate_verified, scene, scene_verified,     │ │
│ │       junction, gd_landmark)                            │ │
│ │   if REJECT: 跳过建 node, 继续                           │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [H] 建 keyframe node (通过白名单分类)                      │ │
│ │   add TopoNode(position_name=final_name_cn)             │ │
│ │   add to SceneGraph / PoseGraph                         │ │
│ │   add odom edge (last_kf_node → new_node)               │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

       (循环结束)
       ▼
┌───────────────────────────────────────────────────────────────┐
│ _finalize()  core/online_mapper_core.py:573                   │
│                                                                │
│ 1. _create_door_plate_nodes()                                  │
│    - 从 DoorPlateTracker 取 bbox 最大帧的 PlateObservation      │
│    - 只接受 MultiFrameVoter.confirmed_names()                  │
│    - 再过 NodeCategoryClassifier, REJECT 丢弃                  │
│    - 重复语义名跳过 (避免与 keyframe node 重复)                 │
│    - 建 TopoNode, 但 **不加边** (邻接关系留给后面重建)          │
│                                                                │
│ 2. ColocationMerger.merge(nodes, features, pose_graph)         │
│    - 强信号: VPR sim ≥ 0.85 单独触发合并                        │
│    - 弱信号: frame_gap ≤ 8 AND spatial ≤ 0.5m (两者 AND)       │
│    - 按 category rank 选 anchor (SHOP>ROOM>FUNC>LANDMARK>JUN)  │
│    - SHOP+FUNC 拼接名字 (e.g. "DEEPROUTE.AI前台")              │
│    - anchor.frame_idx = min(anchor.fi, sub.fi) (首次发现)      │
│    - _apply_node_alias: 从 topo/pose/scene/features 删除 sub   │
│                                                                │
│ 3. _rebuild_topology_neighbors_spatial(k_spatial=1, k_temporal=1) │
│    - 清空所有 edges                                             │
│    - Spatial K=1: 每个 node 找 pose 最近的 1 个邻居, 双向 union │
│    - Temporal K=1: 按 frame_idx 排序, 相邻 node 加边             │
│    - 总边集 = 两者并集                                          │
│                                                                │
│ 4. _generate_names() — 从 node.position_name 属性读取           │
│                                                                │
│ 5. NameDeduplicator.resolve()                                  │
│    - 同名节点 VPR 相似度 ≥ 0.78 → merge                         │
│    - 否则 → _1/_2/... 后缀 (加 CN 和 EN 两个后缀)               │
│                                                                │
│ 6. writer.write_node() 写每个 node 目录                         │
│                                                                │
│ 7. ConnectionBuilder (包装 AutoSubImageExtractor)               │
│    - 对每个 node 的 neighbors (来自 topo.edges) 生成            │
│      next_positions: Qwen 打点 + 3-scale crop + DINOv3 CLS +   │
│      走廊中间帧匹配 + Hungarian + sim ≥ 0.40 阈值过滤            │
│    - patch 回 node_position_info.json                          │
│                                                                │
│ 8. 写 scene_graph.json / pose_graph.json /                     │
│    online_mapping_log.jsonl / metrics.json                     │
└───────────────────────────────────────────────────────────────┘
       ▼
  merged_labeled_data/
```

### 3.2 主要调用链

主循环入口: `core/online_mapper_core.py:142 OnlineMapperCore.run()`.
外部 CLI: `run_online_map.py:1` → 构造 `OnlineMapperConfig` → `OnlineMapperCore(cfg).run()`.

---

## 4. Geometry 层细节

### 4.1 `DepthEstimator` (geometry/depth_estimator.py)

**职责**: 单目深度估计, 用于 occupancy 栅格更新与 VO scale.

**模型**: `depth-anything/Depth-Anything-V2-Small-hf` (HuggingFace), 通过 transformers `pipeline("depth-estimation")` 调用.

**输入**: BGR ndarray (cv2 读取的图像).
**输出**: numpy depth map (相对深度, 非 metric), 若模型不可用返回 None.

**关键参数**:
- `cfg.depth_model_id` — HF 模型 ID
- `cfg.depth_device` — 默认 `"cuda:0"`
- `cfg.enable_depth` — 开关

**调参影响**: Small 模型足够快 (~60ms/帧, 49 帧总 ~3s). 如果要更准的边界可换 Large, 但显存占用会从 ~1GB 涨到 ~4GB.

### 4.2 `MonoVO` (geometry/visual_odometry.py:21)

**职责**: 替换 r1 的常速代理, 提供真实单目视觉里程计.

**算法** (VisualOdometry.estimate at line 32):
1. 灰度化 + ORB 特征 (1000 点)
2. BFMatcher(HAMMING, crossCheck) 匹配前后帧
3. `cv2.findEssentialMat(RANSAC, focal=700, prob=0.999, threshold=1.5)`
4. `cv2.recoverPose` → R, t (t 是单位方向)
5. **Scale**: 取 depth map 中央 60% ROI 的 median × 0.05 (经验系数)
6. 2D 平面映射: `dtrans = |t_xz| * scale`, `drot = yaw(R)`
7. Clamp: `dtrans ∈ [0, 3m]`, `drot ∈ [-1, 1 rad]`
8. Fallback: 任何异常 → 沿用上一帧的 `(dtrans, drot)`, 不再 hardcode 0.5/0.02

**关键参数**:
- `focal=700` — camera 假定焦距 (像素); 4 相机是同一型号, 这个值对 memory_test_data 足够
- `scale 常数 0.05` — 把 Depth-Anything 相对深度映射到"米". 不是真实 metric, 只用于图层软约束

**已知限制**: 没有 ground truth, 无法量化 ATE/RPE. 纯粹用于粗粒度轨迹, **不是** 高精度 SLAM.

### 4.3 `PoseGraph` (geometry/pose_graph.py)

**职责**: 维护一个轻量 2D pose graph (x,y,theta), 支持 odom 边 + loop 边, 闭环时做优化.

**数据结构**:
- `PoseNode(node_id, x, y, theta)`
- `PoseEdge(a, b, dx, dy, dtheta, info, kind ∈ {"odom","loop"})`

**优化** (`PoseGraph.optimize(iters=20)`):
- 用 `scipy.optimize.least_squares` (Levenberg-Marquardt)
- 残差: 每条边预测相对位姿 - 边实际位姿 (加权 `sqrt(info)`)
- 锚点: 固定第一个节点 (加 `1e3` 强约束)
- 闭环触发时调用, 平滑全图

**为什么不用 GTSAM**: 本数据集 ≤ 20 个节点, scipy LM 足够 (< 100ms/优化). 到 100+ node 规模时应切换 GTSAM.

### 4.4 `OccupancyGrid` (geometry/occupancy.py)

**职责**: 2D 栅格占据地图, 用于 frontier 探索 + 信息增益判断 (关键帧触发条件之一).

**参数**:
- `cfg.grid_size = 200` — 200×200 cells
- `cfg.grid_resolution = 0.2` — 每 cell 0.2m, 总覆盖 40×40m

**输入**: robot_pose + depth_map 中行 downsampled 到 64 点, 以 FOV=1.2 rad 投射 raycast.

**输出**: `integrate()` 返回 `info_gain = (new_free - prev_free) / total_cells`, 被 KeyframeSelector 累积.

---

## 5. Topology 层细节

### 5.1 `KeyframeSelector` (topology/keyframe_selector.py)

**职责**: 决定当前帧是否触发 keyframe.

**触发条件** (OR 关系, 在 `should_trigger()`):
1. `frame_idx - last_kf_frame_idx < min_keyframe_frame_interval` → 直接拒 (防抖)
2. VPR 相似度 `< cfg.vpr_dissim_threshold` (默认 0.50) → 触发 (场景已变)
3. 累积位移 `≥ cfg.accumulated_translation` (默认 1.5m) → 触发
4. 累积旋转 `≥ cfg.accumulated_rotation` (默认 0.6 rad) → 触发
5. 累积信息增益 `≥ cfg.info_gain_threshold` (默认 0.05) → 触发

**注意**: 这只是**候选**. 最终是否建 node 还要过 `NodeCategoryClassifier` (见 §6.6). REJECT 的候选 keyframe 会被丢弃.

**调参**:
- 降 `vpr_dissim_threshold` → 场景差异更大才触发, keyframe 更少
- 升 `accumulated_translation` → 更长移动才触发, keyframe 更稀
- `min_keyframe_frame_interval` 防止连续帧都触发 (推荐 3 以上)

### 5.2 `JunctionDetector` (geometry/junction_detector.py:38)

**职责**: 路口类型判定, 用于让 classifier 在"无语义但在十字路口"时也能接受该 node.

**算法**:
1. 对 4 个 camera 各跑 DepthEstimator
2. 取中央 30%×30% ROI 的 median depth
3. 阈值 `OPEN_DEPTH_THRESH = 1.8m` → "open" 方向 (能看很远 = 可通行)
4. 统计 open camera 数:
   - 4 → `CROSS` (十字)
   - 3 → `T_JUNCTION` (丁字)
   - 2 opposite (1+3 或 2+4) → `CORRIDOR` (走廊, 不建 node)
   - 2 adjacent → `T_JUNCTION` (拐角)
   - ≤1 → `DEAD_END`

**camera 布局假设**: `camera_1 前 / camera_2 右 / camera_3 后 / camera_4 左`.

**实测表现**: 在 `memory_test_data/` 这种 360° 办公室开放空间, 4 相机几乎处处看到通道, junction 信号区分度弱; 主要靠 semantic classifier reject 无标识 keyframe. 在有真实走廊结构的数据集上才会发挥作用.

### 5.3 `LoopCloser` (topology/loop_closure.py:13)

**职责**: 全局闭环检测.

**核心改进 (r1→r2)**:
1. **Auto-tune 阈值** (`current_threshold()` at :31): 积累所有 sim 观测, `current_threshold = min(cfg=0.78, max(0.65, mean + 2σ))`. 自适应 sequence 的 VPR 分布.
2. **几何验证** (`geometric_verify` at :55): ORB(800) + `findFundamentalMat(RANSAC, 3.0, 0.99)` + inlier ≥ 15.
3. **每帧触发** (在 `core.run()` 里每帧调用): 修复 r1 只在 keyframe 创建时检测的 bug. 语义合并不建新 node 时依然能 fire.
4. **3-frame 防抖**: `_last_lc_frame` 记录, 避免连续帧重复报告同一闭环.

**输出**: 在 `self.topo.edges` 加 loop 边, 在 `self.pose_graph` 加 info=sim 的 loop edge, 触发 `pose_graph.optimize(20)`.

### 5.4 `TopoGraph` / `TopoNode` (topology/graph.py)

**数据结构**:
```python
@dataclass
class TopoNode:
    node_id: str
    timestamp: str             # 来自 stream frame
    frame_idx: int             # 被 ColocationMerger 改成 min(anchor, sub)
    cameras: Dict[str, str]    # camera_id -> image path
    landmark_name: str = ""
    room: str = ""
    floor: str = "F1"
    neighbors: Set[str]        # 运行时 attach
    representative_score: float = 0.0
    # 额外 monkey-patch 的属性 (见 core:348-350):
    # position_name: str
    # position_name_eng: str
    # category: str (NodeCategory.value)

class TopoGraph:
    nodes: Dict[str, TopoNode]
    edges: Set[Tuple[str, str]]  # 无向, (min, max) 规范化
```

### 5.5 `ConnectionBuilder` (topology/connection_builder.py:167)

**职责**: 为每个 node 生成 `next_positions` (朝向邻居的裁剪图 + 相似度 + landmark 名).

**设计**:
- 通过子类 `ThresholdedSubImageExtractor(AutoSubImageExtractor)` (:20) **复用** `offline_mapper.auto_sub_image_extractor.AutoSubImageExtractor` 的核心流程:
  1. 4 camera 并行跑 Qwen PointGrounder 定位"通道正中间位置"
  2. Y 坐标修正 (避免地板反光点过低)
  3. 裁 3-scale crop (big/mid/small)
  4. DINOv3 CLS 特征
  5. 走廊中间帧特征匹配 (fallback: 邻居节点自身 4 相机全图特征)
  6. **Hungarian 算法** 做 camera→neighbor 分配
- **子类只改动一处**: 在 Hungarian 匹配后加 `sim ≥ SIM_THRESHOLD (默认 0.40)` 阈值过滤 (:90-100). 低于阈值的匹配**不**产生 crop/next_position.

**为什么不直接 import**: `offline_mapper/auto_sub_image_extractor.py` 原版 Hungarian 后无过滤, 线性走廊场景会输出 garbage 匹配. 子类化重写 `generate_next_positions()` 约 100 行, 保持 offline_mapper 不被修改.

**调参**:
- `cfg.connection_sim_threshold` (默认 0.40) — Hungarian 匹配相似度下限
- 过高 → 遗漏真实邻居; 过低 → 让线性走廊场景出现错误连接

### 5.6 `_rebuild_topology_neighbors_spatial` (core/online_mapper_core.py:664)

**这是 r6 的核心修复**. 在 `_finalize` 里 coloc merge 之后调用, 彻底重建邻接关系:

```
ALGORITHM:
1. Clear self.topo.edges and all node.neighbors
2. Spatial KNN (k_spatial=1):
   for nid in nodes:
       dists = [(d(pose[nid], pose[other]), other) for other in nodes]
       add edges to top-k_spatial nearest (bidirectional union)
3. Temporal KNN (k_temporal=1):
   ordered = sorted(nodes by frame_idx)
   for i in range(len(ordered)-1):
       add edge (ordered[i], ordered[i+1])
   (也加 k>1 的 prev/next)
4. final edges = spatial ∪ temporal
```

**为什么 K=1 而不是 K=2**:
在 memory_test_data 上, K=2 会让 `DEEPROUTE.AI前台` 把 `NEUMANN强电井` 拉成 2nd-nearest 邻居, 重新引入 `5↔7` 错误直边 — 空间上 `关爱室` 才是正确的中间节点. K=1 严格取最近的一个, 产生的链正是:
```
打印区 ─ 前台 ─ NEUMANN强电井 ─ 关爱室 ─ DEEPROUTE.AI前台
```
这条链反映真实空间顺序.

**为什么需要 temporal 并集**: 纯 K=1 spatial 有可能留下孤立 cluster (比如两个相距远的 room 各自内部 K=1 连成环). temporal K=1 按 frame_idx 串联保底连通.

**调参**: 增大 `k_spatial` 会让每个 node 多连几个空间邻居, 拓扑更密但可能出现跨越关键中间节点的 shortcut. 不推荐改动.

### 5.7 `FrontierNBV` (topology/frontier_nbv.py)

**职责**: 从 occupancy grid 算 frontier, 按 `info_weight * info - cost_weight * dist + sem_weight * sem` 评分, 选 top5 记录到 log (用于在线决策可视化).

**注意**: 在当前 run 中只做**记录**, 不真正驱动探索 (因为 memory_test_data 是预录序列, 机器人不会听 NBV 指挥).

---

## 6. Semantics 层细节

### 6.1 `OpenSetDetector` (semantics/open_set_detector.py)

**模型**: `IDEA-Research/grounding-dino-base` (HuggingFace Grounding-DINO).

**接口**: `detect(bgr_image, queries: List[str]) → List[{label, score, bbox, norm_bbox, area}]`

**两种调用模式**:
1. 每帧 door plate 扫描 (便宜): `queries=["door plate", "room number sign"]`
2. keyframe 触发时全量检测: 使用 `DEFAULT_QUERIES` (door plate / room number / printer / trash / white chair / stool / elevator / fire extinguisher / potted plant / vending machine / sofa / table / monitor)

**过滤**: `box_threshold=0.30, text_threshold=0.25`.

### 6.2 `AutoLandmarkNamer` (借自 offline_mapper/auto_landmark_namer.py)

`online_mapper` 不重新实现这个, 而是直接 import + 实例化:
```python
from offline_mapper.auto_landmark_namer import AutoLandmarkNamer
self.namer = AutoLandmarkNamer(use_qwen=True, gpu=cfg.qwen_gpu)
```

**提供的方法** (通过其内部 `QwenNamingServer`):
- `describe_scene(b64)` → `{name_cn, name_en}` 场景 2-5 字中文名
- `identify_landmark(b64)` → `{name_cn, name_en}` 最显著地标名
- `detect_text(b64)` → `{found, text, name_cn, name_en}` 门牌/招牌文字识别
- `_chat(prompt, b64, max_tokens)` — 底层 vLLM HTTP 调用, `online_mapper` 用它送自定义 STRICT prompt

**GPU 共享**: `AutoLandmarkNamer` 和 `Qwen35PointGrounder` (ConnectionBuilder 用) 共享同一个 vLLM 8199 实例, **无需 GPU load/unload dance**.

### 6.3 `DoorPlateTracker` (semantics/door_plate_tracker.py)

**职责**: 跨帧累积某个门牌/招牌的所有观测, 提供"代表帧"选择.

**数据结构**:
```python
@dataclass
class PlateObservation:
    frame_idx, timestamp, cameras, camera, bbox, score, text,
    name_cn, name_en, pose, area  # area 自动计算 = bbox 面积

class DoorPlateTracker:
    _observations: Dict[str, List[PlateObservation]]  # key = text or name_cn

    def add(obs)               # 追加一条观测
    def best(key) → obs         # 返回 bbox 面积最大的那条 (代表帧)
    def all_best() → {k: obs}
```

**为什么取 bbox 最大**: 代表机器人最接近门牌时的视角, 此时 OCR 最清晰, pose 最准确.

### 6.4 `HallucinationFilter` (semantics/hallucination_filter.py)

这是 `online_mapper` 最重要的**幻觉防御**模块. 467 行, 四个子组件:

#### 6.4.1 `STRICT_DETECT_TEXT_PROMPT` (:43)

比 `offline_mapper.PROMPT_DETECT_TEXT` 更严格:
- 明确要求 `confidence: "low|medium|high"`
- 明确列出"要报告什么"(门牌/房间号/玻璃刻字/店招) 和"不要报告什么"(海报/屏幕/标语墙/安全出口)
- 明确指令"照抄字符, 不翻译, 不释义, 不发明"
- 指示包括英文品牌名 (`DEEPROUTE.AI` 这类)
- 输出 JSON schema: `{found, text, name_cn, name_en, confidence}`

#### 6.4.2 `QwenVerifier` (:156)

二次 yes/no 验证, 复用同一 vLLM 实例:
- `verify_text(img, claim)`: 严格 prompt "图中是否真的有文字 X?" 只回答 是/否
- `verify_scene(img, claim)`: 根据 `looks_specific(claim)` (:108) 自动路由:
  - 具体 (含数字/英文字母/`号室`/`会议室`/长 CJK 专有名词) → 走严格 text-verify
  - 通用 (打印/办公/前台 等 2-5 字纯 CJK 类别词) → 走宽松 scene-verify

**`looks_specific` 启发式 (:108)**:
```python
def looks_specific(name):
    if any(c.isdigit() or c.isascii() for c in name): return True
    if re.search(r"[〇一二三四五六七八九十百千]+[号室]", name): return True
    if "会议室" in name and len(name) > 3: return True
    if len(name) <= 5 and any(kw in name for kw in GENERIC_CATEGORY_KEYWORDS):
        return False
    return True
```

**为什么两套 prompt**: r3 发现一套统一的严格 prompt 会误杀 "打印区"/"前台" 这种通用功能区. r4 分流后, 通用类走宽松, 具体名走严格.

#### 6.4.3 `MultiFrameVoter` (:240)

**投票规则** (`is_confirmed()` at :296):
名字被确认当且仅当 **任一** 成立:
1. 出现在 `≥ MIN_FRAMES=2` 个不同帧
2. 单帧内 `≥ MIN_CAMERAS=2` 个不同 camera 同时报
3. **单帧白名单 fast-pass**: 任一 vote confidence ∈ {medium, high} AND 名字 (含数字 OR 匹配 `SINGLE_FRAME_WHITELIST_KEYWORDS`). 关键字列表:
   ```
   母婴, 茶水, 茶歇, 打印, 复印, 前台, 接待, 休息, 会议室,
   电梯, 楼梯, 卫生间, 洗手间, 强电, 弱电, 配电, 母婴室,
   关爱, 关爱室, 哺乳, 哺乳室, care
   ```

**为什么放宽到 medium**: r4 的 high-only 太严, 会漏掉"关爱室"这种短暂出现但真实的门牌.

#### 6.4.4 `merge_substring_variants` (:317)

**OCR 残缺变体合并**: 在 `confirmed_names()` 调用前 idempotently 触发一次.
算法:
```
按 name 长度降序排列
for 长 in 每个 key:
    for 短 in 后续 key:
        if 短 is 严格子串 of 长 AND len(长) - len(短) ≤ 2:
            长._votes += 短._votes
            delete 短
```
例如: `NEUMANN` (1 票) + `EUMANN` (1 票, partial OCR) → 合并为 `NEUMANN` (2 票), 通过 min_frames/min_cameras=2.

#### 6.4.5 `NameDeduplicator` (:365)

finalize 阶段跑, 处理全局重名 (不同物理位置但 classifier 给了相同名字, e.g. 两个"前台"):
- 按 frame_idx 升序分组
- 每组内两两比较 VPR cyclic_cosine:
  - `sim ≥ 0.78` → merge sub into anchor (记 alias)
  - 否则 → 加后缀 `_1/_2/...` (CN 和 EN 都加)

**触发场景**: coloc merger 没合并掉的、真的是两个不同位置的同名 node (不同楼的"前台", 或不同区域的"打印区").

### 6.5 `NodeCategoryClassifier` (semantics/node_category.py:244)

**核心**: 7 大类 + 决策树 + 双语 canonical 名.

#### 6.5.1 类别枚举 (`NodeCategory` at :32)

```python
JUNCTION_CROSS       十字路口
JUNCTION_T           丁字 / 拐角
ROOM_NUMBERED        编号房间 (101 / 10号房间 / A301)
ROOM_NAMED           命名会议室 / 通用 X室
FUNCTION_AREA        打印区 / 前台 / 茶水间 / 母婴室 / 休息区 / 零食区 / 工位区 / 办公区
SHOP                 店铺/品牌招牌 (DEEPROUTE.AI 等)
LANDMARK_FACILITY    电梯/楼梯/卫生间/强电井/配电间/消防...
REJECT               丢弃, 不建 node
```

#### 6.5.2 白名单 (:48 起)

`FUNCTION_AREA_WHITELIST` 把多个同义词映射到 canonical 中文名:
```
打印/复印/打印区       → 打印区
前台/接待/reception   → 前台
茶水/茶歇/水吧        → 茶水间
母婴/哺乳/母婴室      → 母婴室
关爱/关爱室/care      → 关爱室
休息/lounge/休闲      → 休息区
零食/snack            → 零食区
工位                  → 工位区
open office           → 办公区
```

`LANDMARK_FACILITY_WHITELIST`:
```
电梯/elevator/lift    → 电梯口
楼梯/stair            → 楼梯口
卫生间/洗手间/厕所    → 卫生间
强电                  → 强电井
弱电                  → 弱电井
配电                  → 配电间
水井                  → 水井间
消防                  → 消防设施
```

#### 6.5.3 REJECT_KEYWORDS (:94)

命中即拒:
```
绿植/植物/盆栽/花盆
椅子/凳子/沙发/桌/chair/stool/sofa/table/desk
装饰/画/海报/poster/decoration
墙/wall
地毯/carpet/rug
screen/monitor/屏幕/电视/tv
白板/whiteboard
门/door (单独出现, 没有其它上下文)
# r5 新增: 公司文化墙 / 标语类
文化/标语/口号/原则/slogan/culture
# 通用泛指词
标识/房间标识/网络出口
```

#### 6.5.4 CN_EN_MAP (:94 起) + `cn_to_en(cn)` (:121)

静态映射表 (20+ 条), 保留 `_N` 后缀, Latin 品牌 pass-through:
```
打印区     → Printing Area
前台       → Reception
茶水间     → Tea Room
母婴室     → Mother-Baby Room
关爱室     → Care Room
休息区     → Lounge
零食区     → Snack Area
工位区     → Workspace
办公区     → Office Area
电梯口     → Elevator
楼梯口     → Stairs
卫生间     → Restroom
强电井     → Electrical Closet
弱电井     → Network Closet
配电间     → Power Room
水井间     → Water Room
消防设施   → Fire Equipment
十字路口   → Cross Junction
丁字路口   → T Junction
```

`cn_to_en("前台_2")` → `"Reception_2"`; `cn_to_en("DEEPROUTE.AI")` → `"DEEPROUTE.AI"` (pass-through).

#### 6.5.5 决策树 (`classify` at :261)

按优先级:
```
A. verified plate 含房号 (has_room_number)            → ROOM_NUMBERED
B. verified plate 含 "会议室/meeting/conference"      → ROOM_NAMED
C. plate 或 verified scene 命中 FUNCTION_AREA 白名单  → FUNCTION_AREA
D. plate 或 verified scene 命中 LANDMARK 白名单       → LANDMARK_FACILITY
D2. verified plate 是 2-6 字 CJK 以 "室" 结尾且非 REJECT → ROOM_NAMED (通用 X室 规则)
E. verified plate 是 4-30 字符 Latin (不在 REJECT)    → SHOP
F. 无语义 AND junction == CROSS/T                     → JUNCTION_*
G. 其它                                               → REJECT
```

**预检查**: `gd_reject = matches_reject(gd_landmark)` — 如果 Grounding-DINO 给的 landmark 命中 REJECT_KEYWORDS, 在 F 步骤里禁止 junction-only 接受.

### 6.6 `ColocationMerger` (semantics/colocation_merger.py:54)

**职责**: 把物理上重合但类别不同的节点合并 (e.g. `DEEPROUTE.AI` 招牌 + `前台_2` keyframe 是同一个前台).

#### 6.6.1 合并条件 (`_should_merge` at :87)

```python
# 强信号: VPR 相似度极高 → 合并 (单独触发)
if cyclic_cosine(feats_a, feats_b) >= 0.85:
    return True, "vpr"
# 弱信号: 帧间隔小 AND 空间距离小 (AND, 防止 VO 不准误合并)
if frame_gap <= 8 AND spatial_dist <= 0.5m:
    return True, "frame+spatial"
return False
```

**为什么 AND 不 OR**: r5 第一次实现用 `OR (frame|vpr|spatial)`, 结果在本数据集 (VO 烂, 所有 pose 聚集在 1m 内) 5 个 merge 把所有 node 合并成 2 个. `AND` 后只有真的物理重合才合并.

#### 6.6.2 Anchor 选择 (`_rank` at :105)

按 category 优先级:
```
SHOP              rank 7   (最高)
ROOM_NAMED        rank 6
ROOM_NUMBERED     rank 5
FUNCTION_AREA     rank 4
LANDMARK_FACILITY rank 3
JUNCTION_CROSS    rank 2
JUNCTION_T        rank 1
REJECT            rank 0
```

#### 6.6.3 名字拼接 (`_combined_name` at :110)

**仅当** anchor 是 SHOP 且 sub ∈ {FUNCTION, LANDMARK, ROOM_NAMED, ROOM_NUMBERED}:
```
new_cn = f"{anchor.cn}{sub.cn_without_dedup_suffix}"
new_en = f"{anchor.en} {sub.en_without_dedup_suffix}"
```
例子: `DEEPROUTE.AI` (SHOP) + `前台_2` (FUNCTION) → `DEEPROUTE.AI前台` / `DEEPROUTE.AI Reception`.

其它情况: 只保留 anchor 的名字.

**例外**: 实测里 `NEUMANN` (被分类器识别为 SHOP, 因为是 4-30 字符 Latin) + `强电井` (LANDMARK) → `NEUMANN强电井`. 这是预期行为, Neumann 是会议室品牌名.

#### 6.6.4 Anchor frame_idx 取 min (r6 修复)

```python
anchor.frame_idx = min(anchor.frame_idx, sub.frame_idx)
```
表示"机器人首次到达该位置". 让合并节点在 trajectory 时间序中反映首次发现, 影响 `_rebuild_topology_neighbors_spatial` 里 temporal 邻居的排序.

### 6.7 `SceneGraph` (semantics/scene_graph.py)

层次场景图: `floor → room → node → objects`.
写出到 `scene_graph.json` (见 §7.4).

目前 `room` 分组多数是 `"unknown"`, 只有门牌 semantic node 填 canonical 名. 完整 room 聚类留作 future work.

### 6.8 `semantic_dedup` (semantics/semantic_dedup.py)

在 keyframe 触发瞬间调用: 如果 new landmark/room 与已有 node 的 landmark/room 一样 AND VPR ≥ 0.65 → merge (不建新 node, 只加边). 这是**在线**合并, 与 finalize 阶段的 `ColocationMerger`/`NameDeduplicator` 互补.

---

## 7. 输出 schema

### 7.1 目录结构

```
online_mapper/output/
├── merged_labeled_data/           ⭐ 主输出, schema 与 offline_mapper 100% 兼容
│   ├── 1/
│   │   ├── 1770097720_camera_1.jpg
│   │   ├── 1770097720_camera_2.jpg
│   │   ├── 1770097720_camera_3.jpg
│   │   ├── 1770097720_camera_4.jpg
│   │   ├── crops/
│   │   │   └── ...__big__.jpg  ...__mid__.jpg  ...__small__.jpg
│   │   └── node_position_info.json
│   ├── 2/  ...
│   └── 8/  ...
├── scene_graph.json              层次场景图 floor→room→node→objects
├── pose_graph.json               SE2 pose graph (nodes + edges)
├── online_mapping_log.jsonl      每帧决策日志
└── metrics.json                  统计与调参命中率
```

### 7.2 `node_position_info.json` schema

```json
{
  "self_position": {
    "position_id": "1",
    "position_name": "打印区",
    "position_name_eng": "Printing Area",
    "camera_1": "1770097720_camera_1.jpg",
    "camera_2": "1770097720_camera_2.jpg",
    "camera_3": "1770097720_camera_3.jpg",
    "camera_4": "1770097720_camera_4.jpg"
  },
  "next_positions": [
    {
      "position_id": "2",
      "position_name": "前台",
      "position_name_eng": "Reception",
      "camera_name": "camera_1",
      "landmark_name": "reception desk",
      "landmark_name_eng": "reception desk",
      "big_box":   "0.365,0.311,0.634,0.648",
      "mid_box":   "0.382,0.333,0.617,0.626",
      "small_box": "0.398,0.353,0.601,0.606",
      "pixel_box": "",
      "crop_image_path":  "crops/..._big__.jpg",
      "crop_image_paths": {
        "big":   "crops/..._big__.jpg",
        "mid":   "crops/..._mid__.jpg",
        "small": "crops/..._small__.jpg"
      }
    }
  ]
}
```

**必填字段** (validator 检查):
- `self_position.position_id / position_name / position_name_eng / camera_1..4`
- `next_positions[].position_id / position_name / position_name_eng / camera_name / landmark_name / landmark_name_eng / big_box / mid_box / small_box / pixel_box / crop_image_path / crop_image_paths`

### 7.3 `scene_graph.json`

```json
{
  "floors": {
    "F1": {
      "unknown": ["1", "2"],
      "关爱室":  ["8"]
    }
  },
  "nodes": {
    "1": {
      "room": "unknown",
      "objects": [
        {"label": "table", "bbox": [...], "score": 0.51, "camera": "camera_1", "plate_text": ""}
      ]
    }
  }
}
```

### 7.4 `pose_graph.json`

```json
{
  "nodes": [
    {"id": "1", "x": 0.0,   "y": 0.0,    "theta": 0.0}
  ],
  "edges": [
    {"a": "1", "b": "2", "dx": 0.47, "dy": -0.46, "dtheta": -1.6, "info": 1.0, "kind": "odom"},
    {"a": "2", "b": "7", "dx": 0.0,  "dy": 0.0,   "dtheta": 0.0,  "info": 0.78, "kind": "loop"}
  ]
}
```

### 7.5 `online_mapping_log.jsonl` (每帧一行)

```json
{
  "frame_idx": 0,
  "ts": "1770097720",
  "vpr_sim_to_last": null,
  "info_gain": 0.000375,
  "vo_dtrans": 0.0,
  "vo_drot": 0.0,
  "robot_pose": [0.0, 0.0, 0.0],
  "occupancy": {"free": 15, "occ": 13, "unknown": 39972},
  "keyframe": true,
  "reason": "first_frame",
  "junction": {"kind": "cross", "open_cams": ["camera_1","camera_2","camera_3","camera_4"], "n_open": 4},
  "category_decision": {"category": "function_area", "name": "打印区", "reason": "function_area scene='打印区'"},
  "nbv_pick": {...},
  "loop_closure": [...]
}
```

额外可能的字段: `semantic_merge_into`, `category_decision` (REJECT 时), `loop_closure`.

### 7.6 `metrics.json`

```json
{
  "n_nodes": 5,
  "n_edges": 4,
  "n_loop_closures": 2,
  "n_semantic_merges": 0,
  "n_frames": 49,
  "n_keyframes_triggered": 8,
  "n_door_plates": 32,
  "n_connections": 8,
  "n_named_landmarks": 8,
  "vo_mode": "real",
  "runtime_s": {"depth": 3.1, "vpr": 25.7, "detect": 6.0, "vo": 4.8, "total": 199.5},
  "kf_accepted_by_category": {"function_area": 4},
  "kf_rejected_by_category": 4,
  "plate_voter": {
    "total_names_seen": 12,
    "confirmed": 5,
    "rejected": 6,
    "confirmed_names": ["DEEPROUTE.AI", "强电井", "NEUMANN", "关爱室", "文化墙标语"],
    "rejected_names":  ["文化墙", "工程师文化原则", "文化标语", "网络出口", "房间标识", "办公室"]
  },
  "plate_drops_low_conf": 0,
  "plate_drops_verify": 4,
  "plate_drops_unconfirmed": 6,
  "plate_drops_category": 2,
  "coloc_merge": {
    "pairs_examined": 17,
    "merges": 3,
    "by_reason": {"frame+spatial": 1, "vpr": 2},
    "aliases": {"3":"2", "6":"7", "4":"5"}
  },
  "topology_rebuild": {"k_spatial": 1, "k_temporal": 1, "spatial_edges": 3, "temporal_edges": 1, "total_edges": 4},
  "name_dedup": {"groups_processed": 0, "merges": 0, "suffixed": 0},
  "loop_threshold_used": 0.684
}
```

---

## 8. 依赖与环境

### 8.1 Python env

- **Conda env**: `internvla`, 位于 `/home/ubuntu/miniconda3/envs/internvla`
- Python 解释器: `/home/ubuntu/miniconda3/envs/internvla/bin/python`

### 8.2 关键 Python 包

```
torch, torchvision         (GPU CUDA)
transformers (>=4.40)      Grounding-DINO / Depth-Anything-V2
opencv-python              cv2 — ORB / BFMatcher / findEssentialMat / findFundamentalMat
numpy, scipy               pose graph LM / Hungarian
requests                   vLLM HTTP client
faiss                      offline_mapper.NodeDistanceEstimator 用
vllm (>=0.18)              在 qwen3 env 单独装; online_mapper 只通过 HTTP 调用
```

### 8.3 模型权重 / 缓存路径

| 模型 | 位置 | 大小 |
|---|---|---|
| SelaVPR | offline_mapper 缓存 | ~500MB |
| Depth-Anything-V2-Small | HF cache: `~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf` | ~100MB |
| Grounding-DINO-base | HF cache: `models--IDEA-Research--grounding-dino-base` | ~700MB |
| Qwen3.5-9B | `/home/ubuntu/Disk/models/Qwen3.5-9B` | ~18GB |

### 8.4 GPU 布局

```
GPU0: SelaVPR + Depth-Anything-V2 + Grounding-DINO + DINOv3  (~5GB 常驻)
GPU1: Qwen3.5-9B vLLM (port 8199)                            (~38GB)
GPU2, GPU3: 空闲
```

**重要**: `AutoLandmarkNamer` (describe_scene / detect_text) 和 `Qwen35PointGrounder` (ConnectionBuilder 用) 共享同一个 GPU1 vLLM 实例, **不需要** 像 offline_mapper 的 Phase 1.5b 那样做 stop/start dance.

### 8.5 vLLM 启动脚本

`deploy/start_qwen_vllm.sh`:
```bash
#!/bin/bash
GPU_ID=${1:-1}
PORT=${2:-8199}
MODEL_PATH="$HOME/Disk/models/Qwen3.5-9B"
CONDA_ENV="qwen3"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve "${MODEL_PATH}" \
    --port ${PORT} --dtype bfloat16 --max-model-len 4096 \
    --max-num-seqs 8 --gpu-memory-utilization 0.85 \
    --enable-prefix-caching --served-model-name qwen3.5-9b \
    --trust-remote-code --no-enable-log-requests
```

---

## 9. 启动方式

### 9.1 启 vLLM

```bash
cd /home/ubuntu/Disk/codes/jianxiong/MemoryNav
nohup bash deploy/start_qwen_vllm.sh 1 8199 > /tmp/qwen_vllm.log 2>&1 &

# 等待 ready (约 60-90s 加载权重)
while ! curl -s --noproxy '*' http://localhost:8199/v1/models | grep -q qwen3.5-9b; do
    sleep 5
done
echo "vLLM ready"
```

### 9.2 端到端跑 online_mapper

```bash
cd /home/ubuntu/Disk/codes/jianxiong/MemoryNav
/home/ubuntu/miniconda3/envs/internvla/bin/python online_mapper/run_online_map.py \
    --input  memory_test_data \
    --output online_mapper/output/merged_labeled_data \
    --log_level INFO
```

**输出**:
```
=== Online Mapping Done ===
Output: online_mapper/output/merged_labeled_data
Metrics: {...}
```

**CLI 选项** (`run_online_map.py:11`):
- `--input` 输入数据目录 (默认 `memory_test_data`)
- `--output` 输出目录
- `--vpr_config` VPR 配置路径 (默认 `deploy/vpr_config.yaml`)
- `--no_depth` 关闭深度估计
- `--no_grounding_dino` 关闭开放集检测
- `--log_level` DEBUG/INFO/WARNING/ERROR

### 9.3 Schema 校验 (必须通过)

```bash
cd /home/ubuntu/Disk/codes/jianxiong/MemoryNav
/home/ubuntu/miniconda3/envs/internvla/bin/python \
    offline_mapper/validate_output.py \
    online_mapper/output/merged_labeled_data
```
期望输出:
```
📊 总计: 5 节点, 8 连接, 24 裁剪图像
✅ 通过: 5/5 节点
🎉 所有验证通过!
```

### 9.4 单元测试

```bash
cd /home/ubuntu/Disk/codes/jianxiong/MemoryNav
/home/ubuntu/miniconda3/envs/internvla/bin/python -m pytest online_mapper/tests/ -v
```

### 9.5 合成闭环测试 (证明闭环路径能 fire)

```bash
cd /home/ubuntu/Disk/codes/jianxiong/MemoryNav
/home/ubuntu/miniconda3/envs/internvla/bin/python \
    online_mapper/tests/test_loop_closure_synth.py
```
脚本会: 构造 `/tmp/memory_test_synth` = 正向序列 + 反向尾 15 帧, 跑 `OnlineMapperCore`, 断言 `n_loop_closures >= 1`.

---

## 10. 配置项 (config.py)

全部在 `online_mapper/config.py`, 构造方式: `OnlineMapperConfig()`.

| 参数 | 默认值 | 含义 | 调大影响 | 调小影响 |
|---|---|---|---|---|
| **IO** | | | | |
| `input_dir` | `memory_test_data` | 数据集目录 | — | — |
| `output_dir` | `online_mapper/output/merged_labeled_data` | 输出目录 | — | — |
| `vpr_config_path` | `deploy/vpr_config.yaml` | SelaVPR 配置 | — | — |
| **关键帧触发** | | | | |
| `vpr_dissim_threshold` | 0.50 | VPR 相似度低于则触发 keyframe | 更少 keyframe | 更多 keyframe |
| `accumulated_translation` | 1.5 m | 累积位移阈值 | 更稀疏 | 更密 |
| `accumulated_rotation` | 0.6 rad | 累积旋转阈值 | 更稀疏 | 更密 |
| `info_gain_threshold` | 0.05 | 占据新增比例阈值 | 更稀疏 | 更密 |
| `min_keyframe_frame_interval` | 3 | 连续 keyframe 最小帧间隔 (防抖) | 跳过更多 | 可能连触 |
| **闭环** | | | | |
| `loop_closure_min_gap` | 8 | 闭环检测最小帧间隔 | 更难 fire | 可能误触 |
| `loop_closure_vpr_threshold` | 0.78 | 闭环 VPR 上限 (auto-tune 取 min) | 更严 | 更松 |
| `loop_closure_top_k` | 5 | 最多报告的候选数 | — | — |
| `loop_closure_geom_verify` | True | 是否做 ORB+F-matrix 几何验证 | — | — |
| `loop_closure_min_inliers` | 15 | 几何验证 inlier 数门槛 | 更严 | 更松 |
| **NBV** | | | | |
| `nbv_info_weight` | 1.0 | 信息增益权重 | 更探索 | 更保守 |
| `nbv_cost_weight` | 0.3 | 距离成本权重 | 更近 | 更远 |
| `nbv_semantic_weight` | 0.5 | 语义权重 | 更语义驱动 | 更几何 |
| **Depth** | | | | |
| `depth_model_id` | `depth-anything/Depth-Anything-V2-Small-hf` | HF 模型 | — | — |
| `depth_device` | `cuda:0` | GPU | — | — |
| `enable_depth` | True | 开关 | — | — |
| **Semantics** | | | | |
| `enable_grounding_dino` | True | 开关 | — | — |
| `grounding_dino_model_id` | `IDEA-Research/grounding-dino-base` | HF 模型 | — | — |
| `enable_qwen_naming` | True | 开关, 关闭后走 placeholder | 真实中文名 | 占位符 |
| `qwen_base_url` | `http://localhost:8199/v1` | vLLM endpoint | — | — |
| `qwen_gpu` | `"1"` | 仅在 subprocess 回退时用 | — | — |
| **门牌** | | | | |
| `enable_door_plate_detection` | True | 开关 | — | — |
| `door_plate_min_score` | 0.30 | GD box_threshold | 更严 | 更松 |
| **Connection** | | | | |
| `enable_real_connections` | True | 用真实 Qwen 打点 vs dummy crop | — | — |
| `connection_sim_threshold` | 0.40 | Hungarian 匹配相似度下限 | 丢弃更多 | 可能有错匹配 |
| **VO** | | | | |
| `enable_real_vo` | True | 用 MonoVO vs 常速代理 | — | — |
| **Occupancy** | | | | |
| `grid_resolution` | 0.2 m | 每 cell | 粗 | 细 (cost↑) |
| `grid_size` | 200 | 栅格边长 | 大覆盖 | 小覆盖 |
| **其他** | | | | |
| `start_id` | 1 | 第一个节点 ID | — | — |

**硬编码的"准配置"项** (不在 config.py 里, 但是调参重点):
- `ColocationMerger.VPR_SIM_THRESHOLD = 0.85` (`semantics/colocation_merger.py:55`)
- `ColocationMerger.SPATIAL_DIST_THRESHOLD = 0.5` m (`:56`)
- `ColocationMerger.FRAME_GAP_THRESHOLD = 8` (`:54`)
- `NameDeduplicator(merge_vpr_threshold=0.78)` (`online_mapper_core.py:600`)
- `MultiFrameVoter(min_frames=2, min_cameras=2)` (`online_mapper_core.py:65`)
- `_rebuild_topology_neighbors_spatial(k_spatial=1, k_temporal=1)` (`online_mapper_core.py:598`)
- `JunctionDetector.OPEN_DEPTH_THRESH = 1.8` m (`junction_detector.py:39`)
- `MonoVO.focal = 700` (`visual_odometry.py:22`)
- `MonoVO` 的 depth scale 常数 `0.05` (`visual_odometry.py:80`)

---

## 11. 迭代历史 (r1→r6)

| 轮 | 核心问题 | 修复 | nodes | edges | 关键指标 |
|:---:|---|---|:---:|:---:|---|
| r1 | 基线 | 流式 VPR + 4 触发 + scipy LM + 占据 + GD + 层次 scene graph | 10 | 9 | n_loop=0, 0 真实 next_positions (全 dummy crop), enable_qwen_naming=False |
| r2 | Qwen 缺失, dummy crop, 无 VO, 闭环不 fire | 启动 vLLM 8199; 新 `ConnectionBuilder` 子类化 `AutoSubImageExtractor` 加 sim≥0.40 阈值; 新 `DoorPlateTracker` bbox-max 代表帧; 新 `MonoVO` (ORB+EssentialMat); 闭环改 auto-tune+几何验证+每帧触发 | 13 | 5 | n_loop=2, 23 真实连接, 23 命名 landmark, 7 门牌, 但部分 plate node 是幻觉 |
| r3 | 幻觉 node (101号房间 / 爱普路特 / 纽布里茨会议室), 重名 "前台" | 新 `hallucination_filter.py`: STRICT prompt + `QwenVerifier` + `MultiFrameVoter(≥2 frames)` + `NameDeduplicator`; 分 scene/text 两套 verify prompt | 7 | 6 | 幻觉清零, 前台_1/前台_2 后缀去重, 但误杀了应该保留的真名 |
| r4 | r3 过度保守, 要求类别白名单 + 抑制走廊中段 | 新 `node_category.py`: 7 类 + WHITELIST/REJECT; 新 `junction_detector.py`: 4-cam depth 路口; 单帧白名单 fast-pass (需 high confidence); 强制分类器 gate 每个 keyframe 创建 | 5 | 4 | 4 keyframe 被 classifier reject; DEEPROUTE.AI 通过 brand fast-pass; 纯净输出但漏掉关爱室 |
| r5 | 关爱室漏检, DEEPROUTE.AI 与 前台 应合并, EN 字段是中文, 强电井单邻居 | `_scan_door_plates` 扩 crop margin + 小 crop 时用整图; 新 `ColocationMerger` (强 VPR / 弱 frame+spatial AND); `CN_EN_MAP` + `cn_to_en()`; MultiFrameVoter 单帧白名单降到 medium; substring 变体合并 (EUMANN→NEUMANN); door-plate 双向 prev/next 连边 | 5 | 6 | 关爱室回归; DEEPROUTE.AI前台 合并; NEUMANN强电井 合并; EN 是真英文; 但 stale 5↔7 边导致 关爱室 "挂末端" |
| r6 | 关爱室位置不对, 应夹在 强电井 和 DEEPROUTE.AI前台 之间 | 取消 `_create_door_plate_nodes` 里 prev/next 连边; 新 `_rebuild_topology_neighbors_spatial(k_spatial=1, k_temporal=1)` 清空重建; coloc merge 锚点 `frame_idx = min(anchor, sub)`; `_apply_node_alias` 同步 `node_frame_idx` | 5 | 4 | 主链 `打印区─前台─NEUMANN强电井─关爱室─DEEPROUTE.AI前台`; 5↔7 stale 边消除; 双向一致 |

**r6 最终节点清单**:
| id | cn | en | frame_idx | pose |
|:---:|---|---|:---:|---|
| 1 | 打印区 | Printing Area | 0 | (0.00, 0.00) |
| 2 | 前台 | Reception | 16 | (0.47, -0.47) |
| 7 | NEUMANN强电井 | NEUMANN Electrical Closet | 38 | (-0.19, -0.98) |
| 8 | 关爱室 | Care Room | 41 | (-0.31, -1.04) |
| 5 | DEEPROUTE.AI前台 | DEEPROUTE.AI Reception | 47 | (-0.59, -1.19) |

**r6 最终邻接 (4 条边, 双向 = 8 个 next_position)**:
```
1 ── 2 ── 7 ── 8 ── 5
```

---

## 12. 已知限制 / 未来工作

1. **GTSAM 未集成**: 当前用 scipy LM, 到 100+ 节点规模时应切换.
2. **Metric scale 不准**: MonoVO 的 scale 用 Depth-Anything 相对深度 × 经验常数 0.05. 没有 GT, 无法量化 ATE/RPE. 生产部署应接真实 IMU/odom.
3. **Scene graph `room` 聚类未打磨**: 多数节点 room 还是 `"unknown"`. 只有门牌 semantic node 填 canonical 名. 可补一层 "描述类似场景 → 同 room" 聚类.
4. **JunctionDetector 在 360° 开放空间区分度弱**: memory_test_data 是办公室, 4 相机几乎处处看到通道, 所有 keyframe 都报 CROSS. 在真走廊结构的数据上才有意义.
5. **门牌检测依赖 Qwen**: vLLM 未就绪时, 门牌节点数 = 0 (但 keyframe-route 的 FUNCTION_AREA 仍可以通过 describe_scene 命中白名单).
6. **`CN_EN_MAP` 是静态表**: 没命中的新词会保留中文 fallback. 未来可调 Qwen 做翻译 (但要防 VLM 翻译幻觉).
7. **更大数据集未压测**: 当前所有迭代都在 49 帧上调参. 200+ 帧时 coloc/dedup 阈值可能需要重新调.
8. **ConnectionBuilder 性能**: Qwen 打点每个 camera ~1s, 4 camera × 5 node ~20s. 大图时需要批处理或缓存.

---

## 13. FAQ / 排错

### 13.1 vLLM 没起来

**症状**: `curl http://localhost:8199/v1/models` connection refused.

**步骤**:
1. `ss -tlnp | grep 8199` — 端口是否监听?
2. `ps aux | grep vllm` — 进程是否活?
3. `tail /tmp/qwen_vllm.log` — 看启动日志, 常见问题:
   - OOM: 别的 GPU1 进程占用, `nvidia-smi` 看谁在用, kill 掉
   - `CXXABI_1.3.15 not found`: 漏加 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
   - 端口已占用: 换 port, 相应改 `config.qwen_base_url`
4. 重启: `bash deploy/start_qwen_vllm.sh 1 8199` (前台); 或 `nohup ... &` 后台

**Fallback**: 把 `config.enable_qwen_naming=False`. 仍能跑, 但所有节点会是 `节点_N` 占位符命名, 门牌检测失效.

### 13.2 GPU OOM

**症状**: `torch.cuda.OutOfMemoryError` 或模型加载挂起.

**排查**:
1. `nvidia-smi` — 哪个 GPU 吃了多少
2. GPU0 应该 < 6GB (VPR+Depth+GD+DINOv3); GPU1 应该 ~38GB (Qwen)
3. 如果 GPU0 满: 有别的进程, 或者 Depth/GD 没释放 — 重启 python
4. 如果 GPU1 满: 只能跑一个 vLLM. 杀掉旧 vllm 进程再启

### 13.3 validator 失败

**症状**: `validate_output.py` 报错.

**常见原因**:
- `node_position_info.json` 缺字段 (`position_name_eng` / `crop_image_paths`) — 查 `output/merged_data_writer.py:22` 的 `write_node`
- crop 文件不存在 — 查 `ConnectionBuilder` 是否跑完, 有没有异常 (日志搜 `ConnectionBuilder failed`)
- 坐标超范围 — `big_box` 值需要在 [0,1], 查 `AutoSubImageExtractor._make_square_crop`

### 13.4 节点太多

**症状**: 输出 20+ 个节点, 大多重复.

**排查**:
1. `metrics.plate_voter.confirmed_names` 看通过投票的名字
2. `metrics.coloc_merge.merges` 看合并数
3. `metrics.name_dedup.merges/suffixed` 看去重数
4. 如果 plate_voter 放行太多幻觉: 看 `rejected_names` 里有没有应该通过的 → 说明 voter 阈值不合理
5. 如果同名节点没合并: 升 `NameDeduplicator.merge_vpr_threshold` 或降到 0.72
6. 调 `KeyframeSelector` 更严 (`vpr_dissim_threshold` 降到 0.40, `accumulated_translation` 升到 2.0)

### 13.5 节点太少 / 漏检

**症状**: 明知道有门牌/会议室但没出现在拓扑里.

**排查步骤** (参考 r5/r6 的调查方法):
1. **用中性 prompt 让 Qwen 扫所有 49×4 帧**, 确认目标是否真的在数据里:
   ```bash
   ssh ubuntu@10.24.99.217 "cd /path && python -c \"
   import cv2, base64, requests
   S=requests.Session(); S.trust_env=False
   def ask(p, claim):
       img=cv2.imread(p); _,b=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,85])
       b64=base64.b64encode(b).decode()
       r=S.post('http://localhost:8199/v1/chat/completions', json={
           'model':'qwen3.5-9b','max_tokens':40,'temperature':0,
           'messages':[{'role':'user','content':[
               {'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{b64}'}},
               {'type':'text','text':f'图中是否清晰可见文字 \\\"{claim}\\\"? 只回答 是/否'}]}]
       }).json()
       return r['choices'][0]['message']['content'].strip()
   # 扫全部帧
   import glob
   for p in sorted(glob.glob('memory_test_data/*_camera_*.jpg')):
       r = ask(p, '你要找的名字')
       if '是' in r: print(p, r)
   \""
   ```
   - 全部 "否": 数据集里**没有这个东西**, 不是 bug
   - 只有 1 帧: 单帧白名单 fast-pass 应该接受, 查 `MultiFrameVoter.is_confirmed` 逻辑
   - 多帧: 查为什么 `_scan_door_plates` 没调到它
2. **检查 keyframe 是否触发**: `jq 'select(.frame_idx == X)' online_mapping_log.jsonl` 看那一帧
3. **检查 Grounding-DINO 给了 bbox 吗**: 手动跑 `OpenSetDetector.detect` 在那一帧 (参考 `docs/online_mapper.md §13.5` 的脚本示例)
4. **检查 bbox crop 大小**: 如果 < 300 px 短边, 确认 `_scan_door_plates` 走了"直接用全图"回退路径
5. **检查 Qwen STRICT prompt 结果**: 可能 `confidence == "low"` 被丢 → metrics `plate_drops_low_conf`
6. **检查 voter**: metrics `plate_voter.rejected_names` 里是否有目标 → 说明投票没通过
7. **检查 classifier**: metrics `plate_drops_category` → 分类器 REJECT 了

### 13.6 有幻觉重现

**症状**: 拓扑里出现不存在的名字.

**排查**:
1. 先用 §13.5 的中性 prompt 扫, 确认是否真是幻觉
2. 看 `metrics.plate_voter.confirmed_names` — 谁混进去了?
3. 看 `metrics.kf_accepted_by_category` — 哪一类通过的?
4. 加强对应的 WHITELIST/REJECT:
   - 如果是 scene_describe 幻觉: 加 `_VERIFY_SCENE_PROMPT_TEMPLATE` 严格度
   - 如果是文化墙/标语类: 加进 `REJECT_KEYWORDS`
5. 调紧 voter: `MIN_FRAMES=3` 或关闭 `allow_single_frame_whitelist`

### 13.7 拓扑主链顺序不对 (r6 问题)

**症状**: 某节点在 next_positions 里出现在不合理的位置.

**步骤**:
1. Dump 所有 node 的 `frame_idx` 和 pose:
   ```bash
   cat online_mapper/output/pose_graph.json | jq '.nodes'
   ```
2. 手算空间距离矩阵 (参考 r6 诊断方法):
   ```python
   import json, itertools
   poses = {n['id']: (n['x'], n['y']) for n in json.load(open('pose_graph.json'))['nodes']}
   for a, b in itertools.combinations(poses.keys(), 2):
       dx, dy = poses[a][0]-poses[b][0], poses[a][1]-poses[b][1]
       print(f'{a}↔{b}: {(dx*dx+dy*dy)**0.5:.3f}m')
   ```
3. 看 `metrics.topology_rebuild` — spatial_edges 和 temporal_edges 分别多少?
4. 如果 stale 边没清: 查 `_rebuild_topology_neighbors_spatial` 是否在 coloc merge 之后被调用 (core:594 附近)
5. 如果空间 K=1 给了错链: 换 K=2, 但要预期会多出冗余短路边

### 13.8 闭环不 fire

**症状**: 有重访场景但 `n_loop_closures = 0`.

**排查**:
1. `metrics.loop_threshold_used` 是多少? 如果 > 0.85 说明 auto-tune 分布偏高
2. `tail online_mapping_log.jsonl` 看每帧的 `vpr_sim_to_last` 分布
3. 手动降 `cfg.loop_closure_vpr_threshold` 到 0.65 试试
4. 检查 ORB 几何验证: 设 `cfg.loop_closure_geom_verify=False` 先绕过几何
5. 跑合成测试 `tests/test_loop_closure_synth.py` 证明代码路径正常

### 13.9 next_positions 指向错误邻居

**症状**: crop 画面看起来不是邻居方向.

**排查**:
1. 看 `ConnectionBuilder` 日志 (INFO level), 找 `KEEP`/`DROP` 行, sim 值是多少
2. `metrics.connection_sim_threshold` 默认 0.40; 如果 matches 都在 0.3-0.4 之间, 说明数据集走廊特征弱
3. 调 `cfg.connection_sim_threshold` 到 0.50 更严
4. 检查邻居节点是否真的是空间邻居 — 很多时候 ConnectionBuilder 没问题, 是拓扑邻接选错了

---

## 文档维护

本文档与代码同步维护. 每次重大修改 (r6+) 应:
1. 更新对应的章节 (架构 / 模块 / 数据流)
2. 在 §11 加一行迭代表格
3. 更新 §10 如果有新 config 项
4. 更新 §13 如果有新故障模式

相关文档:
- `online_mapper/RESULTS.md` — 每轮迭代的详细 before/after
- `online_mapper/DESIGN.md` — 早期设计草案 (历史参考)
- `online_mapper/README.md` — 快速上手
