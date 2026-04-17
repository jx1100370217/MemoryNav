# online_mapper 设计文档 (v2.4.0)

> 当前版本: **v2.4.0** — 新增 WebSocket 建图模式与双模式路由
>
> 代码根: `MemoryNav/online_mapper/`
> 第三方源码: `MemoryNav/third_party/vggt_space/` (VGGT, .gitignore)
> 模型权重: `MemoryNav/pretrained/` (.gitignore, 各模型本地化路径)
>
> 本文档以 v2.4.0 架构为准. 历史迭代见末尾"迭代历史"章节, r1→r6 的早期 metrics 见 `online_mapper/RESULTS.md`.

---

## 1. 定位与能力

`online_mapper` 是**流式在线主动建图模块**, 在机器人边走边拍的场景下实时构建用于导航的**高质量语义拓扑图**.

### 特性一览

| 维度 | **online_mapper (v2.4.0)** |
|---|---|
| 时序 | 流式, 逐帧决策 (`process_frame` + `finalize`) |
| 几何前端 | **VGGT-1B** 单次推理同时输出 depth / pose / dense point cloud |
| VO | 复用 VGGT pose, 零额外推理 |
| 占据栅格 | VGGT dense point map 直填 (替代 1D ray-cast) |
| 关键帧 | VPR + 累积位移 + 累积旋转 + 信息增益 + 路口 + 语义白名单 |
| 闭环 | 全局 VPR + 几何验证, 每帧检测, auto-tune |
| 节点命名 | **结构化** `NodeName(category, organization, nearby_plates, ...)`, 多帧投票 + 二次验证 + 类别白名单 + 防串扰 |
| 显示拼接 | `category·organization`, 例如 `前台·DEEPROUTE.AI` (不再有 `DEEPROUTE.AI前台` 粘连) |
| cam→neighbor 匹配 | 视觉 CLS Hungarian + 几何方向先验 (`cos(robot_ang - cam_ang)`, α=0.2) |
| 接入方式 | CLI 一次性跑 (`run_online_map.py`) **或** WebSocket 流式建图模式 (`ws_proxy_with_memory.py` 的 mapping 模式) |
| 输出 | `merged_labeled_data/` schema + 结构化命名字段 + `scene_graph.json` / `pose_graph.json` / `metrics.json` / `online_mapping_log.jsonl` + 可视化 PNG (`viz/visualize.py`) |

### v2.3.0 测试集 (memory_test_data 49 帧, GPU0 L40 + Qwen3.5-9B vLLM)

```
打印区 ─── 前台 ─── 强电井·NEUMANN ─── 关爱室 ─── 前台·DEEPROUTE.AI
```

- 5 nodes / 4 edges / 8 next_positions / 2 loop closures
- VGGT 加载 ~10s (一次性), 滑窗 4 帧推理 ~230 ms/frame
- 显存峰值 ~8 GB
- 全部节点结构化双语命名, 零幻觉, 零跨节点串扰

---

## 2. 架构总览

按 **几何 / 拓扑 / 语义** 三层解耦, 由 `OnlineMapperCore` 编排.

```
                      OnlineMapperCore (core/online_mapper_core.py)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
  ┌─────▼──────┐           ┌──────▼──────┐           ┌──────▼──────────────┐
  │ Geometry   │           │ Topology    │           │ Semantics           │
  │ (几何)     │           │ (拓扑)      │           │ (语义)              │
  ├────────────┤           ├─────────────┤           ├─────────────────────┤
  │ VGGTBackend│ ◄─单例    │ KeyframeSel │           │ OpenSetDetector     │
  │ ├VGGTDepth │           │ LoopCloser  │           │ DoorPlateTracker    │
  │ ├VGGTVO    │ ─零推理   │ TopoGraph   │           │ MultiFrameVoter     │
  │ │  (复用)  │           │ ConnBuilder │ ─几何先验 │ QwenVerifier        │
  │ ├Occupancy │ ◄─dense   │ ConnBuilder │ ─几何先验 │ NodeCategoryClf     │
  │ │  pointcl │           │ JunctionDet │           │ ColocationMerger    │
  │ │  直填    │           │ FrontierNBV │           │ NodeName (结构化)   │
  │ └PoseGraph │           │             │           │ NameDeduplicator    │
  │  (DA-V2/   │           │             │           │ SceneGraph          │
  │   ORB 旧后  │           │             │           │                     │
  │   端可切回)│           │             │           │                     │
  └────────────┘           └─────────────┘           └─────────────────────┘
        │                         │                         │
        └────────── pretrained/ + third_party/vggt_space/ ────┘
                   (vggt-1b, depth-anything-v2-small-hf, grounding-dino-base, ...)
```

### 2.1 文件树

```
online_mapper/
├── config.py                            全局配置 OnlineMapperConfig
├── run_online_map.py                    CLI 入口
├── README.md / RESULTS.md / DESIGN.md   说明 / 历史
├── core/
│   ├── online_mapper_core.py           ⭐ 主编排器 (~870 行)
│   └── stream_loader.py                 流式帧加载
├── geometry/
│   ├── vggt_backend.py                 ⭐ VGGT-1B 单例 + 滑窗封装 (NEW v2.2)
│   ├── depth_estimator.py              DA-V2 + VGGTDepthEstimator + 工厂
│   ├── visual_odometry.py              MonoVO + VGGTVisualOdometry + 工厂
│   ├── occupancy.py                    OccupancyGrid (1D ray-cast + dense 点云直填)
│   ├── pose_graph.py                   scipy LM pose graph
│   └── junction_detector.py            4-camera depth 路口判定
├── topology/
│   ├── keyframe_selector.py            多触发关键帧选择
│   ├── loop_closure.py                 auto-tune + ORB 几何验证
│   ├── connection_builder.py           next_positions (几何方向先验)
│   └── graph.py                        TopoGraph / TopoNode
├── semantics/
│   ├── open_set_detector.py            Grounding-DINO 封装
│   ├── door_plate_tracker.py           门牌多帧代表帧选择
│   ├── hallucination_filter.py         STRICT prompt + QwenVerifier + MultiFrameVoter + NameDedup
│   ├── node_category.py                节点类别分类器 + CN/EN 映射
│   ├── node_naming.py                  ⭐ 结构化命名 NodeName (NEW v2.3)
│   ├── colocation_merger.py            同位置节点合并 (用 NodeName.merge_names)
│   └── scene_graph.py                  层次场景图
└── io/
    ├── __init__.py
    └── merged_data_writer.py           输出 merged_labeled_data + 结构化字段

third_party/vggt_space/                  VGGT 源码 (从 HF Space facebook/vggt 下载)
pretrained/                              所有模型权重 (.gitignore)
├── vggt-1b/                            facebook/VGGT-1B (model.safetensors 5.0G)
├── depth-anything-v2-small-hf/         backup/legacy depth backend
├── grounding-dino-base/                IDEA-Research/grounding-dino-base
├── dinov3_vitb16.safetensors           VPR backbone (NodeDistanceEstimator)
└── yolov8n.pt                          辅助
```

---

## 3. 数据流 / 主循环

### 3.1 高层 pipeline (`OnlineMapperCore.run()` core/online_mapper_core.py:142)

```
StreamLoader yields frame  (4 cameras + timestamp + frame_idx 0..N)
        │
        ▼
┌────────────────────────────────────────────────────────────────────┐
│ 每帧主循环                                                          │
├────────────────────────────────────────────────────────────────────┤
│  1. depth.estimate(camera_1)                                       │
│     → VGGTBackend 滑窗推理 (默认 window=4)                         │
│       cache: last_depth, last_extri, prev_extri,                   │
│              last_world_points, last_points_camera                 │
│  2. _vo_motion(camera_1, depth)                                    │
│     → VGGTVisualOdometry 读取 last/prev_extri 计算                 │
│       (dtrans_m, dyaw_rad), 零额外推理                             │
│  3. 累积 robot pose (x, y, theta)                                  │
│  4. occ.integrate_pointcloud(last_points_camera, robot_pose)       │
│     → 高度过滤 + 稀疏采样 + 投影到 mapper world + 标 OCC/FREE      │
│  5. vpr.extract_camera_features(4 cams)                            │
│  6. loop_closer.detect(feats, node_features)                       │
│     → 全局 top-k 候选 + ORB 几何验证 + auto-tune 阈值              │
│  7. _scan_door_plates(frame, fidx)                                 │
│     → GD 检测每帧 → 严格 prompt → Qwen 二次验证 →                  │
│       voter.add(NameVote) + door_tracker.add(PlateObservation)     │
│  8. keyframe_selector.should_trigger?                              │
│     是 → 进入 keyframe creation:                                   │
│       8a. junction_detector.classify(4 cams)                       │
│       8b. 选 confirmed plate (functional > brand 优先)             │
│       8c. namer.describe_scene + verifier.verify_scene             │
│       8d. category_clf.classify(plate, scene, junction, gd_lm)     │
│       8e. 若 ACCEPTED:                                             │
│            创建 TopoNode                                           │
│            node.name_struct = NodeName(category=..., category_en=) │
│            或若 SHOP: NodeName(organization=brand_text)            │
│            收集 GD landmarks 进 nearby_landmarks                   │
│            添加 spatial / loop edges                               │
└────────────────────────────────────────────────────────────────────┘
        │
        ▼ (流结束)
┌────────────────────────────────────────────────────────────────────┐
│ _finalize() — 终结化阶段                                            │
├────────────────────────────────────────────────────────────────────┤
│  1. _create_door_plate_nodes() (两阶段)                            │
│     第一遍: functional/landmark plate (强电井, 关爱室) 创建独立 node│
│     第二遍: brand-like plate (DEEPROUTE.AI, NEUMANN) attach 到帧距 │
│            ≤12 且 category 非空的 functional node 作 organization  │
│            attach 时仅重定位 timestamp+cameras (display 层),       │
│            不动 frame_idx/pose, 避免 coloc 误合并                  │
│  2. ColocationMerger.merge()                                       │
│     用 NodeName.merge_names 融合 anchor + other (替代字符串拼接)   │
│  3. _rebuild_topology_neighbors_spatial(k_spatial=1, k_temporal=1) │
│     spatial KNN + temporal KNN 重建邻接                            │
│  4. _generate_names() 优先从 name_struct.display_cn() 渲染         │
│  5. NameDeduplicator.resolve()                                     │
│     按 (category, organization) 元组分组, VPR 高相似合并,          │
│     其余写 instance_suffix=_N                                      │
│  6. writer.write_node() → merged_labeled_data/<id>/                │
│     输出 self_position + 结构化字段                                │
│  7. ConnectionBuilder.build_for_node(pose_graph=...)               │
│     - GroundedPointer 在 4 相机上找 "通道正中间位置" (Y-fix)       │
│     - DINOv3 CLS feature 算 cam_crop_features                      │
│     - 走廊中间帧 corridor_features (visual 路径)                   │
│     - sim_matrix + 几何方向先验 (cos(robot_ang - cam_ang))         │
│       ALPHA=0.6, 反向硬惩罚 (cos<-0.3 → -1.0)                      │
│     - Hungarian 匹配 → cam ↔ nb 1-to-1                             │
│  8. 写 scene_graph.json / pose_graph.json /                        │
│     online_mapping_log.jsonl / metrics.json                        │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 关键调用链

```
OnlineMapperCore.__init__
  ├─ build_depth_estimator(cfg) → VGGTDepthEstimator (depth_backend="vggt")
  │     └─ VGGTBackend.get(model_path, device, dtype) (单例)
  ├─ build_visual_odometry(cfg, self.depth) → VGGTVisualOdometry(depth_estimator)
  ├─ MonoVO 仍可用 (vo_backend="orb")
  ├─ OpenSetDetector(cfg) → grounding-dino-base
  ├─ AutoLandmarkNamer(use_qwen=True, gpu=cfg.qwen_gpu) → vLLM client
  ├─ NodeCategoryClassifier()
  └─ MergedDataWriter(cfg.output_dir)
```

---

## 4. Geometry 层 (v2.2.0+)

### 4.1 VGGTBackend (geometry/vggt_backend.py)

进程内单例, 懒加载 `pretrained/vggt-1b/model.pt`. 接口:

- `VGGTBackend.get(model_path, device, dtype="bf16") → 单例`
- `infer_bgr_list(bgr_list) → dict`
  输出 (按帧拆 list):
  ```
  depth         : list[HxW float] (米, VGGT 自洽尺度)
  depth_conf    : list[HxW float] (expp1 激活, ≥1.0)
  world_points  : list[HxWx3 float] (VGGT-world frame, 即窗口首帧 cam frame)
  extri         : list[3x4 float] (cam-from-world, X_cam = R*X_w + T)
  intri         : list[3x3 float]
  ```

`VGGTSlidingWindow(backend, window_size=4)`: 维护 BGR ring buffer 提供时序上下文.
- `push_and_infer(bgr) → 返回最新帧 + prev (倒数第二帧, 同坐标系)`
- `infer_stateless(bgr) → 单帧推理, 不入栈 (junction_detector 旁路用)`

### 4.2 DepthEstimator (geometry/depth_estimator.py)

工厂 `build_depth_estimator(cfg)` 根据 `cfg.depth_backend` 返回:
- `"da_v2"` → 旧 `DepthEstimator` (transformers pipeline + Depth-Anything-V2-Small + 伪 metric 归一化), 用于回退
- `"vggt"` (默认) → `VGGTDepthEstimator` (维护 ring buffer, 缓存 last/prev_extri, last_points_camera)

接口契约 (两者一致):
- `.available: bool`
- `.estimate(bgr) → HxW float (米)`
- `.estimate_stateless(bgr) → HxW float (单帧, 不污染滑窗)`

VGGTDepthEstimator 额外缓存:
- `last_extri`, `last_intri`, `prev_extri`, `prev_intri`
- `last_depth_conf`
- `last_world_points` (VGGT-world frame)
- `last_points_camera` (转到 last camera frame, 给 Occupancy 用)

### 4.3 VisualOdometry (geometry/visual_odometry.py)

工厂 `build_visual_odometry(cfg, depth_estimator)` 返回:
- `"orb"` → `MonoVO` (ORB + EssentialMatrix + recoverPose, 单帧 ~100ms)
- `"vggt"` (默认) → `VGGTVisualOdometry(depth_estimator)` 复用 VGGT extrinsics

VGGTVisualOdometry.estimate(bgr, depth_map) 算法:
```
last_extri, prev_extri 来自同一次 VGGT 推理 (同坐标系)
R_curr = last_extri[:3,:3]; T_curr = last_extri[:3,3]
R_prev = prev_extri[:3,:3]; T_prev = prev_extri[:3,3]
C_curr = -R_curr.T @ T_curr   # 相机在世界系位置
C_prev = -R_prev.T @ T_prev
dtrans = ||C_curr - C_prev||
R_rel = R_curr @ R_prev.T     # cam_curr_from_cam_prev
dyaw = atan2(R_rel[0,2], R_rel[2,2])  # 绕 y 轴 (OpenCV camera frame)
```

实测: 49 帧 ORB 4.87s, VGGT 0.004s (≈1200×).

### 4.4 OccupancyGrid (geometry/occupancy.py)

支持两种集成:
- `integrate(robot_pose, depth_row, fov)` (legacy 1D ray-cast, DA-V2 路径)
- **`integrate_pointcloud(points_camera, robot_x, robot_y, robot_theta, conf)`** (v2.2.0 新增)
  - 输入: HxWx3 camera frame 点云 + 当前机器人位姿 + 置信度
  - 流程: conf 过滤 (默认 ≥1.0) → z 范围过滤 ([0.05, 10] m) → 高度过滤 ([-1.5, 1.5] m) →
    随机稀疏采样 (默认 6000 点) → camera frame 转 robot local (forward=z, left=-x) →
    旋转 robot_theta + 平移 → 标 OCC + 沿 robot→OCC 射线等距采样标 FREE
  - 比 1D ray-cast 信息量约 3× (28 free / 588 occ vs 8 / 207)

### 4.5 PoseGraph / JunctionDetector

未变. JunctionDetector 改用 `depth.estimate_stateless(img)` 隔离 4 个相机的单帧推理, 避免污染 VGGT 滑窗.

---

## 5. Topology 层

### 5.1 KeyframeSelector (topology/keyframe_selector.py)

未变. 多触发条件 OR:
- `vpr_dissim_threshold` (默认 0.50)
- `accumulated_translation` (默认 1.5 m)
- `accumulated_rotation` (默认 0.6 rad)
- `info_gain_threshold` (默认 0.05)
- `min_keyframe_frame_interval` (默认 3)

### 5.2 LoopCloser (topology/loop_closure.py)

每帧 detect, top-k=5, ORB 几何验证 (`min_inliers=15`). auto-tune 阈值 (取连续不命中时下调).

### 5.3 ConnectionBuilder (topology/connection_builder.py) ⭐ v2.3.0 重写匹配

子类化 `online_mapper.topology.AutoSubImageExtractor`, 在 Hungarian 匹配后增加 (1) 阈值过滤 (2) 几何方向先验.

#### 5.3.1 几何方向先验

修复纯视觉匹配在线性走廊场景下的 cam↔neighbor 错配:

```python
cam_angles = {camera_1: 0, camera_2: -π/2, camera_3: π, camera_4: π/2}
for (cam_id, nb_id):
    nb_pose = neighbor's (x, y, theta)
    dx, dy = nb.x - my.x, nb.y - my.y
    world_ang = atan2(dy, dx)
    robot_ang = wrap(world_ang - my.theta)   # 邻居在机器人本体系下的角度
    diff = wrap(robot_ang - cam_angles[cam_id])
    score = cos(diff)   # 1.0 = 完美对齐, -1 = 反向

final_sim_matrix = visual_sim + 0.6 * angular_score
if angular_score < -0.3: sim -= 1.0  # 反向相机硬惩罚
```

要求: `pose_graph` 通过 `build_for_node(..., pose_graph=...)` 传入, 每个 node 携带 `(x, y, theta)`.

### 5.4 TopoGraph / spatial-KNN 邻接重建

`_rebuild_topology_neighbors_spatial(k_spatial=1, k_temporal=1)` 在 finalize 阶段清空所有边后, 用空间 KNN ∪ 时间 KNN 重建. 防止 keyframe 链 + door-plate 临时边污染最终拓扑.

---

## 6. Semantics 层 (v2.3.0 重写命名)

### 6.1 OpenSetDetector (semantics/open_set_detector.py)

`grounding-dino-base` 封装, 支持文本 query 检测. 默认 query 列表:
```
door plate, room number sign, printer, trash can, white chair, stool,
elevator, fire extinguisher, potted plant, vending machine, sofa, table, monitor
```

### 6.2 DoorPlateTracker / MultiFrameVoter / QwenVerifier

未变, 三层防幻觉:
1. STRICT prompt (要求 Qwen 输出 confidence, 不确定返回 false)
2. QwenVerifier 用整张相机图二次问 "图中是否真有文字 X?"
3. MultiFrameVoter: ≥2 distinct frames, 或同帧 ≥2 cameras, 或单帧白名单 fast-pass; 还做子串合并 `EUMANN→NEUMANN`.

### 6.3 NodeCategoryClassifier (semantics/node_category.py)

未变. 决策树: ROOM_NUMBERED > ROOM_NAMED > FUNCTION_AREA > LANDMARK_FACILITY > 通用 X室 > SHOP > JUNCTION > REJECT.

### 6.4 NodeName 结构化命名 ⭐ NEW v2.3.0 (semantics/node_naming.py)

#### 6.4.1 数据结构

```python
@dataclass
class NodeName:
    category: str = ""              # 主类型 (中文 canonical), 来自 NodeCategoryClassifier
    category_en: str = ""
    organization: str = ""          # 主关联实体 (品牌/门牌主名), 原文
    nearby_plates: list = []        # 同 node 看到的其他门牌
    nearby_landmarks: list = []     # GD 检测到的物体
    instance_suffix: str = ""       # 全局重名后缀

    def display_cn(self) -> str:
        if self.organization and self.category and self.organization != self.category:
            base = f"{self.category}·{self.organization}"  # 中点分隔
        else:
            base = self.category or self.organization
        return f"{base}{self.instance_suffix}"

    def dedup_key(self) -> tuple:
        return (self.category, self.organization)  # 全局唯一性键
```

#### 6.4.2 organization 选择

`select_organization(plate_obs, category)` 评分:
- brand-like (Latin 大写起头) +100
- camera_1 (前向) +30
- bbox 面积 +0..20
- 投票次数 +0..20

返回 (organization, nearby_plates).

#### 6.4.3 merge_names(anchor, other)

替代旧的字符串拼接 `_combined_name`:
- category 取 `_SEMANTIC_RANK` 更高一方 (功能区/房间 > SHOP)
- organization 优先 brand-like
- nearby_plates / nearby_landmarks 取并集 (去重)

#### 6.4.4 全局唯一性

`NameDeduplicator` 改为按 `name_struct.dedup_key()` 元组分组. 同 `(category, organization)` 算重复:
- VPR 高相似 → 合并 (alias)
- 否则写 `instance_suffix=_2/_3...`, `display_cn()` 自动渲染

### 6.5 ColocationMerger (semantics/colocation_merger.py)

`_CATEGORY_RANK` 调整: 功能区/房间 > SHOP (SHOP 7→3, FUNCTION_AREA 4→5).
SHOP 不再抢占 anchor, 仅作 organization 附加.

`_combined_name` 改为调用 `NodeName.merge_names`:
```python
a = anchor.name_struct
b = other.name_struct
merge_names(a, b)
return a.display_cn(), a.display_en()
```

### 6.6 门牌两阶段归属 (core/online_mapper_core.py:_create_door_plate_nodes)

修复 v2.2.0 的 `EUMANN关爱室` 串扰 bug.

```
第一遍: functional/landmark plate (强电井, 关爱室)
        创建独立 door-plate node, 跳过 brand-like SHOP

第二遍: brand-like plate (DEEPROUTE.AI, NEUMANN)
        for each plate:
          找帧距 ≤12 且 category 非空的 functional/room node
          if 找到:
              attach 为 organization (若已有 brand: 比较投票数, 高 1.5x 才替换)
              旧 organization 进 nearby_plates
              重定位 timestamp + cameras 到 brand best frame (display 层)
              不动 frame_idx + pose (避免 coloc 误合并)
          else:
              创建 standalone SHOP node
```

keyframe plate 选择优先级也改: functional CJK > brand-like Latin (避免 keyframe 因先看到 NEUMANN 就被误标 SHOP).

### 6.7 SceneGraph (semantics/scene_graph.py)

未变. 层次化 floor → room → object_id 结构.

---

## 7. 输出 schema

### 7.1 目录结构

```
output_dir/                                  (cfg.output_dir, 默认 online_mapper/output/merged_labeled_data)
├── 1/                                       node_id (按 next_node_id 顺序)
│   ├── 1770097720_camera_1.jpg              4 路相机原图 (display 帧)
│   ├── 1770097720_camera_2.jpg
│   ├── 1770097720_camera_3.jpg
│   ├── 1770097720_camera_4.jpg
│   ├── crops/                               next_position 子图 (node 自身相机帧)
│   │   ├── 1770097770_camera_3__1__big__...jpg
│   │   ├── 1770097770_camera_3__1__mid__...jpg
│   │   └── 1770097770_camera_3__1__small__...jpg
│   └── node_position_info.json              ⭐ 节点元数据
├── 2/ ...
└── ...

# 顶层 (output_dir 同级)
scene_graph.json                            层次场景图
pose_graph.json                             pose graph nodes + edges
online_mapping_log.jsonl                    每帧决策日志
metrics.json                                总指标
```

### 7.2 node_position_info.json schema (v2.3.0)

```json
{
  "self_position": {
    "position_id": "5",
    "position_name": "前台·DEEPROUTE.AI",
    "position_name_eng": "Reception · DEEPROUTE.AI",
    "category": "前台",
    "category_eng": "Reception",
    "organization": "DEEPROUTE.AI",
    "nearby_plates": [],
    "nearby_landmarks": ["white chair", "table"],
    "instance_suffix": "",
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
      "big_box": "0.515,0.311,0.785,0.648",
      "mid_box": "...",
      "small_box": "...",
      "pixel_box": "",
      "crop_image_path": "crops/1770097829_camera_1__7__big__...jpg",
      "crop_image_paths": {"big": "...", "mid": "...", "small": "..."},
      "position_name_eng": "Care Room",
      "landmark_name_eng": "green wall"
    }
  ]
}
```

字段说明:
- `position_name` / `position_name_eng`: display_cn / display_en, 向后兼容旧消费者
- `category` / `category_eng`: 主类型 (前台 / 强电井 / ...), v2.3.0 新增
- `organization`: 品牌/门牌主名, v2.3.0 新增
- `nearby_plates`: 同 node 其他门牌, debug 用
- `nearby_landmarks`: GD 检测物体, debug 用
- `instance_suffix`: 全局重名后缀 (`_2`, `_3`, ...), 多栋楼场景

### 7.3 metrics.json 部分新字段 (v2.3.0)

- `kf_accepted_by_category`: 含 SHOP 计数
- `plate_attached_to_keyframe`: 第二阶段 brand attach 次数
- `plate_drops_*`: 各类 drop 计数

VGGT runtime breakdown:
```json
"runtime_s": {
  "depth": 13.78,
  "vpr": 20.54,
  "detect": 6.89,
  "vo": 0.0036,
  "name": 0.0,
  "total": 206.46
}
```

---

## 8. 配置项 (online_mapper/config.py)

```python
@dataclass
class OnlineMapperConfig:
    # IO
    input_dir: str = "memory_test_data"
    output_dir: str = "online_mapper/output/merged_labeled_data"
    vpr_config_path: str = "deploy/vpr_config.yaml"

    # 关键帧
    vpr_dissim_threshold: float = 0.50
    accumulated_translation: float = 1.5
    accumulated_rotation: float = 0.6
    info_gain_threshold: float = 0.05

    # 闭环
    loop_closure_min_gap: int = 8
    loop_closure_vpr_threshold: float = 0.78
    loop_closure_top_k: int = 5
    loop_closure_geom_verify: bool = True
    loop_closure_min_inliers: int = 15

    # frontier
    nbv_info_weight: float = 1.0
    nbv_cost_weight: float = 0.3
    nbv_semantic_weight: float = 0.5

    # depth (v2.2.0+)
    depth_model_id: str = "pretrained/depth-anything-v2-small-hf"
    depth_device: str = "cuda:0"
    enable_depth: bool = True
    depth_backend: str = "vggt"             # "da_v2" | "vggt"
    vggt_model_path: str = "pretrained/vggt-1b/model.pt"
    vggt_window_size: int = 4
    vggt_dtype: str = "bf16"                # bf16 | fp16

    # semantics
    enable_grounding_dino: bool = True
    grounding_dino_model_id: str = "pretrained/grounding-dino-base"
    enable_qwen_naming: bool = True
    qwen_base_url: str = "http://localhost:8199/v1"
    qwen_gpu: str = "1"
    enable_door_plate_detection: bool = True
    door_plate_min_score: float = 0.30
    enable_real_connections: bool = True
    connection_sim_threshold: float = 0.40

    # VO (v2.2.0+)
    enable_real_vo: bool = True
    vo_backend: str = "vggt"                # "orb" | "vggt"

    # occupancy (v2.2.0+)
    grid_resolution: float = 0.2
    grid_size: int = 200
    occ_backend: str = "vggt"               # "depth_row" | "vggt"

    start_id: int = 1
    min_keyframe_frame_interval: int = 3
```

---

## 9. 启动方式

### 9.1 准备 (一次性)

```bash
# 1. clone vggt 源码到 third_party (从 HF Space, 因为 GitHub clone 可能不通)
mkdir -p third_party
huggingface-cli download facebook/vggt --repo-type space \
  --local-dir third_party/vggt_space

# 2. 下载 VGGT-1B 权重
huggingface-cli download facebook/VGGT-1B --local-dir pretrained/vggt-1b

# 3. 下载备用 depth-anything / grounding-dino
huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf \
  --local-dir pretrained/depth-anything-v2-small-hf
huggingface-cli download IDEA-Research/grounding-dino-base \
  --local-dir pretrained/grounding-dino-base

# 4. 启 Qwen3.5-9B vLLM (GPU 1, 端口 8199, 命名 / 兜底打点)
./deploy/start_qwen_vllm.sh 1 8199

# 5. 启 Qwen3.5-0.8B vLLM (GPU 0, 端口 8198, 意图分类 + 路径叙述, 仅 ws_proxy nav 分支使用)
./deploy/start_qwen08_vllm.sh 0 8198
```

### 9.2 端到端建图 (CLI)

```bash
conda activate internvla
cd MemoryNav
CUDA_VISIBLE_DEVICES=0 python online_mapper/run_online_map.py \
  --input memory_test_data \
  --output online_mapper/output/merged_labeled_data \
  --log_level INFO
```

**注意**: 不要加 `--no_grounding_dino`, 否则门牌检测路径完全不跑, 不会有 SHOP / 关爱室 等 node.

### 9.2.x WebSocket 建图模式 (双模式 ws_proxy)

`deploy/ws_proxy_with_memory.py` 同时承载**导航 (nav)** 和**建图 (mapping)** 两种模式, 通过 `session_state['mode']` 路由。所有请求保持统一形状 `{id, task, pts, images}`, 由 `task` 字段驱动模式切换:

- 默认 `mode='nav'`, 走记忆导航三层策略 + 意图分类路由 (导航 / 询问位置 / 要求指路).
- 发送 `task="mapping"` 进入建图模式: 首帧服务端自动创建独立 `MappingSession`, 之后每帧喂入 `OnlineMapperCore.process_frame`.
- 发送 `task="stop_mapping"` 触发 `finalize` + 可视化, 返回 summary, 切回 nav (请求仍带 images, 服务端不消费).
- 当 client 在 mapping 模式下发送其他 task 值 (包括导航指令), 服务端自动 `finalize` 当前 session 然后切回 nav.
- `{"command": "mapping_status"}` 查询当前 session 进度 (不驱动模式切换, 仅查询).

关键实现要点 (踩过的坑):

- `MappingSession.__init__` / `feed_raw` / `finalize` 都通过 `asyncio.to_thread` 走线程池, 避免 CPU-heavy 同步调用阻塞 asyncio 事件循环造成 ws ping 超时 (`1011 keepalive timeout`).
- `feed_raw(camera_b64)` 直接 `base64.b64decode + open(w, 'wb')` 写原 JPEG 字节, **不做解码/重编码**, 保证与 `run_online_map.py` cv2.imread 的像素完全一致 (否则 GD/Qwen 分数漂移会改变门牌检测结果).
- `shared_vpr_extractor` 传入记忆导航的 SelaVPR 实例, mapping session 不重复加载 VPR 模型.
- 产物路径: `deploy/logs/mapping_output/session_{ts}_{client_id}/`, 临时帧 `deploy/logs/mapping_frames/session_*/` finalize 后自动清理.
- 断线 (`websockets.ConnectionClosed`) 会在 `handle_client` 的 `finally` 中自动 finalize 保住数据.

端到端测试:

```bash
# 一个终端启 ws_proxy
cd MemoryNav
python deploy/ws_proxy_with_memory.py

# 另一个终端跑建图客户端 (全量 49 帧)
python tests/test_memory_ws.py --mode mapping
```

测试脚本汇总打印拓扑/关键帧/门牌/runtime 分解 + 产物路径. `--mode nav` 则跑原有记忆导航回放.

### 9.3 切回旧后端 (回归测试)

```python
cfg = OnlineMapperConfig(
    depth_backend="da_v2",   # Depth-Anything-V2-Small
    vo_backend="orb",        # ORB MonoVO
    occ_backend="depth_row", # 1D ray-cast
)
```

或 `git checkout v2.1.0`.

### 9.4 schema 校验

`merged_labeled_data/{node_id}/node_position_info.json` 应包含:
- `self_position`: `position_id` / `position_name` / `position_name_eng` / `camera_1..4`
- `next_positions`: list of `{position_id, camera_name, landmark_name, big_box/mid_box/small_box, crop_image_paths{big,mid,small}, position_name_eng, landmark_name_eng}`

可用 `jq` 快速检查:

```bash
for d in online_mapper/output/merged_labeled_data/*/; do
  jq -e '.self_position.position_id and (.next_positions | type == "array")' $d/node_position_info.json > /dev/null \
    || echo "FAIL $d"
done
```

---

## 10. 迭代历史

### v2.1.0 (基线: DA-V2 + ORB + 1D ray-cast)
- 最早稳定版, online_mapper 三层架构 + 多帧投票 + 类别白名单
- 字符串拼接命名, 例如 `DEEPROUTE.AI前台`
- 详细 r1→r6 metrics 见 `online_mapper/RESULTS.md`

### v2.2.0-alpha (VGGT depth)
- `geometry/vggt_backend.py`: VGGT-1B 单例 + 滑窗
- `VGGTDepthEstimator` 替代 DA-V2, 滑窗 4 帧 ~230 ms/frame
- `cfg.depth_backend = "vggt"` 默认

### v2.2.0-beta (VGGT VO)
- `VGGTVisualOdometry` 复用 `last_extri/prev_extri`, 零额外推理
- `junction_detector` 改用 `estimate_stateless` 隔离 VGGT 滑窗
- VO 耗时 4.87s → 0.004s (≈1200×)

### v2.2.0 (VGGT 占据栅格)
- `OccupancyGrid.integrate_pointcloud` dense 点云直填
- 信息量比 1D ray-cast 提升约 3×

### v2.3.0 (结构化命名)
- `semantics/node_naming.py`: `NodeName` dataclass + `merge_names` + `select_organization` + `resolve_global_uniqueness`
- `colocation_merger._combined_name` 改为结构化合并 (替代字符串拼接)
- 门牌两阶段归属: functional 先建 node, brand 后 attach (修复 EUMANN 串扰)
- brand attach 仅重定位 display 层 (timestamp + cameras), 不动 topology
- ConnectionBuilder 几何方向先验 (cos 相似度 + 反向硬惩罚) 修复 cam↔neighbor 错配
- writer 输出新增结构化字段 (category, organization, nearby_plates, ...)

### Tag 链 (回滚锚点)

```
v2.1.0           DA-V2 + ORB + 1D ray-cast (VGGT 重构前的稳定快照)
v2.2.0-alpha     +VGGT depth
v2.2.0-beta      +VGGT VO (零额外推理)
v2.2.0           +VGGT 占据栅格 (完整 VGGT 几何前端)
v2.3.0           +结构化命名 + 几何方向先验
```

---

## 11. 已知限制 / 未来工作

1. **VGGT 尺度自洽但偏小**: 实测室内场景 49 帧累积位移 ~8 m, ORB MonoVO 仅 1.4 m. 两者都不是真实物理米, 但 VGGT 内部 (depth + pose + occupancy) 一致, 不影响相对几何.
2. **滑窗大小未调优**: 当前 4 帧, 可试 8/12 看精度是否提升 (代价更慢).
3. **VGGT 单帧 stateless 仅用于 junction**: 多视角约束完全没有, 准确度可能略差; 关键帧主路径仍用滑窗.
4. **VLM 命名仍依赖 Qwen3.5-9B vLLM**: 第三方依赖, 未实现纯本地 fallback.
5. **NodeName 全局唯一性后缀仅在 VPR dedup 后追加**: 真正多栋楼场景未实测, 需要更大数据集验证.
6. **几何方向先验依赖 pose_graph 准确性**: 若 VO 漂移大, 方向也会漂; 当前 VGGT VO 在小场景下足够稳.
7. **门牌 ATTACH_GAP=12 帧**: 适合慢速 (~1Hz) 步行轨迹; 高速移动需要按数据缩短.

---

## 12. FAQ

### Q. 没有 强电井 / 关爱室 / DEEPROUTE.AI前台 节点?
A. 检查是否传了 `--no_grounding_dino`. 必须启用 GD + Qwen vLLM, 否则门牌路径完全不跑.

### Q. EUMANN关爱室 这种串扰?
A. v2.3.0 已修复. 检查 git log 是否在 v2.3.0+ 之后.

### Q. 节点命名出现 "DEEPROUTE.AI前台" 字符串拼接?
A. v2.3.0 已改为 `前台·DEEPROUTE.AI` (中点分隔). 旧版本拼接逻辑在 `_combined_name`, 新版改为调用 `NodeName.merge_names`.

### Q. cam ↔ neighbor 方向匹配错乱 (如 前台 → camera_2 应该是 camera_3)?
A. v2.3.0 已加几何方向先验. 必须保证 `pose_graph` 通过 `build_for_node(..., pose_graph=...)` 传入.

### Q. VGGT OOM?
A. 降 `vggt_window_size` (默认 4 → 2), 或切 `depth_backend="da_v2"` 回退到 DA-V2.

### Q. 想看 VGGT vs DA-V2 / VGGT-VO vs ORB 对比?
A. 改 `cfg.depth_backend / vo_backend / occ_backend` 后跑两遍, 对比 `metrics.json` 与 `online_mapping_log.jsonl`.

### Q. VGGT 输出位姿坐标系约定?
A. extrinsics 是 cam-from-world (`X_cam = R*X_w + T`), 见 `vggt/utils/geometry.py:depth_to_world_coords_points` 注释 "OpenCV camera coordinate convention, cam from world".

---

## 文档维护

- 当前版本: **v2.3.0** (2026-04-08)
- 上次更新: 同步 v2.2.0/v2.3.0 全部架构改动
- 维护建议: 每次 minor tag (v2.x.0) 更新本文档对应章节; patch tag (v2.x.y) 可仅更新迭代历史.
