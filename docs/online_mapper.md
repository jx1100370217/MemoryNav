# online_mapper 设计文档

> 代码根: `memory-nav/online_mapper/`
> 第三方源码: `memory-nav/third_party/vggt_space/` (VGGT, .gitignore)
> 模型权重: `memory-nav/pretrained/` (.gitignore, 各模型本地化路径)

---

## 1. 定位与能力

`online_mapper` 是**流式在线主动建图模块**, 在机器人边走边拍的场景下实时构建用于导航的**高质量语义拓扑图**.

### 特性一览

| 维度 | online_mapper |
|---|---|
| 时序 | 流式, 逐帧决策 (`process_frame` + `finalize`) |
| 几何前端 | VGGT-1B 单次推理同时输出 depth / pose / dense point cloud |
| VO | 复用 VGGT pose, 零额外推理 |
| 占据栅格 | VGGT dense point map 直填 (+ 1D ray-cast 兜底) |
| 可通行度 | 基于 VGGT camera-frame 点云估计地面平面, 给 next_position 点修正用 |
| 关键帧 | VPR + 累积位移 + 累积旋转 + 信息增益 + 路口 + 语义白名单 |
| 闭环 | 全局 VPR + ORB 几何验证, 每帧检测, auto-tune 阈值 |
| 节点命名 | 结构化 `NodeName(category, organization, nearby_plates, ...)`, 多帧投票 + 二次验证 + 类别白名单 + 防串扰 |
| 显示拼接 | `category·organization`, 例如 `外卖柜区·EXHIOH` |
| scene describe | 4 相机并行 Qwen 描述 + specificity rank 投票 + 时序共识 (temporal consensus) |
| cam→neighbor 匹配 | 视觉 DINOv3 patch 滑动匹配 Hungarian + 几何方向先验 (`cos(robot_ang - cam_ang)`, 同 segment ALPHA=0.5 / 跨 300s 断档 ALPHA=0) + person-occlusion 惩罚 + traversability 通道修正 (`resolve_crop_point` 推到 walkable segment 中心) |
| 接入方式 | CLI 一次性跑 (`run_online_map.py`) **或** WebSocket 流式建图模式 (`ws_proxy_with_memory.py` 的 mapping 模式) |
| 输出 | `merged_labeled_data/` schema + 结构化命名字段 + `scene_graph.json` / `pose_graph.json` / `metrics.json` / `online_mapping_log.jsonl` / `plate_voter_dump.json` + 可视化 PNG |

### 样例运行 (memory_test_data 281 帧, GPU0 L40 + Qwen3.5-9B vLLM)

园区漫游 281 帧样本, 最终拓扑链:

```
电梯厅 → 前台 → C座 → H座电梯 → A座 → B座入口 → 外卖柜区·EXHIOH → 2号外卖柜
```

- n_nodes = 8, n_edges = 7, n_loop_closures = 13, n_connections = 17, n_named_landmarks = 14
- n_keyframes_triggered = 44; kf_accepted_by_category = {building_landmark: 8, function_area: 6, landmark_facility: 4}, kf_rejected = 26
- plate_voter: 24 confirmed / 23 rejected (含 `13号楼` / `1号楼` / `D座` / `D栋` 等 BUILDING_LANDMARK 幻觉被 reject)
- runtime_s.total ≈ 1356s; depth ≈ 61s, vpr ≈ 113s, detect ≈ 34s, plate_scan ≈ 953s, vo ≈ 0.02s

---

## 2. 架构总览

按 **几何 / 拓扑 / 语义** 三层解耦, 由 `OnlineMapperCore` 编排.

```
OnlineMapperCore (core/online_mapper_core.py)
├── Geometry        几何前端
│   ├── VGGTBackend + VGGTSlidingWindow  (单例, bf16 滑窗推理)
│   ├── DepthEstimator  (VGGT / DA-V2 工厂)
│   ├── VisualOdometry  (VGGT / ORB 工厂, VGGT 零额外推理)
│   ├── OccupancyGrid   (dense 点云直填 / 1D ray-cast)
│   ├── PoseGraph       (含 motion_theta)
│   ├── JunctionDetector
│   └── Traversability  (地面平面可通行度)
├── Topology        拓扑
│   ├── KeyframeSelector
│   ├── LoopCloser      (VPR top-k + ORB 几何验证 + auto-tune)
│   ├── ConnectionBuilder (next_positions, 几何先验 + 可通行度 + 人遮挡)
│   ├── AutoSubImageExtractor  (DINOv3 Hungarian 匹配基类)
│   └── TopoGraph / TopoNode
└── Semantics       语义
    ├── OpenSetDetector  (Grounding-DINO-base)
    ├── DoorPlateTracker / MultiFrameVoter / HallucinationFilter / QwenVerifier
    ├── NodeCategoryClassifier  (决策树 + canonical 归一 + 白名单)
    ├── NodeName + NameDeduplicator
    ├── ColocationMerger  (category 守卫 + anchor tie-break)
    └── SceneGraph
```

### 2.1 文件树

```
online_mapper/
├── config.py                            全局配置 OnlineMapperConfig
├── run_online_map.py                    CLI 入口
├── README.md / RESULTS.md / DESIGN.md   说明 / 历史
├── core/
│   ├── online_mapper_core.py            主编排器
│   └── stream_loader.py                 流式帧加载
├── geometry/
│   ├── vggt_backend.py                  VGGT-1B 单例 + 滑窗封装
│   ├── depth_estimator.py               DA-V2 + VGGTDepthEstimator + 工厂
│   ├── visual_odometry.py               MonoVO + VGGTVisualOdometry + 工厂
│   ├── occupancy.py                     OccupancyGrid (dense 点云直填 + 1D ray-cast 兜底)
│   ├── pose_graph.py                    scipy LM pose graph (含 motion_theta 字段)
│   ├── junction_detector.py             4-camera depth 路口判定
│   └── traversability.py                VGGT 点云地面平面 → 像素级可通行度图
├── topology/
│   ├── keyframe_selector.py             多触发关键帧选择
│   ├── loop_closure.py                  auto-tune + ORB 几何验证
│   ├── connection_builder.py            next_positions 生成 (方向先验 / 可通行度 / 人遮挡)
│   ├── auto_sub_image_extractor.py      DINOv3 视觉 Hungarian 匹配基类
│   └── graph.py                         TopoGraph / TopoNode
├── semantics/
│   ├── open_set_detector.py             Grounding-DINO 封装
│   ├── door_plate_tracker.py            门牌多帧代表帧选择
│   ├── hallucination_filter.py          STRICT prompt + QwenVerifier + MultiFrameVoter
│   ├── node_category.py                 节点类别分类器 + canonical 归一 + 白名单
│   ├── node_naming.py                   结构化命名 NodeName + NameDeduplicator
│   ├── colocation_merger.py             同位置节点合并 (category 守卫 + anchor tie-break)
│   └── scene_graph.py                   层次场景图
└── io/
    ├── __init__.py
    └── merged_data_writer.py            输出 merged_labeled_data + 结构化字段

third_party/vggt_space/                  VGGT 源码 (从 HF Space facebook/vggt 下载)
pretrained/                              所有模型权重 (.gitignore)
├── vggt-1b/                             facebook/VGGT-1B (model.safetensors 5.0G)
├── depth-anything-v2-small-hf/          备用 depth backend
├── grounding-dino-base/                 IDEA-Research/grounding-dino-base
├── dinov3_vitb16.safetensors            VPR / sub-image matching backbone
└── yolov8n.pt                           辅助
```

---

## 3. 数据流 / 主循环

### 3.1 高层 pipeline (`OnlineMapperCore.run()` core/online_mapper_core.py)

```
StreamLoader yields frame  (4 cameras + timestamp + frame_idx 0..N)
  │
  ▼ 每帧主循环
  1. depth.estimate(camera_1)
       VGGTBackend 滑窗推理 (window=4), cache last/prev_extri, last_points_camera
  2. vo.estimate(camera_1)
       VGGTVisualOdometry 读 last/prev_extri 算 (dtrans, dyaw), 零额外推理
  3. 累积 robot pose (x, y, theta)
  4. occ.integrate_pointcloud(last_points_camera, robot_pose)
       高度过滤 + 稀疏采样 + world 投影 + 标 OCC/FREE
  5. vpr.extract_camera_features(4 cams)
  6. loop_closer.detect(feats, node_features)
       全局 top-k + ORB 几何验证 + auto-tune
  7. _scan_door_plates(frame, fidx)
       GD 每帧检测 → STRICT prompt → QwenVerifier 二次验证 →
       voter.add(NameVote) + door_tracker.add(PlateObservation)
  8. keyframe_selector.should_trigger?
     是 → keyframe creation:
       8a. junction_detector.classify(4 cams)
       8b. 选 confirmed plate (functional CJK > brand-like Latin)
       8c. ⭐ multi-cam describe_scene (4 cam 并行 Qwen) → 投票:
           candidates 按 specificity rank 排序
             FUNCTION_AREA(0) > LANDMARK_FACILITY(1) > BUILDING_LANDMARK(2) > generic(3)
           ≥2 cam 一致 → 采纳 winner; 4 cam 全不同 → skip 节点
           winner 走 canonical 归一 (电梯厅 / 外卖柜区 ...)
       8d. ⭐ temporal consensus (连续帧确认):
           consensus ≥ 3/4 → 强信号, 立即 verified
           consensus == 2/4 + winner ∈ FUNCTION_AREA canonical 值 (前台 / 打印区
             / 关爱室 / 外卖柜区 / 保安亭 等) → func_area bypass, 立即 verified
             (FUNCTION_AREA 是 Qwen 低幻觉的具体 landmark, 独立场景也可信)
           consensus == 2/4 + junction ∈ {CROSS, T_JUNCTION} → 路口豁免, 立即
             verified (机器人在路口短暂瞥见 landmark 也是真实信号)
           consensus == 2/4 + winner 出现在近 3 帧 `_recent_scene_winners` deque
             → temporal confirmed, verified
           其余 → tentative (scene_describe=None, 本帧不建 node, winner 仍入队
             等后续确认)
           ⚠ LANDMARK_FACILITY (含 '电梯厅') 不参与 bypass — 防 N11 电梯厅_2
             类多相机联合幻觉
       8e. category_clf.classify(plate, scene, junction, gd_lm)
       8f. hallucination_filter.is_confirmed (含 BL 专用门槛)
       8g. 若 ACCEPTED:
             创建 TopoNode, name_struct=NodeName(...),
             收集 GD landmarks, 添加 spatial / loop edges

(流结束) → _finalize()
  1. _create_door_plate_nodes()  两阶段
       第一遍: functional / landmark plate 创建独立 door-plate node
                (node._from_plate_best=True)
       第二遍: brand-like plate attach 到帧距 ≤12 且 category 非空的 functional/room node,
                写 organization, 旧 org 进 nearby_plates
                ⭐ RELOCATE-DISPLAY (仅动 timestamp + cameras):
                  - target 由 keyframe trigger 创建 → relocate 到 brand best-view
                  - target 由 plate best-view 创建 (_from_plate_best) → keep
                  - brand 在别处副标 (brand.pos ≠ target.pos) → keep
  2. _merge_by_canonical_name()  同 base building / 同 canonical name 强制合并
       ⭐ BUILDING_LANDMARK 取 base ('C座入口' / 'C座大堂' → 'C座'),
          LANDMARK_FACILITY 取 canonical
       ⭐ 空间聚类守卫 (3 m): 同 canonical name 但物理位置距离 > 3 m 不合并
          (例: 起点电梯厅 vs H 座旁电梯都叫 '电梯厅' canonical)
       ⭐ anchor 改为 latest frame_idx: 沿路径走过 landmark 时, frame_idx 更大的
          keyframe 一般离 landmark 更近 (plate bbox 更大, cam crop 方向先验和 DINOv3
          视觉 sim 都更稳)
       ⭐ Pass B/C: 复合 plate (H座电梯 / 2号外卖柜 / 11层电梯) 优先重命名 anchor
  3. ColocationMerger.merge()
       ⭐ category mismatch guard: 不跨类合并
       ⭐ anchor tie-break: 取 frame_idx 更小的为 anchor
       NodeName.merge_names 融合 category / organization / plates / landmarks
  4. _rebuild_topology_neighbors_spatial(k_spatial=1, k_temporal=1)
       spatial KNN ∪ temporal KNN 重建邻接, 清空 keyframe/plate 临时边
       ⭐ cross-gap filter: 相邻 keyframe ts > 60s 且无 bridging keyframe → 拒该时间边
  5. _generate_names() 优先用 name_struct.display_cn()
  6. NameDeduplicator.resolve()
       按 dedup_key=(category, organization) 分组, VPR 高相似合并 (alias),
       其余写 instance_suffix=_N
  7. writer.write_node() → merged_labeled_data/<id>/
  8. ConnectionBuilder.build_for_node(pose_graph=...)
       视觉 DINOv3 patch 滑动匹配 + 几何方向先验 + person-occlusion 惩罚 + traversability 校正
       Hungarian 匹配 → cam ↔ nb 1-to-1
  9. 写 scene_graph.json / pose_graph.json /
     online_mapping_log.jsonl / metrics.json / plate_voter_dump.json
```

### 3.2 关键调用链

```
OnlineMapperCore.__init__
  ├─ build_depth_estimator(cfg) → VGGTDepthEstimator (depth_backend="vggt")
  │     └─ VGGTBackend.get(model_path, device, dtype)  (单例)
  ├─ build_visual_odometry(cfg, self.depth) → VGGTVisualOdometry(depth_estimator)
  ├─ OpenSetDetector(cfg) → grounding-dino-base
  ├─ AutoLandmarkNamer(use_qwen=True, gpu=cfg.qwen_gpu) → vLLM client
  ├─ NodeCategoryClassifier()
  ├─ Traversability (懒加载 via compute_traversability_map)
  └─ MergedDataWriter(cfg.output_dir)
```

---

## 4. Geometry 层

### 4.1 VGGTBackend (geometry/vggt_backend.py)

进程内单例, 懒加载 `pretrained/vggt-1b/model.pt`. 接口:

- `VGGTBackend.get(model_path, device, dtype="bf16") → 单例`
- `infer_bgr_list(bgr_list) → dict`
  输出 (按帧拆 list):
  ```
  depth         : list[HxW float] (米, VGGT 自洽尺度)
  depth_conf    : list[HxW float] (expp1 激活, ≥ 1.0)
  world_points  : list[HxWx3 float] (VGGT-world frame, 即窗口首帧 cam frame)
  extri         : list[3x4 float] (cam-from-world, X_cam = R*X_w + T)
  intri         : list[3x3 float]
  ```

`VGGTSlidingWindow(backend, window_size=4)`: 维护 BGR ring buffer 提供时序上下文.
- `push_and_infer(bgr) → 返回最新帧 + prev (倒数第二帧, 同坐标系)`
- `infer_stateless(bgr) → 单帧推理, 不入栈 (junction_detector 旁路用)`

### 4.2 DepthEstimator (geometry/depth_estimator.py)

工厂 `build_depth_estimator(cfg)` 根据 `cfg.depth_backend` 返回:
- `"vggt"` (默认) → `VGGTDepthEstimator` (维护 ring buffer, 缓存 last/prev_extri, last_points_camera)
- `"da_v2"` → 旧 `DepthEstimator` (transformers pipeline + Depth-Anything-V2-Small + 伪 metric 归一化)

接口契约 (两者一致):
- `.available: bool`
- `.estimate(bgr) → HxW float (米)`
- `.estimate_stateless(bgr) → HxW float (单帧, 不污染滑窗)`

VGGTDepthEstimator 额外缓存:
- `last_extri`, `last_intri`, `prev_extri`, `prev_intri`
- `last_depth_conf`
- `last_world_points` (VGGT-world frame)
- `last_points_camera` (转到 last camera frame, 给 Occupancy / Traversability 用)

### 4.3 VisualOdometry (geometry/visual_odometry.py)

工厂 `build_visual_odometry(cfg, depth_estimator)` 返回:
- `"vggt"` (默认) → `VGGTVisualOdometry(depth_estimator)` 复用 VGGT extrinsics
- `"orb"` → `MonoVO` (ORB + EssentialMatrix + recoverPose, 单帧 ~100ms)

VGGTVisualOdometry 读取同一次 VGGT 推理得到的 `last_extri / prev_extri` (同坐标系), 由
`C = -R.T @ T` 算两帧相机中心, `dtrans = ||C_curr - C_prev||`; `R_rel = R_curr @ R_prev.T`,
`dyaw = atan2(R_rel[0,2], R_rel[2,2])` (绕 OpenCV camera frame y 轴).

单帧约 0.004s, 相比 ORB 提速约 1000×.

### 4.4 OccupancyGrid (geometry/occupancy.py)

支持两种集成:
- `integrate(robot_pose, depth_row, fov)`: 1D ray-cast (legacy, DA-V2 路径)
- `integrate_pointcloud(points_camera, robot_x, robot_y, robot_theta, conf)` (默认 VGGT 路径):
  - conf 过滤 (≥ 1.0) → z 范围 [0.05, 10m] → 高度 [-1.5, 1.5m] → 随机稀疏采样 (6000 点)
  - camera frame → robot local (forward = z, left = -x) → 旋转 + 平移
  - 标 OCC + 沿 robot → OCC 射线等距采样标 FREE

### 4.5 PoseGraph (geometry/pose_graph.py)

scipy LM pose graph. `PoseNode` 数据结构含字段 `x, y, theta, motion_theta`.

`motion_theta` 基于相邻 keyframe 的位置增量计算 **motion-based heading**:

```python
motion_theta_i = atan2(y_i - y_{i-1}, x_i - x_{i-1})
```

它是 `ConnectionBuilder` 几何方向先验的主信号, 用来绕开 VGGT yaw 在平移场景下的漂移 (VGGT 自身 `theta` 做 fallback).

### 4.6 Traversability (geometry/traversability.py)

从 VGGT camera-frame 点云 (`last_points_camera`) 估计像素级可通行度, 给 `ConnectionBuilder` 在做 next_position crop 时把 Qwen 指的点修正到通道中央.

**接口**:

- `estimate_ground_y(points_camera, image_bottom_frac=0.33) → float`
  取底部 1/3 画面、z ∈ [0.3, 12m] 的有效点, 返回 y 的 75 分位
  (相机 y-down, ground 为最大 y)

- `compute_traversability_map(points_camera, y_ground=None) → HxW float32`
  逐像素分类:
  - GROUND band: `|Δy - y_ground| ≤ 0.30m` → 1.0
  - Obstacle: `y < y_ground - 0.25m` → 0.0
  - 其余 (无效 depth / 远距) → 0.4 (unknown)

- `validate_point(trav_map, cx, cy, radius)`
  半径窗口内 traversable ≥ 55% **且** obstacle ≤ 25% 才通过.

- `detect_vertical_obstacle_columns(points_camera, min_col_coverage=0.15, bottom_frac=0.5) → bool[W]`
  柱子 / 墙 / 椅背等竖向 obstacle 的逐列 mask. 关键: **只扫底部 50% 画面**, 因为
  室内天花板 (2.5m 以上) 会撑满每列上半画面的 above-ground 像素, 全列扫会把
  所有列误判为 obstacle. 柱子 / 墙 / 椅背的根部都伸到底半画面, 仍能被覆盖.

- ⭐ `resolve_crop_point(trav_map, preferred_cx, target_y_frac=0.48, edge_margin_frac=0.10, max_cx_offset_frac=0.30, points_camera=None) → (cx, row) | None`
  Qwen cx 选通道, walkable segment center 决定精确位置:
  1. cy 固定到 target row (默认 48% 高度, 地面带)
  2. 计算 walkable column mask (target row ± 10 px ∪ 排除边缘 10% ∪ 排除
     `detect_vertical_obstacle_columns` 标定的柱子列)
  3. 提取连续 walkable segment (≤5 px 间隙合并)
  4. 选包含 preferred_cx 的 segment, 否则选 segment 中心最接近 preferred_cx 的
  5. cx 推到 segment 中心 (受 30% 画面宽偏移上限约束)

  设计动机: Qwen 给的点常落在 obstacle 边缘 (绿植墙右沿 / 闸机左沿), 单点
  walkable 但 crop 半径 (~259 px) 内有 obstacle, 视觉中心其实在墙上. 推到
  walkable segment 中心可以让 crop 真正落在通道中央.

- `find_best_traversable_point(trav_map, preferred_cx=None, ...) → (cx, row) | None`
  无 preferred_cx 时 (Qwen 全失败兜底) 取最宽 walkable segment 中心. 同样应用
  柱子列 mask 与边缘 10% 屏蔽.

---

## 5. Topology 层

### 5.1 KeyframeSelector (topology/keyframe_selector.py)

多触发条件 OR:
- `vpr_dissim_threshold` (默认 0.50)
- `accumulated_translation` (默认 1.5 m)
- `accumulated_rotation` (默认 0.6 rad)
- `info_gain_threshold` (默认 0.05)
- `min_keyframe_frame_interval` (默认 3)

路口与语义白名单命中作为额外触发, 路径在 8a / 8e / 8f.

### 5.2 LoopCloser (topology/loop_closure.py)

每帧 detect, top-k = 5, ORB 几何验证 (`min_inliers=15`), auto-tune 阈值 (连续不命中时下调).

### 5.3 ConnectionBuilder (topology/connection_builder.py)

子类化 `AutoSubImageExtractor`, 在视觉 Hungarian 匹配后增加 (1) 阈值过滤 (默认 0.40) (2) 几何方向先验 (同 segment ALPHA=0.5 / 跨 300 s 断档 ALPHA=0) (3) person-occlusion 惩罚 (4) traversability 推到 walkable segment 中心 (5) cx ∈ [15% W, 85% W] 硬约束.

#### 5.3.1 几何方向先验

修复纯视觉匹配在线性走廊场景下的 cam↔neighbor 错配. 使用权威 camera azimuth (`memory_nav/coord_transform.py:_DEFAULT_AZIMUTHS`, 由 `cam/params.yaml` T_ic 推算, 逆时针正, y 轴向左):

```python
cam_angles = {
    camera_1: +39.42°,  camera_2: -35.84°,
    camera_3: -142.04°, camera_4: +143.52°,
}
for (cam_id, nb_id):
    # 时间断档判定: 同 segment vs 跨真断档
    gap_ms = abs(my.timestamp - nb.timestamp)
    if gap_ms > GAP_FUSE_MS:        # 300_000 ms (300s)
        # 跨真断档 (memory_test_data Part1→Part2 间隔 533s) 时 VO 朝向已不可信
        gap_mask[i][j] = True
        alpha[i][j] = 0.0           # 完全靠 visual sim
        continue                     # geo_bonus 不算
    nb_pose = neighbor's (x, y, theta)
    dx, dy = nb.x - my.x, nb.y - my.y
    world_ang = atan2(dy, dx)
    # 优先用 motion_theta (基于相邻 keyframe 位移的 heading),
    # fallback my.theta (VGGT pose 朝向)
    heading = my.motion_theta if my.motion_theta is not None else my.theta
    robot_ang = wrap(world_ang - heading)
    diff = wrap(robot_ang - cam_angles[cam_id])
    score = cos(diff)               # 1.0 = 完美对齐, -1 = 反向
    geo_bonus[i][j] = score
    if score < -0.3: sim -= 1.0     # 反向相机硬惩罚 (仅同 segment)

final_sim_matrix = visual_sim + alpha * geo_bonus
# 同 segment alpha = ALPHA_NORMAL = 0.5  (几何先验主导)
# 跨断档 alpha = 0                       (geo_bonus 清零, visual sim 主导)
```

为什么不用单一 ALPHA: 同 segment 内相邻 keyframe 间隔 30~120 s, VO 朝向相对可靠, 几何先验救得住 cam 错配; 跨真断档时机器人位置/朝向都被 VO 漂移污染, 强行加几何先验反而会让方向倒过来扣分.

阈值 300 s 的依据: 真实数据断档 (memory_test_data Part1→Part2 间隔 533 s) 远高于同 segment 间隔, 把阈值放在中间能精准捕获.

要求: `pose_graph` 通过 `build_for_node(..., pose_graph=...)` 传入, 每个 node 携带 `(x, y, theta, motion_theta, timestamp)`.

#### 5.3.2 Person-occlusion 惩罚

每个候选 cam 图跑 GD 查 `person`, 中心 40% × 40% 区域的 person 覆盖率 > 0.15 时给该 cam 列相似度 `-0.30` (防止邻居刚好走到机器人正前方被抓成 next_position anchor).

#### 5.3.3 Traversability 校正 + cx 硬约束

Qwen 给的原始点 (qx, qy) 过 `compute_traversability_map(points_camera)` + `resolve_crop_point(..., preferred_cx=qx)` (推到 walkable segment 中心模式): 若返回点 cx ∈ [15% W, 85% W] 则用返回点替换; 否则保留 Qwen 原点. 画面最外 15% 列在 VGGT 多视图拼接下常出现"伪可通行"像素, 硬屏蔽掉避免边缘噪声污染 crop.

**Qwen 全失败 fallback**: 4 路 cam 全部失锚时, 改用 `find_best_traversable_point(...)` (无 preferred_cx) 取最宽 walkable segment 中心; 它再失败就退到画面正中.

**几何投影也走 traversability 校验** (Round 7): 当 Qwen `cx ≈ 0.500` 触发居中 fallback 时, 走 `_project_target_to_camera(self_pose, target_pose, cam_id)` 用 pose graph 算 target 在画面里的预期 cx, 然后**同样**走 `resolve_crop_point` 校验通道, 让几何投影点也只挑通道中央; 当走廊上有立柱投影到 target cx 时不至于直接落在柱子上.

### 5.4 AutoSubImageExtractor (topology/auto_sub_image_extractor.py)

`ConnectionBuilder` 的基类, **直接复用** `memory_nav.sub_image_matcher.DINOv3Strategy` 的 `extract_patch_grid` / `adaptive_crop_target_size` / `match_grids` 公共 API, 与记忆导航子图匹配使用同一份实现, 避免重复代码 (commit `be8643c`). 在此基础上提供 4 相机并行 crop 特征 + 走廊中间帧 corridor_features 的匹配管线.

**POINT_PROMPT**: `"前方通道最远处的地面中心点"` (`Qwen35PointGrounder` query)

**pointer backend**: 历史上曾尝试 `qwen / gdino / molmo / gsam2` 多 pointer 切换, 实测下 Qwen 单 pointer + traversability 推中央的组合最稳, 已**只保留 qwen** (`refactor(crop): 只保留 qwen pointer + P0 优化`, `34aa578`). `__init__` 仍接 `**kwargs` 吸收上层传入的 `namer` / `cfg` / `pointer_backend` 参数, 保持向后兼容.

### 5.5 TopoGraph / spatial-KNN 邻接重建

`_rebuild_topology_neighbors_spatial(k_spatial=1, k_temporal=1)` 在 finalize 阶段清空所有边后, 用空间 KNN ∪ 时间 KNN 重建, 防止 keyframe 链 + door-plate 临时边污染最终拓扑.

**cross-gap filter**: 相邻 keyframe 时间差 > 60s 且中间无 bridging keyframe → 拒绝该时间边. 应对数据采集中断流场景 (例如挪动机器人时不断帧, 但恢复后不应直接连回断流前的 keyframe).

---

## 6. Semantics 层

### 6.1 OpenSetDetector (semantics/open_set_detector.py)

`grounding-dino-base` 封装, 支持文本 query 检测. 默认 query 列表:

```
door plate, room number sign, printer, trash can, white chair, stool,
elevator, fire extinguisher, potted plant, vending machine, sofa, table, monitor
```

### 6.2 DoorPlateTracker (semantics/door_plate_tracker.py)

维护每个 plate 的 best observation (max bbox area 的 frame).

### 6.3 MultiFrameVoter + HallucinationFilter (semantics/hallucination_filter.py)

三层防幻觉:
1. STRICT prompt (要求 Qwen 输出 confidence, 不确定返回 false)
2. `QwenVerifier` 用整张相机图二次问 "图中是否真有文字 X?"
3. `MultiFrameVoter`: 多帧 / 多相机投票确认

**确认规则**:
- `MIN_FRAMES = 2`, `MIN_CAMERAS = 2`
- `MIN_MAX_AREA` 通用门槛;
  BUILDING_LANDMARK 单独使用 `_BUILDING_LANDMARK_MIN_AREA = 1200` (户外远距 OCR 面积天然小, 若按通用门槛会全部被拒)
- ⭐ **`_BUILDING_LANDMARK_MIN_VOTES = 4`** BL name 必须累计 vote 记录 ≥ 4 才 confirmed.
  实测结果: BL 幻觉 votes ≤ 3 (如 `13号楼` / `1号楼` / `D座` / `D栋`),
  真实 BL 最小 votes = 4 (`B座`). 阈值设在 4 能把幻觉与真 label 分开.
- ⭐ **BUILDING_LANDMARK 禁用单帧白名单 fast-pass**: 普通白名单命中 (FUNCTION_AREA / LANDMARK_FACILITY) 时单帧即可 verified, 但 `is_bl == True` 时跳过该 fast-pass, 避免 "13号楼" 单帧偷渡.
- 子串合并: `EUMANN → NEUMANN`; 大写 Latin 4-8 字符 edit-distance ≤ 2 模糊聚类.

**`_BUILDING_LANDMARK_RE`**:

```
^([A-Za-z]座(入口|大堂)?|[A-Za-z]栋(入口|大堂)?|\d+号楼|\d+栋)$
```

### 6.4 NodeCategoryClassifier (semantics/node_category.py)

决策树:

```
ROOM_NUMBERED > ROOM_NAMED > FUNCTION_AREA > LANDMARK_FACILITY
> BUILDING_LANDMARK > 通用 X室 > SHOP > JUNCTION > REJECT
```

**Canonical 归一 (词表)**:

- 电梯类: `电梯 / 电梯口 / 电梯厅 / 电梯间` → **电梯厅**
- 外卖柜类: `快递柜 / 快递柜区 / 外卖柜 / 外卖柜区 / 储物柜 / locker / 智能取餐柜` → **外卖柜区**

canonical 归一放在所有白名单判断之前, 保证同义词在 voter / dedup / display 三处看到的是同一个 key.

**Scene describe → BUILDING_LANDMARK 分支**: 当 Qwen `describe_scene` 的候选词匹配 `_BUILDING_LANDMARK_RE` 时, 直接分到 BUILDING_LANDMARK 类别, 不经过 FUNCTION_AREA / LANDMARK_FACILITY 路径.

**白名单**:
- `FUNCTION_AREA_WHITELIST`: 前台 / 打印区 / 休息区 (已移除 "休息" 泛词) / 外卖柜区 / 电梯厅 等
- `LANDMARK_FACILITY_WHITELIST`: 电梯 / 安全出口 / 消防栓 / B座入口 等
- `BUILDING_LANDMARK_PATTERNS`: `[A-Z]座` / `[A-Z]座入口` / `\d+号楼` / `[A-Z]栋` 等

### 6.5 NodeName 结构化命名 (semantics/node_naming.py)

#### 6.5.1 数据结构

```python
@dataclass
class NodeName:
    category: str = ""              # 主类型 (中文 canonical), 来自 NodeCategoryClassifier
    category_en: str = ""
    organization: str = ""          # 主关联实体 (品牌/门牌主名), 原文保留
    nearby_plates: list = []        # 同 node 看到的其他门牌
    nearby_landmarks: list = []     # GD 检测到的物体
    instance_suffix: str = ""       # 全局重名后缀

    def display_cn(self) -> str:
        if self.organization and self.category and self.organization != self.category:
            base = f"{self.category}·{self.organization}"   # 中点分隔
        else:
            base = self.category or self.organization
        return f"{base}{self.instance_suffix}"

    def dedup_key(self) -> tuple:
        return (self.category, self.organization)           # 全局唯一性键
```

#### 6.5.2 organization 选择

`select_organization(plate_obs, category)` 评分:

- brand-like (Latin 大写起头) +100
- camera_1 (前向) +30
- bbox 面积 +0..20
- 投票次数 +0..20

返回 `(organization, nearby_plates)`.

#### 6.5.3 merge_names(anchor, other)

- category 取 `_SEMANTIC_RANK` 更高一方 (功能区/房间 > SHOP)
- organization 优先 brand-like
- nearby_plates / nearby_landmarks 取并集 (去重)

#### 6.5.4 全局唯一性

`NameDeduplicator` 按 `name_struct.dedup_key()` 元组分组. 同 `(category, organization)` 视为重复:
- VPR 高相似 → 合并 (alias)
- 否则写 `instance_suffix = _2 / _3 / ...`, `display_cn()` 自动渲染

### 6.6 _merge_by_canonical_name (core/online_mapper_core.py)

finalize 阶段在 ColocationMerger 之前先做的"同 base 强制合并". 不依赖 VPR / 帧距, 仅按命名 canonical:

- BUILDING_LANDMARK (X 座 / X 栋 / X 号楼) 取裸 base name 收纳, 自动吃掉 `X 座入口` / `X 座大堂` / `X 座电梯` 等 variant
- LANDMARK_FACILITY 用原 canonical 收纳 (例: 同 canonical `电梯厅` 的多个节点)

**关键守卫**:

- ⭐ **空间聚类** (`SPATIAL_MERGE_DIST_M = 3.0`, `91fdc4f`): 同 canonical 但欧氏距离 > 3 m 不合并. 例如机器人起点电梯厅 vs H 座旁边的电梯都被 canonical 化成"电梯厅", 但物理位置远, 应当保留两个独立 node.
- ⭐ **anchor 选 latest frame_idx** (`ca28d7b`): 沿路径走过 landmark 时, 后到的 keyframe 一般离 landmark 更近 (plate bbox 更大, cam crop 方向先验和 DINOv3 视觉相似度都更稳). 旧版本用最早帧, 容易锁在远视角.
- **复合 plate 优先 (Pass B)**: 如同时有 `H 座` 和 `H 座电梯` 节点, anchor 重命名成更具体的 `H 座电梯`. specificity 排序: `X 座电梯 / X 座楼梯` > `X 号楼电梯` > `X 层电梯`.
- **数字 plate 命名 (Pass C)**: `<N>号柜 + 储物柜区` → `<N>号储物柜`, `<N>号柜 + 外卖柜区` → `<N>号外卖柜`.

### 6.7 ColocationMerger (semantics/colocation_merger.py)

`VPR_SIM_THRESHOLD = 0.85` 强信号单独触发合并; `frame + spatial` 弱信号组合触发 (帧距 + 欧氏距离).

**关键规则**:

- ⭐ **`_category_mismatch` guard**: 不同 category 禁止合并. 这是为了防止 BUILDING_LANDMARK 错吸 SHOP, 或 FUNCTION_AREA 吞 LANDMARK_FACILITY.
- ⭐ **anchor tie-break**: 两节点合并时取 `frame_idx` 更小的为 anchor, 把另一方的 ts / cameras / next_positions / name 合并进来. 早帧 anchor 能把功能中心的 "第一次看到" 作为 display 帧, 避免被后续副标拖走. (注意与 `_merge_by_canonical_name` 的 latest-anchor 不同, 二者目的不同: 该函数走帧距 + spatial, anchor 早一些可以让"第一次看到"作锚; canonical-name merge 走命名归一, 后到帧 plate 更大更值得作锚.)
- `_combined_name` 调用 `NodeName.merge_names`.

### 6.8 门牌两阶段归属 (core/online_mapper_core.py:_create_door_plate_nodes)

**第一遍**: functional / landmark plate (例如 `打印区`, `关爱室`) 创建独立 `door-plate node`, 打标 `node._from_plate_best = True`. 需满足:

- `plate_voter.is_confirmed(name)` (受上述 BL `MIN_VOTES` / `MIN_FRAMES` / `MIN_MAX_AREA` 约束)
- 分类 ≠ REJECT 且 ≠ SHOP (brand 留给第二遍)
- canonical name 未被现有 keyframe 节点占用

**第二遍**: brand-like plate (例如 `EXHIOH`, `DEEPROUTE.AI`) attach:

- 搜帧距 ≤ 12 且 category 非空的 functional / room node
- 写 organization; 若已占且 `new_votes > 1.5 × old_votes` 才替换, 旧 org 进 `nearby_plates`
- 否则创建独立 SHOP node (保留为 standalone)

**RELOCATE-DISPLAY 规则** (仅动 `timestamp + cameras`, 不动 `frame_idx / pose / VPR feat`):

| target 来源 | brand 位置 | 行为 |
|---|---|---|
| keyframe trigger 创建 (无 `_from_plate_best`) | brand.pos ≈ target.pos | ⭐ relocate timestamp + cameras 到 brand best-view 帧 |
| plate best-view 创建 (`_from_plate_best=True`) | any | keep 原帧 (plate 帧已是到门口的近视角) |
| 任意 target | brand.pos ≠ target.pos (brand 是别处副标) | keep target 原帧避免离开功能中心 |

这样 display 层会显示 brand 的近视角抓图, 而拓扑层仍然锚在 keyframe 起点, coloc/连接都不受影响.

### 6.9 SceneGraph (semantics/scene_graph.py)

层次化 floor → room → object_id 结构, finalize 末尾按最终 node room 重建 floors 索引.

---

## 7. 输出 schema

### 7.1 目录结构

```
output_dir/                                  (cfg.output_dir, 默认 online_mapper/output/merged_labeled_data)
├── <node_id>/                               (按 next_node_id 顺序, 从 cfg.start_id=1 起)
│   ├── <ts>_camera_1.jpg                    4 路相机原图 (display 帧)
│   ├── <ts>_camera_2.jpg
│   ├── <ts>_camera_3.jpg
│   ├── <ts>_camera_4.jpg
│   ├── crops/                               next_position 子图 (node 自身相机帧)
│   │   └── <ts>_camera_X__<target_id>__<size>__<x>_<y>_<w>_<h>.jpg
│   └── node_position_info.json              ⭐ 节点元数据
├── ...

# 顶层 (output_dir 同级)
scene_graph.json                            层次场景图
pose_graph.json                             pose graph nodes + edges (含 motion_theta)
online_mapping_log.jsonl                    每帧决策日志
metrics.json                                总指标
plate_voter_dump.json                       plate 投票原始记录 (含 confirmed + rejected)
```

### 7.2 node_position_info.json

以样例运行的 node 14 (外卖柜区·EXHIOH) 为例:

```json
{
  "self_position": {
    "position_id": "14",
    "position_name": "外卖柜区·EXHIOH",
    "position_name_eng": "Locker Area · EXHIOH",
    "category": "外卖柜区",
    "category_eng": "Locker Area",
    "organization": "EXHIOH",
    "nearby_plates": [],
    "nearby_landmarks": [],
    "instance_suffix": "",
    "camera_1": "1776152264641_camera_1.jpg", "camera_2": "...",
    "camera_3": "...", "camera_4": "..."
  },
  "next_positions": [
    {
      "position_id": "13", "position_name": "B座入口",
      "camera_name": "camera_4", "landmark_name": "椅子",
      "big_box": "0.515,0.311,0.785,0.648",
      "mid_box": "...", "small_box": "...", "pixel_box": "",
      "crop_image_path": "crops/1776152264641_camera_4__13__big__....jpg",
      "crop_image_paths": {"big": "...", "mid": "...", "small": "..."},
      "position_name_eng": "B座入口", "landmark_name_eng": "chair"
    }
  ]
}
```

字段说明:

- `position_name` / `position_name_eng`: `display_cn()` / `display_en()`, 向后兼容旧消费者
- `category` / `category_eng`: 主类型 (外卖柜区 / 前台 / 电梯厅 / ...)
- `organization`: 品牌 / 门牌主名 (EXHIOH / DEEPROUTE.AI / ...), 可为空
- `nearby_plates`: 同 node 其他门牌, debug 用
- `nearby_landmarks`: GD 检测物体, debug 用
- `instance_suffix`: 全局重名后缀 (`_2`, `_3`, ...), 多栋楼场景使用

### 7.3 metrics.json 字段

```
n_nodes / n_edges / n_loop_closures / n_keyframes_triggered
kf_accepted_by_category: {function_area, landmark_facility, building_landmark, shop, room_*, junction, ...}
kf_rejected_by_category
plate_voter: {confirmed_names, rejected_names, min_frames, min_cameras}
plate_drops_verify       (Qwen verify 失败)
plate_drops_category     (category=REJECT)
plate_drops_unconfirmed  (voter 未确认)
plate_drops_already_attached
plate_attached_to_keyframe
coloc_merge: {pairs_examined, merges, aliases, by_reason}
same_name_merge
topology_rebuild
runtime_s: {depth, vpr, detect, vo, name, total, plate_scan}
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
    min_keyframe_frame_interval: int = 3
    # 闭环
    loop_closure_min_gap: int = 8
    loop_closure_vpr_threshold: float = 0.78
    loop_closure_top_k: int = 5
    loop_closure_geom_verify: bool = True
    loop_closure_min_inliers: int = 15
    # depth
    depth_model_id: str = "pretrained/depth-anything-v2-small-hf"
    depth_backend: str = "vggt"             # "vggt" | "da_v2"
    vggt_model_path: str = "pretrained/vggt-1b/model.pt"
    vggt_window_size: int = 4
    vggt_dtype: str = "bf16"
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
    # VO
    vo_backend: str = "vggt"                # "vggt" | "orb"
    # occupancy
    grid_resolution: float = 0.2
    grid_size: int = 200
    occ_backend: str = "vggt"               # "vggt" | "depth_row"
    start_id: int = 1
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

# 3. 下载 grounding-dino (必需; 命名路径不可缺)
huggingface-cli download IDEA-Research/grounding-dino-base \
  --local-dir pretrained/grounding-dino-base

# 4. 启 Qwen3.5-9B vLLM (GPU 1, 端口 8199, 命名 / 二次验证 / 场景描述)
./deploy/start_qwen_vllm.sh 1 8199

# 5. 启 Qwen3.5-0.8B vLLM (GPU 0, 端口 8198, 仅 ws_proxy nav 分支意图分类使用)
./deploy/start_qwen08_vllm.sh 0 8198
```

### 9.2 端到端建图 (CLI)

```bash
conda activate internvla
cd memory-nav
CUDA_VISIBLE_DEVICES=0 python online_mapper/run_online_map.py \
  --input memory_test_data \
  --output online_mapper/output/merged_labeled_data
```

**注意**: 不要加 `--no_grounding_dino`, 否则门牌检测路径完全不跑, 不会有功能区 / 房间 / SHOP 等 node.

### 9.2.x WebSocket 建图模式 (双模式 ws_proxy)

`deploy/ws_proxy_with_memory.py` 同时承载**导航 (nav)** 和**建图 (mapping)** 两种模式, 通过 `session_state['mode']` 路由. 所有请求保持统一形状 `{id, task, pts, images}`, 由**四类意图分类**驱动 (Qwen3.5-0.8B vLLM + 关键词规则兜底):

- 默认 `mode='nav'`. 意图分为 `navigate` / `ask_location` / `ask_direction` / `mapping` 四类.
- `mapping` 意图 (自然语言 "开始建图" 或硬编码 `task="mapping"`) 触发 `MappingSession`: 首帧自动创建独立 session, 之后每帧喂入 `OnlineMapperCore.process_frame`.
- `mapping` 意图的 stop 子类 ("停止建图" / `task="stop_mapping"`) 触发 `finalize` + 可视化, 返回 summary, 切回 nav.
- 在 mapping 模式下发送非 mapping 意图, 服务端自动 `finalize` 当前 session 并切回 nav.
- `{"command": "mapping_status"}` 查询当前 session 进度 (不驱动模式切换).

关键实现要点:

- `MappingSession.__init__` / `feed_raw` / `finalize` 通过 `asyncio.to_thread` 走线程池, 避免 CPU-heavy 调用阻塞 asyncio 事件循环造成 ws ping 超时 (`1011 keepalive timeout`).
- `feed_raw(camera_b64)` 直接 `base64.b64decode + open(w, 'wb')` 写原 JPEG 字节, **不做解码 / 重编码**, 保证与 `run_online_map.py` cv2.imread 像素一致 (否则 GD/Qwen 分数漂移会改变门牌检测结果).
- `shared_vpr_extractor` 传入记忆导航的 SelaVPR 实例, mapping session 不重复加载.
- 产物: `deploy/logs/mapping_output/session_{ts}_{client_id}/`; 临时帧 `deploy/logs/mapping_frames/session_*/` finalize 后清理.
- 断线 (`websockets.ConnectionClosed`) 在 `handle_client` 的 `finally` 中自动 finalize 保住数据.

端到端测试:

```bash
python deploy/ws_proxy_with_memory.py                   # 终端 1: 启 ws_proxy
python tests/test_memory_ws.py --mode mapping           # 终端 2: 建图客户端
```

`--mode nav` 则跑记忆导航回放.

### 9.3 schema 校验

`merged_labeled_data/{node_id}/node_position_info.json` 应包含:

- `self_position`: `position_id / position_name / position_name_eng / category / organization / camera_1..4`
- `next_positions`: list of `{position_id, camera_name, landmark_name, big_box/mid_box/small_box, crop_image_paths{big,mid,small}, position_name_eng, landmark_name_eng}`

可用 `jq` 快速检查:

```bash
for d in online_mapper/output/merged_labeled_data/*/; do
  jq -e '.self_position.position_id and (.next_positions | type == "array")' \
    $d/node_position_info.json > /dev/null \
    || echo "FAIL $d"
done
```

---

## 10. 已知限制

1. **VGGT 尺度自洽但偏小**: VGGT 内部 (depth + pose + occupancy) 一致, 相对几何可用, 但绝对尺度不是真实物理米; 当前配置 (`grid_resolution=0.2`, `accumulated_translation=1.5`) 在 VGGT 尺度下调过, 切回 ORB 需重新调.
2. **滑窗大小未大范围调优**: 默认 4, 调 8/12 可能更精但更慢, 尚未系统 benchmark.
3. **VGGT 单帧 stateless 仅用于 junction**: 多视图时序约束不存在, junction 判定准度略低于主路径.
4. **VLM 命名依赖第三方 Qwen3.5-9B vLLM**: 无纯本地 fallback, 服务没起 mapper 命名会全部退到 default.
5. **门牌 `ATTACH_GAP = 12 帧`**: 适合慢速步行 (≈1 Hz); 高速移动或帧率高的数据需要按实际缩短.
6. **几何方向先验依赖 pose_graph 准确性**: motion_theta 基于相邻 keyframe 位移 atan2, 在 VO 漂移大或走直线几乎零位移时会不稳.
7. **BUILDING_LANDMARK `MIN_VOTES = 4` 门槛**: 在极低照明 / 单次经过 / 仅一个相机看到的场景下可能把真 label 也误杀, 需要按数据放宽.

---

## 11. FAQ

### Q. 没有功能区 / 门牌 / SHOP 节点?

A. 检查是否传了 `--no_grounding_dino`, 或 Qwen vLLM 端口未启. 门牌检测依赖 GD + Qwen 二次验证, 两者任一缺位整条路径都不会建 node.

### Q. 节点显示名是 `category·organization` 的中点分隔结构, 为什么?

A. `NodeName.display_cn()` 用中点防止 CJK + Latin 粘连, 例如 `外卖柜区·EXHIOH`. organization 可为空 (纯功能区), 此时退化为单段 `category` (例如 `电梯厅`).

### Q. cam ↔ neighbor 方向匹配错乱 (如下一节点该在 camera_3 却被标到 camera_2)?

A. 检查 `build_for_node` 是否通过 `pose_graph=...` 传进去. 几何方向先验默认开启, 同 segment ALPHA=0.5, 反向硬惩罚 -1.0; 没有 pose_graph 时 prior 全部退化成视觉相似度, 线性走廊场景下会错配. 若 self/neighbor timestamp 差距 > 300 s 自动判跨段, 几何先验置零让纯视觉决策, 防 VO 漂移污染先验.

### Q. `D座` / `13号楼` 等 BUILDING_LANDMARK 幻觉怎么被挡掉?

A. BUILDING_LANDMARK 单独有两层阈值: `MIN_VOTES = 4` 要求 vote 记录累计 ≥ 4, 单帧白名单 fast-pass 被禁用. 幻觉通常 votes ≤ 3 在 voter 层就被 reject; 即便 Qwen 某帧把玻璃幕墙读成 "D座", 少于 4 次累积也不会走到建 node 阶段.

### Q. 为什么有些 node 的 display 帧 timestamp 跟 keyframe_trigger 帧不同?

A. 这是 brand attach 的 RELOCATE-DISPLAY 行为: brand 是在相近帧抓到的近视角门牌, 会把 display 帧 (仅 timestamp + cameras) 挪到 brand best-view. frame_idx / pose / VPR feat 保持原 keyframe, 不影响拓扑 / coloc / 连接.

### Q. 多栋楼相同功能区怎么区分?

A. `NameDeduplicator` 按 `dedup_key = (category, organization)` 分组. 不同楼会产生不同 organization (不同品牌/门牌主名) → 不同 key, 不会触发重名. 真发生重名时 VPR 高相似合并, 否则写 `instance_suffix = _2 / _3 / ...` 由 `display_cn()` 渲染.

### Q. 起点电梯厅和 H 座旁电梯都被合并成一个 node 怎么办?

A. `_merge_by_canonical_name` 已加空间聚类守卫 (`SPATIAL_MERGE_DIST_M = 3.0`): 同 canonical name 但物理欧氏距离 > 3 m 的节点不再合并, 即使命名 canonical 一致也保留为独立 node, 各自走 `instance_suffix` 后缀去重.

### Q. cross-gap filter 触发条件是什么?

A. spatial-KNN 邻接重建时, 时间边两端 keyframe 的 timestamp 差 > 60s 且中间没有 bridging keyframe → 拒绝该时间边. 用于处理数据采集中途挂机 / 挪动机器人的场景.

### Q. VGGT OOM?

A. 降 `vggt_window_size` (默认 4 → 2), 或切 `depth_backend="da_v2"` 回退到 DA-V2 路径 (同时建议把 `vo_backend` 切回 `"orb"`, `occ_backend` 切回 `"depth_row"`).

### Q. VGGT 位姿坐标系约定?

A. extrinsics 为 cam-from-world (`X_cam = R*X_w + T`), 见 `vggt/utils/geometry.py:depth_to_world_coords_points` 注释 "OpenCV camera coordinate convention, cam from world".
