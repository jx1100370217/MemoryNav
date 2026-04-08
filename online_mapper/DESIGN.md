# Online Active Mapping Module — DESIGN

## 目标
为 MemoryNav 增加一个**在线主动建图**模块 `online_mapper/`,
与离线 `offline_mapper/` 解耦, 但产出 schema 完全兼容的 `merged_labeled_data/` 目录,
使现有导航 runtime 无需改动即可消费。

## 三层架构

### Layer 1 — 几何 (geometry/)
- `depth_estimator.py`: Depth-Anything-V2-Small (HF cached) 单目相对深度。
  无 metric scale → 用 frame-to-frame 中位数对齐，degrade gracefully。
- `pose_graph.py`: scipy.optimize.least_squares 实现的轻量 2D pose graph
  (x,y,θ)。VPR 闭环作为相对位姿约束加入并求解。
- `occupancy.py`: 2D 占据栅格 (numpy)。每帧由前向相机 + depth → free/occupied
  cells，按帧间相对位移累积。
- 帧间位移 (VO) 估计: 若 GTSAM 不可用，回退到 ORB+findEssentialMat 或常速 1m/帧
  代理。日志会标记使用的方法。

### Layer 2 — 拓扑 (topology/)
- `keyframe_selector.py`: 多触发条件
  - VPR 相似度 < 阈值 (复用 NodeDistanceEstimator)
  - 累积位移 > Δd 或累积旋转 > Δθ
  - 信息增益 > τ (occupancy 新增 free cells 占比)
- `loop_closure.py`: 全图 VPR 检索 (top-K) + 帧间隔 > N + 几何验证
  (ORB matches > M) → 加入位姿图作为闭环边。
- `graph.py`: 真正的图结构 (邻接表)，支持节点 >2 邻居。
- `frontier_nbv.py`: 占据栅格 frontier 检测 + NBV 评分
  (information gain - traversal cost + semantic novelty)。
  即便回放模式也会在每个 keyframe 选出 NBV 并写日志。

### Layer 3 — 语义 (semantics/)
- `open_set_detector.py`:
  - 优先 Grounding-DINO (HF cached) 做开放集物体/门牌检测
  - 同时调用 Qwen vLLM (若 8199 端口在线) 做 OCR / 房间命名,
    否则回退到启发式 fallback namer
- `scene_graph.py`: floor → room → node → object 层次图，
  以 JSON (`scene_graph.json`) 持久化。
- `semantic_dedup.py`: 同 room/landmark 且 VPR 相似的新 keyframe → 合并到旧 node。
- 门牌位置 bug 修复: 一旦某个 plate 在多个候选帧出现, 选 bbox 面积最大的帧
  作为该 semantic node 的代表帧。

## 数据流
```
memory_test_data/  (49 帧 × 4 cam)
       │
       ▼
StreamLoader  ──► 每个时刻给出 4 cam 图像
       │
       ▼
OnlineMapperCore (主循环)
   ├── Layer1: depth + 相对位姿 → pose_graph + occupancy
   ├── Layer2: 多触发关键帧? → 添加 node
   │           全局 VPR 闭环? → 添加边 + 优化 pose_graph
   │           frontier/NBV → 日志
   ├── Layer3: open-set 检测 + 房间识别
   │           语义去重 / 门牌帧回溯
   └── 写日志 (online_mapping_log.jsonl)
       │
       ▼
完成后 MergedDataWriter → merged_labeled_data/{node_id}/...
                       + scene_graph.json
                       + pose_graph.json
                       + metrics.json
```

## Schema 兼容
严格遵守 `offline_mapper/validate_output.py`:
- `node_position_info.json` 顶层: `self_position`, `next_positions`
- self_position 字段: position_id, position_name, position_name_eng, camera_1..4
- next_positions[i]: position_id, position_name, camera_name, landmark_name,
  big_box, mid_box, small_box, crop_image_paths{big,mid,small}, position_name_eng,
  landmark_name_eng
- crops/ 子目录, 文件名: `{ts}_camera_{N}__{idx}__{size}__{x}_{y}_{w}_{h}.jpg`

## 模拟在线说明
真实机器人未连上时, StreamLoader 按时间戳排序逐帧 yield, 模拟"流式"输入。
每帧主循环只看到当前帧 & 之前帧, 不允许 peek 未来帧 (这是和 offline_mapper 的关键区别)。
NBV 决策、闭环、关键帧触发都基于"当前为止"的信息。
