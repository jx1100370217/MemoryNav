# ws_proxy_with_memory.py 架构流程图说明

本文档说明了 `deploy/ws_proxy_with_memory.py` 的详细架构和流程图。

## 生成的流程图

### 1. 模型架构图 (ws_proxy_model_architecture.png)

**文件大小**: 2.1 MB  
**描述**: 完整的系统模型架构图，包含所有组件及其详细的shape信息

#### 主要模块:

1. **📥 输入层 (WebSocket数据)**
   - 前置相机 (front_1): [480, 640, 3] RGB图像
   - 环视相机 (camera_1~4): 4个环视视角，每个 [480, 640, 3]
   - 导航指令: 文本字符串
   - 深度图 & 位姿: 可选输入

2. **🔍 LongCLIP 视觉特征提取器**
   - **预处理**: Resize & Normalize → [3, 224, 224]
   - **Vision Encoder**: 
     - Conv1: Patch Embedding [B, 3, 224, 224] → [B, 768, 7, 7]
     - Transformer (12 Layers): Hidden Size 768, 12 Heads
     - Self-Attention + FFN
   - **特征投影**: Linear [B, 768] → [B, 512]
   - **归一化**: L2 Normalization → 特征向量 [512]

3. **🔄 环视相机特征融合**
   - 提取环视特征: 对 camera_1~4 分别提取特征
   - 加权融合: 每个相机权重 0.25
   - 输出: 融合特征 [512]

4. **🧠 Qwen3-VL 场景描述生成器 (关键帧)**
   - **触发条件**: pixel_target ≠ None (关键帧检测)
   - **处理器**: Qwen3-VL Processor
     - 输入: 4张环视图像
     - Image Preprocessing + Prompt Construction
   - **模型**: Qwen2.5-VL-8B
     - Vision Encoder: 提取图像特征
     - Language Decoder: 生成描述
     - Max New Tokens: 256
     - Device: cuda:1
   - **输出**:
     - scene_description: 场景文本描述
     - semantic_labels: 语义标签列表

5. **🎯 视觉位置识别 (VPR)**
   - **FAISS 索引**: IndexFlatIP (内积搜索)
     - Feature Dimension: 512
     - Database: features [N, 512], node_ids [N], timestamps [N]
   - **相似度搜索**: Top-K Search (k=10)
   - **回环检测**: 
     - 条件: similarity > 0.85 且 time_gap > 30s
     - 输出: (node_id, similarity) 或 None

6. **🗺️ 拓扑地图管理器**
   - **创建/更新节点**: TopologicalNode
     - node_id, visual_feature [512]
     - rgb_image [480, 640, 3]
     - surround_images: Dict
     - timestamp, scene_description, semantic_labels
     - pixel_target [2] (关键帧)
     - is_keyframe: bool
   - **NetworkX 有向图**: 
     - 节点: TopologicalNode
     - 边: (from, to, actions)
     - 最短路径搜索
   - **GraphRAG 语义地图**:
     - node_metadata: 场景描述和语义标签
     - label_index: 标签索引
     - 语义搜索功能

7. **🤖 InternVLA-N1 导航模型**
   - **历史帧缓冲**: 最大8帧
     - rgb_list, depth_list, pose_list
     - 采样策略: np.linspace
   - **视觉编码器**:
     - 输入: RGB+Depth+Pose
     - Resize: (384, 384)
     - CNN Backbone
     - Multi-scale Feature Extraction
   - **语言编码器**:
     - Tokenization + Embedding
   - **多模态融合**:
     - Vision-Language Fusion
     - Cross-attention
   - **解码器** - 双系统输出:
     1. **离散动作序列**: output_action: List[int]
        - 0=STOP, 1=前进, 2=左转, 3=右转, 5=向下看
     2. **连续轨迹**: output_trajectory: [33, 2]
        - 增量坐标 [dx, dy]
     3. **像素目标**: output_pixel: [y, x]
        - 关键帧标记

8. **📤 输出转换**
   - **动作转换器**:
     - 离散动作 → 机器人控制: [[x, y, yaw]]
       - x = forward × 0.25m
       - yaw = (left - right) × π/24
     - 轨迹点 → 累积坐标: cumsum(delta_xy)
   - **像素目标归一化**:
     - [y, x] → [x/640, y/480]
   - **WebSocket 响应**: JSON格式
     - status, id, pts
     - task_status: "executing" / "end"
     - action: [[x, y, yaw], ...]
     - pixel_target: [x, y]
     - memory_info: {...}

---

### 2. 记忆记录流程图 (ws_proxy_memory_recording_flow.png)

**文件大小**: 723 KB  
**描述**: 展示了系统如何记录导航路线和生成视觉记忆

#### 流程步骤:

1. **🎬 启动记忆记录**
   - 接收 `START_MEMORY` 指令
   - 初始化 RouteMemory:
     - route_id = f"route_{timestamp}"
     - node_sequence = []
     - action_history = []
     - keyframe_indices = []
     - visual_features = []

2. **🔁 导航循环 (每帧)**
   - **接收帧数据**: RGB图像 [480, 640, 3] + 环视图像 + 导航指令
   - **提取视觉特征**: 
     - LongCLIP: camera_1~4
     - 环视融合 [512]
     - VPR 回环检测
   - **InternVLA-N1 推理**:
     - 输出动作序列
     - 输出像素目标 (关键帧)
   - **是否关键帧？**: pixel_target ≠ None
   - **VLM 场景描述生成** (关键帧):
     - Qwen3-VL: camera_1~4
     - 生成 scene_description
     - 提取 semantic_labels
   - **添加拓扑节点**:
     - TopologicalNode (包含所有语义信息)
   - **记录导航步骤**:
     - node_sequence.append(node_id)
     - action_history.append(action)
     - visual_features.append(feature)
     - 如果是关键帧:
       - keyframe_indices.append(idx)
       - keyframe_images.append(rgb)
   - **实时保存到磁盘** (每个关键帧):
     - {route_id}.pkl
     - {route_id}_features.npy
     - {route_id}_keyframes/

3. **🛑 停止记录**
   - 接收 `STOP_MEMORY` 指令
   - route.is_complete = True
   - **保存完整路线**:
     - 路线元数据 .pkl
     - 视觉特征 .npy
     - 关键帧图像 .jpg
     - 语义图:
       - semantic_metadata.json
       - semantic_graph.json
   - **路线统计**:
     - total_nodes
     - total_keyframes
     - duration
     - semantic_nodes

---

### 3. 推理流程图 (ws_proxy_inference_flow.png)

**文件大小**: 2.1 MB  
**描述**: 完整的推理流程，包含记忆复用机制

#### 流程步骤:

1. **📡 WebSocket 输入**
   - 接收消息: JSON格式
     - id: robot_id
     - pts: timestamp
     - task: instruction
     - images: {front_1, camera_1~4}
     - depth: base64 (可选)
     - pose: [4,4] (可选)
   - 解码图像:
     - Base64 → PIL.Image
     - Resize to (640, 480)
     - Convert to numpy array

2. **⚡ 特殊指令处理**
   - 检查特殊指令:
     - STOP / RETURN
     - START_MEMORY / STOP_MEMORY
     - turn left / turn right / go straight
   - **直接控制**:
     - turn left → [0, 0, π/12]
     - turn right → [0, 0, -π/12]
     - go straight → [1, 0, 0]
     - 跳过模型推理
   - **返回导航**:
     - 启动返回导航器
     - 使用拓扑图路径或轨迹回溯

3. **🔄 记忆复用检查**
   - **检查记忆复用**:
     - 条件: 任务刚开始 (request_count=0) 或任务刚变化
     - 搜索匹配路线: find_matching_route(instruction)
   - **加载匹配路线**:
     - 从内存或磁盘加载:
       - route_id, action_history
       - keyframe_indices, visual_features
   - **复用动作**:
     - action = route.action_history[step]
     - step += 1
     - **跳过模型推理**
     - skipped_inference = True
   - **检查复用完成**: step >= total_steps

4. **🧠 正常推理流程**
   - **检查任务变化**:
     - instruction != last_task?
     - 如果变化: Agent.reset()
   - **InternVLA-N1 推理**:
     - agent.step(rgb, depth, pose, instruction, intrinsic, look_down)
     - 历史帧采样 (最多8帧)
     - 多模态编码与融合
   - **检测动作5**:
     - output_action[0] == 5 (向下看)
   - **Look-Down 推理**:
     - agent.step(同样输入, look_down=True)
     - 重新推理获取精细动作
   - **双系统输出**:
     1. output_action: List[int] - 离散动作序列
     2. output_trajectory: [33, 2] - 连续轨迹点 (增量)
     3. output_pixel: [y, x] - 像素目标 (关键帧)

5. **💾 记忆处理**
   - **提取环视特征**:
     - LongCLIP(camera_1~4)
     - 环视融合 [512]
   - **VPR 回环检测**:
     - FAISS 搜索
     - is_revisited?
   - **VLM 处理 (关键帧)**:
     - pixel_target ≠ None?
     - Qwen3-VL 生成:
       - scene_description
       - semantic_labels
   - **更新拓扑图**:
     - 添加/更新节点
     - 添加边 (动作)
     - 更新 FAISS 索引
     - 更新 GraphRAG
   - **记录步骤 (如果记录中)**:
     - if route_memory.is_recording():
       - record_step(node_id, action)
       - if is_keyframe:
         - save_keyframe_image()

6. **📤 输出处理**
   - **动作转换**:
     - 离散动作 → [x, y, yaw]
     - 轨迹点 → 累积坐标
   - **小动作检测**:
     - 33个点 && 所有值 < 0.5?
     - 自动转换为 STOP
   - **像素目标归一化**:
     - [y, x] → [x/640, y/480]
   - **构建响应**: JSON
     - status, id, pts
     - task_status
     - action
     - pixel_target
     - memory_info
   - **可视化保存 (关键帧)**:
     - 标注图像
     - 环视拼接图
     - 元数据 JSON
   - **WebSocket 发送响应**: JSON 响应
   - **下一帧**: 循环继续

---

## 关键特性

### 1. 环视相机特征融合
- 仅使用 camera_1~4 四个环视相机 (不包含 front_1)
- 每个相机权重 0.25，加权融合后归一化
- 用于 VPR 位置识别

### 2. 关键帧检测
- 基于 pixel_target 是否为 None 判断
- 关键帧时触发:
  - VLM 场景描述生成 (Qwen3-VL)
  - 语义标签提取
  - 关键帧图像保存
  - 可视化结果保存

### 3. 记忆复用机制
- 匹配指令: find_matching_route(instruction)
- 跳过模型推理，直接使用历史动作
- 大幅提高推理速度和一致性

### 4. 双系统输出
- **离散动作**: 0=STOP, 1=前进, 2=左转, 3=右转, 5=向下看
- **连续轨迹**: 33个增量点 [dx, dy]
- **像素目标**: [y, x] 关键帧标记

### 5. Look-Down 机制
- 检测动作5时自动触发
- 重新推理获取精细动作
- 适用于复杂场景导航

---

## Shape 信息汇总

| 数据类型 | Shape | 说明 |
|---------|-------|------|
| 前置相机图像 | [480, 640, 3] | RGB uint8 |
| 环视相机图像 | [480, 640, 3] × 4 | camera_1~4 |
| 深度图 | [480, 640] | float32 |
| 位姿矩阵 | [4, 4] | float32 |
| 内参矩阵 | [4, 4] | float32 |
| LongCLIP 输入 | [3, 224, 224] | 预处理后 |
| LongCLIP 特征 | [512] | L2归一化 |
| 融合特征 | [512] | 环视加权融合 |
| InternVLA 输入 | [384, 384, 3] | Resize后 |
| 历史帧缓冲 | 最多8帧 | 动态采样 |
| 离散动作 | List[int] | 变长序列 |
| 连续轨迹 | [33, 2] | 增量坐标 |
| 像素目标 | [2] | [y, x] |
| 机器人控制 | [[x, y, yaw], ...] | 变长序列 |
| 归一化像素 | [2] | [x_norm, y_norm] |

---

## 依赖库

- **graphviz**: 流程图生成
- **websockets**: WebSocket 服务
- **torch**: 深度学习框架
- **transformers**: Qwen3-VL 模型
- **faiss-cpu/faiss-gpu**: 高效相似度搜索
- **networkx**: 拓扑图管理
- **numpy, opencv-python, pillow**: 图像处理

---

## 运行脚本

生成流程图:
```bash
cd /home/ubuntu/Disk/codes/jianxiong/MemoryNav
python docs/generate_ws_proxy_architecture.py
```

输出文件:
- `docs/ws_proxy_model_architecture.png` - 模型架构图
- `docs/ws_proxy_memory_recording_flow.png` - 记忆记录流程图
- `docs/ws_proxy_inference_flow.png` - 推理流程图

---

## 总结

本系统是一个集成了 **视觉记忆导航** 功能的 WebSocket 代理服务，主要特点:

1. ✅ **多模态输入**: 前置相机 + 4个环视相机 + 深度 + 位姿
2. ✅ **视觉位置识别 (VPR)**: 基于 LongCLIP + FAISS 的高效回环检测
3. ✅ **语义场景理解**: 基于 Qwen3-VL 的场景描述和标签提取
4. ✅ **拓扑地图管理**: NetworkX 有向图 + GraphRAG 语义索引
5. ✅ **记忆复用**: 匹配历史路线，跳过推理，提升效率
6. ✅ **双系统导航**: 离散动作 + 连续轨迹双输出
7. ✅ **关键帧机制**: 基于 pixel_target 的智能关键帧检测
8. ✅ **实时持久化**: 每个关键帧实时保存到磁盘

该系统适用于移动机器人的长时记忆导航任务，能够记录和复用导航经验，提高导航效率和鲁棒性。










