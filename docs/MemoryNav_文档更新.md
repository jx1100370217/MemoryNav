# 纯视觉记忆导航方案-MemoryNav — 文档更新稿

> 说明：此文件根据最新代码（截止 2026-03-30）更新。
> 标注了所有与旧文档的差异，方便主人对照更新飞书。
> 🔴 = 需修改  🟢 = 新增  🔵 = 不变

---

# 纯视觉记忆导航方案-MemoryNav

## 一. 输入

文本指令：前往A8档案室
视觉图片：去畸变的4张鱼眼相机图 camera_1~camera_4 + 前置相机图 front_1

### WebSocket 请求格式

系统通过 WebSocket 接收机器人端的请求，端口 9528，每帧请求包含以下字段：

```json
{
  "id": "robot_001",
  "pts": 1710000000000,
  "task": "前往C8前台",
  "images": {
    "front_1": "<base64 RGB图>",
    "camera_1": "<base64 鱼眼图>",
    "camera_2": "<base64 鱼眼图>",
    "camera_3": "<base64 鱼眼图>",
    "camera_4": "<base64 鱼眼图>"
  }
}
```

- **task**: 自然语言导航指令（如 "前往C8前台"），也支持直接控制指令（"turn left"、"turn right"、"go straight"、"STOP"）
- **images.front_1**: 前置 RGB 图，用于模型推理（调整为 640×480）
- **images.camera_1~4**: 四方向鱼眼相机图（服务端自动去畸变），用于 VPR 定位和子图匹配

🟢 **变更：鱼眼去畸变由服务端自动完成**
- 启动时从 `cam/params.yaml` 加载各相机内参和畸变系数
- 每帧推理前自动调用 `cv2.remap` 做柱面投影去畸变
- `cam/params.yaml` 缺失时自动跳过，不影响服务运行

---

## 二. 导航步骤

系统采用**三层导航策略**：

1. **记忆引导**：通过记忆拓扑图规划路径，每步返回目标相机方向 + 三级注意力子图匹配区域
2. **VPR 持续验证**：每帧请求都用 camera_1~4 做 VPR 判断是否到达下一节点
3. 🔴 **模型兜底**：子图匹配失败时，用 **Qwen3.5-9B 两步推理打点模型**进行兜底定位（替代原 InternVLN 模型推理）

### 2.1 定位

#### 2.1.1 导航起点

基于当前位置的 camera_1~camera_4 通过视觉位置识别（VPR）确定起点 node。

VPR 定位流程：
1. 对 camera_1~4 的图像分别提取 VPR 特征向量
2. 与记忆拓扑图中每个节点存储的 4 个相机特征做相似度比较
3. 支持两种匹配模式（通过 `vpr_config.yaml` 的 `order_invariant` 配置）：
   - **循环移位匹配（SelaVPR++）**：`order_invariant: false`，利用相机空间排列约束，对 4 个相机进行循环移位匹配（共 4 种 shift），选取最佳 shift 得到匹配节点和朝向偏移（heading_offset）
   - **无序贪心匹配（AnyLoc/MegaLoc/EffoVPR）**：`order_invariant: true`，忽略相机顺序，每张 query 图与 memory 中所有相机比较，贪心二分匹配找到最佳节点
4. 🔴 相似度超过阈值时返回匹配节点。**置信度 = 4 相机平均余弦相似度**（不再使用最佳 vs 次佳差异计算）

🟢 **VPR 方法可配置**
- 支持多种 VPR 方法：`selavpr`（默认推荐）、`anyloc`、`megaloc`、`effovpr`
- 通过 `deploy/vpr_config.yaml` 统一配置，一处修改全局生效

🟢 **目的地查找支持多种查询格式**
- 直接节点名："C8前台区"
- 自然语言任务："前往C8前台"、"go to A8 front desk"（自动去除前缀"前往/去/go to"等）
- 节点 ID："8"
- 子串模糊搜索

#### 2.1.2 导航终点

语义匹配目的地节点。

🔵 **此部分基本无变化**

### 2.2 定路线

基于拓扑图最短路径规划。

导航计划（NavigationPlan）包含：
- 起点/终点节点信息（ID、中文名、英文名）
- 完整路径节点序列
- 逐步导航步骤（NavigationStep），每步包含：
  - from_node → to_node（起止节点）
  - camera_name：目标所在的相机方向（camera_1~camera_4）
  - landmark_name：注意力目标地标名称
  - 🔴 **crop_image_paths**：**三级注意力子图路径** `{"big": ..., "mid": ..., "small": ...}`
  - 🔴 crop_image_path：指向 big 尺度的兼容引用

### 2.3 定方向

确定了行驶路线后，就是逐段按照规划路线移动到终点。该部分的核心是返回给机器人控制端**运动目标点**。

#### 2.3.1 子图匹配定位

🔴 **子图匹配方案：DINOv3 密集特征匹配（三级级联 + 全相机遍历）**

1. **特征提取**：使用 DINOv3 (ViT-B/16, 通过 timm 加载) 提取两张图的 patch token 密集特征图
2. **级联匹配**：对每个相机（camera_1~camera_4）执行三级级联匹配：**small → mid → big**
3. **全局最优**：遍历所有 4 个相机，选择 confidence 最高的匹配结果（不再局限于记忆中标注的 camera_name）
4. **最佳位置**：找到相似度最高的位置作为匹配结果

🔴 **置信度阈值**：匹配成功阈值为 **0.60**（`SUB_MATCH_CONFIDENCE_THRESHOLD`）

🟢 **best_fail_camera 追踪**
- 即使所有相机都匹配失败，也记录得分最高的 camera
- 该 camera 用于后续遮挡检测（而非静态的 `step.camera_name`）

匹配结果（SubImageMatchResult）：
- 像素坐标 bounding box（x_min, y_min, x_max, y_max）
- 百分比 bounding box（top_left_pct / bottom_right_pct / center_pct）
- confidence：匹配置信度 [0, 1]
- method：使用的匹配方案名称

#### 2.3.2 帧间相似度缓存

🔴 **帧间相似度基于 DINOv2 VPR 特征的余弦相似度（零额外推理成本）**

核心逻辑（`_cache_or_reuse_sub_match`）：

- **匹配成功（conf ≥ 0.60）**：采纳结果，更新缓存（保存当前帧的 VPR 特征用于下次比较）
- **匹配失败（conf < 0.60）**：
  - 若前后帧 DINOv2 特征余弦相似度 ≥ **0.70**（`FRAME_SIMILARITY_THRESHOLD`），认为场景几乎没变，**复用上一帧的匹配框**
  - 否则认为场景变化大，**清除缓存**
- **无缓存可用**：返回 None，触发 Qwen3.5 兜底

优势（相比 SSIM）：
- 复用 VPR 流程已提取的特征，**零额外推理成本**
- 语义级比较，对光照变化鲁棒

🟢 **匹配结果可视化**
- 匹配成功：绿色框 + 红色中心点
- 缓存复用：黄色框 + 蓝色中心点
- 低置信度：灰色框 + 灰色中心点
- 保存到 `deploy/logs/images/` 目录

#### 2.3.3 遮挡检测

🟢 **新增：YOLOv8n 视觉遮挡检测**

当子图匹配失败时（无论 VPR 是否成功），系统自动对注意力相机执行遮挡检测：

1. **触发条件**：子图匹配失败即触发，不依赖 VPR 结果
2. **相机选择**：使用子图匹配得分最高（但低于阈值）的 camera
3. **检测类别**：person、backpack、umbrella、handbag、suitcase
4. **遮挡判定**：单个遮挡物 bbox 面积占比 ≥ **25%** → 判定为遮挡
5. **遮挡时行为**：输出 `action: [0, 0, 0]`（原地等待），清除子图匹配缓存
6. **未遮挡时行为**：使用 Qwen3.5 打点继续导航

#### 2.3.4 Qwen3.5 兜底打点

🔴 **兜底模型：Qwen3.5-9B 两步推理打点**

当子图匹配失败且未遮挡时，使用 Qwen3.5-9B VLM 对当前步骤的 landmark_name 进行视觉打点定位。

🟢 **两步推理法**（解决了 Qwen3.5 "无论目标是否存在都强行输出坐标"的 BUG）：

1. **存在性检测**（`check_existence`）：
   - Prompt: `Is "xxx" visible in this image? Answer: yes or no`
   - 仅生成 1~2 个 token，延迟极小
   - 回答 "no" → 直接返回 `not_found`，不执行打点

2. **条件打点**（仅在确认目标存在时执行）：
   - Prompt: `Point to "xxx" in this image. Output ONLY JSON: {"point": [x, y]}` (坐标范围 0~1000)
   - 输出归一化到 [0, 1] 的坐标

🟢 **Qwen3.5 运行架构**
- 通过**子进程**运行在独立的 `qwen3` conda 环境（transformers 5.x），避免版本冲突
- 主进程通过 stdin/stdout JSON 行协议通信
- 默认运行在 GPU 1

🟢 **多相机搜索策略**（`predict_on_camera`）
- 优先尝试 step 指定的 target_camera
- 失败后遍历其余相机
- 找到置信度 ≥ 0.5 的结果即提前退出

🟢 **打点结果可视化**
- 橙色十字准星 + 同心圆 + 中文地标名称标注
- 保存到 `deploy/logs/images/`

#### 2.3.5 像素→机器人坐标转换

🟢 **新增：完整物理管线坐标转换**

将 `pixel_target: [x_norm, y_norm]` 通过以下管线转换为机器人运动坐标：

```
x_norm → 柱面水平角 → + 相机方位角 → 全局 yaw
y_norm → 柱面垂直角 → 俯仰角 → 距离估算（相机高度 + pitch_up）
yaw + distance → (x_forward, y_lateral)
```

🟢 **侧面相机旋转处理**
- camera_3/camera_4 匹配成功时：通过坐标转换获取实际 yaw 角度，输出原地旋转动作 `[0, 0, yaw_rad]`
- camera_3/camera_4 Qwen3.5 兜底时：输出固定旋转 `[0, 0, 0.785]`（约45°）

覆盖全部导航决策路径：子图匹配成功、帧间缓存、Qwen3.5 兜底。

#### 2.3.6 VPR 持续验证与步骤切换

🔴 **重大变更：移除趋势判断，简化为遮挡检测 + Qwen3.5 兜底**

导航决策流程（每帧）：

```
1. 执行子图匹配 (全4相机 × 3级cascade)
2. 执行 Lookahead 下一步子图匹配
3. 子图匹配失败时:
   ├─ 遮挡检测 (对子图匹配得分最高的 camera)
   │   ├─ 遮挡 → action=[0,0,0] 原地等待
   │   └─ 未遮挡 → Qwen3.5 打点继续导航
   │              └─ 打点也失败 → 重发记忆引导
4. VPR 匹配成功:
   ├─ 匹配到目标节点 + sim ≥ 0.70 (VPR_ARRIVE_THRESHOLD):
   │   ├─ 最后一步 → 直接 advance
   │   ├─ 下一步子图匹配成功 (conf ≥ 0.60) → advance
   │   └─ 下一步子图匹配未成功 → VPR HELD 暂不切换
   └─ 匹配到其他节点 / sim < 0.70 → 继续当前步骤
5. VPR 匹配失败:
   ├─ 子图匹配成功 → 继续用子图匹配结果导航
   ├─ 子图匹配失败 + Qwen3.5 有结果 → 用打点结果导航
   └─ 子图匹配失败 + Qwen3.5 无结果 → 重发记忆引导
```

🟢 **Lookahead 双重确认**
- 每帧对当前步骤和下一步同时做子图匹配（lookahead 不走缓存逻辑）
- VPR 匹配到目标节点时，需下一步子图匹配成功（conf ≥ 0.60）才 advance
- 最后一步无需 lookahead，直接 advance
- advance 后使用 lookahead 结果驱动新步骤

### 2.4 导航状态机

系统通过 `MemoryNavState` 管理导航的完整状态：

状态流转：
```
idle → step_init → verifying → (回到 step_init / fallback / occluded / completed)
```

| 状态 | 说明 |
|------|------|
| idle | 空闲，无导航计划 |
| step_init | 步骤初始化，首次返回记忆引导 |
| verifying | VPR 持续验证中 |
| fallback | VPR 丢失 + 打点无结果，重发记忆引导 |
| occluded | 检测到遮挡，原地等待 |
| qwen35_fallback | VPR 丢失 + 未遮挡，Qwen3.5 打点导航 |
| completed | 导航完成，已到达终点 |

状态重置触发条件：
- task 变化（新任务）
- STOP 指令
- 导航完成

### WebSocket 响应格式

🔴 **响应格式（完整字段，与最新代码同步）**

```json
{
  "status": "success",
  "id": "robot_001",
  "pts": 1770097767,
  "task_status": "executing",
  "action": [[0.0, 0.0, 0.0]],
  "pixel_target": [0.45, 0.62],
  "camera_name": "camera_2",
  "landmark_name": "电梯门",
  "landmark_name_eng": "elevator door",
  "position_name_eng": "C8 front desk",
  "crop_image_paths": {
    "big": "path/to/big.jpg",
    "mid": "path/to/mid.jpg",
    "small": "path/to/small.jpg"
  },
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
  "fallback_instruction": null,
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

🟢 **新增/变更响应字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| camera_name | string | 子图匹配命中的相机（或得分最高相机） |
| landmark_name | string | 当前步骤注意力地标（中文） |
| crop_image_paths | object | 三级子图路径 {big, mid, small} |
| sub_image_match | object | 子图匹配详细结果 |
| fallback_instruction | string\|null | Qwen3.5 兜底时使用的 landmark 名称 |
| memory_active | bool | 是否处于记忆导航模式 |
| memory_info.frame_similarity | float\|null | 帧间 DINOv2 余弦相似度 |
| memory_info.cache_action | string\|null | 缓存操作：accepted/reused/cleared/no_cache |
| memory_info.consecutive_occlusions | int | 连续遮挡次数 |
| memory_info.occlusion | object\|null | 遮挡检测详细结果 |
| memory_info.lookahead_conf | float\|null | Lookahead 下一步子图匹配置信度 |
| memory_info.lookahead_found | bool\|null | Lookahead 下一步子图匹配是否成功 |
| memory_info.coord_transform | object\|null | 坐标转换调试信息 |

🟢 **WebSocket 控制命令**

| 命令 | 说明 |
|------|------|
| reset | 重置 Agent + 记忆导航状态 |
| session_status | 查看会话状态 |
| toggle_memory | 切换记忆导航开关 |
| memory_status | 查看记忆导航详情（含可用目的地列表） |
| reset_memory | 仅重置记忆状态（Agent 历史保留） |

---

## 三. 优化方向

🔵 **此部分无变化**（保持原样）

---

## 变更摘要（与旧文档对比）

### 🔴 需要修改的内容

1. **二. 导航步骤 → 第 3 点"模型兜底"**：InternVLN → Qwen3.5-9B 两步推理打点
2. **2.1.1 VPR 置信度**：改为"4 相机平均余弦相似度"
3. **2.2 NavigationStep 字段**：`crop_image_path` → `crop_image_paths`（三级）
4. **2.3.1 子图匹配方案**：SuperPoint + LightGlue → **DINOv3 密集特征 + 三级级联 + 全相机遍历**
5. **2.3.2 帧间相似度**：SSIM → **DINOv2 VPR 特征余弦相似度**（零额外开销）
6. **2.3 兜底模型**：InternVLN → **Qwen3.5-9B 两步推理法**
7. **导航决策**：移除趋势判断（Case B 跳步 / Case C 重规划 / Case D 趋势检测），替换为遮挡检测 + Qwen3.5 兜底
8. **子图匹配阈值**：统一为 **0.60**（`SUB_MATCH_CONFIDENCE_THRESHOLD`）

### 🟢 需要新增的内容

1. VPR 方法可配置（selavpr/anyloc/megaloc/effovpr），支持无序匹配模式
2. 鱼眼去畸变（自动从 cam/params.yaml 加载）
3. 像素→机器人坐标转换（完整物理管线）
4. 侧面相机旋转处理（camera_3/camera_4 输出旋转而非前进）
5. YOLOv8n 遮挡检测（子图匹配失败即触发，不依赖 VPR，面积阈值 25%）
6. Qwen3.5 两步推理法详细说明
7. Lookahead 双重确认机制（VPR 到达 + 下一步子图匹配）
8. VPR 到达阈值 `VPR_ARRIVE_THRESHOLD = 0.70`
9. best_fail_camera 追踪
10. WebSocket 控制命令列表
11. 完整响应字段（含 occlusion、lookahead、coord_transform）
