# 纯视觉记忆导航方案-MemoryNav — 文档更新稿

> 说明：此文件是根据最新代码（截止 2026-03-19）对飞书文档的更新稿。
> 标注了所有与旧文档的差异，方便主人对照更新飞书。
> 🔴 = 需修改  🟢 = 新增  🔵 = 不变

---

# 纯视觉记忆导航方案-MemoryNav

## 一. 输入

文本指令：前往A8档案室
视觉图片：去畸变的4张鱼眼相机图 camera_1~camera_4 + 前置相机图 front_1

（系统架构图保持不变）

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

- **task**: 自然语言导航指令（如 "前往C8前台"、"go to A8 front desk"），也支持直接控制指令（"turn left"、"turn right"、"go straight"、"STOP"）
- **images.front_1**: 前置 RGB 图，用于 InternVLA 模型推理（调整为 640×480）
- **images.camera_1~4**: 四方向鱼眼相机图(已去畸变)，用于 VPR 定位和子图匹配

🔵 **此部分无变化**

---

## 二. 导航步骤

系统采用**三层导航策略**：

1. **记忆引导**：通过记忆拓扑图规划路径，每步返回目标相机方向 + 注意力子图匹配区域
2. **VPR 持续验证**：每帧请求都用 camera_1~4 做 VPR 定位，判断是否到达下一节点
3. 🔴 **模型兜底**：VPR 丢失或子图匹配失败时，用 **Qwen3.5-9B 打点模型**进行兜底定位（替代原 InternVLN 模型推理）

> 旧文档写"InternVLN 模型推理生成动作"，代码中兜底已改为 Qwen3.5-9B 打点。InternVLA 仍作为最终 fallback 保留（按需加载），但优先使用 Qwen3.5。

### 2.1 定位

#### 2.1.1 导航起点

🔵 **此部分基本无变化**

基于当前位置的 camera_1~camera_4 通过视觉位置识别（VPR）确定起点 node。

VPR 定位流程：
1. 对 camera_1~4 的图像分别提取 VPR 特征向量
2. 与记忆拓扑图中每个节点存储的 4 个相机特征做相似度比较
3. 支持**环视循环移位匹配（Circular Shift）**：考虑机器人朝向不同，对 4 个相机进行循环移位匹配（共 4 种 shift），选取最佳 shift 得到匹配节点和朝向偏移（heading_offset）
4. 🔴 相似度超过阈值（默认 0.70）时，返回匹配节点。**置信度 = 4 相机平均余弦相似度**（不再使用最佳 vs 次佳差异计算）

🟢 **新增：VPR 方法可配置**
- 支持多种 VPR 方法：`selavpr`（默认）、`anyloc`、`megaloc`、`effovpr`
- 通过 `vpr_config_loader.py` 配置

🟢 **新增：无序匹配模式（AnyLoc 模式）**
- 当 `order_invariant=True` 时，不依赖 camera_id 的对应关系
- 每张 query 图与 memory 中所有相机比较，通过贪心最优匹配找到最佳节点

#### 2.1.2 导航终点

🔵 **此部分基本无变化**（语义匹配目的地节点）

🟢 **新增细节：目的地查找支持多种查询格式**
- 直接节点名："C8前台区"
- 自然语言任务："前往C8前台"、"go to A8 front desk"（自动去除前缀"前往/去/go to"等）
- 节点 ID："8"
- 子串模糊搜索

### 2.2 定路线

🔵 **此部分无变化**（基于拓扑图最短路径规划）

导航计划（NavigationPlan）包含：
- 起点/终点节点信息（ID、中文名、英文名）
- 完整路径节点序列
- 逐步导航步骤（NavigationStep），每步包含：
  - from_node → to_node（起止节点）
  - camera_name：目标所在的相机方向（camera_1~camera_4）
  - landmark_name：注意力目标地标名称
  - 🔴 **crop_image_paths**：**三级注意力子图路径** `{"big": ..., "mid": ..., "small": ...}`（替代原单一 crop_image_path）
  - 🔴 crop_image_path：向 big 尺度的兼容引用
  - 🔴 ~~pixel_box~~：**已移除**，子图定位改由实时子图匹配完成

🟢 **新增：跳步机制**
- 当 VPR 匹配到路径中更后面的节点时，自动跳过中间步骤

🟢 **新增：重规划机制**
- 当 VPR 以高置信度（≥0.8）匹配到路径外的节点时，自动从新位置重新规划路径

### 2.3 定方向

确定了行驶路线后，就是逐段（比如 12→4）按照规划路线移动到终点。该部分的核心是返回给机器人控制端**运动目标点**。

#### 2.3.1 子图匹配定位

🔴 **重大变更：子图匹配方案已从 SuperPoint + LightGlue 更换为 DINOv3 密集特征匹配**

目前的方案是通过**注意力子图**（记忆中保存的关键地标 crop 图）和当前 camera_1~camera_4 进行图像匹配，从而确定机器人的目标方向。

🔴 **子图匹配器（SubImageMatcher）基于 DINOv3 密集特征匹配**：

1. **特征提取**：使用 DINOv3 (ViT-B/16, 通过 timm 加载) 提取两张图的 patch token 密集特征图
2. **滑动窗口匹配**：在相机图特征图上滑动窗口，计算与子图特征图的余弦相似度
3. **最佳位置**：找到相似度最高的位置作为匹配结果

🔴 **置信度阈值**：匹配成功阈值为 **0.35**（`SUB_MATCH_CONFIDENCE_THRESHOLD`）

🔴 **级联匹配策略（Cascade Matching）**：
- 对每个相机（camera_1~camera_4）执行三级级联匹配：**small → mid → big**
- 遍历所有 4 个相机，选择 confidence 最高的匹配结果
- 不再局限于记忆中标注的 camera_name，而是全局搜索最佳匹配

匹配结果（SubImageMatchResult）：
- 像素坐标 bounding box（x_min, y_min, x_max, y_max）
- 百分比 bounding box（与 WebUI 一致的 top_left_pct / bottom_right_pct / center_pct）
- confidence：匹配置信度 [0, 1]
- method：使用的匹配方案名称

#### 2.3.2 帧间相似度缓存

🔴 **帧间相似度已从 SSIM 改为基于 DINOv2 VPR 特征的余弦相似度**

核心逻辑（`_cache_or_reuse_sub_match`）：

- **匹配成功（conf ≥ 0.35）**：采纳结果，更新缓存（保存当前帧的 VPR 特征用于下次比较）
- **匹配失败（conf < 0.35）**：
  - 若前后帧 DINOv2 特征余弦相似度 ≥ **0.70**（`FRAME_SIMILARITY_THRESHOLD`），认为场景几乎没变，**复用上一帧的匹配框**
  - 否则认为场景变化大，**清除缓存**
- **无缓存可用**：返回 None，触发 Qwen3.5 兜底

优势（相比 SSIM）：
- 复用 VPR 流程已提取的特征，**零额外推理成本**
- 语义级比较，对光照变化鲁棒，对微小运动不过度敏感

🟢 **匹配结果可视化**：
- 匹配成功：绿色框 + 红色中心点
- 缓存复用：黄色框 + 蓝色中心点
- 低置信度：灰色框 + 灰色中心点
- 保存到 `deploy/logs/images/` 目录

#### 2.3.3 Qwen3.5 兜底打点

🔴 **重大变更：兜底模型从 InternVLN 改为 Qwen3.5-9B 打点模型**

当子图匹配失败且无缓存可用时，使用 Qwen3.5-9B VLM 对当前步骤的 landmark_name 进行视觉打点定位。

🟢 **两步推理法**（解决了 Qwen3.5 "无论目标是否存在都强行输出坐标"的 BUG）：

1. **存在性检测**（`check_existence`）：
   - Prompt: `Is "xxx" visible in this image? Answer: yes or no`
   - 仅生成 1~2 个 token，延迟极小
   - 回答 "no" → 直接返回 `not_found`，不执行打点

2. **条件打点**（仅在确认目标存在时执行）：
   - Prompt: `Point to "xxx" in this image. Output ONLY JSON: {"point": [x, y]}` (坐标范围 0~1000)
   - 输出归一化到 [0, 1] 的坐标

🟢 **Qwen3.5 运行架构**：
- 通过**子进程**运行在独立的 `qwen3` conda 环境（transformers 5.x），避免版本冲突
- 主进程通过 stdin/stdout JSON 行协议通信
- 默认运行在 GPU 1

🟢 **多相机搜索策略**（`predict_on_camera`）：
- 优先尝试 step 指定的 target_camera
- 失败后遍历其余相机
- 找到置信度 ≥ 0.5 的结果即提前退出

🟢 **打点结果可视化**：
- 橙色十字准星 + 同心圆 + 中文地标名称标注
- 保存到 `deploy/logs/images/`

> InternVLA 仍保留作为最终兜底（按需加载，默认不启动），在 Qwen3.5 也无法处理时使用。

#### 2.3.4 VPR 持续验证与趋势检测

🔵 **此部分基本无变化**，但补充一些代码中的细节：

相似度趋势检测（VPR 丢失时）：
系统持续记录当前帧与源节点、目标节点的 VPR 相似度，通过滑动窗口（TREND_WINDOW=2）分析趋势：

| 趋势 | 含义 | 动作 |
|------|------|------|
| source↓ target↑ | 远离源、接近目标 → 方向正确 | 🔴 优先使用 Qwen3.5 兜底 action（如有），否则直行 1m |
| source↓ target↓ | 远离源、远离目标 → 偏离路线 | 重发记忆引导纠偏 |
| 连续偏离 ≥ 5 次 | 严重偏离 | 强制 advance 到下一步 |
| 其他/趋势不明 | 可能在源节点附近 | 重发记忆引导 |
| 样本不足 | 刚开始走 | 走 InternVLA 兜底推理 |

🟢 **新增细节：趋势正确时的动作选择**
- 如果有 Qwen3.5 兜底打点结果（子图匹配失败时自动触发），使用其 pixel_target
- 如果子图匹配成功，使用子图匹配的 pixel_target
- 否则发送直行 1m 指令 `[[1.0, 0.0, 0.0]]`

### 2.4 导航状态机

🔵 **此部分基本无变化**

系统通过 `MemoryNavState` 管理导航的完整状态：

状态流转：
```
idle → step_init → verifying → (回到 step_init / fallback / completed)
```

| 状态 | 说明 |
|------|------|
| idle | 空闲，无导航计划 |
| step_init | 步骤初始化，首次返回记忆引导 |
| verifying | VPR 持续验证中 |
| fallback | VPR 丢失，走模型兜底推理 |
| completed | 导航完成，已到达终点 |

🟢 **新增状态：trend_go_straight**
- 趋势检测判断方向正确时的临时状态（在 memory_info.phase 中返回）

状态重置触发条件：
- task 变化（新任务）
- STOP 指令
- 导航完成

### WebSocket 响应格式

🔴 **响应格式已大幅扩展**

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
    "consecutive_misses": 0
  },
  "message": "记忆导航: C8走廊 → C8前台区 (步骤1/2)"
}
```

🟢 **新增响应字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| camera_name | string | 子图匹配命中的相机名称 |
| landmark_name | string | 当前步骤的注意力目标地标（中文） |
| landmark_name_eng | string | 地标英文名 |
| position_name_eng | string | 目标节点英文名 |
| crop_image_paths | object | 三级子图路径 {big, mid, small} |
| sub_image_match | object | 子图匹配详细结果 |
| fallback_instruction | string\|null | Qwen3.5 兜底时使用的 landmark 名称 |
| memory_active | bool | 是否处于记忆导航模式 |
| memory_info | object | 记忆导航状态详情 |
| memory_info.frame_similarity | float\|null | 帧间 DINOv2 余弦相似度 |
| memory_info.cache_action | string\|null | 缓存操作：accepted/reused/cleared/no_cache |

🟢 **新增 WebSocket 控制命令**：

| 命令 | 说明 |
|------|------|
| reset | 重置 Agent + 记忆导航状态 |
| session_status | 查看会话状态 |
| toggle_memory | 切换记忆导航开关 |
| memory_status | 查看记忆导航详情（含可用目的地列表） |
| reset_memory | 仅重置记忆状态（Agent 历史保留） |

---

## 三. 优化方向

🔵 **此部分无变化**（3.1~3.5 均为规划中的优化方向，保持原样）

---

## 变更摘要

### 🔴 需要修改的内容（共 7 处）

1. **二. 导航步骤 → 第 3 点"模型兜底"**：InternVLN → Qwen3.5-9B 打点模型
2. **2.1.1 VPR 置信度**：不再是"最佳 vs 次佳差异"，而是"4 相机平均余弦相似度"
3. **2.2 NavigationStep 字段**：`pixel_box` 已移除，`crop_image_path` 改为 `crop_image_paths`（三级）
4. **2.3.1 子图匹配方案**：SuperPoint + LightGlue + Homography → **DINOv3 密集特征匹配**
5. **2.3.2 帧间相似度**：SSIM → **DINOv2 VPR 特征余弦相似度**（零额外开销）
6. **2.3.3 兜底模型**：InternVLN → **Qwen3.5-9B 两步推理法**（存在性检测 + 条件打点）
7. **WebSocket 响应格式**：大幅扩展，新增多个字段

### 🟢 需要新增的内容（共 6 处）

1. VPR 方法可配置（selavpr/anyloc/megaloc/effovpr）
2. 无序匹配模式（AnyLoc 模式）
3. 目的地查找的多种查询格式
4. 跳步机制 & 重规划机制
5. Qwen3.5 两步推理法详细说明
6. WebSocket 控制命令列表
