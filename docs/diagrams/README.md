# 记忆导航系统流程图索引

## 功能特性 

本系统包含以下功能的流程图文档:

1. **语义引导VPR匹配** - 使用语义标签辅助视觉位置识别
2. **空间一致性验证** - 防止拓扑跳跃式误匹配
3. **模糊指令匹配** - 支持语义相似的指令匹配
4. **同义词扩展** - 28组中英文同义词字典
5. **跨语言支持** - 支持中英文混合指令

## 图表列表

| 图表名称 | 文件名 | 说明 |
|---------|--------|------|
| 系统总览图 (记忆开启) | `memory_navigation_overview.png` | [查看PDF](memory_navigation_overview.pdf) |
| 记忆关闭时的导航流程 | `memory_disabled_flow.png` | [查看PDF](memory_disabled_flow.pdf) |
| 相机图像使用分工 | `camera_usage_diagram.png` | [查看PDF](camera_usage_diagram.pdf) |
| VPR回环检测流程 (语义引导) | `vpr_detection_flow.png` | [查看PDF](vpr_detection_flow.pdf) |
| 拓扑地图管理流程 | `topological_map_flow.png` | [查看PDF](topological_map_flow.pdf) |
| 记忆回放决策流程 (模糊匹配) | `memory_replay_flow.png` | [查看PDF](memory_replay_flow.pdf) |
| 语义搜索流程 (同义词+模糊) | `semantic_search_flow.png` | [查看PDF](semantic_search_flow.pdf) |
| 完整导航流水线 | `full_navigation_pipeline.png` | [查看PDF](full_navigation_pipeline.pdf) |
| 特征提取详细流程 | `feature_extraction_detail.png` | [查看PDF](feature_extraction_detail.pdf) |
| 路线记录流程 | `route_recording_flow.png` | [查看PDF](route_recording_flow.pdf) |
| 配置参数说明 | `config_parameters.png` | [查看PDF](config_parameters.pdf) |

## 查看建议

### 记忆功能开启时 (memory_enabled=True)
1. **系统总览图** - 首先查看，了解整体架构和核心功能
2. **相机图像使用分工** - 理解前置相机和环视相机的分工
3. **完整导航流水线** - 了解数据流向
4. **VPR回环检测流程** - 理解语义引导的核心记忆匹配逻辑
5. **语义搜索流程** - 理解同义词扩展和模糊匹配
6. **拓扑地图管理流程** - 理解节点管理策略
7. **记忆回放决策流程** - 理解模糊匹配的回放决策
8. **特征提取详细流程** - 理解多相机特征处理
9. **路线记录流程** - 理解路线记忆机制
10. **配置参数说明** - 参数调优参考 (包含配置参数)

### 记忆功能关闭时 (memory_enabled=False)
1. **记忆关闭时的导航流程** - 了解关闭记忆时的简化流程

## 相机使用说明

| 相机 | 记忆开启时 | 记忆关闭时 |
|------|-----------|-----------|
| front_1 (前置) | 仅模型推理 | 仅模型推理 |
| camera_1~4 (环视) | 仅记忆模块 | 不使用 |

## 核心功能详情

### 语义引导VPR
- 语义标签数据库: 每个节点存储语义标签集合
- Jaccard相似度: 计算语义标签重叠度
- 语义加分/降分: 匹配加分+0.1, 冲突降分-0.05

### 空间一致性验证
- 拓扑跳跃检测: 候选节点与上次确认节点距离 ≤ 5
- 防止误匹配: 过滤掉空间上不合理的候选

### 模糊指令匹配
- 指令解析: 提取位置、动作、返回标记
- 加权相似度: 位置(3.0) + 动作(1.0) + 字符串(0.5)
- 阈值: 相似度 ≥ 0.65 视为匹配

### 同义词扩展
- 28组中英文同义词: 前台/reception, 走廊/corridor...
- 跨语言支持: "Go to front desk" ≈ "去前台"
- Levenshtein模糊匹配: 最大编辑距离 2

## 生成时间

生成于: 2026-01-26 
