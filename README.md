<div align="center">

# 🧠 MemoryNav

**视觉记忆导航系统 | Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-1.4.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v1.4.0)

基于视觉位置识别（VPR）和拓扑地图的机器人记忆导航系统

[English](README_EN.md) | **中文**

</div>

---

## 📖 简介

MemoryNav 是一个面向移动机器人的视觉记忆导航系统。系统通过 4 个环视鱼眼相机采集图像，利用 VPR 技术在预建的拓扑记忆图中定位，结合 InternVLA 视觉语言模型生成导航动作，实现"记住去过的地方，再走一次"的记忆导航能力。

### 核心能力

- **🔍 多方案 VPR 定位**：支持 4 种 SOTA 视觉位置识别方案，统一配置文件一键切换
- **🗺️ 拓扑记忆图**：自动从标注数据构建节点-边拓扑图，支持最短路径规划
- **🔄 循环移位匹配**：4 相机循环移位算法，支持任意朝向下的定位与偏转角估计
- **🎯 子图匹配导航**：基于 SuperPoint + LightGlue 的注意力子图定位，实时在相机图中匹配导航目标 *(v1.4.0 新增)*
- **🤖 VLA 兜底推理**：VPR 丢失时自动切换 InternVLA 模型继续导航
- **🌐 WebSocket 服务**：实时流式接收图像、返回导航指令
- **⚙️ 统一配置管理**：所有 VPR 参数集中在 `deploy/vpr_config.yaml`，一处修改全局生效

---

## 🆕 v1.4.0 更新亮点

> **架构升级：从角度导航到子图匹配导航**

| 特性 | v1.3.0（旧方案） | v1.4.0（新方案） |
|------|------------------|------------------|
| **边模型** | `angle + pixel_position + stitch_image` | `camera_name + crop_image + pixel_box` |
| **导航方式** | 服务端计算转向角度和像素目标 | 下发注意力子图，客户端自行匹配定位 |
| **目标定位** | 固定角度 + 像素坐标 | SuperPoint+LightGlue 实时子图匹配 |
| **灵活性** | 依赖精确标定 | 自适应视角变化，鲁棒性更强 |
| **可视化** | 基础拓扑图展示 | 新增子图匹配验证页面 |

---

## 🏗️ 系统架构

```
MemoryNav/
├── deploy/                         # 部署模块
│   ├── vpr_config.yaml             # VPR 统一配置文件
│   ├── memory_nav/                 # 核心记忆导航包
│   │   ├── vpr_config_loader.py    # 统一配置加载器
│   │   ├── memory_models.py        # 数据模型 (Node, Edge, Plan, VPRResult)
│   │   ├── memory_graph.py         # 拓扑图 (BFS/Dijkstra 路径规划)
│   │   ├── memory_vpr.py           # VPR 匹配引擎 (循环移位 + 无序匹配)
│   │   ├── memory_builder.py       # 记忆构建器 (从标注数据构建拓扑图)
│   │   ├── memory_navigator.py     # 导航器主接口
│   │   ├── sub_image_matcher.py    # 子图匹配器 (SuperPoint + LightGlue) ⭐ NEW
│   │   ├── vpr_factory.py          # VPR 提取器工厂
│   │   ├── anyloc_extractor.py     # AnyLoc (DINOv2 + VLAD)
│   │   ├── megaloc_extractor.py    # MegaLoc (DINOv2 + OT聚合)
│   │   ├── effovpr_extractor.py    # EffoVPR (DINOv2 多层CLS token)
│   │   └── selavpr_extractor.py    # SelaVPR++ (DINOv2 + MultiConv)
│   ├── ws_proxy_with_memory.py     # WebSocket 代理服务 (主入口)
│   └── build_memory.sh             # 记忆构建脚本
├── internnav/                      # InternNav 导航框架
├── scripts/                        # 工具脚本
│   └── memory_visualization_server.py  # 记忆图可视化服务 (含子图匹配验证)
├── tests/                          # 测试
│   ├── test_memory_nav.py          # 记忆模块单元测试
│   └── test_memory_ws.py           # WebSocket 集成测试 (详细日志版)
└── docs/                           # 文档
```

---

## 🎯 子图匹配导航（v1.4.0 新方案）

v1.4.0 引入基于 **SuperPoint + LightGlue** 的子图匹配，取代旧版的角度+像素坐标方案：

### 工作原理

1. **记忆构建时**：为每条边标注 `camera_name`（目标所在相机）和 `crop_image`（注意力子图）
2. **导航执行时**：从记忆中取出 crop 子图，在当前对应相机的实时画面中进行特征匹配
3. **目标定位**：SuperPoint 提取关键点 → LightGlue 匹配 → 计算目标区域百分比坐标
4. **回退机制**：匹配失败时使用记忆中的 `pixel_box` 作为估计值

### 边数据结构

```yaml
# 旧方案 (v1.3.0)
edge:
  angle: 37.5              # 转向角度
  pixel_position: [0.48, 0.52]  # 像素目标
  stitch_image: "stitch.jpg"    # 拼接图

# 新方案 (v1.4.0)
edge:
  camera_name: "camera_2"       # 目标所在相机
  landmark_name: "电梯"          # 地标名称
  pixel_box: [120, 80, 200, 160]  # (x, y, w, h) 像素框
  crop_image_path: "crop_elevator.jpg"  # 注意力子图
```

---

## ✨ VPR 方案对比

MemoryNav 支持 **4 种** VPR 方案，通过 `deploy/vpr_config.yaml` 统一切换：

| 方案 | 参数值 | 发表 | 特征维度 | Backbone | 特点 |
|------|--------|------|---------|----------|------|
| **SelaVPR++** ⭐ | `selavpr` | T-PAMI 2025 | 4096D | DINOv2-L + MultiConv | **推荐方案**，支持 hashing+rerank，官方最强配置 |
| **MegaLoc** | `megaloc` | CVPR 2025 | 8448D | DINOv2-B + OT聚合 | 综合性能最强，多数据集 SOTA |
| **EffoVPR** | `effovpr` | arXiv 2024 | 3072D | DINOv2-B 多层CLS | 轻量快速，适合实时场景 |
| **AnyLoc** | `anyloc` | RA-L 2023 | 可配置 | DINOv2-B + VLAD | 经典稳定，聚类数可调 |

---

## ⚙️ 统一配置

所有 VPR 相关参数集中在 `deploy/vpr_config.yaml` 中管理，修改后重启服务即可生效：

```yaml
# VPR 方法: selavpr | megaloc | effovpr | anyloc
vpr_method: selavpr

# GPU 设备
device: "cuda:0"

# VPR 相似度阈值 (各方案独立设置)
similarity_threshold:
  selavpr: 0.60
  megaloc: 0.60
  effovpr: 0.80
  anyloc: 0.70

# SelaVPR++ 专用配置
selavpr:
  backbone: dinov2-large      # dinov2-base (2048D) 或 dinov2-large (4096D)
  aggregation: gem            # gem, boq, salad
  use_hashing: true           # 开启深度哈希
  use_rerank: true            # 开启重排 (需 use_hashing=true)

# AnyLoc 专用配置
anyloc:
  dino_model: dinov2_vitb14
  agg_mode: vlad
  num_clusters: 32
  domain: indoor
  max_img_size: 630
```

**切换方案只需修改 `vpr_method` 一行**，以下模块自动读取统一配置：
- `ws_proxy_with_memory.py` — WebSocket 导航服务
- `memory_visualization_server.py` — 可视化服务
- `memory_builder.py` / `memory_navigator.py` — 核心模块
- `build_memory.sh` — 构建脚本

> ⚠️ 切换 VPR 方案后需要重新构建记忆缓存：`bash deploy/build_memory.sh`

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav
pip install -r requirements/base.txt
pip install -e .
```

### 配置 VPR 方案

编辑 `deploy/vpr_config.yaml`，选择你需要的 VPR 方案和参数。

### 构建记忆库

```bash
# 自动从 vpr_config.yaml 读取 VPR 方案
bash deploy/build_memory.sh

# 或指定参数覆盖
bash deploy/build_memory.sh --method megaloc --gpu 0
```

### 启动导航服务

```bash
# 自动从 vpr_config.yaml 读取配置
python deploy/ws_proxy_with_memory.py
```

### Python API

```python
from deploy.memory_nav import MemoryNavigator

# 自动使用 vpr_config.yaml 中的配置
navigator = MemoryNavigator(vpr_method='selavpr', device='cuda:0')
navigator.load_memory(path='memory_cache.pkl', data_dir='merged_labeled_data')

# VPR 定位
images = {'camera_1': img1, 'camera_2': img2, 'camera_3': img3, 'camera_4': img4}
features = {cam: navigator.extractor.extract(img) for cam, img in images.items()}
result = navigator.vpr.locate(features)
print(f"定位: {result.matched_node_name}, 相似度: {result.similarity:.4f}")

# 规划导航
plan = navigator.navigate_to("前台", camera_images=images)
for step in plan['plan']['steps']:
    print(f"  → {step['to_node']['name']}, camera={step['camera_name']}, landmark={step['landmark_name']}")

# 子图匹配（导航执行中）
match = navigator.match_current_step(images)
if match and match['match']['found']:
    print(f"目标定位: ({match['match']['center_x_pct']:.1f}%, {match['match']['center_y_pct']:.1f}%)")
```

---

## 📡 WebSocket 协议

### 请求格式

```json
{
    "id": "robot_01",
    "pts": 1709558400,
    "task": "导航到前台",
    "images": {
        "front_1": "<base64>",
        "camera_1": "<base64>",
        "camera_2": "<base64>",
        "camera_3": "<base64>",
        "camera_4": "<base64>"
    }
}
```

### 响应格式

```json
{
    "status": "success",
    "id": "robot_01",
    "task_status": "executing",
    "action": [[0.5, 0.0, 0.1]],
    "memory_active": true,
    "memory_info": {
        "phase": "verifying",
        "current_step": 1,
        "total_steps": 3,
        "from_node": "大厅",
        "to_node": "前台",
        "camera_name": "camera_2",
        "landmark_name": "电梯",
        "crop_image_path": "merged_labeled_data/node_5/crop_elevator.jpg",
        "pixel_box": [120, 80, 200, 160],
        "vpr_similarity": 0.85,
        "vpr_confidence": 0.85,
        "vpr_matched_node": "node_5",
        "heading_offset": -37.5,
        "consecutive_misses": 0
    },
    "sub_image_match": {
        "camera_name": "camera_2",
        "landmark_name": "电梯",
        "match": {
            "found": true,
            "confidence": 0.92,
            "center_x_pct": 48.5,
            "center_y_pct": 52.1,
            "x_min_pct": 30.2,
            "y_min_pct": 35.8,
            "x_max_pct": 66.8,
            "y_max_pct": 68.4
        }
    }
}
```

### 控制命令

| 命令 | 说明 |
|------|------|
| `reset` | 重置 Agent 和记忆状态 |
| `toggle_memory` | 切换记忆导航开关 |
| `memory_status` | 查看记忆导航详情 |
| `reset_memory` | 仅重置记忆状态 |
| `session_status` | 查看会话状态 |

---

## 📐 相机布局

系统使用 4 个鱼眼相机（等角投影，HFOV=190°）：

```
            前方 (0°)
              ↑
     cam_1 (-37.5°)  cam_2 (+37.5°)
              │
     cam_4 (-142.5°) cam_3 (+142.5°)
              ↓
            后方 (180°)
```

循环移位匹配支持 4 种朝向偏移：`0°`, `-75°`, `180°`, `+105°`

---

## 🧪 测试

```bash
# 单元测试
python -m pytest tests/test_memory_nav.py -v

# WebSocket 集成测试 (含逐帧VPR决策日志 + 统计报告 + 相似度趋势图)
python tests/test_memory_ws.py
```

测试输出包含：
- 📊 逐帧 VPR 匹配详情（相似度、置信度、匹配节点、决策类型）
- 📈 VPR 相似度变化趋势 ASCII 图
- 📋 统计报告（匹配率、节点分布、决策分布、Phase 分布）

---

## 📚 引用

如果本项目对您的研究有帮助，请引用相关 VPR 论文：

```bibtex
@article{selavprpp2025,
  title={SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition},
  author={Lu, Feng and Jin, Tong and others},
  journal={IEEE T-PAMI},
  year={2026},
  volume={48},
  number={3},
  pages={2731-2748}
}

@inproceedings{megaloc2025,
  title={MegaLoc: One Retrieval to Place Them All},
  author={Berton, Gabriele and Masone, Carlo},
  booktitle={CVPR Workshops},
  year={2025}
}

@article{effovpr2024,
  title={Effective Foundation Model Utilization for Visual Place Recognition},
  author={Tzachor, Issar and others},
  journal={arXiv:2405.18065},
  year={2024}
}

@article{anyloc2023,
  title={AnyLoc: Towards Universal Visual Place Recognition},
  author={Keetha, Nikhil and others},
  journal={IEEE RA-L},
  year={2023}
}
```

---

## 📄 License

本项目采用 [MIT License](LICENSE) 开源协议。
