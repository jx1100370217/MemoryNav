<div align="center">

# 🧠 MemoryNav

**视觉记忆导航系统 | Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-1.2.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v1.2.0)

基于视觉位置识别（VPR）和拓扑地图的机器人记忆导航系统

[English](README_EN.md) | **中文**

</div>

---

## 📖 简介

MemoryNav 是一个面向移动机器人的视觉记忆导航系统。系统通过 4 个环视鱼眼相机采集图像，利用 VPR 技术在预建的拓扑记忆图中定位，结合 InternVLA 视觉语言模型生成导航动作，实现"记住去过的地方，再走一次"的记忆导航能力。

### 核心能力

- **🔍 多方案 VPR 定位**：支持 5 种 SOTA 视觉位置识别方案，通过参数一键切换
- **🗺️ 拓扑记忆图**：自动从标注数据构建节点-边拓扑图，支持最短路径规划
- **🔄 循环移位匹配**：4 相机循环移位算法，支持任意朝向下的定位与偏转角估计
- **🤖 VLA 兜底推理**：VPR 丢失时自动切换 InternVLA 模型继续导航
- **🌐 WebSocket 服务**：实时流式接收图像、返回导航指令

---

## 🏗️ 系统架构

```
MemoryNav/
├── deploy/                         # 部署模块
│   ├── memory_nav/                 # 核心记忆导航包
│   │   ├── memory_models.py        # 数据模型 (Node, Edge, Plan, VPRResult)
│   │   ├── memory_graph.py         # 拓扑图 (BFS/Dijkstra 路径规划)
│   │   ├── memory_vpr.py           # VPR 匹配引擎 (循环移位 + 无序匹配)
│   │   ├── memory_builder.py       # 记忆构建器 (从标注数据构建拓扑图)
│   │   ├── memory_navigator.py     # 导航器主接口
│   │   ├── vpr_factory.py          # VPR 提取器工厂 ⭐
│   │   ├── anyloc_extractor.py     # AnyLoc (DINOv2 + VLAD)
│   │   ├── megaloc_extractor.py    # MegaLoc (DINOv2 + OT聚合) ⭐
│   │   ├── effovpr_extractor.py    # EffoVPR (DINOv2 多层GeM) ⭐
│   │   └── selavpr_extractor.py    # SelaVPR++ (DINOv2 + MultiConv) ⭐
│   ├── ws_proxy_with_memory.py     # WebSocket 代理服务 (主入口)
│   └── build_memory.sh             # 记忆构建脚本
├── internnav/                      # InternNav 导航框架
│   ├── agent/                      # 导航智能体 (InternVLA 等)
│   ├── model/                      # 模型定义
│   ├── env/                        # 环境接口 (Habitat, 真机)
│   └── evaluator/                  # 评估模块
├── scripts/                        # 工具脚本
│   └── memory_visualization_server.py  # 记忆图可视化服务
├── tests/                          # 测试
│   ├── test_memory_nav.py          # 记忆模块单元测试
│   └── test_memory_ws.py           # WebSocket 集成测试
└── docs/                           # 文档
```

---

## ✨ VPR 方案对比

MemoryNav v1.2.0 支持 **5 种** VPR 方案，通过 `vpr_method` 参数或 `VPR_METHOD` 环境变量切换：

| 方案 | 参数值 | 发表 | 特征维度 | Backbone | 特点 |
|------|--------|------|---------|----------|------|
| **MegaLoc** | `megaloc` | CVPR 2025 | 8448D | DINOv2-B + OT聚合 | 综合性能最强，多数据集 SOTA |
| **SelaVPR++** | `selavpr` | T-PAMI 2025 | 2048/4096D | DINOv2-B/L + MultiConv | 参数高效适配，支持哈希重排 |
| **EffoVPR** | `effovpr` | arXiv 2024 | 768D | DINOv2-B 多层GeM | 超紧凑特征，适合实时场景 |
| **AnyLoc** | `anyloc` | RA-L 2023 | 6144D | DINOv2-B + VLAD | 经典稳定，默认方案 |
| **LongCLIP** | `longclip` | - | 768D | LongCLIP | 兼容旧版本 |

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav
pip install -r requirements/base.txt
pip install -e .
```

### 构建记忆库

```bash
# 从标注数据构建记忆拓扑图
python -c "
from deploy.memory_nav import MemoryBuilder

builder = MemoryBuilder(vpr_method='anyloc', device='cuda:0')
graph, vpr = builder.build_from_directory(
    'path/to/merged_labeled_data',
    save_path='memory_cache.pkl'
)
print(f'构建完成: {graph.get_stats()}')
"
```

### 启动导航服务

```bash
# 默认使用 AnyLoc
python deploy/ws_proxy_with_memory.py

# 使用 MegaLoc（推荐）
VPR_METHOD=megaloc python deploy/ws_proxy_with_memory.py

# 使用 EffoVPR（轻量快速）
VPR_METHOD=effovpr python deploy/ws_proxy_with_memory.py

# 使用 SelaVPR++
VPR_METHOD=selavpr python deploy/ws_proxy_with_memory.py
```

### Python API

```python
from deploy.memory_nav import MemoryNavigator

# 创建导航器（指定 VPR 方案）
navigator = MemoryNavigator(vpr_method='megaloc', device='cuda:0')
navigator.load_memory(path='memory_cache.pkl', data_dir='merged_labeled_data')

# VPR 定位
images = {'camera_1': img1, 'camera_2': img2, 'camera_3': img3, 'camera_4': img4}
features = {cam: navigator.extractor.extract(img) for cam, img in images.items()}
result = navigator.vpr.locate(features)
print(f"定位: {result.matched_node_name}, 相似度: {result.similarity:.4f}")

# 规划导航
plan = navigator.navigate_to("前台", start_node_id=result.matched_node_id)
for step in plan.steps:
    print(f"  → {step.to_node_name}, angle={step.angle:.1f}°")
```

---

## 🔧 VPR 方案详细配置

### 使用工厂函数

```python
from deploy.memory_nav.vpr_factory import create_vpr_extractor

# MegaLoc
extractor, dim, order_inv = create_vpr_extractor('megaloc', device='cuda:0')

# EffoVPR (自定义层数和维度)
extractor, dim, order_inv = create_vpr_extractor('effovpr', device='cuda:0', config={
    'dino_model': 'dinov2_vitb14',
    'output_dim': 128,       # 超紧凑 128D
    'layers': [8, 9, 10, 11] # 使用最后4层
})

# SelaVPR++ (使用 Large backbone)
extractor, dim, order_inv = create_vpr_extractor('selavpr', device='cuda:0', config={
    'backbone': 'dinov2-large',  # 4096D 输出
    'aggregation': 'gem'
})

# AnyLoc (自定义聚类数)
extractor, dim, order_inv = create_vpr_extractor('anyloc', device='cuda:0', config={
    'agg_mode': 'vlad',
    'num_clusters': 64,
    'domain': 'indoor'
})
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
    "pixel_target": [0.48, 0.52],
    "angle": 37.5,
    "memory_active": true,
    "memory_info": {
        "current_node": "大厅",
        "target_node": "前台",
        "remaining_steps": 3,
        "vpr_similarity": 0.85,
        "heading_offset": -37.5
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

# WebSocket 集成测试
python tests/test_memory_ws.py
```

---

## 📚 引用

如果本项目对您的研究有帮助，请引用相关 VPR 论文：

```bibtex
@inproceedings{megaloc2025,
  title={MegaLoc: One Retrieval to Place Them All},
  author={Berton, Gabriele and Masone, Carlo},
  booktitle={CVPR Workshops},
  year={2025}
}

@article{selavprpp2025,
  title={SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition},
  author={Lu, Feng and Jin, Tong and others},
  journal={IEEE T-PAMI},
  year={2026}
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
