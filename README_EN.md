<div align="center">

# 🧠 MemoryNav

**Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-1.2.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v1.2.0)

A robot memory navigation system based on Visual Place Recognition (VPR) and topological mapping

**English** | [中文](README.md)

</div>

---

## 📖 Introduction

MemoryNav is a visual memory navigation system for mobile robots. It captures images from 4 surround-view fisheye cameras, localizes the robot within a pre-built topological memory graph using VPR, and generates navigation actions via the InternVLA vision-language model — enabling "remember where you've been, navigate there again" capability.

### Key Features

- **🔍 Multi-Method VPR**: 5 state-of-the-art VPR backends, switchable via a single parameter
- **🗺️ Topological Memory Graph**: Auto-built from labeled data with shortest-path planning
- **🔄 Cyclic Shift Matching**: 4-camera cyclic shift algorithm for orientation-invariant localization
- **🤖 VLA Fallback**: Automatic fallback to InternVLA when VPR loses track
- **🌐 WebSocket Service**: Real-time image streaming and navigation command output

---

## 🏗️ Architecture

```
MemoryNav/
├── deploy/                         # Deployment
│   ├── memory_nav/                 # Core memory navigation package
│   │   ├── memory_models.py        # Data models (Node, Edge, Plan, VPRResult)
│   │   ├── memory_graph.py         # Topological graph (BFS/Dijkstra planning)
│   │   ├── memory_vpr.py           # VPR matching engine (cyclic shift + order-invariant)
│   │   ├── memory_builder.py       # Memory builder (build graph from labeled data)
│   │   ├── memory_navigator.py     # Navigator main interface
│   │   ├── vpr_factory.py          # VPR extractor factory ⭐
│   │   ├── anyloc_extractor.py     # AnyLoc (DINOv2 + VLAD)
│   │   ├── megaloc_extractor.py    # MegaLoc (DINOv2 + OT aggregation) ⭐
│   │   ├── effovpr_extractor.py    # EffoVPR (DINOv2 multi-layer GeM) ⭐
│   │   └── selavpr_extractor.py    # SelaVPR++ (DINOv2 + MultiConv) ⭐
│   ├── ws_proxy_with_memory.py     # WebSocket proxy service (main entry)
│   └── build_memory.sh             # Memory build script
├── internnav/                      # InternNav framework
│   ├── agent/                      # Navigation agents (InternVLA, etc.)
│   ├── model/                      # Model definitions
│   ├── env/                        # Environment interfaces (Habitat, real robot)
│   └── evaluator/                  # Evaluation modules
├── scripts/                        # Utility scripts
│   └── memory_visualization_server.py  # Memory graph visualization
├── tests/                          # Tests
└── docs/                           # Documentation
```

---

## ✨ VPR Methods

MemoryNav v1.2.0 supports **5 VPR methods**, switchable via `vpr_method` parameter or `VPR_METHOD` environment variable:

| Method | Parameter | Venue | Feature Dim | Backbone | Highlights |
|--------|-----------|-------|-------------|----------|------------|
| **MegaLoc** | `megaloc` | CVPR 2025 | 8448D | DINOv2-B + OT | Best overall, SOTA on most benchmarks |
| **SelaVPR++** | `selavpr` | T-PAMI 2025 | 2048/4096D | DINOv2-B/L + MultiConv | Parameter-efficient, hashing support |
| **EffoVPR** | `effovpr` | arXiv 2024 | 768D | DINOv2-B multi-layer GeM | Ultra-compact, real-time friendly |
| **AnyLoc** | `anyloc` | RA-L 2023 | 6144D | DINOv2-B + VLAD | Classic and stable, default method |
| **LongCLIP** | `longclip` | - | 768D | LongCLIP | Legacy compatibility |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav
pip install -r requirements/base.txt
pip install -e .
```

### Build Memory Graph

```python
from deploy.memory_nav import MemoryBuilder

builder = MemoryBuilder(vpr_method='anyloc', device='cuda:0')
graph, vpr = builder.build_from_directory(
    'path/to/merged_labeled_data',
    save_path='memory_cache.pkl'
)
print(f'Built: {graph.get_stats()}')
```

### Launch Navigation Service

```bash
# Default: AnyLoc
python deploy/ws_proxy_with_memory.py

# MegaLoc (recommended)
VPR_METHOD=megaloc python deploy/ws_proxy_with_memory.py

# EffoVPR (lightweight)
VPR_METHOD=effovpr python deploy/ws_proxy_with_memory.py

# SelaVPR++
VPR_METHOD=selavpr python deploy/ws_proxy_with_memory.py
```

### Python API

```python
from deploy.memory_nav import MemoryNavigator

# Create navigator with chosen VPR method
navigator = MemoryNavigator(vpr_method='megaloc', device='cuda:0')
navigator.load_memory(path='memory_cache.pkl', data_dir='merged_labeled_data')

# VPR localization
images = {'camera_1': img1, 'camera_2': img2, 'camera_3': img3, 'camera_4': img4}
features = {cam: navigator.extractor.extract(img) for cam, img in images.items()}
result = navigator.vpr.locate(features)
print(f"Located: {result.matched_node_name}, similarity: {result.similarity:.4f}")

# Plan navigation
plan = navigator.navigate_to("Reception", start_node_id=result.matched_node_id)
for step in plan.steps:
    print(f"  → {step.to_node_name}, angle={step.angle:.1f}°")
```

---

## 🔧 Advanced VPR Configuration

### Using the Factory Function

```python
from deploy.memory_nav.vpr_factory import create_vpr_extractor

# MegaLoc
extractor, dim, order_inv = create_vpr_extractor('megaloc', device='cuda:0')

# EffoVPR with custom config
extractor, dim, order_inv = create_vpr_extractor('effovpr', device='cuda:0', config={
    'dino_model': 'dinov2_vitb14',
    'output_dim': 128,          # Ultra-compact 128D
    'layers': [8, 9, 10, 11]    # Last 4 layers
})

# SelaVPR++ with Large backbone
extractor, dim, order_inv = create_vpr_extractor('selavpr', device='cuda:0', config={
    'backbone': 'dinov2-large',  # 4096D output
    'aggregation': 'gem'
})

# AnyLoc with custom clusters
extractor, dim, order_inv = create_vpr_extractor('anyloc', device='cuda:0', config={
    'agg_mode': 'vlad',
    'num_clusters': 64,
    'domain': 'indoor'
})
```

---

## 📡 WebSocket Protocol

### Request

```json
{
    "id": "robot_01",
    "pts": 1709558400,
    "task": "Navigate to reception",
    "images": {
        "front_1": "<base64>",
        "camera_1": "<base64>",
        "camera_2": "<base64>",
        "camera_3": "<base64>",
        "camera_4": "<base64>"
    }
}
```

### Response

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
        "current_node": "Lobby",
        "target_node": "Reception",
        "remaining_steps": 3,
        "vpr_similarity": 0.85,
        "heading_offset": -37.5
    }
}
```

### Control Commands

| Command | Description |
|---------|-------------|
| `reset` | Reset agent and memory state |
| `toggle_memory` | Toggle memory navigation on/off |
| `memory_status` | View memory navigation details |
| `reset_memory` | Reset memory state only |
| `session_status` | View session status |

---

## 📐 Camera Layout

The system uses 4 fisheye cameras (equidistant projection, HFOV=190°):

```
             Front (0°)
               ↑
     cam_1 (-37.5°)  cam_2 (+37.5°)
               │
     cam_4 (-142.5°) cam_3 (+142.5°)
               ↓
             Rear (180°)
```

Cyclic shift matching supports 4 heading offsets: `0°`, `-75°`, `180°`, `+105°`

---

## 🧪 Testing

```bash
# Unit tests
python -m pytest tests/test_memory_nav.py -v

# WebSocket integration test
python tests/test_memory_ws.py
```

---

## 📚 Citation

If this project helps your research, please cite the relevant VPR papers:

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

This project is licensed under the [MIT License](LICENSE).
