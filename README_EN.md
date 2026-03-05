<div align="center">

# 🧠 MemoryNav

**Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-1.3.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v1.3.0)

A robot memory navigation system based on Visual Place Recognition (VPR) and topological mapping

**English** | [中文](README.md)

</div>

---

## 📖 Introduction

MemoryNav is a visual memory navigation system for mobile robots. It captures images from 4 surround-view fisheye cameras, localizes the robot within a pre-built topological memory graph using VPR, and generates navigation actions via the InternVLA vision-language model — enabling "remember where you've been, navigate there again" capability.

### Key Features

- **🔍 Multi-Method VPR**: 4 state-of-the-art VPR backends, switchable via a single config file
- **🗺️ Topological Memory Graph**: Auto-built from labeled data with shortest-path planning
- **🔄 Cyclic Shift Matching**: 4-camera cyclic shift algorithm for orientation-invariant localization
- **🤖 VLA Fallback**: Automatic fallback to InternVLA when VPR loses track
- **🌐 WebSocket Service**: Real-time image streaming and navigation command output
- **⚙️ Unified Configuration**: All VPR parameters in `deploy/vpr_config.yaml`, one change applies everywhere

---

## 🏗️ Architecture

```
MemoryNav/
├── deploy/                         # Deployment
│   ├── vpr_config.yaml             # Unified VPR config ⭐ (NEW)
│   ├── memory_nav/                 # Core memory navigation package
│   │   ├── vpr_config_loader.py    # Unified config loader ⭐ (NEW)
│   │   ├── memory_models.py        # Data models (Node, Edge, Plan, VPRResult)
│   │   ├── memory_graph.py         # Topological graph (BFS/Dijkstra planning)
│   │   ├── memory_vpr.py           # VPR matching engine (cyclic shift + order-invariant)
│   │   ├── memory_builder.py       # Memory builder (build graph from labeled data)
│   │   ├── memory_navigator.py     # Navigator main interface
│   │   ├── vpr_factory.py          # VPR extractor factory
│   │   ├── anyloc_extractor.py     # AnyLoc (DINOv2 + VLAD)
│   │   ├── megaloc_extractor.py    # MegaLoc (DINOv2 + OT aggregation)
│   │   ├── effovpr_extractor.py    # EffoVPR (DINOv2 multi-layer CLS token)
│   │   └── selavpr_extractor.py    # SelaVPR++ (DINOv2 + MultiConv)
│   ├── ws_proxy_with_memory.py     # WebSocket proxy service (main entry)
│   └── build_memory.sh             # Memory build script
├── internnav/                      # InternNav framework
├── scripts/                        # Utility scripts
│   └── memory_visualization_server.py  # Memory graph visualization
├── tests/                          # Tests
│   ├── test_memory_nav.py          # Unit tests
│   └── test_memory_ws.py           # WebSocket integration test (detailed logging)
└── docs/                           # Documentation
```

---

## ✨ VPR Methods

MemoryNav v1.3.0 supports **4 VPR methods**, configurable via `deploy/vpr_config.yaml`:

| Method | Parameter | Venue | Feature Dim | Backbone | Highlights |
|--------|-----------|-------|-------------|----------|------------|
| **SelaVPR++** ⭐ | `selavpr` | T-PAMI 2025 | 4096D | DINOv2-L + MultiConv | **Recommended**, hashing+rerank, official best config |
| **MegaLoc** | `megaloc` | CVPR 2025 | 8448D | DINOv2-B + OT | Best overall, SOTA on most benchmarks |
| **EffoVPR** | `effovpr` | arXiv 2024 | 3072D | DINOv2-B multi-layer CLS | Lightweight, real-time friendly |
| **AnyLoc** | `anyloc` | RA-L 2023 | Configurable | DINOv2-B + VLAD | Classic and stable, tunable clusters |

---

## ⚙️ Unified Configuration

All VPR parameters are managed in `deploy/vpr_config.yaml`. Changes take effect after restarting the service:

```yaml
# VPR method: selavpr | megaloc | effovpr | anyloc
vpr_method: selavpr

# GPU device
device: "cuda:0"

# Similarity thresholds (per-method)
similarity_threshold:
  selavpr: 0.60
  megaloc: 0.60
  effovpr: 0.80
  anyloc: 0.70

# SelaVPR++ specific config
selavpr:
  backbone: dinov2-large      # dinov2-base (2048D) or dinov2-large (4096D)
  aggregation: gem            # gem, boq, salad
  use_hashing: true           # Enable deep hashing
  use_rerank: true            # Enable re-ranking (requires use_hashing=true)

# AnyLoc specific config
anyloc:
  dino_model: dinov2_vitb14
  agg_mode: vlad
  num_clusters: 32
  domain: indoor
  max_img_size: 630
```

**To switch VPR methods, just change the `vpr_method` line.** All modules read from this config:
- `ws_proxy_with_memory.py` — WebSocket navigation service
- `memory_visualization_server.py` — Visualization service
- `memory_builder.py` / `memory_navigator.py` — Core modules
- `build_memory.sh` — Build script

> ⚠️ After switching VPR methods, rebuild the memory cache: `bash deploy/build_memory.sh`

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav
pip install -r requirements/base.txt
pip install -e .
```

### Configure VPR Method

Edit `deploy/vpr_config.yaml` to select your VPR method and parameters.

### Build Memory Graph

```bash
# Automatically reads VPR method from vpr_config.yaml
bash deploy/build_memory.sh

# Or override with command-line args
bash deploy/build_memory.sh --method megaloc --gpu 0
```

### Launch Navigation Service

```bash
# Automatically reads config from vpr_config.yaml
python deploy/ws_proxy_with_memory.py
```

### Python API

```python
from deploy.memory_nav import MemoryNavigator

# Uses config from vpr_config.yaml
navigator = MemoryNavigator(vpr_method='selavpr', device='cuda:0')
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
        "phase": "verifying",
        "current_step": 1,
        "total_steps": 3,
        "from_node": "Lobby",
        "to_node": "Reception",
        "vpr_similarity": 0.85,
        "vpr_confidence": 0.85,
        "vpr_matched_node": "node_5",
        "heading_offset": -37.5,
        "consecutive_misses": 0
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

# WebSocket integration test (with per-frame VPR decision logs + stats + trend chart)
python tests/test_memory_ws.py
```

Test output includes:
- 📊 Per-frame VPR matching details (similarity, confidence, matched node, decision type)
- 📈 VPR similarity trend ASCII chart
- 📋 Statistics report (match rate, node distribution, decision distribution, phase distribution)

---

## 📚 Citation

If this project helps your research, please cite the relevant VPR papers:

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

This project is licensed under the [MIT License](LICENSE).
