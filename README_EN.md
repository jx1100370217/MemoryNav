<div align="center">

# 🧠 MemoryNav

**Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-1.8.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v1.8.0)

A robot memory navigation system based on Visual Place Recognition (VPR) and topological maps

[Chinese](README.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | **English**

</div>

---

## 📖 Introduction

MemoryNav is a visual memory navigation system for mobile robots. It collects images via 4 omnidirectional fisheye cameras, uses VPR to localize within a pre-built topological memory graph, and falls back to Qwen3.5-9B visual-language model for grounding — enabling "remember where you've been, walk it again" navigation.

### Key Capabilities

- **🔍 Multi-scheme VPR Localization**: Supports 4 SOTA VPR methods, switchable via a single config file
- **🗺️ Topological Memory Graph**: Automatically builds a node-edge topology from annotated data; supports shortest-path planning (BFS/Dijkstra)
- **🔄 Cyclic-shift Matching**: 4-camera cyclic-shift algorithm for orientation-agnostic localization and heading estimation
- **🎯 DINOv3 Sub-image Matching**: Dense patch feature matching with sliding-window cosine similarity to localize navigation targets in real-time camera frames
- **💾 Match Caching**: Automatically reuses the last successful match when confidence drops, ensuring navigation continuity
- **📤 Unified Output Format**: Consistent response schema regardless of memory mode — always provides `pixel_target`
- **🤖 Qwen3.5 Fallback Grounding**: Falls back to Qwen3.5-9B VLM when VPR/sub-image matching fails, using Chinese landmark names directly
- **📷 Fisheye Undistortion**: Loads camera intrinsics from `cam/params.yaml` at startup; applies cylindrical-projection undistortion to all input images before VPR and sub-image matching
- **🧭 Pixel→Robot Coordinate Transform**: Converts normalized `pixel_target` to robot motion coordinates `[x_forward, y_lateral, 0.0]` via a full cylindrical-angle + camera-azimuth + depression-angle pipeline
- **🌐 WebSocket Service**: Real-time streaming of images and navigation commands
- **⚙️ Unified Config**: All VPR parameters centralized in `deploy/vpr_config.yaml`

---

## 🏗️ Architecture

```
MemoryNav/
├── cam/                            # Multi-eye fisheye camera ROS2 node
│   ├── params.yaml                 # Camera intrinsics, extrinsics (T_ic), distortion coeffs
│   ├── fisheye_undist.h            # GPU-accelerated fisheye undistortion (CUDA)
│   ├── main.cc / main.h            # ROS2 node main program
│   ├── video.h                     # V4L2 video capture
│   └── tools/                      # Standalone tools (no ROS2/CUDA required)
│       ├── fisheye_undist_cpu.h    # CPU undistortion (numpy/cv2 port basis)
│       ├── fisheye_to_cylindrical.cpp  # Fisheye-to-cylindrical CLI tool
│       └── batch_undistort.py      # Batch undistortion script
├── deploy/                         # Deployment module
│   ├── vpr_config.yaml             # Unified VPR config
│   ├── memory_nav/                 # Core memory navigation package
│   │   ├── vpr_config_loader.py    # Config loader
│   │   ├── memory_models.py        # Data models (Node, Edge, Plan, VPRResult)
│   │   ├── memory_graph.py         # Topology graph (BFS/Dijkstra)
│   │   ├── memory_vpr.py           # VPR matching engine (cyclic-shift + order-invariant)
│   │   ├── memory_builder.py       # Memory builder (topology from annotated data)
│   │   ├── memory_navigator.py     # Navigator main interface
│   │   ├── sub_image_matcher.py    # Sub-image matcher (DINOv3 dense features)
│   │   ├── fisheye_undistort.py    # 🆕 Fisheye undistortion (ported from cam/tools/fisheye_undist_cpu.h)
│   │   ├── coord_transform.py      # 🆕 Pixel→robot coordinate transform (full cylindrical pipeline)
│   │   ├── qwen35_point_grounder.py # Qwen3.5 grounding wrapper (fallback model)
│   │   ├── qwen35_grounding_server.py # Qwen3.5 subprocess inference server
│   │   ├── vpr_factory.py          # VPR extractor factory
│   │   ├── anyloc_extractor.py     # AnyLoc (DINOv2 + VLAD)
│   │   ├── megaloc_extractor.py    # MegaLoc (DINOv2 + OT aggregation)
│   │   ├── effovpr_extractor.py    # EffoVPR (DINOv2 multi-layer CLS)
│   │   └── selavpr_extractor.py    # SelaVPR++ (DINOv2 + MultiConv)
│   ├── ws_proxy_with_memory.py     # WebSocket proxy service (main entry)
│   └── build_memory.sh             # Memory build script
├── internnav/                      # InternNav navigation framework
├── scripts/
│   └── memory_visualization_server.py  # Memory graph visualization (sub-image + model grounding)
├── tests/
│   ├── test_memory_nav.py          # Memory module unit tests
│   └── test_memory_ws.py           # WebSocket integration tests (detailed logging)
└── docs/
```

---

## 📷 Fisheye Undistortion

The system automatically undistorts all 4 fisheye camera images before VPR matching and sub-image matching:

### How It Works

1. At startup, loads per-camera intrinsics (`xi, fx, fy, cx, cy`) and distortion coefficients (`k1, k2, p1, p2`) from `cam/params.yaml`
2. Pre-computes a cylindrical-projection remap table once per camera (with optional `pitch_up` offset)
3. Before each inference frame, applies `cv2.remap` for near-zero-cost undistortion
4. Gracefully skips undistortion if `cam/params.yaml` is missing — service continues normally

```python
from deploy.memory_nav.fisheye_undistort import FisheyeUndistorter

undistorter = FisheyeUndistorter.from_yaml("cam/params.yaml")
# Batch undistort all 4 camera images
perspective_images = undistorter.undistort_batch(camera_images)
```

---

## 🧭 Pixel → Robot Coordinate Transform

Converts normalized `pixel_target: [x_norm, y_norm]` to robot motion coordinates `[x_forward, y_lateral, 0.0]` via a full physical pipeline:

### Pipeline

```
x_norm → cylindrical horizontal angle → + camera azimuth → global yaw
y_norm → cylindrical vertical angle → depression angle → distance estimate (camera height + pitch_up)
yaw + distance → (x_forward, y_lateral)
```

### Parameters (`coord_transform.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_FOV` | 180° | Cylindrical image field of view |
| `DEFAULT_WIDTH` | 1920 | Cylindrical image width |
| `DEFAULT_HEIGHT` | 1536 | Cylindrical image height |
| `DEFAULT_CAMERA_HEIGHT` | 1.0 m | Camera height above ground |
| `DEFAULT_PITCH_UP` | 15° | Undistortion pitch_up offset |
| `MIN_DISTANCE` / `MAX_DISTANCE` | 0.3 m / 30.0 m | Distance estimation range |

### Camera Azimuths (hardcoded, derived from `cam/params.yaml` T_ic)

| Camera | Azimuth |
|--------|---------|
| camera_1 | +39.42° |
| camera_2 | −35.84° |
| camera_3 | −142.04° |
| camera_4 | +143.52° |

### New Response Field

Coordinate transform results are appended to `memory_info.coord_transform`:

```json
"memory_info": {
    "coord_transform": {
        "yaw_global_deg": -12.3,
        "depression_deg": 8.5,
        "distance": 2.4,
        "elapsed_ms": 0.3
    }
}
```

---

## 🎯 Sub-image Matching Navigation

Dense patch feature sub-image matching using **DINOv3**:

### How It Works

1. **Memory Build**: Each edge is annotated with `camera_name` (target camera) and `crop_image` (attention crop)
2. **Navigation**: The crop sub-image is retrieved from memory and matched against the real-time camera frame
3. **Target Localization**: DINOv3 ViT-B/16 extracts dense patch tokens → sliding window + unfold acceleration → cosine similarity argmax → output as `pixel_target`
4. **Match Threshold**: Confidence ≥ 0.6 counts as a successful match
5. **Cache**: On low-confidence frames, the last successful result is reused; cache clears on step transitions
6. **Fallback**: If no cache, uses `pixel_box` from memory as an estimate

### Edge Data Structure

```yaml
edge:
  camera_name: "camera_2"              # Target camera
  landmark_name: "Elevator"            # Landmark name
  pixel_box: [120, 80, 200, 160]       # (x, y, w, h) pixel box
  crop_image_path: "crop_elevator.jpg" # Attention crop
```

### Output Format

All responses always include `pixel_target: [x, y]` (normalized 0–1) and robot `action`:

| Scenario | pixel_target Source | action Source | memory_active |
|----------|---------------------|---------------|---------------|
| Memory off + InternVLA | `output_pixel / image_size` | InternVLA | not present |
| Memory on + sub-match hit | `sub_image_match.match.center_pct` | coord_transform | `true` |
| Memory on + cache reuse | last successful match | coord_transform | `true` |
| Memory on + Qwen3.5 fallback | Qwen3.5 grounding coords | coord_transform | `true` |

---

## ✨ VPR Method Comparison

MemoryNav supports **4 VPR methods**, switchable via `deploy/vpr_config.yaml`:

| Method | Key | Venue | Feature Dim | Backbone | Notes |
|--------|-----|-------|------------|----------|-------|
| **SelaVPR++** ⭐ | `selavpr` | T-PAMI 2025 | 4096D | DINOv2-L + MultiConv | **Recommended**, hashing+reranking |
| **MegaLoc** | `megaloc` | CVPR 2025 | 8448D | DINOv2-B + OT | Best overall, multi-dataset SOTA |
| **EffoVPR** | `effovpr` | arXiv 2024 | 3072D | DINOv2-B multi-CLS | Lightweight, real-time friendly |
| **AnyLoc** | `anyloc` | RA-L 2023 | configurable | DINOv2-B + VLAD | Classic, stable |

---

## ⚙️ Unified Config

All VPR parameters are managed in `deploy/vpr_config.yaml`:

```yaml
# VPR method: selavpr | megaloc | effovpr | anyloc
vpr_method: selavpr

# GPU device
device: "cuda:0"

# Per-method similarity thresholds
similarity_threshold:
  selavpr: 0.60
  megaloc: 0.60
  effovpr: 0.80
  anyloc: 0.70

# SelaVPR++ specific
selavpr:
  backbone: dinov2-large
  aggregation: gem
  use_hashing: true
  use_rerank: true

# AnyLoc specific
anyloc:
  dino_model: dinov2_vitb14
  agg_mode: vlad
  num_clusters: 32
  domain: indoor
  max_img_size: 630
```

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

### Camera Setup (Optional)

Place your calibration file at `cam/params.yaml`. The service will automatically load fisheye undistortion on startup.

### Configure VPR Method

Edit `deploy/vpr_config.yaml` to select the VPR method and parameters.

### Build Memory

```bash
bash deploy/build_memory.sh
# Override parameters
bash deploy/build_memory.sh --method megaloc --gpu 0
```

### Start Navigation Service

```bash
python deploy/ws_proxy_with_memory.py
```

### Python API

```python
from deploy.memory_nav import MemoryNavigator

navigator = MemoryNavigator(vpr_method='selavpr', device='cuda:0')
navigator.load_memory(path='memory_cache.pkl', data_dir='merged_labeled_data')

images = {'camera_1': img1, 'camera_2': img2, 'camera_3': img3, 'camera_4': img4}
features = {cam: navigator.extractor.extract(img) for cam, img in images.items()}
result = navigator.vpr.locate(features)
print(f"Localized: {result.matched_node_name}, similarity: {result.similarity:.4f}")

plan = navigator.navigate_to("Reception", camera_images=images)
match = navigator.match_current_step(images)
if match and match['match']['found']:
    center = match['match']['center_pct']
    print(f"Target: ({center['x']:.3f}, {center['y']:.3f})")
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
    "action": [[0.5, -0.1, 0.0]],
    "pixel_target": [0.485, 0.521],
    "memory_active": true,
    "camera_name": "camera_2",
    "landmark_name": "Elevator",
    "memory_info": {
        "phase": "verifying",
        "current_step": 1,
        "total_steps": 3,
        "vpr_similarity": 0.85,
        "coord_transform": {
            "yaw_global_deg": -12.3,
            "depression_deg": 8.5,
            "distance": 2.4,
            "elapsed_ms": 0.3
        }
    }
}
```

### Control Commands

| Command | Description |
|---------|-------------|
| `reset` | Reset agent and memory state |
| `toggle_memory` | Toggle memory navigation on/off |
| `memory_status` | Show memory navigation details |
| `reset_memory` | Reset memory state only |
| `session_status` | Show session status |

---

## 📐 Camera Layout

4 fisheye cameras (equiangular projection, HFOV=190°):

```
           Front (0°)
              ↑
     cam_1 (-37.5°)  cam_2 (+37.5°)
              │
     cam_4 (-142.5°) cam_3 (+142.5°)
              ↓
           Back (180°)
```

Cyclic-shift matching supports 4 heading offsets: `0°`, `-75°`, `180°`, `+105°`

---

## 🧪 Testing

```bash
# Unit tests
python -m pytest tests/test_memory_nav.py -v

# WebSocket integration test (per-frame logs + stats + similarity trend chart)
python tests/test_memory_ws.py
```

---

## 📋 Changelog

### v1.8.0

- **🆕 Fisheye Undistortion**: Added `deploy/memory_nav/fisheye_undistort.py`, ported from `cam/tools/fisheye_undist_cpu.h`
  - Loads 4-camera intrinsics at startup; pre-computes cylindrical remap tables
  - Automatically undistorts input images before each inference frame
  - Gracefully skips if `cam/params.yaml` is missing
- **🆕 Pixel→Robot Coordinate Transform**: Added `deploy/memory_nav/coord_transform.py`
  - Full pipeline: cylindrical angle → camera azimuth → global yaw + depression angle → distance → `[x_forward, y_lateral, 0.0]`
  - Applied to all three navigation decision paths (sub-match, cache, Qwen3.5 fallback)
  - Debug fields (yaw, depression, distance, latency) included in response
- **🆕 cam/ Directory**: Added multi-eye fisheye camera ROS2 node source and `params.yaml`
  - `cam/tools/`: Standalone fisheye-to-cylindrical CLI (no ROS2/CUDA dependency)
- **Startup Log**: Added fisheye undistortion status and coord-transform module status

### v1.7.0

- **Qwen3.5 Fallback Grounding**: Replaces InternVLA fallback with Qwen3.5-9B VLM
  - Uses Chinese `landmark_name` directly; subprocess mode avoids transformers conflicts
- **InternVLA On-demand Loading**: Not loaded by default; loaded on demand to save GPU memory
- **Visualization**: Added model grounding tab in `memory_visualization_server.py`

### v1.6.0

- **Sub-image Matching Simplified**: Removed SuperPoint+LightGlue, kept DINOv3 dense matching only
- **Unified Threshold**: Confidence threshold set to 0.6
- **Frame Similarity Upgrade**: SSIM replaced by DINOv2 inter-frame similarity (threshold 0.70)
- **3-scale Cascade Matching**: small/mid/big crops + full-camera sweep for robustness

### v1.5.0

- **Unified Output Format**: Memory-off response matches `ws_proxy.py` exactly; always includes `pixel_target`
- **Sub-image Match Cache**: Reuses last successful result on low-confidence frames

### v1.4.0

- **Sub-image Matching Navigation**: Upgraded from angle-based to SuperPoint+LightGlue
- **Edge Model Rework**: `angle + pixel_position` → `camera_name + crop_image + pixel_box`

### v1.3.0

- **Unified Config**: All VPR parameters centralized in `deploy/vpr_config.yaml`

### v1.2.0

- **Multi-VPR Support**: Added SelaVPR++, MegaLoc, EffoVPR
- **VPR Factory Pattern**: Unified extractor interface

### v1.1.0

- **Memory Navigation Service**: WebSocket proxy + VPR localization + path planning

### v1.0.0

- **Base Framework**: Topological memory graph, AnyLoc VPR, InternVLA inference

---

## 📚 Citation

```bibtex
@article{selavprpp2025,
  title={SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition},
  author={Lu, Feng and Jin, Tong and others},
  journal={IEEE T-PAMI},
  year={2026}
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
