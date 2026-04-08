<div align="center">

# 🧠 MemoryNav

**Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-2.2.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v2.2.0)

A robot memory navigation system based on Visual Place Recognition (VPR) and topological maps

[Chinese](README.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | **English**

</div>

---

## 📖 Introduction

MemoryNav is a visual memory navigation system for mobile robots. It collects images via 4 omnidirectional fisheye cameras, uses VPR to localize within a pre-built topological memory graph, detects visual occlusion with YOLOv8n, and falls back to Qwen3.5-9B visual-language model for grounding — enabling "remember where you've been, walk it again" navigation.

### Key Capabilities

- **🗺️ Offline mapping** (`offline_mapper/`): 3-phase pipeline (VPR node creation → semantic augmentation → connection generation) automatically builds topological navigation graphs from image sequences — no manual annotation required
- **🛰️ Online active mapping** (`online_mapper/`): Three-layer architecture (Geometry + Topology + Semantics) for streaming online mapping, with category whitelist, multi-frame hallucination voting, co-location merging, real VO, loop closure verification, bilingual naming, and spatial-KNN neighbour rebuild. Output schema is fully compatible with `offline_mapper/`. See [`docs/online_mapper.md`](docs/online_mapper.md) for the full design.
- **🔍 Multi-scheme VPR Localization**: Supports 4 SOTA VPR methods, switchable via a single config file
- **🗺️ Topological Memory Graph**: Automatically builds a node-edge topology from annotated data; supports shortest-path planning (BFS/Dijkstra)
- **🔄 Cyclic-shift Matching**: 4-camera cyclic-shift algorithm for orientation-agnostic localization and heading estimation
- **🎯 DINOv3 Sub-image Matching**: Dense patch feature matching with 3-scale cascade (small→mid→big) across all 4 cameras, sliding-window cosine similarity
- **💾 Frame-level Cache Reuse**: DINOv2 VPR feature-based inter-frame similarity (zero extra inference cost); intelligently reuses last successful match when current match fails
- **🔭 Lookahead Dual Confirmation**: Validates both VPR localization and next-step sub-image matching before advancing steps
- **📤 Unified Output Format**: Consistent response schema regardless of memory mode — always provides `pixel_target`
- **🚧 YOLOv8n Occlusion Detection**: Automatically detects camera occlusion (pedestrians, objects) when sub-image matching fails; stops in place when occluded, resumes after clearance
- **🤖 Qwen3.5 Fallback Grounding**: Two-step inference (existence check + conditional grounding) with Qwen3.5-9B VLM when not occluded but sub-image matching fails
- **📷 Fisheye Undistortion**: Loads camera intrinsics from `cam/params.yaml` at startup; applies cylindrical-projection undistortion before VPR and sub-image matching
- **🧭 Pixel→Robot Coordinate Transform**: Converts normalized `pixel_target` to robot motion coordinates `[x_forward, y_lateral, 0.0]` via cylindrical-angle + camera-azimuth + depression-angle pipeline
- **🔄 Side Camera Rotation**: camera_3/camera_4 matches output in-place rotation actions to orient the robot toward the target
- **🌐 WebSocket Service**: Real-time streaming of images and navigation commands
- **⚙️ Unified Config**: All VPR parameters centralized in `deploy/vpr_config.yaml`

---

## 🏗️ Architecture

```
MemoryNav/
├── memory_nav/                     # Core memory navigation module
│   ├── memory_navigator.py         # Navigator main interface
│   ├── memory_models.py            # Data models (Node, Edge, Plan, VPRResult)
│   ├── memory_graph.py             # Topology graph (BFS/Dijkstra)
│   ├── memory_vpr.py               # VPR matching engine (cyclic-shift + order-invariant)
│   ├── memory_builder.py           # Memory builder (topology from annotated data)
│   ├── sub_image_matcher.py        # Sub-image matcher (DINOv3 dense features)
│   ├── occlusion_detector.py       # YOLOv8n occlusion detector
│   ├── fisheye_undistort.py        # Fisheye undistortion (cylindrical projection)
│   ├── coord_transform.py          # Pixel→robot coordinate transform
│   ├── qwen35_point_grounder.py    # Qwen3.5 grounding wrapper (fallback model)
│   ├── qwen35_grounding_server.py  # Qwen3.5 subprocess inference server
│   ├── vpr_factory.py              # VPR extractor factory
│   ├── vpr_config_loader.py        # Unified config loader
│   ├── selavpr_extractor.py        # SelaVPR++ (DINOv2 + MultiConv)
│   ├── megaloc_extractor.py        # MegaLoc (DINOv2 + OT aggregation)
│   ├── effovpr_extractor.py        # EffoVPR (DINOv2 multi-layer CLS)
│   ├── anyloc_extractor.py         # AnyLoc (DINOv2 + VLAD)
│   └── selavpr_model/              # SelaVPR++ model code
├── deploy/                         # Deployment entry points
│   ├── ws_proxy_with_memory.py     # WebSocket proxy service (main entry)
│   ├── vpr_config.yaml             # Unified VPR config
│   ├── build_memory.sh             # Memory build script
│   └── start_server.sh             # Server start script
├── cam/                            # Multi-eye fisheye camera
│   ├── params.yaml                 # Camera intrinsics & extrinsics
│   └── tools/                      # Standalone tools (no ROS2/CUDA)
├── scripts/
│   └── memory_visualization_server.py  # Visualization (sub-image + grounding + occlusion)
├── pretrained/                     # Pretrained models (YOLOv8n, DINOv3, etc.)
├── merged_labeled_data/            # Memory annotated data
├── offline_mapper/                 # 🗺️ Offline mapping module (formerly auto_mapper)
│   ├── run_auto_map.py             # Entry script
│   ├── auto_mapper_core.py         # Core controller (3-phase pipeline)
│   ├── node_distance_estimator.py  # VPR node distance estimation
│   ├── auto_landmark_namer.py      # Qwen3.5 scene naming (vLLM)
│   ├── semantic_node_detector.py   # Door plate / sign text recognition
│   ├── auto_node_generator.py      # Node directory & metadata generation
│   ├── auto_sub_image_extractor.py # Grounding crop + corridor frame matching
│   └── validate_output.py          # Output format validation
├── online_mapper/                  # 🛰️ Online active mapping (3-layer architecture)
│   ├── run_online_map.py           # Entry script
│   ├── config.py                   # Global config (OnlineMapperConfig)
│   ├── core/online_mapper_core.py  # ⭐ Main orchestrator (streaming loop)
│   ├── geometry/                   # Geometry layer
│   │   ├── depth_estimator.py      #   Depth-Anything-V2 wrapper
│   │   ├── visual_odometry.py      #   ORB + EssentialMatrix VO
│   │   ├── pose_graph.py           #   scipy LM pose graph
│   │   ├── junction_detector.py    #   4-camera depth junction detector
│   │   └── occupancy.py            #   2D occupancy grid
│   ├── topology/                   # Topology layer
│   │   ├── keyframe_selector.py    #   Multi-trigger keyframe selection
│   │   ├── loop_closure.py         #   Auto-tune + ORB geometric verification
│   │   ├── connection_builder.py   #   next_positions builder (subclasses offline_mapper)
│   │   └── graph.py                #   TopoGraph / TopoNode
│   └── semantics/                  # Semantics layer
│       ├── open_set_detector.py    #   Grounding-DINO wrapper
│       ├── door_plate_tracker.py   #   Door plate representative frame selection
│       ├── hallucination_filter.py # ⭐ STRICT prompt + QwenVerifier + MultiFrameVoter
│       ├── node_category.py        # ⭐ 7-class whitelist classifier + CN/EN map
│       ├── colocation_merger.py    # ⭐ Co-location node merger
│       └── scene_graph.py          #   Hierarchical scene graph
├── tests/
│   └── test_memory_ws.py           # WebSocket integration tests
└── docs/
    └── online_mapper.md            # 📘 Full online_mapper design doc (13 chapters)
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
from memory_nav.fisheye_undistort import FisheyeUndistorter

undistorter = FisheyeUndistorter.from_yaml("cam/params.yaml")
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

### Side Camera Rotation

When camera_3 or camera_4 (rear-facing) matches successfully, the system does not output a forward action. Instead, it computes the actual yaw angle via coordinate transform and outputs an in-place rotation `[0, 0, yaw_rad]` to orient the robot toward the target. For Qwen3.5 fallback on side cameras, a fixed rotation of `[0, 0, 0.785]` (~45°) is output.

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

---

## 🎯 Sub-image Matching Navigation

Dense patch feature sub-image matching using **DINOv3**:

### How It Works

1. **Memory Build**: Each edge is annotated with `camera_name` (target camera) and 3-scale `crop_image` (big/mid/small attention crops)
2. **Navigation**: All 4 cameras are scanned with small→mid→big cascade per camera; the global highest confidence match is selected
3. **Target Localization**: DINOv3 ViT-B/16 extracts dense patch tokens → sliding window + unfold acceleration → cosine similarity argmax → output as `pixel_target`
4. **Match Threshold**: Confidence ≥ `SUB_MATCH_CONFIDENCE_THRESHOLD` (currently **0.60**) counts as a successful match
5. **Frame Cache**: On low-confidence frames, DINOv2 VPR feature inter-frame similarity (threshold **0.70**) determines whether to reuse the last successful result; cache clears on step transitions
6. **Fallback**: If no cache, triggers Qwen3.5 fallback grounding

### Output Format

All responses always include `pixel_target: [x, y]` (normalized 0–1) and robot `action`:

| Scenario | pixel_target Source | action Source | memory_active |
|----------|---------------------|---------------|---------------|
| Memory on + sub-match hit | `sub_image_match.match.center_pct` | coord_transform | `true` |
| Memory on + cache reuse | last successful match | coord_transform | `true` |
| Memory on + Qwen3.5 fallback | Qwen3.5 grounding coords | coord_transform | `true` |
| Memory on + occlusion | none (wait in place) | `[0, 0, 0]` | `true` |
| Memory on + side camera match | `sub_image_match.match.center_pct` | in-place rotation `[0, 0, yaw]` | `true` |

---

## 🚧 Occlusion Detection

When sub-image matching fails (regardless of VPR result), the system automatically performs occlusion detection on the attention camera:

### How It Works

1. **Trigger**: Sub-image matching failure triggers detection — independent of VPR result
2. **Camera Selection**: Uses the camera with the highest sub-image match score (below threshold), not the static `step.camera_name`
3. **YOLOv8n Inference**: Detects nearby foreground objects (person, backpack, umbrella, handbag, suitcase), computes bbox area ratio
4. **Occlusion Criteria**: Single object area ratio ≥ **25%** (default) → judged as occluded
5. **When Occluded**: Outputs `action: [0, 0, 0]` (wait in place), clears sub-image match cache
6. **When Not Occluded**: Falls back to Qwen3.5 grounding with fixed "center of corridor + depth" wayfinding strategy

### Navigation Decision Flow

```
Per-frame processing:
  ├─ Sub-image matching (all 4 cameras × 3-scale cascade)
  ├─ Lookahead next-step sub-image matching
  │
  ├─ When sub-image matching fails:
  │   ├─ YOLOv8n occlusion detection (on highest-scoring camera)
  │   │   ├─ Occluded → action=[0,0,0], wait, clear cache
  │   │   └─ Not occluded → Qwen3.5 grounding (wayfinding: center of corridor + depth)
  │   │                     └─ Also fails → resend memory guidance
  │   └─ (independent of VPR result)
  │
  ├─ VPR match success:
  │   ├─ Target node + sim≥0.70:
  │   │   ├─ Last step → advance directly
  │   │   ├─ Next-step sub-match OK → lookahead confirm → advance
  │   │   └─ Next-step sub-match fail → VPR HELD, don't advance yet
  │   └─ Other node / sim<0.70 → continue current step
  │
  └─ VPR match failure:
      ├─ Sub-match OK → navigate with sub-match result
      ├─ Sub-match fail + Qwen3.5 OK → navigate with grounding result
      └─ Sub-match fail + Qwen3.5 fail → resend memory guidance
```

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
vpr_method: selavpr
device: "cuda:0"

order_invariant:
  selavpr: false    # cyclic-shift matching + heading estimation
  megaloc: true     # order-invariant greedy matching
  effovpr: true
  anyloc: true

similarity_threshold:
  selavpr: 0.60
  megaloc: 0.60
  effovpr: 0.80
  anyloc: 0.70

selavpr:
  backbone: dinov2-large
  aggregation: gem
  use_hashing: true
  use_rerank: true

anyloc:
  dino_model: dinov2_vitb14
  agg_mode: vlad
  num_clusters: 32
  domain: indoor
  max_img_size: 630
```

> ⚠️ After switching VPR methods, rebuild the memory cache: `bash deploy/build_memory.sh`

---

## 🗂️ Mapping Modules at a Glance

MemoryNav ships **two complementary** mapping modules for two typical usage scenarios:

| Dimension | `offline_mapper/` (offline) | `online_mapper/` (online active) |
|---|---|---|
| **Positioning** | One-shot post-processing after recording | Robot walks and streams, per-frame decisions |
| **Temporal assumption** | All frames visible | Only "already-arrived" frames |
| **Main loop** | 3-phase pipeline (Phase1 → 1.5 → 2) | Per-frame: geometry → VPR → loop → plate scan → KF trigger → classify → build node |
| **Keyframe strategy** | VPR similarity + min frame interval | VPR + accumulated translation + rotation + info gain + junction + semantic whitelist |
| **Loop closure** | First-last comparison (optional) | Global VPR + ORB geometric verification, every frame |
| **Naming** | Qwen describe_scene / detect_text (single-frame) | Multi-frame voting + 2nd-pass verification + category whitelist + CN/EN bilingual |
| **Node filtering** | None (all VPR triggers create nodes) | 7-class whitelist, rejects decorative walls / plants / empty corridors |
| **Hallucination defense** | None | STRICT prompt + QwenVerifier + MultiFrameVoter + substring variant merging |
| **Output schema** | `merged_labeled_data/` | **Fully compatible** with `merged_labeled_data/` + extra `scene_graph.json` / `pose_graph.json` / `online_mapping_log.jsonl` / `metrics.json` |
| **Code relation** | Standalone | Subclasses & reuses `offline_mapper.AutoSubImageExtractor` / `AutoLandmarkNamer` / `NodeDistanceEstimator` without modifying offline_mapper |

Both modules produce **schema-identical outputs**, both can be consumed directly by `deploy/build_memory.sh`.

Full `online_mapper` design doc: **[`docs/online_mapper.md`](docs/online_mapper.md)** (13 chapters, ~47k characters).
`online_mapper` iteration history (r1→r6) and detailed before/after metrics: **[`online_mapper/RESULTS.md`](online_mapper/RESULTS.md)**.

---

## 🗺️ Offline Mapping (offline_mapper)

The offline mapping module (`offline_mapper/`, formerly `auto_mapper`) automatically generates topological navigation graphs from robot-captured image sequences — no manual annotation required.

### 3-Phase Pipeline

```
Phase 1: VPR Coarse-grained Node Creation
  ├─ Scan 4-camera images frame by frame
  ├─ VPR feature extraction → compare similarity with nearest node
  ├─ Similarity < threshold (0.70) → create new node
  └─ Qwen3.5 VLM auto-naming (Chinese/English)

Phase 1.5: Semantic Augmentation
  ├─ Scan intermediate frames between nodes
  ├─ Qwen3.5 text recognition: detect door plates, signs, meeting room names
  ├─ Quality filtering: blacklist excludes meaningless signs (exit, fire, etc.)
  ├─ Name normalization: bare numbers/English auto-completed (10 → Room 10, MOORE → Moore Meeting Room)
  └─ Insert new nodes at spatial positions + renumber

Phase 2: Connection Generation
  ├─ Qwen3.5 PointGrounder for pairwise adjacent node grounding
  ├─ Corridor mid-frame matching: DINOv3 CLS features find best camera
  ├─ Hungarian algorithm: optimal camera → neighbor assignment
  ├─ 3-scale crop (big/mid/small) + Y-coordinate correction
  └─ Output format validation (validate_output.py)
```

### Usage

```bash
# Basic usage
python offline_mapper/run_auto_map.py \
    --input_dir memory_test_data \
    --output_dir offline_mapper/merged_labeled_data \
    --vpr_config deploy/vpr_config.yaml

# Full parameters
python offline_mapper/run_auto_map.py \
    --input_dir memory_test_data \
    --output_dir offline_mapper/merged_labeled_data \
    --vpr_config deploy/vpr_config.yaml \
    --start_id 1 \
    --similarity_threshold 0.70 \
    --min_frame_interval 5 \
    --use_qwen_naming \
    --qwen_gpu 1
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | `memory_test_data` | Input image directory (containing `*_camera_*.jpg` files) |
| `--output_dir` | `offline_mapper/merged_labeled_data` | Output directory (`merged_labeled_data` format) |
| `--vpr_config` | `deploy/vpr_config.yaml` | VPR config file path |
| `--start_id` | `1` | Starting node ID |
| `--similarity_threshold` | `0.70` | VPR similarity threshold (below = create new node) |
| `--min_frame_interval` | `5` | Minimum frame interval (prevents over-dense nodes) |
| `--use_qwen_naming` | `true` | Use Qwen3.5 auto-naming |
| `--qwen_gpu` | `1` | GPU device for Qwen3.5 |
| `--dry_run` | `false` | Validate input only, skip execution |

### Prerequisites

1. **vLLM Service**: Start Qwen3.5 vLLM server for scene naming and text recognition
   ```bash
   bash deploy/start_qwen_vllm.sh
   ```
2. **VPR Model**: Pretrained weights for the VPR method configured in `deploy/vpr_config.yaml`
3. **DINOv3 Model**: For corridor mid-frame matching during crop extraction

### Output Format

Auto-mapping output is fully compatible with manually annotated `merged_labeled_data/`:

```
merged_labeled_data/
├── 1/                          # Node 1
│   ├── position_info.json      # Node metadata (name, camera images, connections)
│   ├── <timestamp>_camera_1.jpg
│   ├── <timestamp>_camera_2.jpg
│   ├── <timestamp>_camera_3.jpg
│   ├── <timestamp>_camera_4.jpg
│   └── crops/                  # 3-scale attention crops
│       ├── to_2_camera_2_big.jpg
│       ├── to_2_camera_2_mid.jpg
│       └── to_2_camera_2_small.jpg
├── 2/
│   └── ...
```

Use auto-mapped data directly for memory building:

```bash
bash deploy/build_memory.sh --data_dir offline_mapper/merged_labeled_data
```

### Core Components

| Component | Description |
|-----------|-------------|
| `offline_mapper/auto_mapper_core.py` | Core controller orchestrating the 3-phase pipeline |
| `node_distance_estimator.py` | VPR feature comparison for new node decisions |
| `auto_landmark_namer.py` | Qwen3.5 vLLM scene description + landmark naming |
| `semantic_node_detector.py` | Door plate/sign text recognition + name normalization + blacklist filtering |
| `auto_node_generator.py` | Node directory creation, metadata JSON generation |
| `auto_sub_image_extractor.py` | PointGrounding + DINOv3 corridor frame matching + Hungarian assignment + crop extraction |
| `validate_output.py` | Output format integrity validation |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav
pip install -r requirements/core_requirements.txt
pip install -e .
```

### Camera Setup (Optional)

Place your calibration file at `cam/params.yaml`. The service will automatically load fisheye undistortion on startup.

### Configure VPR Method

Edit `deploy/vpr_config.yaml` to select the VPR method and parameters.

### Build Memory

```bash
bash deploy/build_memory.sh
bash deploy/build_memory.sh --method megaloc --gpu 0
```

### Start Navigation Service

```bash
python deploy/ws_proxy_with_memory.py
```

### Python API

```python
from memory_nav import MemoryNavigator

navigator = MemoryNavigator(vpr_method='selavpr', device='cuda:0')
navigator.load_memory(path='memory_nav/memory_cache', data_dir='merged_labeled_data')

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
    }
}
```

### Control Commands

| Command | Description |
|---------|-------------|
| `reset` | Reset agent and memory state |
| `toggle_memory` | Toggle memory navigation on/off |
| `memory_status` | Show memory navigation details (incl. available destinations) |
| `reset_memory` | Reset memory state only (agent history preserved) |
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
python -m pytest tests/unit_test/test_basic.py -v
python tests/test_memory_ws.py
```

---

## 📋 Changelog

### v2.3.0

- **🛰️ Online Active Mapping Module** (`online_mapper/`): A brand-new streaming online mapping module with a 3-layer architecture (Geometry + Topology + Semantics), complementary to `offline_mapper/`
  - **Geometry layer**: Monocular ORB+EssentialMatrix VO, Depth-Anything-V2, scipy LM pose graph, 2D occupancy grid, 4-camera depth junction detector
  - **Topology layer**: Multi-trigger keyframes (VPR + translation + rotation + info gain), auto-tune + ORB-verified global loop closure, spatial-KNN ∪ temporal-adjacent neighbour rebuild
  - **Semantics layer**: STRICT prompt + QwenVerifier 2nd-pass verification + MultiFrameVoter multi-frame voting + substring variant merging + 7-class whitelist (NodeCategoryClassifier) + ColocationMerger co-location merge + CN/EN bilingual naming + NameDeduplicator suffix-based dedup
  - Output schema 100% compatible with `offline_mapper/`; additionally writes `scene_graph.json` / `pose_graph.json` / `online_mapping_log.jsonl` / `metrics.json`
  - **Test data (49 frames) final result**: 5 high-quality nodes (Printing Area / Reception / NEUMANN Electrical Closet / Care Room / DEEPROUTE.AI Reception), 0 hallucinations, 0 duplicates, 2 loop closures, validator 5/5 passed
  - Full design doc: **[`docs/online_mapper.md`](docs/online_mapper.md)** (~47k characters, 13 chapters)
  - Iteration history (r1→r6): [`online_mapper/RESULTS.md`](online_mapper/RESULTS.md)
- **🔁 `auto_mapper/` renamed to `offline_mapper/`**: Aligned with `online_mapper/` to clearly distinguish offline vs online mapping scenarios. Internal class names (`AutoMapperCore` / `AutoSubImageExtractor` etc.) are intentionally preserved; only import paths are migrated.

### v2.2.0

- **🆕 Auto-mapping Module**: New `auto_mapper/` module (renamed to `offline_mapper/` in v2.3.0) for fully automatic topological graph generation from image sequences
  - 3-phase pipeline: VPR node creation → semantic augmentation (door plate/sign detection) → connection generation (grounding + crop)
  - Qwen3.5 vLLM inference backend for scene naming, text recognition, and point grounding
  - Semantic node detector auto-identifies meeting room names, room numbers, and other navigation-relevant signs
  - DINOv3 corridor mid-frame matching + Hungarian algorithm for optimal camera→neighbor assignment
  - 4-camera parallel vLLM calls (Phase 1.5: 1.3x speedup, Phase 2: 1.6x speedup, overall 315s→238s)
  - Output fully compatible with manually annotated `merged_labeled_data/` — ready for memory building

### v2.1.0 (Documentation sync)

- **📝 Occlusion threshold corrected**: 35% → **25%** (matching code default)
- **📝 Sub-image match threshold corrected**: 0.65 → **0.60** (matching `SUB_MATCH_CONFIDENCE_THRESHOLD`)
- **📝 Occlusion trigger corrected**: Now documented as triggered on **sub-image match failure** (independent of VPR result)
- **📝 Side camera rotation**: Documented camera_3/camera_4 in-place rotation logic
- **📝 VPR arrival threshold**: Documented `VPR_ARRIVE_THRESHOLD = 0.70`
- **📝 Order-invariant matching**: Documented `order_invariant` config option

### v2.0.0

- **🆕 YOLOv8n Occlusion Detection**: Added `memory_nav/occlusion_detector.py`
  - Detects person, backpack, umbrella, handbag, suitcase using YOLOv8n (6MB, ~30ms GPU)
  - Occlusion = single object bbox area ratio ≥ 25% of frame
  - Occluded → `action: [0, 0, 0]` (wait); cleared → resume
  - Not occluded → Qwen3.5 grounding (fixed "center of corridor + depth" wayfinding)
- **🔄 Simplified Navigation Logic**: Removed legacy trend-based detection
- **🎯 best_fail_camera**: Tracks highest-scoring camera even on failure
- **🖥️ Occlusion Detection Tab**: Added in visualization server

### v1.9.0

- **🔭 Lookahead Dual Confirmation**: VPR + next-step sub-image match required for advance
- **🎯 Unified Threshold**: `SUB_MATCH_CONFIDENCE_THRESHOLD = 0.60`

### v1.8.0

- **🆕 Fisheye Undistortion** + **Pixel→Robot Coordinate Transform** + **cam/ Directory**

### v1.7.0

- **Qwen3.5 Fallback**: Two-step inference (existence check + conditional grounding)

### v1.6.0

- **DINOv3 only**, unified threshold 0.60, DINOv2 inter-frame similarity, 3-scale cascade

### v1.5.0 — v1.0.0

See Chinese README for full history.

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
