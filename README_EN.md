<div align="center">

# 🧠 MemoryNav

**Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A robot memory navigation system based on Visual Place Recognition (VPR) and topological mapping

**English** | [中文](README.md)

</div>

---

## 📖 Introduction

MemoryNav is a visual memory navigation system for mobile robots, featuring:

- **Visual Place Recognition (VPR)**: Multi-view loop closure detection based on LongCLIP
- **Topological Mapping**: Real-time topological representation of environments
- **Semantic-Guided Navigation**: Enhanced localization with semantic labels
- **Multi-View Fusion**: 4-camera surround system with voting-based matching

## 🏗️ System Architecture

```
MemoryNav System Architecture
├── deploy/                    # Deployment modules
│   ├── memory_modules/        # Core memory modules
│   │   ├── vpr.py            # Visual Place Recognition (v4.0)
│   │   ├── feature_extraction.py  # LongCLIP feature extraction
│   │   ├── surround_fusion.py     # Multi-view fusion
│   │   ├── topological_map.py     # Topological map
│   │   └── config.py              # Configuration
│   ├── visual_memory_system.py    # Visual memory system
│   └── ws_proxy_with_memory.py    # WebSocket proxy service
├── internnav/                 # InternNav navigation framework
│   ├── agent/                 # Navigation agents
│   ├── model/                 # Model definitions
│   │   ├── basemodel/         # Base models (LongCLIP, InternVLA, etc.)
│   │   └── encoder/           # Encoder modules
│   ├── env/                   # Environment interfaces
│   └── evaluator/             # Evaluation modules
└── scripts/                   # Utility scripts
```

## ✨ Key Features

### 1. Multi-View VPR (v4.0)
- Independent FAISS indices for 4 surround cameras
- Voting mechanism for loop closure confirmation
- Adaptive threshold adjustment

```python
# Camera configuration
CAMERA_ANGLES = {
    'camera_1': 37.5,   # Front-right
    'camera_2': -37.5,  # Front-left
    'camera_3': -142.5, # Rear-left
    'camera_4': 142.5   # Rear-right
}
```

### 2. LongCLIP Visual Encoding
- 768-dimensional feature vectors
- L2 normalization
- GPU acceleration support

### 3. Multi-Stage Verification
- Time gap check (>5 seconds)
- Spatial consistency verification
- Semantic label guidance
- Temporal consistency verification

### 4. Threshold System
| Threshold Type | Value | Description |
|---------------|-------|-------------|
| High Confidence | 0.96 | Direct loop confirmation |
| Base Threshold | 0.78 | Requires verification |
| Low Confidence | 0.72 | Record to temporal window |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav

# Install dependencies
pip install -r requirements/base.txt
pip install -e .
```

### Usage Example

```python
from deploy.memory_modules import VisualPlaceRecognition, LongCLIPFeatureExtractor

# Initialize VPR
vpr = VisualPlaceRecognition(feature_dim=768, similarity_threshold=0.78)

# Initialize feature extractor
extractor = LongCLIPFeatureExtractor(
    model_path="path/to/longclip.pt",
    device="cuda:0"
)

# Extract features and add to database
feature = extractor.extract_feature(rgb_image)
vpr.add_feature(feature, node_id=0, timestamp=time.time())

# Loop closure detection
result = vpr.is_revisited(query_feature, current_time)
if result:
    node_id, similarity = result
    print(f"Loop detected: Node {node_id}, Similarity {similarity:.4f}")
```

## 📊 Performance Metrics

Evaluation results on internal test set:

| Metric | Value |
|--------|-------|
| Loop Detection Accuracy | 94.2% |
| Average Query Time | 12ms |
| False Positive Rate | < 2% |

## 🔧 Configuration

Edit `deploy/memory_modules/config.py`:

```python
class MemoryNavigationConfig:
    # VPR parameters
    similarity_threshold = 0.78
    high_confidence_threshold = 0.96
    
    # Multi-view parameters
    use_surround_cameras = True
    surround_weight = 0.25
    
    # Feature extraction
    feature_dim = 768
    longclip_model_path = "path/to/model"
```

## 📚 References

This project builds upon the following works:

- [InternNav](https://github.com/InternRobotics/InternNav) - Navigation foundation model
- [LongCLIP](https://github.com/beichenzbc/Long-CLIP) - Long-text CLIP model
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- DPV-SLAM, ORB-SLAM, TopoNav - VPR methodologies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- InternRobotics team for the InternNav framework
- LongCLIP authors for the visual encoder
- Facebook AI Research for FAISS

---

<div align="center">

**Made with ❤️ for Robot Navigation**

*Built upon [InternNav](https://github.com/InternRobotics/InternNav)*

</div>
