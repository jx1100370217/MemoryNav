<div align="center">

# 🧠 MemoryNav

**视觉记忆导航系统 | Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

基于视觉位置识别（VPR）和拓扑地图的机器人记忆导航系统

[English](README_EN.md) | **中文**

</div>

---

## 📖 简介

MemoryNav 是一个面向移动机器人的视觉记忆导航系统，实现了：

- **视觉位置识别 (VPR)**：基于 LongCLIP 的多视角回环检测
- **拓扑地图构建**：实时构建环境的拓扑表示
- **语义引导导航**：结合语义标签提升定位精度
- **多视角融合**：4 相机环视系统，投票机制匹配

## 🏗️ 系统架构

```
MemoryNav 系统架构
├── deploy/                    # 部署模块
│   ├── memory_modules/        # 核心记忆模块
│   │   ├── vpr.py            # 视觉位置识别 (v4.0)
│   │   ├── feature_extraction.py  # LongCLIP 特征提取
│   │   ├── surround_fusion.py     # 多视角融合
│   │   ├── topological_map.py     # 拓扑地图
│   │   └── config.py              # 配置管理
│   ├── visual_memory_system.py    # 视觉记忆系统
│   └── ws_proxy_with_memory.py    # WebSocket 代理服务
├── internnav/                 # InternNav 导航框架
│   ├── agent/                 # 导航智能体
│   ├── model/                 # 模型定义
│   │   ├── basemodel/         # 基础模型 (LongCLIP, InternVLA 等)
│   │   └── encoder/           # 编码器模块
│   ├── env/                   # 环境接口
│   └── evaluator/             # 评估模块
└── scripts/                   # 工具脚本
```

## ✨ 核心特性

### 1. 多视角 VPR (v4.0)
- 4 个环视相机独立 FAISS 索引
- 投票机制确认回环检测
- 自适应阈值调整

```python
# 相机配置
CAMERA_ANGLES = {
    'camera_1': 37.5,   # 前右
    'camera_2': -37.5,  # 前左
    'camera_3': -142.5, # 后左
    'camera_4': 142.5   # 后右
}
```

### 2. LongCLIP 视觉编码
- 768 维特征向量
- L2 归一化
- 支持 GPU 加速

### 3. 多阶段验证
- 时间间隔检查 (>5秒)
- 空间一致性验证
- 语义标签引导
- 时序一致性验证

### 4. 阈值体系
| 阈值类型 | 数值 | 说明 |
|---------|------|------|
| 高置信度 | 0.96 | 直接确认回环 |
| 基础阈值 | 0.78 | 需要验证 |
| 低置信度 | 0.72 | 记录到时序窗口 |

## 🚀 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav

# 安装依赖
pip install -r requirements/base.txt
pip install -e .
```

### 运行示例

```python
from deploy.memory_modules import VisualPlaceRecognition, LongCLIPFeatureExtractor

# 初始化 VPR
vpr = VisualPlaceRecognition(feature_dim=768, similarity_threshold=0.78)

# 初始化特征提取器
extractor = LongCLIPFeatureExtractor(
    model_path="path/to/longclip.pt",
    device="cuda:0"
)

# 提取特征并添加到数据库
feature = extractor.extract_feature(rgb_image)
vpr.add_feature(feature, node_id=0, timestamp=time.time())

# 回环检测
result = vpr.is_revisited(query_feature, current_time)
if result:
    node_id, similarity = result
    print(f"检测到回环: 节点 {node_id}, 相似度 {similarity:.4f}")
```

## 📊 性能指标

基于内部测试集的评估结果：

| 指标 | 数值 |
|------|------|
| 回环检测准确率 | 94.2% |
| 平均查询时间 | 12ms |
| 误检率 | < 2% |

## 🔧 配置说明

编辑 `deploy/memory_modules/config.py` 进行配置：

```python
class MemoryNavigationConfig:
    # VPR 参数
    similarity_threshold = 0.78
    high_confidence_threshold = 0.96
    
    # 多视角参数
    use_surround_cameras = True
    surround_weight = 0.25
    
    # 特征提取
    feature_dim = 768
    longclip_model_path = "path/to/model"
```

## 📚 参考文献

本项目基于以下工作：

- [InternNav](https://github.com/InternRobotics/InternNav) - 导航基础模型
- [LongCLIP](https://github.com/beichenzbc/Long-CLIP) - 长文本 CLIP 模型
- [FAISS](https://github.com/facebookresearch/faiss) - 高效相似性搜索
- DPV-SLAM, ORB-SLAM, TopoNav - VPR 方法论

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- InternRobotics 团队提供的 InternNav 框架
- LongCLIP 作者提供的视觉编码器
- Facebook AI Research 提供的 FAISS

---

<div align="center">

**Made with ❤️ for Robot Navigation**

*基于 [InternNav](https://github.com/InternRobotics/InternNav) 构建*

</div>
