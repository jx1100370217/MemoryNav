#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Navigation System - 模块化组件

此包包含记忆导航系统的核心模块:
- config: 配置管理
- models: 数据模型（TopologicalNode, RouteMemory）
- feature_extraction: LongCLIP视觉特征提取
- scene_description: VLM场景描述生成
- vpr: 视觉位置识别（VPR）
- semantic_graph: GraphRAG语义图管理
- topological_map: 拓扑图管理
- route_memory: 路线记忆管理
- utils: 工具函数
"""

from .config import MemoryNavigationConfig
from .models import TopologicalNode, RouteMemory
from .feature_extraction import LongCLIPFeatureExtractor
from .scene_description_v5 import SceneDescriptionGeneratorV5 as SceneDescriptionGenerator
from .vpr import VisualPlaceRecognition
from .semantic_graph import SemanticGraphManager
from .topological_map import TopologicalMapManager
from .route_memory import RouteMemoryManager
from .surround_fusion import SurroundCameraFusion
from .return_navigator import ReturnNavigator
from .utils import (
    decode_base64_image,
    decode_base64_depth,
    encode_numpy_to_base64,
    convert_output_action_to_robot_action,
    convert_trajectory_to_robot_action,
)

__all__ = [
    # 配置
    'MemoryNavigationConfig',
    # 数据模型
    'TopologicalNode',
    'RouteMemory',
    # 核心组件
    'LongCLIPFeatureExtractor',
    'SceneDescriptionGenerator',
    'VisualPlaceRecognition',
    'SemanticGraphManager',
    'TopologicalMapManager',
    'RouteMemoryManager',
    'SurroundCameraFusion',
    'ReturnNavigator',
    # 工具函数
    'decode_base64_image',
    'decode_base64_depth',
    'encode_numpy_to_base64',
    'convert_output_action_to_robot_action',
    'convert_trajectory_to_robot_action',
]
