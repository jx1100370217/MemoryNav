#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav v2.0 - 视觉记忆导航模块

使用示例:
    from deploy.memory_nav import MemoryNavigator, MemoryBuilder
    
    # 构建记忆
    builder = MemoryBuilder()
    graph, vpr = builder.build_from_directory("merged_labeled_data")
    
    # 创建导航器
    navigator = MemoryNavigator()
    navigator.set_memory(graph, vpr)
    
    # 导航到目的地
    result = navigator.navigate_to("前台", start_node_id="1")
"""

from .memory_models import (
    MemoryNode,
    MemoryEdge,
    NavigationStep,
    NavigationPlan,
    VPRResult
)
from .memory_graph import MemoryGraph
from .memory_vpr import MemoryVPR
from .memory_builder import MemoryBuilder, LongCLIPExtractor
from .memory_navigator import MemoryNavigator

__version__ = "2.0.0"
__all__ = [
    # 数据模型
    "MemoryNode",
    "MemoryEdge", 
    "NavigationStep",
    "NavigationPlan",
    "VPRResult",
    # 核心模块
    "MemoryGraph",
    "MemoryVPR",
    "MemoryBuilder",
    "MemoryNavigator",
    # 特征提取
    "LongCLIPExtractor",
]
