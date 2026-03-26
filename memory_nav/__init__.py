#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav v2.0 - 视觉记忆导航模块

使用示例:
    from memory_nav import MemoryNavigator, MemoryBuilder
    
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
from .memory_builder import MemoryBuilder
from .anyloc_extractor import AnyLocExtractor
from .megaloc_extractor import MegaLocExtractor
from .effovpr_extractor import EffoVPRExtractor
from .selavpr_extractor import SelaVPRExtractor
from .vpr_factory import create_vpr_extractor
from .memory_navigator import MemoryNavigator
from .sub_image_matcher import SubImageMatcher, SubImageMatchResult, list_strategies, STRATEGY_DISPLAY_NAMES
from .qwen35_point_grounder import Qwen35PointGrounder

__version__ = "3.0.0"
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
    "AnyLocExtractor",
    "MegaLocExtractor",
    "EffoVPRExtractor",
    "SelaVPRExtractor",
    "create_vpr_extractor",
    # 子图匹配
    "SubImageMatcher",
    "SubImageMatchResult",
    "list_strategies",
    "STRATEGY_DISPLAY_NAMES",
    # 模型打点
    "Qwen35PointGrounder",
]
