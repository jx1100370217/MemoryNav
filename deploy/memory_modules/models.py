#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 数据模型模块

包含核心数据结构定义：
- TopologicalNode: 拓扑图节点
- RouteMemory: 导航路线记忆
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class TopologicalNode:
    """
    拓扑图节点 - 增强版 v2.0

    支持完整的导航记忆信息存储：
    - 基础信息：视觉特征、图像
    - 语义信息：场景描述、语义标签、节点名称
    - 导航信息：导航指令、像素目标、前视图特征编码
    - 元数据：时间戳、访问计数
    """
    node_id: int
    visual_feature: np.ndarray
    rgb_image: Optional[np.ndarray] = None
    surround_images: Dict[str, np.ndarray] = field(default_factory=dict)
    timestamp: float = 0.0
    instruction_context: Optional[str] = None
    is_landmark: bool = False
    semantic_label: Optional[str] = None
    visit_count: int = 1

    # ============ 语义信息 (v1.0) ============
    scene_description: Optional[str] = None  # VLM生成的场景描述
    semantic_labels: List[str] = field(default_factory=list)  # 语义标签列表（已去重）
    pixel_target: Optional[List[float]] = None  # 关键帧的像素目标
    is_keyframe: bool = False  # 是否为关键帧

    # ============ 增强信息 (v2.0) ============
    # 节点名称：基于场景描述和语义标签提炼的简短名称，用于快速识别
    node_name: Optional[str] = None

    # 导航指令：当前节点对应的导航指令（普通导航时使用）
    navigation_instruction: Optional[str] = None

    # 前视图特征编码：front_1相机的视觉特征（用于记忆检索）
    front_view_feature: Optional[np.ndarray] = None

    # 前视图图像路径：front_1相机图像的保存路径（用于可视化展示）
    front_view_image_path: Optional[str] = None

    # 创建时间戳：节点首次创建的时间戳（更精确的时间记录）
    created_at: float = 0.0

    # 最后更新时间戳
    updated_at: float = 0.0

    # 像素目标历史：记录多次访问时的所有pixel_target
    pixel_target_history: List[Dict] = field(default_factory=list)

    # 动作历史：记录进入和离开此节点的动作
    entry_actions: List[List[int]] = field(default_factory=list)
    exit_actions: List[List[int]] = field(default_factory=list)

    # ============ 节点来源追踪 (v2.1) ============
    # 节点来源：记录所有合并到此节点的图片时间戳列表
    # 格式: [{"timestamp": "1769394774504", "camera": "camera_1", "merged_at": 1706520000.0}, ...]
    source_timestamps: List[Dict] = field(default_factory=list)


@dataclass
class RouteMemory:
    """导航路线记忆"""
    route_id: str
    start_instruction: str
    start_timestamp: float
    node_sequence: List[int] = field(default_factory=list)
    action_history: List[List[int]] = field(default_factory=list)
    keyframe_indices: List[int] = field(default_factory=list)
    visual_features: Optional[np.ndarray] = None
    keyframe_images: List[np.ndarray] = field(default_factory=list)
    is_complete: bool = False
    end_timestamp: float = 0.0
