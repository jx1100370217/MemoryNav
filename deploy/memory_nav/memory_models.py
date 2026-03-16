#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav v3.0 - 数据模型

定义记忆导航系统的核心数据结构（新方案）：
- MemoryEdge: 记忆边 → camera_name + crop 子图（取代 angle + pixel_position + stitch）
- MemoryNode: 记忆节点（不变）
- NavigationStep: 导航步骤（适配新方案）

新方案：
  旧: angle + pixel_position + stitch_image → 机器人直接转向
  新: camera_name + crop_image + 子图匹配 → 机器人控制端自行生成目标
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class MemoryEdge:
    """
    记忆边 - 节点间的连接信息

    存储从当前节点到目标节点的导航线索：
    - camera_name: 注意力子图所在的相机 (camera_1~camera_4)
    - landmark_name: 注意力目标的地标名称（如"电梯"、"办公桌"）
    - crop_image_paths: 三级注意力子图路径 {"big": ..., "mid": ..., "small": ...}
    - big_box/mid_box/small_box: 归一化 bounding box
    """
    target_node_id: str               # 目标节点ID
    target_node_name: str             # 目标节点名称
    camera_name: str                  # 相机名称 (camera_1~camera_4)
    landmark_name: str                # 地标名称
    crop_image_paths: Dict[str, str] = field(default_factory=dict)  # 三级子图路径 {"big": ..., "mid": ..., "small": ...}
    big_box: Tuple[float, ...] = ()    # normalized bbox (big)
    mid_box: Tuple[float, ...] = ()    # normalized bbox (mid)
    small_box: Tuple[float, ...] = ()  # normalized bbox (small)
    target_node_name_eng: str = ""    # 目标节点英文名称
    landmark_name_eng: str = ""       # 地标英文名称
    crop_image_path: str = ""         # 向 big 的兼容引用

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'target_node_id': self.target_node_id,
            'target_node_name': self.target_node_name,
            'target_node_name_eng': self.target_node_name_eng,
            'camera_name': self.camera_name,
            'landmark_name': self.landmark_name,
            'landmark_name_eng': self.landmark_name_eng,
            'crop_image_paths': self.crop_image_paths,
            'crop_image_path': self.crop_image_path,
            'big_box': list(self.big_box),
            'mid_box': list(self.mid_box),
            'small_box': list(self.small_box),
        }

    @classmethod
    def from_dict(cls, data: Dict, base_path: str = "") -> 'MemoryEdge':
        """从字典创建"""
        # 解析 crop_image_paths
        crop_image_paths = data.get('crop_image_paths', {})

        # 解析 normalized boxes
        def _parse_float_box(raw):
            if isinstance(raw, str) and raw:
                return tuple(float(x) for x in raw.split(','))
            elif isinstance(raw, (list, tuple)):
                return tuple(float(x) for x in raw)
            return ()

        big_box = _parse_float_box(data.get('big_box', ()))
        mid_box = _parse_float_box(data.get('mid_box', ()))
        small_box = _parse_float_box(data.get('small_box', ()))

        return cls(
            target_node_id=str(data.get('position_id', '')),
            target_node_name=data.get('position_name', ''),
            target_node_name_eng=data.get('position_name_eng', ''),
            camera_name=data.get('camera_name', ''),
            landmark_name=data.get('landmark_name', ''),
            landmark_name_eng=data.get('landmark_name_eng', ''),
            crop_image_paths=crop_image_paths,
            crop_image_path=data.get('crop_image_path', ''),
            big_box=big_box,
            mid_box=mid_box,
            small_box=small_box,
        )


@dataclass
class MemoryNode:
    """
    记忆节点 - 拓扑图中的位置节点

    包含：
    - 节点基本信息（ID、名称）
    - 4张环视相机图（用于VPR）
    - 环视图视觉特征编码
    - 连接到其他节点的边
    """
    node_id: str                      # 节点ID (position_id)
    node_name: str                    # 节点名称 (position_name)
    node_name_eng: str = ""           # 节点英文名称 (position_name_eng)

    # 环视相机图路径
    camera_images: Dict[str, str] = field(default_factory=dict)
    # 格式: {'camera_1': 'xxx_camera_1.jpg', ...}

    # 环视图视觉特征（用于VPR）
    camera_features: Dict[str, np.ndarray] = field(default_factory=dict)
    # 格式: {'camera_1': feature_vector, ...}

    # 融合后的节点特征（所有相机特征的融合）
    fused_feature: Optional[np.ndarray] = None

    # 连接边（通往其他节点）
    edges: List[MemoryEdge] = field(default_factory=list)

    # 元数据
    base_path: str = ""               # 节点数据所在目录
    timestamp: str = ""               # 时间戳（从文件名提取）

    def get_edge_to(self, target_node_id: str) -> Optional[MemoryEdge]:
        """获取通往指定节点的边"""
        for edge in self.edges:
            if edge.target_node_id == target_node_id:
                return edge
        return None

    def get_neighbor_ids(self) -> List[str]:
        """获取所有相邻节点ID"""
        return [edge.target_node_id for edge in self.edges]

    def to_dict(self) -> Dict:
        """转换为字典（不含特征向量）"""
        return {
            'node_id': self.node_id,
            'node_name': self.node_name,
            'node_name_eng': self.node_name_eng,
            'camera_images': self.camera_images,
            'edges': [e.to_dict() for e in self.edges],
            'neighbor_count': len(self.edges),
            'has_features': self.fused_feature is not None
        }


@dataclass
class NavigationStep:
    """
    导航步骤 - 从一个节点到下一个节点的导航信息

    返回：
    - camera_name: 目标所在的相机
    - landmark_name: 注意力目标地标
    - crop_image_paths: 三级子图路径 {"big": ..., "mid": ..., "small": ...}
    """
    from_node_id: str                 # 起始节点ID
    from_node_name: str               # 起始节点名称
    to_node_id: str                   # 目标节点ID
    to_node_name: str                 # 目标节点名称
    camera_name: str                  # 目标相机 (camera_1~camera_4)
    landmark_name: str                # 注意力地标名称
    crop_image_paths: Dict[str, str] = field(default_factory=dict)  # 三级子图路径
    step_index: int = 0               # 步骤序号
    from_node_name_eng: str = ""
    to_node_name_eng: str = ""
    landmark_name_eng: str = ""
    crop_image_path: str = ""         # 向 big 的兼容引用

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'step_index': self.step_index,
            'from_node': {
                'id': self.from_node_id,
                'name': self.from_node_name,
                'name_eng': self.from_node_name_eng,
            },
            'to_node': {
                'id': self.to_node_id,
                'name': self.to_node_name,
                'name_eng': self.to_node_name_eng,
            },
            'camera_name': self.camera_name,
            'landmark_name': self.landmark_name,
            'landmark_name_eng': self.landmark_name_eng,
            'crop_image_paths': self.crop_image_paths,
            'crop_image_path': self.crop_image_path,
        }


@dataclass
class NavigationPlan:
    """
    导航计划 - 完整的路径规划结果
    """
    start_node_id: str
    start_node_name: str
    goal_node_id: str
    goal_node_name: str
    path: List[str]                   # 节点ID序列
    steps: List[NavigationStep]       # 导航步骤列表
    total_steps: int
    success: bool
    start_node_name_eng: str = ""
    goal_node_name_eng: str = ""
    message: str = ""

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'success': self.success,
            'message': self.message,
            'start': {'id': self.start_node_id, 'name': self.start_node_name,
                       'name_eng': self.start_node_name_eng},
            'goal': {'id': self.goal_node_id, 'name': self.goal_node_name,
                      'name_eng': self.goal_node_name_eng},
            'path': self.path,
            'total_steps': self.total_steps,
            'steps': [s.to_dict() for s in self.steps]
        }


@dataclass
class VPRResult:
    """VPR匹配结果"""
    matched_node_id: str
    matched_node_name: str
    similarity: float
    confidence: float
    matched_node_name_eng: str = ""
    camera_scores: Dict[str, float] = field(default_factory=dict)
    heading_offset: float = 0.0
    best_shift: int = 0

    def to_dict(self) -> Dict:
        return {
            'matched_node_id': self.matched_node_id,
            'matched_node_name': self.matched_node_name,
            'matched_node_name_eng': self.matched_node_name_eng,
            'similarity': self.similarity,
            'confidence': self.confidence,
            'camera_scores': self.camera_scores,
            'heading_offset': self.heading_offset,
            'best_shift': self.best_shift
        }
