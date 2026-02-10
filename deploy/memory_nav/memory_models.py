#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav v2.0 - 数据模型

定义记忆导航系统的核心数据结构：
- MemoryNode: 记忆节点
- MemoryEdge: 记忆边（节点间连接）
- NavigationStep: 导航步骤
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class MemoryEdge:
    """
    记忆边 - 节点间的连接信息
    
    存储从当前节点到目标节点的导航信息
    """
    target_node_id: str           # 目标节点ID
    target_node_name: str         # 目标节点名称
    angle: float                  # 绝对地理角度（度）
    pixel_position: Tuple[float, float]  # 归一化像素目标 (x, y)
    stitch_image_path: str        # 前视图路径
    target_node_name_eng: str = ""  # 目标节点英文名称
    stitch_image: Optional[np.ndarray] = None  # 前视图图像（可选加载）
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'target_node_id': self.target_node_id,
            'target_node_name': self.target_node_name,
            'target_node_name_eng': self.target_node_name_eng,
            'angle': self.angle,
            'pixel_position': list(self.pixel_position),
            'stitch_image_path': self.stitch_image_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict, base_path: str = "") -> 'MemoryEdge':
        """从字典创建"""
        pixel_pos = data.get('pixel_position', '0.5,0.5')
        if isinstance(pixel_pos, str):
            x, y = map(float, pixel_pos.split(','))
        else:
            x, y = pixel_pos[0], pixel_pos[1]
        
        return cls(
            target_node_id=str(data['position_id']),
            target_node_name=data.get('position_name', ''),
            target_node_name_eng=data.get('position_name_eng', ''),
            angle=float(data.get('angle', 0)),
            pixel_position=(x, y),
            stitch_image_path=data.get('stitch_image', '')
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
    node_id: str                  # 节点ID (position_id)
    node_name: str                # 节点名称 (position_name)
    node_name_eng: str = ""       # 节点英文名称 (position_name_eng)
    
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
    base_path: str = ""           # 节点数据所在目录
    timestamp: str = ""           # 时间戳（从文件名提取）
    
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
    """
    from_node_id: str             # 起始节点ID
    from_node_name: str           # 起始节点名称
    to_node_id: str               # 目标节点ID
    to_node_name: str             # 目标节点名称
    angle: float                  # 需要转向的绝对地理角度
    pixel_position: Tuple[float, float]  # 像素目标
    stitch_image_path: str        # 前视图路径
    step_index: int               # 步骤序号（在整个路径中的位置）
    from_node_name_eng: str = ""  # 起始节点英文名称
    to_node_name_eng: str = ""    # 目标节点英文名称
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'step_index': self.step_index,
            'from_node': {'id': self.from_node_id, 'name': self.from_node_name, 'name_eng': self.from_node_name_eng},
            'to_node': {'id': self.to_node_id, 'name': self.to_node_name, 'name_eng': self.to_node_name_eng},
            'angle': self.angle,
            'pixel_position': list(self.pixel_position),
            'stitch_image_path': self.stitch_image_path
        }


@dataclass
class NavigationPlan:
    """
    导航计划 - 完整的路径规划结果
    """
    start_node_id: str            # 起点节点ID
    start_node_name: str          # 起点节点名称
    goal_node_id: str             # 终点节点ID
    goal_node_name: str           # 终点节点名称
    path: List[str]               # 节点ID序列
    steps: List[NavigationStep]   # 导航步骤列表
    total_steps: int              # 总步数
    success: bool                 # 是否规划成功
    start_node_name_eng: str = "" # 起点英文名称
    goal_node_name_eng: str = ""  # 终点英文名称
    message: str = ""             # 消息/错误信息
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'success': self.success,
            'message': self.message,
            'start': {'id': self.start_node_id, 'name': self.start_node_name, 'name_eng': self.start_node_name_eng},
            'goal': {'id': self.goal_node_id, 'name': self.goal_node_name, 'name_eng': self.goal_node_name_eng},
            'path': self.path,
            'total_steps': self.total_steps,
            'steps': [s.to_dict() for s in self.steps]
        }


@dataclass 
class VPRResult:
    """
    VPR匹配结果
    """
    matched_node_id: str          # 匹配到的节点ID
    matched_node_name: str        # 匹配到的节点名称
    similarity: float             # 相似度分数
    confidence: float             # 置信度
    matched_node_name_eng: str = ""  # 匹配到的节点英文名称
    camera_scores: Dict[str, float] = field(default_factory=dict)  # 各相机得分
    heading_offset: float = 0.0           # 朝向偏移角度（度，顺时针为正）
    best_shift: int = 0                   # 最佳循环移位 (0-3)
    
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
