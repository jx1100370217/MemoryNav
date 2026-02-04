#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 环视相机融合模块 v2.0

支持两种模式：
1. 融合模式：将四个环视相机特征融合为一个向量
2. 独立模式：保留各视角独立编码，用于多视角VPR检测

v2.0 新增功能：
- 支持保留各视角独立编码
- 多视角相似度计算
- 最佳视角匹配
"""

import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from .config import MemoryNavigationConfig

logger = logging.getLogger(__name__)


@dataclass
class MultiViewFeatures:
    """多视角特征数据类"""
    camera_features: Dict[str, np.ndarray]  # {camera_id: feature}
    fused_feature: Optional[np.ndarray] = None  # 融合后的特征（可选）
    camera_angles: Dict[str, float] = None  # 各相机角度

    def __post_init__(self):
        if self.camera_angles is None:
            self.camera_angles = {
                'camera_1': 37.5,   # 前右
                'camera_2': -37.5,  # 前左
                'camera_3': -142.5, # 后左
                'camera_4': 142.5   # 后右
            }


@dataclass
class MultiViewMatchResult:
    """多视角匹配结果"""
    best_camera: str  # 最佳匹配的相机
    best_similarity: float  # 最佳相似度
    all_similarities: Dict[str, float]  # 各相机相似度
    voting_score: float  # 投票分数（多少视角认为匹配）
    weighted_similarity: float  # 加权平均相似度


class SurroundCameraFusion:
    """
    环视相机特征融合 v2.0

    支持两种模式：
    1. 融合模式：传统加权融合
    2. 独立模式：保留各视角独立特征，支持多视角VPR
    """

    # 相机角度配置 (度)
    CAMERA_ANGLES = {
        'camera_1': 37.5,   # 前右
        'camera_2': -37.5,  # 前左
        'camera_3': -142.5, # 后左
        'camera_4': 142.5   # 后右
    }

    def __init__(self, config: MemoryNavigationConfig):
        self.config = config
        # 仅环视相机 camera_1~4
        self.camera_ids = ['camera_1', 'camera_2', 'camera_3', 'camera_4']
        self.weights = {cam_id: config.surround_weight for cam_id in self.camera_ids}

        # v2.0: 多视角模式参数
        self.multi_view_mode = True  # 默认启用多视角模式
        self.voting_threshold = 0.5  # 投票阈值（多少比例的视角需要匹配）
        self.min_views_for_match = 2  # 最少需要匹配的视角数

    def set_multi_view_mode(self, enabled: bool):
        """设置多视角模式"""
        self.multi_view_mode = enabled
        logger.info(f"[SurroundFusion] 多视角模式: {'启用' if enabled else '禁用'}")

    def fuse_features(self,
                     surround_features: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        融合四个环视相机特征 (camera_1~4) - 传统模式

        Args:
            surround_features: {camera_id: feature} 字典
                camera_1: +37.5° (前右)
                camera_2: -37.5° (前左)
                camera_3: -142.5° (后左)
                camera_4: +142.5° (后右)

        Returns:
            fused_feature: 融合后的特征向量，如果无有效特征则返回 None
        """
        if not self.config.use_surround_cameras or not surround_features:
            return None

        fused = None
        total_weight = 0.0

        for cam_id in self.camera_ids:
            if cam_id in surround_features and surround_features[cam_id] is not None:
                feat = surround_features[cam_id]
                weight = self.weights[cam_id]

                if fused is None:
                    fused = feat * weight
                else:
                    fused += feat * weight
                total_weight += weight

        if fused is None or total_weight == 0:
            return None

        # 归一化
        fused = fused / total_weight
        fused = fused / (np.linalg.norm(fused) + 1e-8)

        return fused

    def get_independent_features(self,
                                 surround_features: Dict[str, np.ndarray],
                                 normalize: bool = True) -> MultiViewFeatures:
        """
        [v2.0] 获取独立的各视角特征

        Args:
            surround_features: {camera_id: feature} 字典
            normalize: 是否归一化特征

        Returns:
            MultiViewFeatures: 包含独立特征的数据结构
        """
        if not surround_features:
            return MultiViewFeatures(
                camera_features={},
                fused_feature=None,
                camera_angles=self.CAMERA_ANGLES.copy()
            )

        camera_features = {}
        for cam_id in self.camera_ids:
            if cam_id in surround_features and surround_features[cam_id] is not None:
                feat = surround_features[cam_id].astype('float32')
                if normalize:
                    feat = feat / (np.linalg.norm(feat) + 1e-8)
                camera_features[cam_id] = feat

        # 同时计算融合特征作为备用
        fused_feature = self.fuse_features(surround_features)

        return MultiViewFeatures(
            camera_features=camera_features,
            fused_feature=fused_feature,
            camera_angles=self.CAMERA_ANGLES.copy()
        )

    def compute_multi_view_similarity(self,
                                      query_features: MultiViewFeatures,
                                      db_features: MultiViewFeatures,
                                      similarity_threshold: float = 0.78) -> MultiViewMatchResult:
        """
        [v2.0] 计算多视角相似度

        对每个视角独立计算相似度，然后进行投票和加权融合

        Args:
            query_features: 查询帧的多视角特征
            db_features: 数据库中的多视角特征
            similarity_threshold: 单视角匹配阈值

        Returns:
            MultiViewMatchResult: 多视角匹配结果
        """
        all_similarities = {}
        matching_views = 0
        total_views = 0
        weighted_sum = 0.0
        total_weight = 0.0

        best_camera = None
        best_similarity = 0.0

        for cam_id in self.camera_ids:
            if (cam_id in query_features.camera_features and
                cam_id in db_features.camera_features):

                query_feat = query_features.camera_features[cam_id]
                db_feat = db_features.camera_features[cam_id]

                # 计算余弦相似度
                similarity = float(np.dot(query_feat, db_feat))
                all_similarities[cam_id] = similarity
                total_views += 1

                # 检查是否超过阈值
                if similarity >= similarity_threshold:
                    matching_views += 1

                # 记录最佳匹配
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_camera = cam_id

                # 加权求和
                weight = self.weights.get(cam_id, 1.0)
                weighted_sum += similarity * weight
                total_weight += weight

        # 计算投票分数
        voting_score = matching_views / total_views if total_views > 0 else 0.0

        # 计算加权平均相似度
        weighted_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0

        return MultiViewMatchResult(
            best_camera=best_camera or 'none',
            best_similarity=best_similarity,
            all_similarities=all_similarities,
            voting_score=voting_score,
            weighted_similarity=weighted_similarity
        )

    def is_multi_view_match(self,
                            match_result: MultiViewMatchResult,
                            similarity_threshold: float = 0.78) -> bool:
        """
        [v2.0] 判断是否为多视角匹配

        使用投票机制：至少一定比例的视角认为匹配

        Args:
            match_result: 多视角匹配结果
            similarity_threshold: 相似度阈值

        Returns:
            bool: 是否匹配
        """
        # 条件1: 投票分数超过阈值
        if match_result.voting_score >= self.voting_threshold:
            return True

        # 条件2: 最佳视角相似度很高（单视角强匹配）
        if match_result.best_similarity >= similarity_threshold + 0.1:
            return True

        # 条件3: 加权平均相似度超过阈值
        if match_result.weighted_similarity >= similarity_threshold:
            return True

        return False

    def get_angle_difference(self, camera1: str, camera2: str) -> float:
        """
        [v2.0] 计算两个相机之间的角度差

        Args:
            camera1: 第一个相机ID
            camera2: 第二个相机ID

        Returns:
            角度差（度）
        """
        angle1 = self.CAMERA_ANGLES.get(camera1, 0)
        angle2 = self.CAMERA_ANGLES.get(camera2, 0)
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)

    def get_camera_count(self, surround_features: Dict[str, np.ndarray]) -> int:
        """获取有效相机数量"""
        if not surround_features:
            return 0
        count = 0
        for cam_id in self.camera_ids:
            if cam_id in surround_features and surround_features[cam_id] is not None:
                count += 1
        return count

    def set_weights(self, weights: Dict[str, float]):
        """设置相机权重"""
        for cam_id, weight in weights.items():
            if cam_id in self.camera_ids:
                self.weights[cam_id] = weight

    def get_front_cameras(self) -> List[str]:
        """获取前向相机列表"""
        return ['camera_1', 'camera_2']  # 前右和前左

    def get_rear_cameras(self) -> List[str]:
        """获取后向相机列表"""
        return ['camera_3', 'camera_4']  # 后左和后右

    def get_camera_by_direction(self, direction: str) -> List[str]:
        """
        根据方向获取相机

        Args:
            direction: 'front', 'rear', 'left', 'right', 'all'

        Returns:
            相机ID列表
        """
        if direction == 'front':
            return ['camera_1', 'camera_2']
        elif direction == 'rear':
            return ['camera_3', 'camera_4']
        elif direction == 'left':
            return ['camera_2', 'camera_3']  # 前左和后左
        elif direction == 'right':
            return ['camera_1', 'camera_4']  # 前右和后右
        else:
            return self.camera_ids.copy()
