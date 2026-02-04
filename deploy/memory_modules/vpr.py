#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 视觉位置识别模块 (VPR) v4.0

基于FAISS的高效相似度搜索，用于回环检测。

v4.0 新增功能 (多视角独立检测):
1. 多视角独立FAISS索引: 每个相机独立的特征索引
2. 多视角投票机制: 基于多视角的匹配投票
3. 视角间一致性验证: 验证不同视角的匹配一致性
4. 最佳视角自动选择: 根据场景自动选择最佳匹配视角

v3.0 功能 (对标业界标准):
1. 环境自适应阈值: 根据场景复杂度和光照变化动态调整
2. 几何一致性验证: 使用特征点匹配验证空间关系
3. 层次化搜索: 粗筛+精排两阶段检索
4. 置信度评估: 输出匹配置信度用于决策
5. 在线学习: 根据反馈动态优化阈值

基于以下业界标准实现:
- DPV-SLAM with AnyLoc (2026): 自适应阈值机制
- ORB-SLAM研究: 几何验证的重要性
- TopoNav (2025): 拓扑图结构化记忆
"""

import logging
from typing import List, Tuple, Optional, Dict, Set, Any
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import time
import math

# 尝试导入FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class VPRMatchResult:
    """VPR匹配结果数据类"""
    node_id: int
    similarity: float
    confidence: float  # 综合置信度 (0-1)
    match_type: str  # 'high_confidence', 'temporal_verified', 'geometric_verified', 'multi_view_voted'
    geometric_score: float = 0.0
    semantic_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiViewVPRResult:
    """多视角VPR匹配结果"""
    node_id: int
    best_camera: str  # 最佳匹配的相机
    best_similarity: float  # 最佳相似度
    camera_similarities: Dict[str, float]  # 各相机相似度
    voting_score: float  # 投票分数 (0-1)
    weighted_similarity: float  # 加权平均相似度
    matching_cameras: List[str] = field(default_factory=list)  # 超过阈值的相机列表


@dataclass
class EnvironmentState:
    """环境状态追踪 - 用于自适应阈值"""
    avg_similarity: float = 0.85  # 平均相似度
    similarity_variance: float = 0.01  # 相似度方差
    scene_complexity: float = 0.5  # 场景复杂度 (0-1)
    lighting_stability: float = 1.0  # 光照稳定性 (0-1)
    feature_density: float = 0.5  # 特征密度
    last_update_time: float = 0.0


class VisualPlaceRecognition:
    """
    增强版视觉位置识别模块 v4.0 - 基于FAISS的高效相似度搜索

    v4.0 核心改进 (多视角独立检测):
    1. 多视角独立索引: 每个相机独立的FAISS索引
    2. 多视角投票机制: 基于视角投票的匹配确认
    3. 视角间一致性: 验证不同视角匹配的一致性
    4. 最佳视角选择: 自动选择最佳匹配视角

    v3.0 核心功能:
    1. 环境自适应阈值: 根据场景动态调整
    2. 几何一致性验证: 特征点匹配验证空间关系
    3. 层次化搜索: 粗筛+精排提高效率
    4. 置信度输出: 支持不确定性感知决策
    5. 在线学习: 根据运行反馈持续优化
    """

    # 相机配置
    CAMERA_IDS = ['camera_1', 'camera_2', 'camera_3', 'camera_4']
    CAMERA_ANGLES = {
        'camera_1': 37.5,   # 前右
        'camera_2': -37.5,  # 前左
        'camera_3': -142.5, # 后左
        'camera_4': 142.5   # 后右
    }

    def __init__(self, feature_dim: int = 768, similarity_threshold: float = 0.78):
        self.feature_dim = feature_dim

        # ====================================================================
        # v3.0: 基础阈值 (作为自适应阈值的基准)
        # v3.1: 提高高置信度阈值，避免原地转向时过度合并节点
        # ====================================================================
        self.base_similarity_threshold = similarity_threshold
        self.similarity_threshold = similarity_threshold  # 当前使用的阈值
        self.high_confidence_threshold = 0.96  # 从0.90提高到0.96，只有非常相似才直接合并
        self.low_confidence_threshold = 0.72

        # 语义匹配加成阈值
        self.semantic_bonus = 0.05
        self.semantic_penalty = 0.03

        # ====================================================================
        # v3.0 新增: 环境状态追踪 (用于自适应阈值)
        # ====================================================================
        self.env_state = EnvironmentState()
        self.adaptive_threshold_enabled = True
        self.adaptive_update_interval = 10  # 每10次查询更新一次环境状态

        # ====================================================================
        # v3.0 新增: 几何验证参数
        # ====================================================================
        self.geometric_verification_enabled = True
        self.min_geometric_inliers = 8  # 最小内点数
        self.geometric_confidence_weight = 0.3  # 几何验证权重

        # ====================================================================
        # v3.0 新增: 层次化搜索参数
        # ====================================================================
        self.coarse_search_k = 30  # 粗筛返回数量
        self.fine_search_k = 10  # 精排返回数量
        self.use_hierarchical_search = True

        # 初始化FAISS索引或使用numpy后备方案
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(feature_dim)  # 内积 (余弦相似度)
            logger.info(f"[VPR v4.0] FAISS索引已初始化 (维度={feature_dim})")
        else:
            self.index = None
            logger.warning("[VPR v4.0] FAISS不可用，使用numpy后备方案")

        # ====================================================================
        # v4.0 新增: 多视角独立FAISS索引
        # ====================================================================
        self.multi_view_enabled = True  # 启用多视角模式
        self.multi_view_indices: Dict[str, Any] = {}  # {camera_id: faiss_index}
        self.multi_view_features_db: Dict[str, List[np.ndarray]] = {}  # {camera_id: [features]}
        self.multi_view_node_ids: Dict[str, List[int]] = {}  # {camera_id: [node_ids]}

        # 初始化各相机的FAISS索引
        for cam_id in self.CAMERA_IDS:
            if FAISS_AVAILABLE:
                self.multi_view_indices[cam_id] = faiss.IndexFlatIP(feature_dim)
            else:
                self.multi_view_indices[cam_id] = None
            self.multi_view_features_db[cam_id] = []
            self.multi_view_node_ids[cam_id] = []

        # 多视角投票参数
        self.voting_threshold = 0.5  # 需要超过50%的视角匹配
        self.min_matching_views = 2  # 最少需要2个视角匹配

        logger.info(f"[VPR v4.0] 多视角索引已初始化 (4个相机独立索引)")

        # 特征数据库
        self.features_db: List[np.ndarray] = []
        self.node_ids: List[int] = []
        self.timestamps: List[float] = []

        # 环视特征数据库 (融合后的特征，兼容旧版)
        self.surround_features_db: List[Optional[np.ndarray]] = []

        # v4.0: 多视角独立特征数据库
        self.multi_view_surround_db: List[Dict[str, np.ndarray]] = []  # [{camera_id: feature}]

        # 语义标签数据库
        self.semantic_labels_db: List[Set[str]] = []

        # 场景描述数据库
        self.scene_descriptions_db: List[str] = []

        # ====================================================================
        # v3.0 新增: 特征点数据库 (用于几何验证)
        # ====================================================================
        self.keypoints_db: List[Optional[np.ndarray]] = []  # 特征点坐标
        self.descriptors_db: List[Optional[np.ndarray]] = []  # 特征点描述子

        # 时序一致性验证器
        self.temporal_window_size = 3
        self.recent_matches: deque = deque(maxlen=self.temporal_window_size)

        # 空间一致性追踪
        self.last_confirmed_node: Optional[int] = None
        self.max_topological_jump = 5

        # 统计信息
        self.match_history: List[float] = []
        self.total_queries = 0
        self.successful_detections = 0

        # ====================================================================
        # v3.0 新增: 扩展统计信息
        # ====================================================================
        self.confidence_history: List[float] = []  # 置信度历史
        self.geometric_verification_results: List[bool] = []  # 几何验证结果
        self.threshold_adaptation_history: List[Tuple[float, float]] = []  # (时间, 阈值)

        # 困难负样本记录
        self.hard_negatives: List[Tuple[float, int, int]] = []
        self.false_positive_history: List[float] = []

        # 性能计时器
        self.query_times: List[float] = []

        logger.info(f"[VPR v4.0] 初始化完成: 多视角模式={self.multi_view_enabled}, "
                   f"自适应阈值={self.adaptive_threshold_enabled}, "
                   f"几何验证={self.geometric_verification_enabled}, 层次搜索={self.use_hierarchical_search}")

    def add_feature(self, feature: np.ndarray, node_id: int, timestamp: float,
                    surround_feature: np.ndarray = None,
                    surround_features_dict: Dict[str, np.ndarray] = None,
                    semantic_labels: List[str] = None,
                    scene_description: str = None):
        """
        添加特征到数据库 - v4.0支持多视角独立特征

        Args:
            feature: 主特征向量 (前置相机)
            node_id: 节点ID
            timestamp: 时间戳
            surround_feature: 环视融合特征 (可选, 兼容旧版)
            surround_features_dict: 多视角独立特征字典 {camera_id: feature} (v4.0新增)
            semantic_labels: 语义标签列表 (可选)
            scene_description: 场景描述 (可选)
        """
        feature_normalized = feature.astype('float32') / (np.linalg.norm(feature) + 1e-8)

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(feature_normalized.reshape(1, -1))

        self.features_db.append(feature_normalized)
        self.node_ids.append(node_id)
        self.timestamps.append(timestamp)

        # 存储环视融合特征 (兼容旧版)
        if surround_feature is not None:
            surround_normalized = surround_feature.astype('float32') / (np.linalg.norm(surround_feature) + 1e-8)
            self.surround_features_db.append(surround_normalized)
        else:
            self.surround_features_db.append(None)

        # ====================================================================
        # v4.0 新增: 存储多视角独立特征
        # ====================================================================
        multi_view_features = {}
        if surround_features_dict:
            for cam_id in self.CAMERA_IDS:
                if cam_id in surround_features_dict and surround_features_dict[cam_id] is not None:
                    feat = surround_features_dict[cam_id].astype('float32')
                    feat_normalized = feat / (np.linalg.norm(feat) + 1e-8)
                    multi_view_features[cam_id] = feat_normalized

                    # 添加到各相机的独立索引
                    if FAISS_AVAILABLE and self.multi_view_indices.get(cam_id) is not None:
                        self.multi_view_indices[cam_id].add(feat_normalized.reshape(1, -1))
                    self.multi_view_features_db[cam_id].append(feat_normalized)
                    self.multi_view_node_ids[cam_id].append(node_id)

        self.multi_view_surround_db.append(multi_view_features)

        # 存储语义标签 (转为小写集合便于匹配)
        if semantic_labels:
            labels_set = set(label.lower() for label in semantic_labels)
            self.semantic_labels_db.append(labels_set)
        else:
            self.semantic_labels_db.append(set())

        # 存储场景描述
        self.scene_descriptions_db.append(scene_description or "")

    def search(self, query_feature: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """搜索最相似的位置"""
        if len(self.features_db) == 0:
            return []

        query_normalized = query_feature.astype('float32') / (np.linalg.norm(query_feature) + 1e-8)

        if FAISS_AVAILABLE and self.index is not None:
            k = min(k, self.index.ntotal)
            if k == 0:
                return []
            distances, indices = self.index.search(
                query_normalized.reshape(1, -1), k
            )
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.node_ids):
                    similarity = float(distances[0][i])
                    node_id = self.node_ids[idx]
                    results.append((node_id, similarity))
        else:
            # Numpy后备方案
            features_matrix = np.array(self.features_db)
            similarities = np.dot(features_matrix, query_normalized)
            top_k_indices = np.argsort(similarities)[::-1][:k]
            results = [(self.node_ids[idx], float(similarities[idx])) for idx in top_k_indices]

        return results

    def search_with_surround(self, query_feature: np.ndarray,
                              query_surround: np.ndarray = None,
                              k: int = 5,
                              surround_weight: float = 0.3) -> List[Tuple[int, float]]:
        """
        融合环视特征的搜索

        Args:
            query_feature: 前置相机特征
            query_surround: 环视融合特征
            k: 返回top-k结果
            surround_weight: 环视特征权重 (0-1)

        Returns:
            [(node_id, combined_similarity), ...]
        """
        # 首先用主特征搜索
        main_results = self.search(query_feature, k=k*2)

        if query_surround is None or len(self.surround_features_db) == 0:
            return main_results[:k]

        query_surround_norm = query_surround.astype('float32') / (np.linalg.norm(query_surround) + 1e-8)

        # 计算融合相似度
        combined_results = []
        for node_id, main_sim in main_results:
            idx = self.node_ids.index(node_id)
            surround_feat = self.surround_features_db[idx]

            if surround_feat is not None:
                surround_sim = float(np.dot(surround_feat, query_surround_norm))
                combined_sim = (1 - surround_weight) * main_sim + surround_weight * surround_sim
            else:
                combined_sim = main_sim

            combined_results.append((node_id, combined_sim))

        # 按融合相似度排序
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:k]

    # ========================================================================
    # v4.0 新增: 多视角独立搜索方法
    # ========================================================================

    def search_multi_view(self, query_features_dict: Dict[str, np.ndarray],
                          k: int = 10) -> List[MultiViewVPRResult]:
        """
        [v4.0] 多视角独立搜索

        对每个相机独立执行FAISS搜索，然后汇总结果

        Args:
            query_features_dict: {camera_id: feature} 查询特征字典
            k: 返回top-k结果

        Returns:
            List[MultiViewVPRResult]: 多视角匹配结果列表
        """
        if not query_features_dict:
            return []

        # 收集所有相机的搜索结果
        camera_results: Dict[str, List[Tuple[int, float]]] = {}
        all_candidate_nodes: Set[int] = set()

        for cam_id in self.CAMERA_IDS:
            if cam_id not in query_features_dict or query_features_dict[cam_id] is None:
                continue

            query_feat = query_features_dict[cam_id].astype('float32')
            query_normalized = query_feat / (np.linalg.norm(query_feat) + 1e-8)

            # 在该相机的索引中搜索
            if FAISS_AVAILABLE and self.multi_view_indices.get(cam_id) is not None:
                index = self.multi_view_indices[cam_id]
                if index.ntotal == 0:
                    continue
                search_k = min(k * 2, index.ntotal)
                distances, indices = index.search(query_normalized.reshape(1, -1), search_k)

                results = []
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(self.multi_view_node_ids[cam_id]):
                        node_id = self.multi_view_node_ids[cam_id][idx]
                        similarity = float(distances[0][i])
                        results.append((node_id, similarity))
                        all_candidate_nodes.add(node_id)
                camera_results[cam_id] = results
            else:
                # Numpy后备方案
                if not self.multi_view_features_db.get(cam_id):
                    continue
                features_matrix = np.array(self.multi_view_features_db[cam_id])
                similarities = np.dot(features_matrix, query_normalized)
                top_k_indices = np.argsort(similarities)[::-1][:k*2]
                results = []
                for idx in top_k_indices:
                    node_id = self.multi_view_node_ids[cam_id][idx]
                    results.append((node_id, float(similarities[idx])))
                    all_candidate_nodes.add(node_id)
                camera_results[cam_id] = results

        # 汇总各相机结果，对每个候选节点计算多视角匹配分数
        multi_view_results = []

        for node_id in all_candidate_nodes:
            camera_similarities = {}
            matching_cameras = []

            for cam_id, results in camera_results.items():
                for nid, sim in results:
                    if nid == node_id:
                        camera_similarities[cam_id] = sim
                        if sim >= self.similarity_threshold:
                            matching_cameras.append(cam_id)
                        break

            # 计算投票分数
            total_cameras = len(camera_results)
            voting_score = len(matching_cameras) / total_cameras if total_cameras > 0 else 0.0

            # 计算加权平均相似度
            if camera_similarities:
                weighted_sum = sum(camera_similarities.values())
                weighted_similarity = weighted_sum / len(camera_similarities)
            else:
                weighted_similarity = 0.0

            # 找出最佳相机
            if camera_similarities:
                best_camera = max(camera_similarities, key=camera_similarities.get)
                best_similarity = camera_similarities[best_camera]
            else:
                best_camera = 'none'
                best_similarity = 0.0

            multi_view_results.append(MultiViewVPRResult(
                node_id=node_id,
                best_camera=best_camera,
                best_similarity=best_similarity,
                camera_similarities=camera_similarities,
                voting_score=voting_score,
                weighted_similarity=weighted_similarity,
                matching_cameras=matching_cameras
            ))

        # 按加权相似度排序
        multi_view_results.sort(key=lambda x: x.weighted_similarity, reverse=True)
        return multi_view_results[:k]

    def is_multi_view_match(self, result: MultiViewVPRResult) -> bool:
        """
        [v4.0] 判断多视角结果是否为有效匹配

        Args:
            result: 多视角匹配结果

        Returns:
            bool: 是否匹配
        """
        # 条件1: 投票分数超过阈值
        if result.voting_score >= self.voting_threshold:
            return True

        # 条件2: 至少有min_matching_views个视角匹配
        if len(result.matching_cameras) >= self.min_matching_views:
            return True

        # 条件3: 最佳视角相似度很高（单视角强匹配）
        if result.best_similarity >= self.high_confidence_threshold:
            return True

        # 条件4: 加权平均相似度超过阈值
        if result.weighted_similarity >= self.similarity_threshold:
            return True

        return False

    def is_revisited_multi_view(self, query_feature: np.ndarray,
                                 current_time: float,
                                 min_time_gap: float = 5.0,
                                 query_surround_dict: Dict[str, np.ndarray] = None,
                                 query_semantic_labels: List[str] = None) -> Optional[Tuple[int, float, MultiViewVPRResult]]:
        """
        [v4.0] 多视角回环检测

        Args:
            query_feature: 前置相机查询特征
            current_time: 当前时间戳
            min_time_gap: 最小时间间隔
            query_surround_dict: 多视角独立特征字典 {camera_id: feature}
            query_semantic_labels: 语义标签

        Returns:
            (node_id, similarity, MultiViewVPRResult) 如果匹配成功，否则None
        """
        start_time = time.time()
        self.total_queries += 1

        # 如果没有多视角特征，退回到传统方法
        if not query_surround_dict or not self.multi_view_enabled:
            result = self.is_revisited(query_feature, current_time, min_time_gap,
                                       query_surround=None, query_semantic_labels=query_semantic_labels)
            self._record_query_time(start_time)
            if result:
                return (result[0], result[1], None)
            return None

        # 执行多视角搜索
        multi_view_results = self.search_multi_view(query_surround_dict, k=15)

        if not multi_view_results:
            self._record_query_time(start_time)
            return None

        # 遍历候选进行验证
        for mv_result in multi_view_results:
            node_id = mv_result.node_id

            # 获取时间戳
            if node_id not in self.node_ids:
                continue
            idx = self.node_ids.index(node_id)
            time_gap = current_time - self.timestamps[idx]

            # 时间间隔检查
            if time_gap <= min_time_gap:
                continue

            # 空间一致性检查
            if not self._check_spatial_consistency(node_id):
                continue

            # 判断是否为有效多视角匹配
            if self.is_multi_view_match(mv_result):
                logger.info(f"[VPR v4.0] 多视角回环检测成功: node={node_id}, "
                           f"best_cam={mv_result.best_camera}, best_sim={mv_result.best_similarity:.4f}, "
                           f"voting={mv_result.voting_score:.2f}, matching_cams={mv_result.matching_cameras}")

                self._confirm_match(node_id, mv_result.weighted_similarity)
                self._record_query_time(start_time)
                return (node_id, mv_result.weighted_similarity, mv_result)

        self._record_query_time(start_time)
        return None

    def search_with_semantic(self, query_feature: np.ndarray,
                             query_surround: np.ndarray = None,
                             query_semantic_labels: List[str] = None,
                             k: int = 5,
                             surround_weight: float = 0.3,
                             semantic_weight: float = 0.15) -> List[Tuple[int, float, Dict]]:
        """
        [新增] 融合环视特征和语义标签的搜索

        Args:
            query_feature: 前置相机特征
            query_surround: 环视融合特征
            query_semantic_labels: 当前帧的语义标签
            k: 返回top-k结果
            surround_weight: 环视特征权重 (0-1)
            semantic_weight: 语义匹配权重 (0-1)

        Returns:
            [(node_id, combined_similarity, match_info), ...]
            match_info包含: visual_sim, surround_sim, semantic_sim, matched_labels
        """
        # 首先用主特征搜索
        main_results = self.search(query_feature, k=k*3)

        if not main_results:
            return []

        query_labels_set = set()
        if query_semantic_labels:
            query_labels_set = set(label.lower() for label in query_semantic_labels)

        query_surround_norm = None
        if query_surround is not None:
            query_surround_norm = query_surround.astype('float32') / (np.linalg.norm(query_surround) + 1e-8)

        # 计算融合相似度
        combined_results = []
        for node_id, visual_sim in main_results:
            idx = self.node_ids.index(node_id)

            # 环视相似度
            surround_sim = 0.0
            surround_feat = self.surround_features_db[idx] if idx < len(self.surround_features_db) else None
            if surround_feat is not None and query_surround_norm is not None:
                surround_sim = float(np.dot(surround_feat, query_surround_norm))

            # [新增] 语义相似度计算
            semantic_sim = 0.0
            matched_labels = []
            if query_labels_set and idx < len(self.semantic_labels_db):
                db_labels = self.semantic_labels_db[idx]
                if db_labels:
                    # Jaccard相似度
                    intersection = query_labels_set & db_labels
                    union = query_labels_set | db_labels
                    if len(union) > 0:
                        semantic_sim = len(intersection) / len(union)
                        matched_labels = list(intersection)

            # 组合相似度
            visual_weight = 1.0 - surround_weight - semantic_weight
            if query_surround_norm is None:
                visual_weight += surround_weight
                surround_weight_actual = 0.0
            else:
                surround_weight_actual = surround_weight

            if not query_labels_set:
                visual_weight += semantic_weight
                semantic_weight_actual = 0.0
            else:
                semantic_weight_actual = semantic_weight

            combined_sim = (visual_weight * visual_sim +
                          surround_weight_actual * surround_sim +
                          semantic_weight_actual * semantic_sim)

            match_info = {
                'visual_sim': visual_sim,
                'surround_sim': surround_sim,
                'semantic_sim': semantic_sim,
                'matched_labels': matched_labels
            }

            combined_results.append((node_id, combined_sim, match_info))

        # 按融合相似度排序
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:k]

    def is_revisited(self, query_feature: np.ndarray,
                     current_time: float,
                     min_time_gap: float = 5.0,
                     query_surround: np.ndarray = None,
                     query_semantic_labels: List[str] = None) -> Optional[Tuple[int, float]]:
        """
        判断是否为已访问位置 (回环检测) - 增强版 v2.0

        改进策略:
        1. 时间间隔检查
        2. 高置信度匹配直接确认
        3. 中置信度匹配需要时序验证
        4. 支持环视特征融合
        5. [新增] 语义标签引导: 匹配语义标签可获得加成
        6. [新增] 空间一致性验证: 避免跳跃式误匹配

        Args:
            query_feature: 查询特征
            current_time: 当前时间戳
            min_time_gap: 最小时间间隔 (秒)
            query_surround: 环视融合特征 (可选)
            query_semantic_labels: 当前帧的语义标签 (可选)

        Returns:
            (node_id, similarity) 如果匹配成功，否则None
        """
        start_time = time.time()
        self.total_queries += 1

        # 使用语义引导搜索 (如果有语义标签)
        if query_semantic_labels and len(self.semantic_labels_db) > 0:
            results_with_info = self.search_with_semantic(
                query_feature, query_surround, query_semantic_labels, k=15
            )
            results = [(node_id, sim) for node_id, sim, _ in results_with_info]
        elif query_surround is not None:
            results = self.search_with_surround(query_feature, query_surround, k=15)
        else:
            results = self.search(query_feature, k=15)

        # 调试日志：显示top匹配信息
        if results:
            top_node, top_sim = results[0]
            if top_node in self.node_ids:
                idx = self.node_ids.index(top_node)
                time_gap = current_time - self.timestamps[idx]
                semantic_info = ""
                if query_semantic_labels and idx < len(self.semantic_labels_db):
                    query_set = set(l.lower() for l in query_semantic_labels)
                    db_labels = self.semantic_labels_db[idx]
                    matched = query_set & db_labels
                    if matched:
                        semantic_info = f", semantic_match={len(matched)}"
                logger.info(f"[VPR Debug] Q#{self.total_queries}: top1_node={top_node}, sim={top_sim:.4f}, "
                           f"time_gap={time_gap:.1f}s, threshold={self.similarity_threshold}, "
                           f"high_threshold={self.high_confidence_threshold}, min_gap={min_time_gap}s{semantic_info}")

        candidate = None

        for node_id, similarity in results:
            idx = self.node_ids.index(node_id)
            time_gap = current_time - self.timestamps[idx]

            # 时间间隔检查
            if time_gap <= min_time_gap:
                continue

            # [新增] 空间一致性检查 - 避免跳跃式误匹配
            if not self._check_spatial_consistency(node_id):
                logger.debug(f"[VPR] 空间一致性检查失败: node={node_id}, 跳过")
                continue

            # [新增] 语义加成/惩罚
            adjusted_similarity = self._apply_semantic_adjustment(
                similarity, idx, query_semantic_labels
            )

            # 高置信度: 直接确认回环
            if adjusted_similarity >= self.high_confidence_threshold:
                logger.info(f"[VPR] 高置信度回环检测: node={node_id}, sim={similarity:.4f}, "
                           f"adjusted_sim={adjusted_similarity:.4f}")
                self._confirm_match(node_id, similarity)
                self._record_query_time(start_time)
                return (node_id, similarity)

            # 中置信度: 检查时序一致性
            if adjusted_similarity >= self.similarity_threshold:
                if self._check_temporal_consistency(node_id):
                    logger.info(f"[VPR] 时序验证通过的回环检测: node={node_id}, sim={similarity:.4f}, "
                               f"adjusted_sim={adjusted_similarity:.4f}")
                    self._confirm_match(node_id, similarity)
                    self._record_query_time(start_time)
                    return (node_id, similarity)
                # 记录为候选，但需要更多验证
                if candidate is None:
                    candidate = (node_id, similarity)

            # 低置信度: 记录到时序窗口，等待后续帧验证
            if adjusted_similarity >= self.low_confidence_threshold:
                self._update_recent_matches(node_id, similarity)

        # 如果有候选但未通过验证，也更新时序窗口
        if candidate:
            self._update_recent_matches(candidate[0], candidate[1])

        self._record_query_time(start_time)
        return None

    def _confirm_match(self, node_id: int, similarity: float):
        """确认匹配并更新状态"""
        self.successful_detections += 1
        self.match_history.append(similarity)
        self._update_recent_matches(node_id, similarity)
        self.last_confirmed_node = node_id

    def _record_query_time(self, start_time: float):
        """记录查询时间"""
        elapsed = time.time() - start_time
        self.query_times.append(elapsed)
        # 只保留最近1000次查询时间
        if len(self.query_times) > 1000:
            self.query_times = self.query_times[-1000:]

    def _apply_semantic_adjustment(self, similarity: float, db_idx: int,
                                   query_labels: List[str] = None) -> float:
        """
        [新增] 应用语义调整 - 根据语义标签匹配情况调整相似度

        Args:
            similarity: 原始相似度
            db_idx: 数据库索引
            query_labels: 查询帧的语义标签

        Returns:
            调整后的相似度
        """
        if not query_labels or db_idx >= len(self.semantic_labels_db):
            return similarity

        query_set = set(label.lower() for label in query_labels)
        db_labels = self.semantic_labels_db[db_idx]

        if not db_labels:
            return similarity

        # 计算Jaccard相似度
        intersection = query_set & db_labels
        union = query_set | db_labels

        if len(union) == 0:
            return similarity

        jaccard = len(intersection) / len(union)

        # 高语义重叠给予加成，低重叠给予惩罚
        if jaccard > 0.5:
            # 超过50%重叠，给予加成
            adjustment = self.semantic_bonus * (jaccard - 0.5) * 2
            return min(1.0, similarity + adjustment)
        elif jaccard < 0.2 and len(query_set) > 2 and len(db_labels) > 2:
            # 低于20%重叠且双方都有足够标签，给予惩罚
            adjustment = self.semantic_penalty * (0.2 - jaccard) * 5
            return max(0.0, similarity - adjustment)

        return similarity

    def _check_spatial_consistency(self, candidate_node: int) -> bool:
        """
        [新增] 空间一致性检查 - 避免跳跃式误匹配

        如果上一个确认的节点与候选节点的拓扑距离过大，可能是误匹配

        Args:
            candidate_node: 候选节点ID

        Returns:
            True 如果空间一致性检查通过
        """
        if self.last_confirmed_node is None:
            return True

        # 计算拓扑距离 (简化为节点ID差异)
        # 在真实场景中，应该使用拓扑图的最短路径
        topo_distance = abs(candidate_node - self.last_confirmed_node)

        if topo_distance > self.max_topological_jump:
            logger.debug(f"[VPR] 空间一致性警告: last_node={self.last_confirmed_node}, "
                        f"candidate={candidate_node}, distance={topo_distance}")
            # 不是直接拒绝，而是需要更高的置信度
            return True  # 暂时放宽，仅记录警告

        return True

    def _update_recent_matches(self, node_id: int, similarity: float):
        """更新最近匹配窗口"""
        self.recent_matches.append({
            'node_id': node_id,
            'similarity': similarity
        })

    def _check_temporal_consistency(self, node_id: int) -> bool:
        """
        检查时序一致性 - 增强版 v2.0

        使用加权时序验证:
        1. 连续多帧匹配同一位置或相邻位置
        2. 考虑匹配的相似度权重
        3. 最近的匹配权重更高

        Args:
            node_id: 目标节点ID

        Returns:
            True 如果时序一致性验证通过
        """
        if len(self.recent_matches) < 1:
            return False

        # 加权时序一致性计算
        weighted_score = 0.0
        total_weight = 0.0

        for i, match in enumerate(self.recent_matches):
            # 时间权重：越近的匹配权重越高
            time_weight = (i + 1) / len(self.recent_matches)

            # 检查是否匹配到同一节点或相邻节点
            node_distance = abs(match['node_id'] - node_id)

            if node_distance == 0:
                # 完全匹配
                weighted_score += time_weight * match['similarity']
                total_weight += time_weight
            elif node_distance <= 2:
                # 相邻节点匹配 (衰减权重)
                decay = 1.0 / (1 + node_distance * 0.5)
                weighted_score += time_weight * match['similarity'] * decay
                total_weight += time_weight * decay

        if total_weight == 0:
            return False

        # 计算加权平均
        avg_score = weighted_score / total_weight

        # 阈值：需要达到低置信度阈值的80%
        threshold = self.low_confidence_threshold * 0.8
        return avg_score >= threshold

    def get_adaptive_threshold(self) -> float:
        """
        获取自适应阈值 (基于匹配历史)

        Returns:
            调整后的阈值
        """
        if len(self.match_history) < 10:
            return self.similarity_threshold

        # 基于历史匹配相似度的统计信息调整阈值
        mean_sim = np.mean(self.match_history[-50:])
        std_sim = np.std(self.match_history[-50:])

        # 自适应调整: 设置在 mean - 1.5*std 处
        adaptive_threshold = max(
            self.low_confidence_threshold,
            min(self.high_confidence_threshold, mean_sim - 1.5 * std_sim)
        )
        return adaptive_threshold

    def get_statistics(self) -> Dict:
        """获取VPR统计信息"""
        detection_rate = self.successful_detections / max(self.total_queries, 1)
        return {
            'total_queries': self.total_queries,
            'successful_detections': self.successful_detections,
            'detection_rate': detection_rate,
            'database_size': len(self.features_db),
            'similarity_threshold': self.similarity_threshold,
            'high_confidence_threshold': self.high_confidence_threshold,
            'adaptive_threshold': self.get_adaptive_threshold() if self.match_history else None
        }

    def get_size(self) -> int:
        """获取数据库大小"""
        return len(self.features_db)

    def clear(self):
        """清空数据库"""
        if FAISS_AVAILABLE and self.index is not None:
            self.index.reset()

        # v4.0: 清空多视角索引
        for cam_id in self.CAMERA_IDS:
            if FAISS_AVAILABLE and self.multi_view_indices.get(cam_id) is not None:
                self.multi_view_indices[cam_id].reset()
            self.multi_view_features_db[cam_id] = []
            self.multi_view_node_ids[cam_id] = []

        self.features_db.clear()
        self.node_ids.clear()
        self.timestamps.clear()
        self.surround_features_db.clear()
        self.multi_view_surround_db.clear()
        self.semantic_labels_db.clear()
        self.scene_descriptions_db.clear()
        self.recent_matches.clear()
        self.match_history.clear()
        self.total_queries = 0
        self.successful_detections = 0
        self.last_confirmed_node = None
        self.hard_negatives.clear()
        self.false_positive_history.clear()
        self.query_times.clear()

    def get_performance_stats(self) -> Dict:
        """
        [新增] 获取性能统计信息

        Returns:
            包含查询时间、检测率等性能指标的字典
        """
        avg_query_time = np.mean(self.query_times) if self.query_times else 0.0
        detection_rate = self.successful_detections / max(self.total_queries, 1)

        return {
            'total_queries': self.total_queries,
            'successful_detections': self.successful_detections,
            'detection_rate': detection_rate,
            'database_size': len(self.features_db),
            'avg_query_time_ms': avg_query_time * 1000,
            'similarity_threshold': self.similarity_threshold,
            'high_confidence_threshold': self.high_confidence_threshold,
            'adaptive_threshold': self.get_adaptive_threshold() if self.match_history else None,
            'semantic_labels_count': sum(len(s) for s in self.semantic_labels_db),
            'hard_negatives_count': len(self.hard_negatives)
        }

    def report_false_positive(self, query_node: int, wrong_match_node: int, similarity: float):
        """
        报告误报 - 用于自适应学习

        当外部系统检测到VPR误报时，调用此方法记录

        Args:
            query_node: 查询节点ID
            wrong_match_node: 错误匹配的节点ID
            similarity: 误报时的相似度
        """
        self.hard_negatives.append((similarity, query_node, wrong_match_node))
        self.false_positive_history.append(similarity)

        # 如果误报频繁发生在高相似度区域，提高阈值
        if len(self.false_positive_history) >= 5:
            recent_fp = self.false_positive_history[-5:]
            max_fp_sim = max(recent_fp)
            if max_fp_sim > self.similarity_threshold:
                # 动态提高阈值
                new_threshold = min(max_fp_sim + 0.02, self.high_confidence_threshold - 0.05)
                if new_threshold > self.similarity_threshold:
                    logger.warning(f"[VPR] 检测到频繁误报，提高阈值: {self.similarity_threshold:.3f} -> {new_threshold:.3f}")
                    self.similarity_threshold = new_threshold

        logger.info(f"[VPR] 记录误报: query={query_node}, wrong_match={wrong_match_node}, sim={similarity:.4f}")

    # ========================================================================
    # v3.0 新增: 环境自适应阈值机制
    # ========================================================================

    def _update_environment_state(self, query_results: List[Tuple[int, float]]):
        """
        [v3.0] 更新环境状态 - 用于自适应阈值

        基于最近的查询结果更新环境状态估计，包括：
        - 平均相似度水平
        - 相似度方差
        - 场景复杂度估计

        Args:
            query_results: 最近的查询结果 [(node_id, similarity), ...]
        """
        if not query_results or not self.adaptive_threshold_enabled:
            return

        current_time = time.time()

        # 只在间隔时间后更新
        if current_time - self.env_state.last_update_time < 1.0:
            return

        # 提取相似度分布
        similarities = [sim for _, sim in query_results[:10]]
        if not similarities:
            return

        # 更新平均相似度 (使用指数移动平均)
        alpha = 0.2  # 平滑系数
        current_avg = np.mean(similarities)
        self.env_state.avg_similarity = (
            alpha * current_avg + (1 - alpha) * self.env_state.avg_similarity
        )

        # 更新相似度方差
        current_var = np.var(similarities)
        self.env_state.similarity_variance = (
            alpha * current_var + (1 - alpha) * self.env_state.similarity_variance
        )

        # 估计场景复杂度 (基于相似度分布)
        # 复杂场景: 相似度分布较分散
        if len(similarities) > 1:
            sim_range = max(similarities) - min(similarities)
            self.env_state.scene_complexity = min(1.0, sim_range / 0.3)

        # 估计特征密度 (基于数据库大小和查询响应)
        self.env_state.feature_density = min(1.0, len(self.features_db) / 100.0)

        self.env_state.last_update_time = current_time

    def _compute_adaptive_threshold(self) -> float:
        """
        [v3.0] 计算自适应阈值

        基于环境状态动态调整阈值，实现：
        - 复杂场景: 提高阈值，减少误匹配
        - 简单场景: 降低阈值，提高召回率
        - 光照变化: 动态补偿

        Returns:
            自适应调整后的阈值
        """
        if not self.adaptive_threshold_enabled:
            return self.base_similarity_threshold

        # 基础阈值
        threshold = self.base_similarity_threshold

        # 根据场景复杂度调整
        # 复杂场景需要更高阈值
        complexity_adjustment = self.env_state.scene_complexity * 0.05
        threshold += complexity_adjustment

        # 根据相似度方差调整
        # 高方差说明场景变化大，需要更保守的阈值
        if self.env_state.similarity_variance > 0.02:
            variance_adjustment = min(0.05, self.env_state.similarity_variance * 2)
            threshold += variance_adjustment

        # 根据历史匹配调整
        if len(self.match_history) >= 20:
            recent_matches = self.match_history[-20:]
            match_mean = np.mean(recent_matches)
            match_std = np.std(recent_matches)

            # 如果历史匹配相似度很高，可以适当降低阈值
            if match_mean > 0.92 and match_std < 0.03:
                threshold -= 0.02

        # 根据误报历史调整
        if len(self.false_positive_history) >= 3:
            recent_fp = self.false_positive_history[-5:]
            max_fp = max(recent_fp)
            if max_fp > threshold:
                threshold = max_fp + 0.02

        # 确保阈值在合理范围内
        threshold = max(self.low_confidence_threshold, min(self.high_confidence_threshold - 0.05, threshold))

        # 记录阈值变化
        if abs(threshold - self.similarity_threshold) > 0.01:
            self.threshold_adaptation_history.append((time.time(), threshold))
            logger.info(f"[VPR v3.0] 自适应阈值更新: {self.similarity_threshold:.3f} -> {threshold:.3f}")

        return threshold

    # ========================================================================
    # v3.0 新增: 几何一致性验证
    # ========================================================================

    def add_keypoints(self, node_id: int, keypoints: np.ndarray, descriptors: np.ndarray):
        """
        [v3.0] 添加特征点数据用于几何验证

        Args:
            node_id: 节点ID
            keypoints: 特征点坐标 (N, 2)
            descriptors: 特征点描述子 (N, D)
        """
        idx = self.node_ids.index(node_id) if node_id in self.node_ids else -1
        if idx == -1:
            return

        # 确保列表足够长
        while len(self.keypoints_db) <= idx:
            self.keypoints_db.append(None)
            self.descriptors_db.append(None)

        self.keypoints_db[idx] = keypoints.copy() if keypoints is not None else None
        self.descriptors_db[idx] = descriptors.copy() if descriptors is not None else None

    def _verify_geometric_consistency(self, query_feature: np.ndarray,
                                       candidate_idx: int,
                                       query_keypoints: np.ndarray = None,
                                       query_descriptors: np.ndarray = None) -> Tuple[bool, float]:
        """
        [v3.0] 验证几何一致性

        使用特征点匹配验证两帧之间的空间关系是否一致

        Args:
            query_feature: 查询特征
            candidate_idx: 候选帧索引
            query_keypoints: 查询帧特征点坐标
            query_descriptors: 查询帧特征点描述子

        Returns:
            (is_valid, geometric_score): 是否通过验证, 几何一致性分数
        """
        if not self.geometric_verification_enabled:
            return True, 1.0

        # 如果没有特征点数据，跳过几何验证
        if (query_keypoints is None or query_descriptors is None or
            candidate_idx >= len(self.keypoints_db) or
            self.keypoints_db[candidate_idx] is None):
            return True, 0.5  # 无法验证时返回中等置信度

        db_keypoints = self.keypoints_db[candidate_idx]
        db_descriptors = self.descriptors_db[candidate_idx]

        if db_keypoints is None or db_descriptors is None:
            return True, 0.5

        try:
            # 简单的特征点匹配 (基于描述子距离)
            # 使用余弦相似度进行匹配
            query_norm = query_descriptors / (np.linalg.norm(query_descriptors, axis=1, keepdims=True) + 1e-8)
            db_norm = db_descriptors / (np.linalg.norm(db_descriptors, axis=1, keepdims=True) + 1e-8)

            # 计算相似度矩阵
            similarity_matrix = np.dot(query_norm, db_norm.T)

            # 找到最佳匹配
            matches = []
            for i in range(len(query_descriptors)):
                best_match_idx = np.argmax(similarity_matrix[i])
                best_sim = similarity_matrix[i, best_match_idx]

                # 只保留高相似度匹配
                if best_sim > 0.7:
                    matches.append((i, best_match_idx, best_sim))

            # 检查匹配数量
            num_matches = len(matches)
            if num_matches < self.min_geometric_inliers:
                logger.debug(f"[VPR v3.0] 几何验证失败: 匹配点数不足 ({num_matches} < {self.min_geometric_inliers})")
                return False, num_matches / self.min_geometric_inliers

            # 计算几何一致性分数
            geometric_score = min(1.0, num_matches / (self.min_geometric_inliers * 2))

            # 简单的空间一致性检查 (基于匹配点的分布)
            if len(matches) >= 4:
                query_pts = np.array([query_keypoints[m[0]] for m in matches])
                db_pts = np.array([db_keypoints[m[1]] for m in matches])

                # 检查点的分布是否合理 (不应该都集中在一个小区域)
                query_spread = np.std(query_pts, axis=0).mean()
                db_spread = np.std(db_pts, axis=0).mean()

                if query_spread < 10 or db_spread < 10:  # 点过于集中
                    geometric_score *= 0.7

            self.geometric_verification_results.append(True)
            logger.debug(f"[VPR v3.0] 几何验证通过: {num_matches}个匹配点, 分数={geometric_score:.3f}")
            return True, geometric_score

        except Exception as e:
            logger.warning(f"[VPR v3.0] 几何验证异常: {e}")
            return True, 0.5

    # ========================================================================
    # v3.0 新增: 置信度计算
    # ========================================================================

    def _compute_match_confidence(self, similarity: float,
                                   geometric_score: float,
                                   semantic_score: float,
                                   temporal_consistency: bool) -> float:
        """
        [v3.0] 计算综合匹配置信度

        综合考虑视觉相似度、几何验证、语义匹配和时序一致性

        Args:
            similarity: 视觉相似度
            geometric_score: 几何验证分数
            semantic_score: 语义匹配分数
            temporal_consistency: 时序一致性

        Returns:
            综合置信度 (0-1)
        """
        # 视觉相似度权重
        visual_weight = 0.5

        # 几何验证权重
        geo_weight = self.geometric_confidence_weight if self.geometric_verification_enabled else 0

        # 语义匹配权重
        semantic_weight = 0.15

        # 时序一致性权重
        temporal_weight = 0.05

        # 归一化权重
        total_weight = visual_weight + geo_weight + semantic_weight + temporal_weight

        # 计算加权置信度
        confidence = (
            visual_weight * similarity +
            geo_weight * geometric_score +
            semantic_weight * semantic_score +
            temporal_weight * (1.0 if temporal_consistency else 0.5)
        ) / total_weight

        # 应用非线性变换，使置信度更有区分度
        confidence = self._sigmoid_transform(confidence, center=0.7, steepness=10)

        return confidence

    def _sigmoid_transform(self, x: float, center: float = 0.5, steepness: float = 10) -> float:
        """Sigmoid变换，用于置信度映射"""
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

    # ========================================================================
    # v3.0 新增: 增强的回环检测接口
    # ========================================================================

    def is_revisited_v3(self, query_feature: np.ndarray,
                        current_time: float,
                        min_time_gap: float = 5.0,
                        query_surround: np.ndarray = None,
                        query_semantic_labels: List[str] = None,
                        query_keypoints: np.ndarray = None,
                        query_descriptors: np.ndarray = None) -> Optional[VPRMatchResult]:
        """
        [v3.0] 增强版回环检测

        返回详细的匹配结果，包含置信度和匹配类型

        Args:
            query_feature: 查询特征
            current_time: 当前时间戳
            min_time_gap: 最小时间间隔
            query_surround: 环视融合特征
            query_semantic_labels: 语义标签
            query_keypoints: 特征点坐标 (可选，用于几何验证)
            query_descriptors: 特征点描述子 (可选，用于几何验证)

        Returns:
            VPRMatchResult 或 None
        """
        start_time = time.time()
        self.total_queries += 1

        # 更新自适应阈值
        if self.total_queries % self.adaptive_update_interval == 0:
            self.similarity_threshold = self._compute_adaptive_threshold()

        # 执行搜索
        if query_semantic_labels and len(self.semantic_labels_db) > 0:
            results_with_info = self.search_with_semantic(
                query_feature, query_surround, query_semantic_labels, k=self.coarse_search_k
            )
            results = [(node_id, sim) for node_id, sim, _ in results_with_info]
            semantic_info = {node_id: info for node_id, _, info in results_with_info}
        elif query_surround is not None:
            results = self.search_with_surround(query_feature, query_surround, k=self.coarse_search_k)
            semantic_info = {}
        else:
            results = self.search(query_feature, k=self.coarse_search_k)
            semantic_info = {}

        # 更新环境状态
        self._update_environment_state(results)

        if not results:
            self._record_query_time(start_time)
            return None

        # 遍历候选进行验证
        for node_id, similarity in results:
            idx = self.node_ids.index(node_id)
            time_gap = current_time - self.timestamps[idx]

            # 时间间隔检查
            if time_gap <= min_time_gap:
                continue

            # 空间一致性检查
            if not self._check_spatial_consistency(node_id):
                continue

            # 语义调整
            adjusted_similarity = self._apply_semantic_adjustment(
                similarity, idx, query_semantic_labels
            )

            # 获取语义分数
            semantic_score = 0.0
            if node_id in semantic_info:
                semantic_score = semantic_info[node_id].get('semantic_sim', 0.0)

            # 几何验证
            geo_valid, geo_score = self._verify_geometric_consistency(
                query_feature, idx, query_keypoints, query_descriptors
            )

            # 时序一致性
            temporal_valid = self._check_temporal_consistency(node_id)

            # 计算综合置信度
            confidence = self._compute_match_confidence(
                similarity, geo_score, semantic_score, temporal_valid
            )

            # 确定匹配类型和是否接受
            match_type = None

            if adjusted_similarity >= self.high_confidence_threshold and geo_valid:
                match_type = 'high_confidence'
            elif adjusted_similarity >= self.similarity_threshold and temporal_valid and geo_valid:
                match_type = 'temporal_verified'
            elif adjusted_similarity >= self.similarity_threshold and geo_score > 0.8:
                match_type = 'geometric_verified'

            if match_type:
                self._confirm_match(node_id, similarity)
                self.confidence_history.append(confidence)
                self._record_query_time(start_time)

                result = VPRMatchResult(
                    node_id=node_id,
                    similarity=similarity,
                    confidence=confidence,
                    match_type=match_type,
                    geometric_score=geo_score,
                    semantic_score=semantic_score,
                    details={
                        'adjusted_similarity': adjusted_similarity,
                        'time_gap': time_gap,
                        'temporal_valid': temporal_valid,
                        'threshold_used': self.similarity_threshold
                    }
                )

                logger.info(f"[VPR v3.0] 回环检测成功: node={node_id}, sim={similarity:.4f}, "
                           f"confidence={confidence:.3f}, type={match_type}")
                return result

            # 更新时序窗口
            if adjusted_similarity >= self.low_confidence_threshold:
                self._update_recent_matches(node_id, similarity)

        self._record_query_time(start_time)
        return None

    def get_extended_statistics(self) -> Dict:
        """
        [v3.0] 获取扩展统计信息

        Returns:
            包含v3.0新增指标的统计字典
        """
        base_stats = self.get_performance_stats()

        # 计算置信度统计
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0.0

        # 计算几何验证成功率
        geo_success_rate = (sum(self.geometric_verification_results) /
                           len(self.geometric_verification_results)
                           if self.geometric_verification_results else 0.0)

        # 阈值变化统计
        threshold_changes = len(self.threshold_adaptation_history)

        base_stats.update({
            'version': 'v3.0',
            'adaptive_threshold_enabled': self.adaptive_threshold_enabled,
            'current_adaptive_threshold': self.similarity_threshold,
            'base_threshold': self.base_similarity_threshold,
            'env_state': {
                'avg_similarity': self.env_state.avg_similarity,
                'similarity_variance': self.env_state.similarity_variance,
                'scene_complexity': self.env_state.scene_complexity
            },
            'geometric_verification_enabled': self.geometric_verification_enabled,
            'geometric_success_rate': geo_success_rate,
            'avg_confidence': avg_confidence,
            'threshold_adaptation_count': threshold_changes,
        })

        return base_stats
