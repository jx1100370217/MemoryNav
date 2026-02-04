#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 增强版视觉位置识别模块 (VPR) v4.0

优化改进:
1. 各视角独立编码: camera_1~4分别进行相似度检测，而非融合后检测
2. 多视角投票机制: 综合多个视角的匹配结果
3. 最佳视角选择: 自动选择最可靠的视角进行匹配
4. 支持拓扑图最短路径规划

作者: Memory Navigation Team
日期: 2026-01-26
"""

import logging
from typing import List, Tuple, Optional, Dict, Set, Any
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import time
import math

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MultiViewMatchResult:
    """多视角匹配结果数据类"""
    node_id: int
    combined_similarity: float
    confidence: float
    view_similarities: Dict[str, float]  # 各视角的相似度
    best_view: str  # 最佳匹配视角
    matched_views: int  # 匹配成功的视角数量
    match_type: str  # 'high_confidence', 'multi_view_verified', 'single_view'
    semantic_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologicalNode:
    """拓扑图节点 - 存储多视角特征"""
    node_id: int
    timestamp: float
    view_features: Dict[str, np.ndarray]  # camera_id -> feature (独立存储)
    fused_feature: Optional[np.ndarray] = None  # 融合特征 (用于快速粗筛)
    semantic_labels: List[str] = field(default_factory=list)
    scene_description: str = ""
    visit_count: int = 1
    is_keyframe: bool = False
    neighbors: List[int] = field(default_factory=list)  # 邻居节点ID列表


class EnhancedVisualPlaceRecognition:
    """
    增强版视觉位置识别模块 v4.0 - 多视角独立编码

    核心改进:
    1. 各视角独立存储和检索，不做特征融合
    2. 多视角投票机制确定最终匹配
    3. 支持任意视角的相似度查询
    4. 视角权重可配置
    """

    # 相机ID和对应的角度
    CAMERA_ANGLES = {
        'camera_1': 37.5,    # 前右
        'camera_2': -37.5,   # 前左
        'camera_3': -142.5,  # 后左
        'camera_4': 142.5,   # 后右
    }

    def __init__(self, feature_dim: int = 512, similarity_threshold: float = 0.78):
        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold
        self.high_confidence_threshold = 0.90
        self.low_confidence_threshold = 0.72

        # 多视角匹配参数
        self.min_matching_views = 2  # 至少需要2个视角匹配
        self.view_match_threshold = 0.75  # 单视角匹配阈值
        self.multi_view_bonus = 0.05  # 多视角匹配加分

        # 视角权重 (可根据实际情况调整)
        self.view_weights = {
            'camera_1': 0.25,
            'camera_2': 0.25,
            'camera_3': 0.25,
            'camera_4': 0.25,
        }

        # 为每个视角创建独立的FAISS索引
        self.view_indices: Dict[str, Any] = {}
        self.view_features_db: Dict[str, List[np.ndarray]] = {}

        if FAISS_AVAILABLE:
            for cam_id in self.CAMERA_ANGLES.keys():
                self.view_indices[cam_id] = faiss.IndexFlatIP(feature_dim)
                self.view_features_db[cam_id] = []
            logger.info(f"[VPR v4.0] 已为4个视角创建独立FAISS索引 (维度={feature_dim})")
        else:
            for cam_id in self.CAMERA_ANGLES.keys():
                self.view_indices[cam_id] = None
                self.view_features_db[cam_id] = []
            logger.warning("[VPR v4.0] FAISS不可用，使用numpy后备方案")

        # 节点数据库
        self.nodes: Dict[int, TopologicalNode] = {}
        self.node_ids: List[int] = []
        self.timestamps: List[float] = []
        self.semantic_labels_db: List[Set[str]] = []

        # 融合特征索引 (用于快速粗筛)
        if FAISS_AVAILABLE:
            self.fused_index = faiss.IndexFlatIP(feature_dim)
        else:
            self.fused_index = None
        self.fused_features_db: List[np.ndarray] = []

        # 时序验证
        self.temporal_window_size = 3
        self.recent_matches: deque = deque(maxlen=self.temporal_window_size)

        # 统计信息
        self.total_queries = 0
        self.successful_detections = 0
        self.match_history: List[float] = []
        self.view_match_counts: Dict[str, int] = {cam_id: 0 for cam_id in self.CAMERA_ANGLES.keys()}

        logger.info(f"[VPR v4.0] 初始化完成: 多视角独立编码模式")

    def add_node(self, node_id: int, view_features: Dict[str, np.ndarray],
                 timestamp: float, semantic_labels: List[str] = None,
                 scene_description: str = "", is_keyframe: bool = False):
        """
        添加节点到数据库 - 各视角独立存储

        Args:
            node_id: 节点ID
            view_features: {camera_id: feature} 各视角特征字典
            timestamp: 时间戳
            semantic_labels: 语义标签列表
            scene_description: 场景描述
            is_keyframe: 是否为关键帧
        """
        # 存储各视角特征
        stored_features = {}
        for cam_id, feature in view_features.items():
            if cam_id in self.CAMERA_ANGLES and feature is not None:
                feature_normalized = feature.astype('float32') / (np.linalg.norm(feature) + 1e-8)
                stored_features[cam_id] = feature_normalized.copy()

                # 添加到对应视角的FAISS索引
                if FAISS_AVAILABLE and self.view_indices[cam_id] is not None:
                    self.view_indices[cam_id].add(feature_normalized.reshape(1, -1))
                self.view_features_db[cam_id].append(feature_normalized)

        # 计算融合特征 (用于快速粗筛)
        fused_feature = self._compute_fused_feature(stored_features)
        if fused_feature is not None:
            if FAISS_AVAILABLE and self.fused_index is not None:
                self.fused_index.add(fused_feature.reshape(1, -1))
            self.fused_features_db.append(fused_feature)

        # 创建节点
        node = TopologicalNode(
            node_id=node_id,
            timestamp=timestamp,
            view_features=stored_features,
            fused_feature=fused_feature,
            semantic_labels=semantic_labels or [],
            scene_description=scene_description,
            is_keyframe=is_keyframe
        )

        self.nodes[node_id] = node
        self.node_ids.append(node_id)
        self.timestamps.append(timestamp)

        # 存储语义标签
        if semantic_labels:
            labels_set = set(label.lower() for label in semantic_labels)
            self.semantic_labels_db.append(labels_set)
        else:
            self.semantic_labels_db.append(set())

        logger.debug(f"[VPR v4.0] 添加节点 {node_id}, 视角数: {len(stored_features)}")

    def _compute_fused_feature(self, view_features: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """计算融合特征 (等权重平均)"""
        if not view_features:
            return None

        fused = None
        count = 0

        for cam_id, feature in view_features.items():
            if feature is not None:
                if fused is None:
                    fused = feature.copy() * self.view_weights.get(cam_id, 0.25)
                else:
                    fused += feature * self.view_weights.get(cam_id, 0.25)
                count += 1

        if fused is None or count == 0:
            return None

        # 归一化
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        return fused

    def search_single_view(self, cam_id: str, query_feature: np.ndarray,
                          k: int = 10) -> List[Tuple[int, float]]:
        """
        在单个视角中搜索最相似的节点

        Args:
            cam_id: 相机ID
            query_feature: 查询特征
            k: 返回top-k结果

        Returns:
            [(node_id, similarity), ...]
        """
        if cam_id not in self.view_features_db or len(self.view_features_db[cam_id]) == 0:
            return []

        query_normalized = query_feature.astype('float32') / (np.linalg.norm(query_feature) + 1e-8)

        if FAISS_AVAILABLE and self.view_indices[cam_id] is not None:
            actual_k = min(k, self.view_indices[cam_id].ntotal)
            if actual_k == 0:
                return []
            distances, indices = self.view_indices[cam_id].search(
                query_normalized.reshape(1, -1), actual_k
            )
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.node_ids):
                    similarity = float(distances[0][i])
                    node_id = self.node_ids[idx]
                    results.append((node_id, similarity))
            return results
        else:
            # Numpy后备方案
            features_matrix = np.array(self.view_features_db[cam_id])
            similarities = np.dot(features_matrix, query_normalized)
            top_k_indices = np.argsort(similarities)[::-1][:k]
            return [(self.node_ids[idx], float(similarities[idx])) for idx in top_k_indices]

    def search_multi_view(self, query_features: Dict[str, np.ndarray],
                         k: int = 10) -> List[MultiViewMatchResult]:
        """
        多视角联合搜索 - 核心改进功能

        Args:
            query_features: {camera_id: feature} 查询特征字典
            k: 返回top-k结果

        Returns:
            List[MultiViewMatchResult] 多视角匹配结果列表
        """
        if not query_features or len(self.node_ids) == 0:
            return []

        # 步骤1: 在各视角中独立搜索
        view_results: Dict[str, Dict[int, float]] = {}
        for cam_id, query_feature in query_features.items():
            if cam_id in self.CAMERA_ANGLES and query_feature is not None:
                results = self.search_single_view(cam_id, query_feature, k=k*2)
                view_results[cam_id] = {node_id: sim for node_id, sim in results}

        if not view_results:
            return []

        # 步骤2: 收集所有候选节点
        all_candidates = set()
        for results in view_results.values():
            all_candidates.update(results.keys())

        # 步骤3: 为每个候选节点计算多视角综合得分
        multi_view_results = []

        for node_id in all_candidates:
            view_similarities = {}
            matched_views = 0
            total_weight = 0.0
            weighted_sum = 0.0
            best_sim = 0.0
            best_view = None

            for cam_id, results in view_results.items():
                if node_id in results:
                    sim = results[node_id]
                    view_similarities[cam_id] = sim

                    weight = self.view_weights.get(cam_id, 0.25)
                    weighted_sum += sim * weight
                    total_weight += weight

                    if sim >= self.view_match_threshold:
                        matched_views += 1

                    if sim > best_sim:
                        best_sim = sim
                        best_view = cam_id

            if total_weight == 0:
                continue

            # 计算综合相似度
            combined_sim = weighted_sum / total_weight

            # 多视角匹配加分
            if matched_views >= self.min_matching_views:
                combined_sim += self.multi_view_bonus * (matched_views - 1)

            # 确定匹配类型
            if combined_sim >= self.high_confidence_threshold:
                match_type = 'high_confidence'
            elif matched_views >= self.min_matching_views:
                match_type = 'multi_view_verified'
            else:
                match_type = 'single_view'

            # 计算置信度
            confidence = self._compute_confidence(combined_sim, matched_views, len(view_results))

            result = MultiViewMatchResult(
                node_id=node_id,
                combined_similarity=combined_sim,
                confidence=confidence,
                view_similarities=view_similarities,
                best_view=best_view or '',
                matched_views=matched_views,
                match_type=match_type
            )
            multi_view_results.append(result)

        # 按综合相似度排序
        multi_view_results.sort(key=lambda x: x.combined_similarity, reverse=True)
        return multi_view_results[:k]

    def _compute_confidence(self, similarity: float, matched_views: int,
                           total_views: int) -> float:
        """计算匹配置信度"""
        # 基础置信度
        base_confidence = similarity

        # 多视角一致性加分
        view_ratio = matched_views / max(total_views, 1)
        view_bonus = 0.1 * view_ratio

        confidence = min(1.0, base_confidence + view_bonus)
        return confidence

    def is_revisited(self, query_features: Dict[str, np.ndarray],
                    current_time: float,
                    min_time_gap: float = 0.5,
                    query_semantic_labels: List[str] = None) -> Optional[MultiViewMatchResult]:
        """
        判断是否为已访问位置 (回环检测) - 多视角版本

        Args:
            query_features: {camera_id: feature} 各视角查询特征
            current_time: 当前时间戳
            min_time_gap: 最小时间间隔
            query_semantic_labels: 语义标签

        Returns:
            MultiViewMatchResult 或 None
        """
        self.total_queries += 1

        # 多视角搜索
        results = self.search_multi_view(query_features, k=15)

        if not results:
            return None

        # 调试日志
        top_result = results[0]
        logger.debug(f"[VPR v4.0] 查询#{self.total_queries}: top1_node={top_result.node_id}, "
                    f"sim={top_result.combined_similarity:.4f}, views={top_result.matched_views}")

        for result in results:
            node_id = result.node_id
            if node_id not in self.nodes:
                continue

            node = self.nodes[node_id]
            time_gap = current_time - node.timestamp

            # 时间间隔检查
            if time_gap <= min_time_gap:
                continue

            # 语义调整
            adjusted_sim = self._apply_semantic_adjustment(
                result.combined_similarity, node_id, query_semantic_labels
            )

            # 高置信度: 直接确认
            if adjusted_sim >= self.high_confidence_threshold:
                self._confirm_match(result)
                logger.info(f"[VPR v4.0] 高置信度回环: node={node_id}, sim={adjusted_sim:.4f}, "
                           f"views={result.matched_views}")
                return result

            # 多视角验证通过
            if adjusted_sim >= self.similarity_threshold and result.matched_views >= self.min_matching_views:
                if self._check_temporal_consistency(node_id):
                    self._confirm_match(result)
                    logger.info(f"[VPR v4.0] 多视角验证回环: node={node_id}, sim={adjusted_sim:.4f}, "
                               f"views={result.matched_views}")
                    return result

            # 更新时序窗口
            if adjusted_sim >= self.low_confidence_threshold:
                self._update_recent_matches(node_id, adjusted_sim)

        return None

    def _apply_semantic_adjustment(self, similarity: float, node_id: int,
                                  query_labels: List[str] = None) -> float:
        """应用语义调整"""
        if not query_labels or node_id not in self.nodes:
            return similarity

        node = self.nodes[node_id]
        if not node.semantic_labels:
            return similarity

        query_set = set(label.lower() for label in query_labels)
        db_labels = set(label.lower() for label in node.semantic_labels)

        # Jaccard相似度
        intersection = query_set & db_labels
        union = query_set | db_labels

        if len(union) == 0:
            return similarity

        jaccard = len(intersection) / len(union)

        if jaccard > 0.5:
            return min(1.0, similarity + 0.05 * (jaccard - 0.5) * 2)
        elif jaccard < 0.2 and len(query_set) > 2 and len(db_labels) > 2:
            return max(0.0, similarity - 0.03 * (0.2 - jaccard) * 5)

        return similarity

    def _confirm_match(self, result: MultiViewMatchResult):
        """确认匹配并更新统计"""
        self.successful_detections += 1
        self.match_history.append(result.combined_similarity)
        self._update_recent_matches(result.node_id, result.combined_similarity)

        # 更新视角统计
        if result.best_view:
            self.view_match_counts[result.best_view] = self.view_match_counts.get(result.best_view, 0) + 1

    def _update_recent_matches(self, node_id: int, similarity: float):
        """更新时序窗口"""
        self.recent_matches.append({
            'node_id': node_id,
            'similarity': similarity
        })

    def _check_temporal_consistency(self, node_id: int) -> bool:
        """检查时序一致性"""
        if len(self.recent_matches) < 1:
            return False

        score = 0.0
        for i, match in enumerate(self.recent_matches):
            time_weight = (i + 1) / len(self.recent_matches)
            node_distance = abs(match['node_id'] - node_id)

            if node_distance == 0:
                score += time_weight * match['similarity']
            elif node_distance <= 2:
                decay = 1.0 / (1 + node_distance * 0.5)
                score += time_weight * match['similarity'] * decay

        return score >= self.low_confidence_threshold * 0.8

    def get_node_by_id(self, node_id: int) -> Optional[TopologicalNode]:
        """获取节点"""
        return self.nodes.get(node_id)

    def get_view_statistics(self) -> Dict:
        """获取各视角匹配统计"""
        return {
            'total_queries': self.total_queries,
            'successful_detections': self.successful_detections,
            'detection_rate': self.successful_detections / max(self.total_queries, 1),
            'database_size': len(self.nodes),
            'view_match_counts': self.view_match_counts.copy(),
            'best_performing_view': max(self.view_match_counts, key=self.view_match_counts.get)
                if self.view_match_counts else None
        }

    def clear(self):
        """清空所有数据"""
        for cam_id in self.CAMERA_ANGLES.keys():
            if FAISS_AVAILABLE and self.view_indices[cam_id] is not None:
                self.view_indices[cam_id].reset()
            self.view_features_db[cam_id] = []

        if FAISS_AVAILABLE and self.fused_index is not None:
            self.fused_index.reset()
        self.fused_features_db = []

        self.nodes.clear()
        self.node_ids = []
        self.timestamps = []
        self.semantic_labels_db = []
        self.recent_matches.clear()
        self.match_history = []
        self.total_queries = 0
        self.successful_detections = 0
        self.view_match_counts = {cam_id: 0 for cam_id in self.CAMERA_ANGLES.keys()}


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # 创建VPR实例
    vpr = EnhancedVisualPlaceRecognition(feature_dim=512)

    # 模拟添加节点
    for i in range(5):
        view_features = {
            'camera_1': np.random.randn(512).astype('float32'),
            'camera_2': np.random.randn(512).astype('float32'),
            'camera_3': np.random.randn(512).astype('float32'),
            'camera_4': np.random.randn(512).astype('float32'),
        }
        vpr.add_node(
            node_id=i,
            view_features=view_features,
            timestamp=time.time() - 10 + i,
            semantic_labels=['test', f'node_{i}']
        )

    # 测试多视角搜索
    query_features = {
        'camera_1': np.random.randn(512).astype('float32'),
        'camera_2': np.random.randn(512).astype('float32'),
        'camera_3': np.random.randn(512).astype('float32'),
        'camera_4': np.random.randn(512).astype('float32'),
    }

    results = vpr.search_multi_view(query_features, k=3)
    print("\n多视角搜索结果:")
    for r in results:
        print(f"  Node {r.node_id}: sim={r.combined_similarity:.4f}, "
              f"views={r.matched_views}, best={r.best_view}")

    # 打印统计
    stats = vpr.get_view_statistics()
    print(f"\n统计信息: {stats}")
