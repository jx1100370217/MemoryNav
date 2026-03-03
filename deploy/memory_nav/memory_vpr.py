#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav v2.1 - 视觉位置识别 (VPR) - 循环移位匹配版

核心改进：
- 4相机循环移位匹配，支持不同朝向下的定位
- 计算机器人当前朝向与记忆库朝向的偏移角度
- 每种shift对应特定的朝向偏移

相机布局（鱼眼等角投影, HFOV=190°）：
- camera_1: 左前37.5° → -37.5°
- camera_2: 右前37.5° → +37.5°
- camera_3: 右后37.5° → +142.5°
- camera_4: 左后37.5° → -142.5° (217.5°)
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .memory_models import VPRResult

logger = logging.getLogger(__name__)


class MemoryVPR:
    """
    记忆VPR模块 v2.1 - 循环移位匹配
    
    通过尝试4种相机排列（循环移位），找到最佳匹配的节点和朝向偏移。
    """
    
    CAMERA_IDS = ['camera_1', 'camera_2', 'camera_3', 'camera_4']
    
    # 相机中心方位角（机器人坐标系，正前方0°，顺时针为正）
    CAMERA_ANGLES = {
        'camera_1': -37.5,   # 左前
        'camera_2': 37.5,    # 右前
        'camera_3': 142.5,   # 右后
        'camera_4': -142.5,  # 左后 (= 217.5°)
    }
    
    # 循环移位对应的朝向偏移（度，顺时针为正）
    # shift=k: query_cam[ORDER[(i+k)%4]] 对应 memory_cam[ORDER[i]]
    # 偏移 = CAMERA_ANGLES[ORDER[0]] - CAMERA_ANGLES[ORDER[k]]
    SHIFT_HEADING_OFFSETS = [
        0.0,     # shift=0: 同向
        -75.0,   # shift=1: 逆时针旋转75° (cam1→cam2方向)
        -180.0,  # shift=2: 掉头180°
        105.0,   # shift=3: 顺时针旋转105°
    ]
    
    def __init__(self, feature_dim: int = 768, similarity_threshold: float = 0.90):
        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold
        
        # 多视角独立索引
        self.camera_indices: Dict[str, any] = {}
        self.camera_node_ids: Dict[str, List[str]] = {}
        self.camera_features: Dict[str, List[np.ndarray]] = {}
        
        # 融合特征索引
        self.fused_index = None
        self.fused_node_ids: List[str] = []
        self.fused_features: List[np.ndarray] = []
        
        # 节点名称映射
        self.node_names: Dict[str, str] = {}
        self.node_names_eng: Dict[str, str] = {}
        
        self._init_indices()
        
        logger.info(f"[MemoryVPR] 初始化完成: dim={feature_dim}, "
                   f"threshold={similarity_threshold}, FAISS={FAISS_AVAILABLE}")
    
    def _init_indices(self):
        for cam_id in self.CAMERA_IDS:
            if FAISS_AVAILABLE:
                self.camera_indices[cam_id] = faiss.IndexFlatIP(self.feature_dim)
            else:
                self.camera_indices[cam_id] = None
            self.camera_node_ids[cam_id] = []
            self.camera_features[cam_id] = []
        
        if FAISS_AVAILABLE:
            self.fused_index = faiss.IndexFlatIP(self.feature_dim)
    
    def add_node_features(self, node_id: str, node_name: str,
                          camera_features: Dict[str, np.ndarray],
                          fused_feature: Optional[np.ndarray] = None,
                          node_name_eng: str = ""):
        """添加节点特征到VPR索引"""
        self.node_names[node_id] = node_name
        self.node_names_eng[node_id] = node_name_eng
        
        for cam_id in self.CAMERA_IDS:
            if cam_id in camera_features and camera_features[cam_id] is not None:
                feat = camera_features[cam_id].astype(np.float32)
                feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
                
                if FAISS_AVAILABLE and self.camera_indices[cam_id] is not None:
                    self.camera_indices[cam_id].add(feat_norm.reshape(1, -1))
                
                self.camera_features[cam_id].append(feat_norm)
                self.camera_node_ids[cam_id].append(node_id)
        
        if fused_feature is not None:
            feat = fused_feature.astype(np.float32)
            feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
            if FAISS_AVAILABLE and self.fused_index is not None:
                self.fused_index.add(feat_norm.reshape(1, -1))
            self.fused_features.append(feat_norm)
            self.fused_node_ids.append(node_id)
    
    def search_single_camera(self, query_feature: np.ndarray, 
                             camera_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """单相机搜索"""
        if camera_id not in self.camera_indices:
            return []
        
        query_norm = query_feature.astype(np.float32)
        query_norm = query_norm / (np.linalg.norm(query_norm) + 1e-8)
        
        if FAISS_AVAILABLE and self.camera_indices[camera_id] is not None:
            index = self.camera_indices[camera_id]
            if index.ntotal == 0:
                return []
            k = min(k, index.ntotal)
            distances, indices = index.search(query_norm.reshape(1, -1), k)
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.camera_node_ids[camera_id]):
                    node_id = self.camera_node_ids[camera_id][idx]
                    similarity = float(distances[0][i])
                    results.append((node_id, similarity))
            return results
        else:
            if not self.camera_features[camera_id]:
                return []
            features_matrix = np.array(self.camera_features[camera_id])
            similarities = np.dot(features_matrix, query_norm)
            top_indices = np.argsort(similarities)[::-1][:k]
            return [(self.camera_node_ids[camera_id][idx], float(similarities[idx])) 
                    for idx in top_indices]
    
    def search_multi_view(self, query_features: Dict[str, np.ndarray],
                          k: int = None) -> Optional[VPRResult]:
        """
        多视角循环移位匹配
        
        核心算法：
        1. 尝试4种循环移位 (shift=0,1,2,3)
        2. 每种shift下，将 query 的相机重新映射到 memory 的相机
        3. 计算每个节点在每种shift下的4相机平均相似度
        4. 选择全局最高平均相似度的 (节点, shift) 组合
        5. 根据最佳shift计算朝向偏移角度
        
        Args:
            query_features: 查询特征 {'camera_1': feat, ...}
            k: 每个相机搜索的top-k数量（None=全部节点）
            
        Returns:
            VPRResult 包含匹配节点、置信度和朝向偏移
        """
        if not query_features:
            return None
        
        query_cam_ids = sorted(query_features.keys())
        n_query_cams = len(query_cam_ids)
        
        # 如果不满4个相机，退回简单匹配（不做循环移位）
        if n_query_cams < 4:
            return self._search_simple(query_features, k or 10)
        
        # 搜索所有节点
        if k is None:
            k = max(len(self.camera_node_ids.get(cam, [])) 
                    for cam in self.CAMERA_IDS)
            k = max(k, 1)
        
        # 全局最佳结果
        best_node_id = None
        best_avg_sim = -1.0
        best_shift = 0
        best_cam_scores = {}
        
        for shift in range(4):
            # 构建移位映射: query_cam → memory_cam
            # shift=k: query 的 CAMERA_IDS[(i+shift)%4] 对应 memory 的 CAMERA_IDS[i]
            node_scores: Dict[str, Dict[str, float]] = {}
            
            for i, mem_cam in enumerate(self.CAMERA_IDS):
                query_cam = self.CAMERA_IDS[(i + shift) % 4]
                
                if query_cam not in query_features:
                    continue
                
                query_feat = query_features[query_cam]
                results = self.search_single_camera(query_feat, mem_cam, k=k)
                
                for node_id, similarity in results:
                    if node_id not in node_scores:
                        node_scores[node_id] = {}
                    node_scores[node_id][mem_cam] = similarity
            
            # 对每个节点，如果4个相机都有结果，计算平均相似度
            for node_id, scores in node_scores.items():
                if len(scores) >= 4:
                    avg_sim = sum(scores.values()) / len(scores)
                    if avg_sim > best_avg_sim:
                        best_avg_sim = avg_sim
                        best_node_id = node_id
                        best_shift = shift
                        best_cam_scores = dict(scores)
        
        if best_node_id is None:
            return None
        
        # 计算朝向偏移
        heading_offset = self.SHIFT_HEADING_OFFSETS[best_shift]
        
        # 置信度 = 平均相似度（不再用 match_ratio 膨胀）
        confidence = best_avg_sim
        
        if best_avg_sim >= self.similarity_threshold:
            result = VPRResult(
                matched_node_id=best_node_id,
                matched_node_name=self.node_names.get(best_node_id, ""),
                matched_node_name_eng=self.node_names_eng.get(best_node_id, ""),
                similarity=best_avg_sim,
                confidence=confidence,
                camera_scores=best_cam_scores,
                heading_offset=heading_offset,
                best_shift=best_shift
            )
            logger.info(
                f"[MemoryVPR] 循环移位匹配: node={best_node_id} "
                f"({self.node_names.get(best_node_id, '')}), "
                f"shift={best_shift}, heading_offset={heading_offset:.1f}°, "
                f"avg_sim={best_avg_sim:.4f}")
            return result
        
        return None
    
    def _search_simple(self, query_features: Dict[str, np.ndarray],
                       k: int = 10) -> Optional[VPRResult]:
        """
        简单匹配（不足4相机时使用，兼容旧逻辑）
        """
        node_scores: Dict[str, Dict] = {}
        
        for cam_id, query_feat in query_features.items():
            if cam_id not in self.CAMERA_IDS:
                continue
            results = self.search_single_camera(query_feat, cam_id, k=k)
            for node_id, similarity in results:
                if node_id not in node_scores:
                    node_scores[node_id] = {
                        'camera_scores': {}, 'total_score': 0.0, 'match_count': 0
                    }
                node_scores[node_id]['camera_scores'][cam_id] = similarity
                node_scores[node_id]['total_score'] += similarity
                if similarity >= self.similarity_threshold:
                    node_scores[node_id]['match_count'] += 1
        
        if not node_scores:
            return None
        
        sorted_nodes = sorted(
            node_scores.items(),
            key=lambda x: (x[1]['match_count'], x[1]['total_score']),
            reverse=True
        )
        
        best_node_id, best_data = sorted_nodes[0]
        n_cams = len(best_data['camera_scores'])
        avg_sim = best_data['total_score'] / n_cams if n_cams > 0 else 0
        
        # 简单匹配时 heading_offset=0, best_shift=0
        if avg_sim >= self.similarity_threshold or best_data['match_count'] >= 2:
            return VPRResult(
                matched_node_id=best_node_id,
                matched_node_name=self.node_names.get(best_node_id, ""),
                matched_node_name_eng=self.node_names_eng.get(best_node_id, ""),
                similarity=avg_sim,
                confidence=avg_sim,
                camera_scores=best_data['camera_scores'],
                heading_offset=0.0,
                best_shift=0
            )
        return None
    
    def locate(self, query_features: Dict[str, np.ndarray]) -> Optional[VPRResult]:
        """定位当前位置（主接口）"""
        result = self.search_multi_view(query_features)
        
        if result:
            logger.info(f"[MemoryVPR] 定位成功: node={result.matched_node_id} "
                       f"({result.matched_node_name}), sim={result.similarity:.4f}, "
                       f"conf={result.confidence:.4f}, "
                       f"heading_offset={result.heading_offset:.1f}°")
        else:
            logger.debug("[MemoryVPR] 定位失败: 无匹配节点")
        
        return result
    
    def search_fused(self, query_feature: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """融合特征搜索"""
        if not self.fused_features:
            return []
        query_norm = query_feature.astype(np.float32)
        query_norm = query_norm / (np.linalg.norm(query_norm) + 1e-8)
        if FAISS_AVAILABLE and self.fused_index is not None and self.fused_index.ntotal > 0:
            k = min(k, self.fused_index.ntotal)
            distances, indices = self.fused_index.search(query_norm.reshape(1, -1), k)
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.fused_node_ids):
                    results.append((self.fused_node_ids[idx], float(distances[0][i])))
            return results
        else:
            features_matrix = np.array(self.fused_features)
            similarities = np.dot(features_matrix, query_norm)
            top_indices = np.argsort(similarities)[::-1][:k]
            return [(self.fused_node_ids[idx], float(similarities[idx])) 
                    for idx in top_indices]
    
    def get_stats(self) -> Dict:
        """获取VPR统计信息"""
        return {
            'feature_dim': self.feature_dim,
            'similarity_threshold': self.similarity_threshold,
            'total_nodes': len(self.fused_node_ids),
            'faiss_available': FAISS_AVAILABLE,
            'camera_stats': {cam_id: len(self.camera_node_ids.get(cam_id, []))
                           for cam_id in self.CAMERA_IDS},
            'camera_angles': self.CAMERA_ANGLES,
            'shift_heading_offsets': self.SHIFT_HEADING_OFFSETS
        }
    

    def get_node_similarity(self, query_features: Dict[str, np.ndarray],
                            node_id: str) -> float:
        """
        获取查询特征与指定节点的最佳相似度（跨所有shift）
        
        即使相似度低于阈值也会返回结果，用于趋势检测。
        
        Args:
            query_features: 查询特征 {'camera_1': feat, ...}
            node_id: 目标节点ID
            
        Returns:
            最佳平均相似度 (float)，如果无法计算返回 0.0
        """
        if not query_features or len(query_features) < 4:
            return 0.0
        
        best_avg_sim = 0.0
        
        for shift in range(4):
            cam_sims = []
            for i, mem_cam in enumerate(self.CAMERA_IDS):
                query_cam = self.CAMERA_IDS[(i + shift) % 4]
                if query_cam not in query_features:
                    continue
                if node_id not in self.camera_node_ids.get(mem_cam, []):
                    continue
                
                idx = self.camera_node_ids[mem_cam].index(node_id)
                query_feat = query_features[query_cam].astype(np.float32)
                query_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)
                stored_feat = self.camera_features[mem_cam][idx]
                sim = float(np.dot(query_norm, stored_feat))
                cam_sims.append(sim)
            
            if len(cam_sims) == 4:
                avg_sim = sum(cam_sims) / 4
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
        
        return best_avg_sim

    def clear(self):
        """清空VPR索引"""
        self._init_indices()
        self.fused_node_ids.clear()
        self.fused_features.clear()
        self.node_names.clear()
        self.node_names_eng.clear()
        logger.info("[MemoryVPR] 索引已清空")
