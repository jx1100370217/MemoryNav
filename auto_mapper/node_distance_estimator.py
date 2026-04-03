#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
节点距离估计器 v4

与导航 MemoryVPR 完全一致的 VPR 逻辑:
- 每个 camera 独立提取特征 + L2 归一化
- 4 camera 逐一比较 + 4 种 shift 循环移位
- 相似度 = 最佳 shift 下的 4 camera 平均 cosine similarity
"""

import os
import sys
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')

try:
    from memory_nav import create_vpr_extractor
    from memory_nav.vpr_config_loader import load_vpr_config
except ImportError as e:
    logging.error(f"Failed to import memory_nav modules: {e}")
    raise


CAMERA_IDS = ['camera_1', 'camera_2', 'camera_3', 'camera_4']


class NodeDistanceEstimator:
    """节点距离估计器 v4 — 与 MemoryVPR 完全一致"""

    def __init__(self,
                 vpr_config_path: str,
                 similarity_threshold: float = 0.50,
                 min_frame_interval: int = 5):
        self.similarity_threshold = similarity_threshold
        self.min_frame_interval = min_frame_interval

        config = load_vpr_config(vpr_config_path)
        vpr_method = config.get('vpr_method', 'selavpr')
        device = config.get('device', 'cuda:0')

        extractor_info = create_vpr_extractor(
            vpr_method=vpr_method, device=device, config=config
        )

        self.vpr_extractor = extractor_info[0]
        self.feature_dim = extractor_info[1] if len(extractor_info) > 1 else None

        # 存储每个 node 的 4 camera 独立特征 (L2 归一化后)
        # {node_id: {camera_id: np.ndarray}}
        self.node_camera_features: Dict[str, Dict[str, np.ndarray]] = {}
        self.node_frames: Dict[str, int] = {}
        self._last_created_frame: int = -999
        self._last_created_id: Optional[str] = None

        logging.info(f"NodeDistanceEstimator v4 (MemoryVPR-consistent): "
                     f"threshold={similarity_threshold}, "
                     f"vpr={vpr_method}, device={device}, dim={self.feature_dim}")

    # ---- 兼容旧接口 (node_features) ----
    @property
    def node_features(self):
        """兼容旧代码访问 node_features 的地方，返回 node_camera_features"""
        return self.node_camera_features

    @node_features.setter
    def node_features(self, value):
        self.node_camera_features = value

    def extract_camera_features(self, camera_images: Dict[str, str]) -> Dict[str, np.ndarray]:
        """提取每个 camera 的独立特征并 L2 归一化 (与 MemoryBuilder 一致)

        Args:
            camera_images: {camera_id: image_path}

        Returns:
            {camera_id: normalized_feature_vector}
        """
        camera_features = {}
        for cam_id in CAMERA_IDS:
            img_path = camera_images.get(cam_id)
            if not img_path or not os.path.exists(img_path):
                continue
            image = cv2.imread(img_path)
            if image is None:
                continue
            feat = self.vpr_extractor.extract(image)
            if isinstance(feat, list):
                feat = np.array(feat)
            feat = feat.astype(np.float32).flatten()
            # L2 归一化
            feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
            camera_features[cam_id] = feat_norm
        return camera_features

    def compute_node_similarity(self,
                                features_a: Dict[str, np.ndarray],
                                features_b: Dict[str, np.ndarray]) -> float:
        """计算两组 camera 特征的相似度 (与 MemoryVPR.get_node_similarity 完全一致)

        算法:
        1. 尝试 4 种循环移位 (shift=0,1,2,3)
        2. 每种 shift 下，将 A 的 camera 重新映射到 B 的 camera
        3. 计算 4 个 camera 的 cosine similarity 平均值
        4. 返回全局最高平均相似度

        Args:
            features_a: {camera_id: L2-normalized feature}
            features_b: {camera_id: L2-normalized feature}

        Returns:
            最佳平均相似度 (float)，范围约 [-1, 1]
        """
        if len(features_a) < 4 or len(features_b) < 4:
            # 不足 4 camera，直接匹配已有的
            sims = []
            for cam_id in CAMERA_IDS:
                if cam_id in features_a and cam_id in features_b:
                    sim = float(np.dot(features_a[cam_id], features_b[cam_id]))
                    sims.append(sim)
            return sum(sims) / len(sims) if sims else 0.0

        best_avg_sim = -1.0

        for shift in range(4):
            cam_sims = []
            for i, mem_cam in enumerate(CAMERA_IDS):
                query_cam = CAMERA_IDS[(i + shift) % 4]
                if query_cam not in features_a or mem_cam not in features_b:
                    continue
                sim = float(np.dot(features_a[query_cam], features_b[mem_cam]))
                cam_sims.append(sim)

            if len(cam_sims) == 4:
                avg_sim = sum(cam_sims) / 4
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim

        return best_avg_sim if best_avg_sim > -1.0 else 0.0

    # ---- 兼容旧接口 ----
    def extract_frame_feature(self, camera_images: Dict[str, str]) -> Dict[str, np.ndarray]:
        """兼容旧接口，返回 camera features dict"""
        return self.extract_camera_features(camera_images)

    def compute_similarity(self, f1, f2) -> float:
        """兼容旧接口

        如果传入的是 dict (新格式)，用 compute_node_similarity。
        """
        if isinstance(f1, dict) and isinstance(f2, dict):
            return self.compute_node_similarity(f1, f2)
        # 不应该走到这里，但保底
        logging.warning("compute_similarity called with non-dict features, fallback")
        return 0.0

    def should_create_node(self,
                           frame_index: int,
                           camera_images: Dict[str, str]) -> Tuple[bool, Dict]:
        info = {
            'frame_index': frame_index,
            'similarities': {},
            'max_similarity': 0.0,
            'last_node_similarity': 0.0,
            'reason': 'unknown'
        }

        try:
            current_features = self.extract_camera_features(camera_images)

            if len(current_features) < 4:
                info['reason'] = 'insufficient_cameras'
                return False, info

            # 第一个节点直接创建
            if not self.node_camera_features:
                info['reason'] = 'first_node'
                return True, info

            # 只检查与最近创建节点的帧间隔
            if frame_index - self._last_created_frame < self.min_frame_interval:
                info['reason'] = 'frame_interval_too_small'
                return False, info

            # 计算与所有已有节点的相似度
            max_sim = -1.0
            for pid, node_feats in self.node_camera_features.items():
                sim = self.compute_node_similarity(current_features, node_feats)
                info['similarities'][pid] = sim
                max_sim = max(max_sim, sim)

            # 与最近创建节点的相似度
            if self._last_created_id and self._last_created_id in self.node_camera_features:
                last_sim = self.compute_node_similarity(
                    current_features, self.node_camera_features[self._last_created_id]
                )
            else:
                last_sim = max_sim

            info['max_similarity'] = max_sim
            info['last_node_similarity'] = last_sim

            if last_sim < self.similarity_threshold:
                info['reason'] = 'low_similarity'
                return True, info
            else:
                info['reason'] = 'high_similarity'
                return False, info

        except Exception as e:
            logging.error(f"Failed to evaluate: {e}")
            info['reason'] = f'error: {e}'
            return False, info

    def register_node(self, position_id: str, frame_index: int,
                      camera_images: Dict[str, str]):
        features = self.extract_camera_features(camera_images)
        self.node_camera_features[position_id] = features
        self.node_frames[position_id] = frame_index
        self._last_created_frame = frame_index
        self._last_created_id = position_id
        logging.info(f"Registered node {position_id} at frame {frame_index} "
                     f"({len(features)} cameras)")

    def get_statistics(self) -> Dict:
        return {
            'total_nodes': len(self.node_camera_features),
            'node_ids': list(self.node_camera_features.keys()),
            'similarity_threshold': self.similarity_threshold,
            'min_frame_interval': self.min_frame_interval,
        }

    def cleanup(self):
        if hasattr(self.vpr_extractor, 'cleanup'):
            self.vpr_extractor.cleanup()
