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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
                 min_frame_interval: int = 5,
                 extractor=None,
                 feature_dim: Optional[int] = None):
        """
        Args:
            extractor: 可选复用的 VPR extractor (如 MemoryNavigator.extractor).
                       传入后不再新建模型实例, 节省显存.
            feature_dim: 配合 extractor 传入时的特征维度 (可选).
        """
        self.similarity_threshold = similarity_threshold
        self.min_frame_interval = min_frame_interval

        if extractor is not None:
            self.vpr_extractor = extractor
            self.feature_dim = feature_dim or getattr(extractor, "feature_dim", None)
            logging.info(f"NodeDistanceEstimator v4: 复用已有 extractor "
                         f"(type={type(extractor).__name__}, dim={self.feature_dim})")
        else:
            config = load_vpr_config(vpr_config_path)
            vpr_method = config.get('vpr_method', 'selavpr')
            device = config.get('device', 'cuda:0')

            extractor_info = create_vpr_extractor(
                vpr_method=vpr_method, device=device, config=config
            )

            self.vpr_extractor = extractor_info[0]
            self.feature_dim = extractor_info[1] if len(extractor_info) > 1 else None
            logging.info(f"NodeDistanceEstimator v4 (MemoryVPR-consistent): "
                         f"threshold={similarity_threshold}, "
                         f"vpr={vpr_method}, device={device}, dim={self.feature_dim}")

        # 存储每个 node 的 4 camera 独立特征 (L2 归一化后)
        # {node_id: {camera_id: np.ndarray}}
        self.node_camera_features: Dict[str, Dict[str, np.ndarray]] = {}
        self.node_frames: Dict[str, int] = {}

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

    def cleanup(self):
        if hasattr(self.vpr_extractor, 'cleanup'):
            self.vpr_extractor.cleanup()
