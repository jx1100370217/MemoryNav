#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
节点距离估计器 v3

改进: 用与最近创建节点的相似度来判断是否建新节点
(而非与所有历史节点的最大相似度)
"""

import os
import sys
import cv2
import torch
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


class NodeDistanceEstimator:
    """节点距离估计器 v3"""

    def __init__(self,
                 vpr_config_path: str,
                 similarity_threshold: float = 0.65,
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

        self.node_features: Dict[str, torch.Tensor] = {}
        self.node_frames: Dict[str, int] = {}
        self._last_created_frame: int = -999
        self._last_created_id: Optional[str] = None

        logging.info(f"NodeDistanceEstimator v3: threshold={similarity_threshold}, "
                     f"vpr={vpr_method}, device={device}, dim={self.feature_dim}")

    def load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        images = []
        for path in image_paths:
            if not os.path.exists(path):
                continue
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
        return images

    def extract_frame_feature(self, camera_images: Dict[str, str]) -> torch.Tensor:
        """提取 VPR 特征"""
        image_paths = [camera_images[c] for c in
                       ['camera_1', 'camera_2', 'camera_3', 'camera_4']
                       if c in camera_images]
        if not image_paths:
            raise ValueError("No valid camera images")

        images = self.load_images(image_paths)
        if not images:
            raise ValueError("No images could be loaded")

        if hasattr(self.vpr_extractor, 'extract_batch'):
            feature = self.vpr_extractor.extract_batch(images)
        elif hasattr(self.vpr_extractor, 'extract'):
            feature = self.vpr_extractor.extract(images[0])
        else:
            raise AttributeError("VPR extractor has no extract method")

        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature)
        elif isinstance(feature, list):
            feature = torch.tensor(feature)
        if feature.dim() > 1:
            feature = feature[0]
        return feature.float()

    def compute_similarity(self, f1: torch.Tensor, f2: torch.Tensor) -> float:
        """计算余弦相似度 [0, 1]"""
        f1 = f1.flatten()
        f2 = f2.flatten()
        sim = torch.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

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
            current_feature = self.extract_frame_feature(camera_images)

            # 第一个节点直接创建
            if not self.node_features:
                info['reason'] = 'first_node'
                return True, info

            # 只检查与最近创建节点的帧间隔
            if frame_index - self._last_created_frame < self.min_frame_interval:
                info['reason'] = 'frame_interval_too_small'
                return False, info

            # 计算与所有已有节点的相似度 (用于信息记录)
            max_sim = 0.0
            for pid, nf in self.node_features.items():
                sim = self.compute_similarity(current_feature, nf)
                info['similarities'][pid] = sim
                max_sim = max(max_sim, sim)

            # 关键改进: 用与最近创建节点的相似度来判断
            # 这样可以更好地捕捉路径上的距离变化
            if self._last_created_id and self._last_created_id in self.node_features:
                last_sim = self.compute_similarity(
                    current_feature, self.node_features[self._last_created_id]
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
        feature = self.extract_frame_feature(camera_images)
        self.node_features[position_id] = feature
        self.node_frames[position_id] = frame_index
        self._last_created_frame = frame_index
        self._last_created_id = position_id
        logging.info(f"Registered node {position_id} at frame {frame_index}")

    def get_statistics(self) -> Dict:
        return {
            'total_nodes': len(self.node_features),
            'node_ids': list(self.node_features.keys()),
            'similarity_threshold': self.similarity_threshold,
            'min_frame_interval': self.min_frame_interval,
        }

    def cleanup(self):
        if hasattr(self.vpr_extractor, 'cleanup'):
            self.vpr_extractor.cleanup()
