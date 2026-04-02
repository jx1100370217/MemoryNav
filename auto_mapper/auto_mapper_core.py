#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动建图核心模块 v4

改进:
1. 先生成所有 node 的 self_position (用 namer 的 Qwen 进程)
2. 停掉 namer 的 Qwen 进程释放显存
3. 启动 PointGrounder，在每个 node 的 4 个 camera 上打点
4. 用 DINOv3 CLS 特征匹配 crop 到邻居 node
5. 更新 next_positions
"""

import os
import sys
import glob
import logging
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')

from .node_distance_estimator import NodeDistanceEstimator
from .auto_node_generator import AutoNodeGenerator
from .auto_sub_image_extractor import AutoSubImageExtractor
from .auto_landmark_namer import AutoLandmarkNamer


class AutoMapperCore:
    """自动建图核心控制器 v4"""

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 vpr_config_path: str,
                 start_id: int = 1,
                 similarity_threshold: float = 0.69,
                 min_frame_interval: int = 5,
                 use_qwen_naming: bool = False,
                 qwen_gpu: str = "1"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.start_id = start_id
        self.current_id = start_id
        self._qwen_gpu = qwen_gpu

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        self.distance_estimator = NodeDistanceEstimator(
            vpr_config_path, similarity_threshold, min_frame_interval
        )

        self.node_generator = AutoNodeGenerator(
            str(self.output_dir), use_qwen_naming
        )

        # sub_extractor 延迟初始化 (避免和 namer 同时占 GPU 显存)
        self.sub_extractor = None

        self.namer = AutoLandmarkNamer(use_qwen=use_qwen_naming, gpu=qwen_gpu)

        self.created_nodes: List[Dict] = []
        self._all_frames: List[Dict] = []
        self._used_names: set = set()

        logging.info(f"AutoMapperCore v4 initialized")
        logging.info(f"Input: {self.input_dir}, Output: {self.output_dir}")
        logging.info(f"Start ID: {start_id}, Threshold: {similarity_threshold}")
        logging.info(f"Qwen naming: {use_qwen_naming}")

    def load_input_data(self) -> List[Dict]:
        image_pattern = str(self.input_dir / "*_camera_*.jpg")
        image_files = glob.glob(image_pattern)
        if not image_files:
            raise FileNotFoundError(f"No camera images in {self.input_dir}")

        timestamps = sorted(set(
            os.path.basename(f).split('_')[0] for f in image_files
        ))
        logging.info(f"Found {len(timestamps)} frames")

        frames = []
        for ts in timestamps:
            images = {}
            for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
                p = self.input_dir / f"{ts}_{cam_id}.jpg"
                if p.exists():
                    images[cam_id] = str(p)
            if images:
                frames.append({'timestamp': ts, 'images': images})

        self._all_frames = frames
        logging.info(f"Loaded {len(frames)} valid frames")
        return frames

    def _make_unique_name(self, name: str) -> str:
        if name not in self._used_names:
            self._used_names.add(name)
            return name
        for i in range(2, 100):
            new_name = f"{name}{i}"
            if new_name not in self._used_names:
                self._used_names.add(new_name)
                return new_name
        return name

    def process_frame(self, frame_index: int, frame_data: Dict) -> Optional[str]:
        timestamp = frame_data['timestamp']
        images = frame_data['images']

        should_create, info = self.distance_estimator.should_create_node(
            frame_index, images
        )

        logging.info(f"Frame {frame_index} (ts: {timestamp}): create={should_create}")
        logging.info(f"  Reason: {info['reason']}, Max sim: {info.get('max_similarity', 0):.3f}")

        if should_create:
            position_id = str(self.current_id)

            pos_names = self.namer.generate_self_position_names(position_id, images)

            orig_name = pos_names['position_name']
            unique_name = self._make_unique_name(orig_name)
            if unique_name != orig_name:
                pos_names['position_name'] = unique_name

            node_dir = self.node_generator.create_node(
                position_id=position_id,
                timestamp=timestamp,
                source_images=images,
                position_names=pos_names,
            )

            self.distance_estimator.register_node(position_id, frame_index, images)

            node_info = {
                'position_id': position_id,
                'timestamp': timestamp,
                'frame_index': frame_index,
                'images': images,
                'node_dir': str(node_dir),
                'position_name': pos_names.get('position_name', ''),
                'position_name_eng': pos_names.get('position_name_eng', ''),
            }
            self.created_nodes.append(node_info)

            logging.info(f"Created node: {position_id} ({pos_names.get('position_name', '')})")
            self.current_id += 1
            return position_id

        return None

    def generate_connections(self):
        """
        v4: 对每个 node 用 generate_next_positions 一次性生成所有连接
        """
        if len(self.created_nodes) < 2:
            logging.info("Less than 2 nodes, no connections")
            return

        logging.info("=== Generating connections (v4: PointGrounding + DINOv3 match) ===")

        for i, node in enumerate(self.created_nodes):
            # 确定邻居: 相邻节点
            neighbors = []
            if i > 0:
                neighbors.append(self.created_nodes[i - 1])
            if i < len(self.created_nodes) - 1:
                neighbors.append(self.created_nodes[i + 1])

            # 首尾连接检测
            if len(self.created_nodes) >= 4:
                if i == 0:
                    last = self.created_nodes[-1]
                    try:
                        feat_first = self.distance_estimator.extract_frame_feature(node['images'])
                        feat_last = self.distance_estimator.extract_frame_feature(last['images'])
                        sim = self.distance_estimator.compute_similarity(feat_first, feat_last)
                        logging.info(f"First-Last similarity: {sim:.3f}")
                        if sim > 0.55 and last not in neighbors:
                            neighbors.append(last)
                    except Exception as e:
                        logging.warning(f"First-last check failed: {e}")
                elif i == len(self.created_nodes) - 1:
                    first = self.created_nodes[0]
                    try:
                        feat_last = self.distance_estimator.extract_frame_feature(node['images'])
                        feat_first = self.distance_estimator.extract_frame_feature(first['images'])
                        sim = self.distance_estimator.compute_similarity(feat_last, feat_first)
                        if sim > 0.55 and first not in neighbors:
                            neighbors.append(first)
                    except Exception as e:
                        logging.warning(f"Last-first check failed: {e}")

            if not neighbors:
                continue

            # 调用 generate_next_positions
            next_positions = self.sub_extractor.generate_next_positions(
                node_info=node,
                neighbor_nodes=neighbors,
                all_frames=self._all_frames,
                qwen_namer=None,
            )

            if next_positions:
                self.node_generator.update_node_connections(
                    node['position_id'], next_positions
                )
                for np_ in next_positions:
                    logging.info(f"Connected: {node['position_id']} -> {np_['position_id']} "
                               f"via {np_['camera_name']}")
            else:
                logging.warning(f"Node {node['position_id']}: no connections generated")

        logging.info("Connection generation completed")

    def run_auto_mapping(self) -> Dict:
        logging.info("Starting automatic mapping v4...")

        frames = self.load_input_data()

        logging.info("Phase 1: Creating nodes (with Qwen naming)...")
        for idx, frame in enumerate(frames):
            self.process_frame(idx, frame)

        # 停掉 namer 的 Qwen 进程，释放显存给 PointGrounder
        logging.info("Phase 1.5: Stopping namer Qwen to free GPU memory...")
        self.namer.stop()

        # 延迟初始化 sub_extractor (PointGrounder + DINOv3)
        logging.info("Phase 2: Starting PointGrounder + DINOv3...")
        self.sub_extractor = AutoSubImageExtractor(qwen_gpu=self._qwen_gpu)

        self.generate_connections()

        stats = self.get_statistics()
        logging.info(f"Completed! {stats['total_nodes']} nodes, {stats['total_connections']} connections")
        return stats

    def get_statistics(self) -> Dict:
        node_stats = self.node_generator.get_statistics()
        distance_stats = self.distance_estimator.get_statistics()
        return {
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'total_nodes': len(self.created_nodes),
            'node_ids': [n['position_id'] for n in self.created_nodes],
            'total_connections': node_stats['total_connections'],
            'similarity_threshold': distance_stats['similarity_threshold'],
            'min_frame_interval': distance_stats['min_frame_interval'],
            'start_id': self.start_id,
            'final_id': self.current_id - 1 if self.created_nodes else self.start_id,
        }

    def cleanup(self):
        try:
            self.distance_estimator.cleanup()
            if self.sub_extractor:
                self.sub_extractor.cleanup()
            self.namer.stop()
            logging.info("Resources cleaned up")
        except Exception as e:
            logging.warning(f"Cleanup: {e}")
