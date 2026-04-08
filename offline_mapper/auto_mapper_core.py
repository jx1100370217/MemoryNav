#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动建图核心模块 v5.2

改进:
1. Phase 1: VPR 粗粒度 node 创建 (用 Qwen 命名)
2. Phase 1.5 (新增): 语义增补 — 扫描中间帧，检测门牌/标识/功能区域，插入新 node
3. Phase 2: 停 Qwen，启动 PointGrounder + DINOv3 生成连接
4. 修复闭环 bug: 首尾连接改为可选参数，默认关闭
5. v5.1: 修复重编号时目录重命名冲突（先全部改临时名，再统一改最终名）
6. v5.2: Phase 1.6 节点去重与合并（VPR 去重 + 同帧合并 + 别名机制）
"""

import os
import sys
import glob
import logging
import cv2
import shutil
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')

from .node_distance_estimator import NodeDistanceEstimator
from .auto_node_generator import AutoNodeGenerator
from .auto_sub_image_extractor import AutoSubImageExtractor
from .auto_landmark_namer import AutoLandmarkNamer
from .semantic_node_detector import SemanticNodeDetector
from .node_dedup_merger import NodeDedupMerger


class AutoMapperCore:
    """自动建图核心控制器 v5.2"""

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 vpr_config_path: str,
                 start_id: int = 1,
                 similarity_threshold: float = 0.50,
                 min_frame_interval: int = 5,
                 use_qwen_naming: bool = False,
                 qwen_gpu: str = "1",
                 enable_loop_closure: bool = False,
                 loop_closure_threshold: float = 0.80,
                 semantic_detection: bool = True,
                 vpr_dedup_threshold: float = 0.60):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.start_id = start_id
        self.current_id = start_id
        self._qwen_gpu = qwen_gpu
        self._enable_loop_closure = enable_loop_closure
        self._loop_closure_threshold = loop_closure_threshold
        self._semantic_detection = semantic_detection
        self._vpr_dedup_threshold = vpr_dedup_threshold

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

        logging.info(f"AutoMapperCore v5.2 initialized")
        logging.info(f"Input: {self.input_dir}, Output: {self.output_dir}")
        logging.info(f"Start ID: {start_id}, Threshold: {similarity_threshold}")
        logging.info(f"Qwen naming: {use_qwen_naming}")
        logging.info(f"Loop closure: {enable_loop_closure} (threshold: {loop_closure_threshold})")
        logging.info(f"Semantic detection: {semantic_detection}")

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


    # ==================================================================
    # Phase 1.5: 语义增补
    # ==================================================================
    def _run_semantic_detection(self):
        """扫描相邻 node 之间的中间帧，检测并插入语义 node"""
        if len(self.created_nodes) < 2:
            logging.info("Less than 2 nodes, skip semantic detection")
            return

        logging.info("=== Phase 1.5: 语义增补 — 扫描中间帧检测门牌/标识/功能区域 ===")

        # 获取 Qwen server 引用
        qwen_server = self.namer._qwen_server
        if not qwen_server or not qwen_server.is_ready:
            logging.warning("Qwen server not ready, skip semantic detection")
            return

        detector = SemanticNodeDetector()
        new_nodes = []

        for i in range(len(self.created_nodes) - 1):
            node_a = self.created_nodes[i]
            node_b = self.created_nodes[i + 1]

            logging.info(f"扫描 node {node_a['position_id']}({node_a['position_name']}) "
                         f"→ node {node_b['position_id']}({node_b['position_name']}) 之间...")

            candidates = detector.scan_gap(
                all_frames=self._all_frames,
                start_frame_idx=node_a['frame_index'],
                end_frame_idx=node_b['frame_index'],
                qwen_server=qwen_server,
                existing_names=self._used_names,
            )

            for cand in candidates:
                # 去重: 确保名称唯一
                orig_name = cand['position_names']['position_name']
                unique_name = self._make_unique_name(orig_name)
                if unique_name != orig_name:
                    cand['position_names']['position_name'] = unique_name

                new_nodes.append(cand)

        if not new_nodes:
            logging.info("语义增补: 未发现新的语义 node")
            return

        logging.info(f"语义增补: 发现 {len(new_nodes)} 个新 node，开始插入并重编号...")

        # 插入新 node 并重新排序
        self._insert_and_renumber(new_nodes)

    def _insert_and_renumber(self, new_nodes: List[Dict]):
        """将新 node 插入 created_nodes 列表，按 frame_index 排序，重新编号

        v5.1 修复: 使用两阶段重命名避免目录名冲突
        阶段 1: 所有目录 → _tmp_rename_<idx>
        阶段 2: _tmp_rename_<idx> → 最终编号
        """
        # 1. 为每个新 node 创建目录和注册 VPR 特征
        for seq_i, cand in enumerate(new_nodes):
            temp_id = f"semantic_{cand['frame_idx']}_{seq_i}"

            node_dir = self.node_generator.create_node(
                position_id=temp_id,
                timestamp=cand['timestamp'],
                source_images=cand['images'],
                position_names=cand['position_names'],
            )

            self.distance_estimator.register_node(temp_id, cand['frame_idx'], cand['images'])

            node_info = {
                'position_id': temp_id,
                'timestamp': cand['timestamp'],
                'frame_index': cand['frame_idx'],
                'images': cand['images'],
                'node_dir': str(node_dir),
                'position_name': cand['position_names']['position_name'],
                'position_name_eng': cand['position_names']['position_name_eng'],
            }
            self.created_nodes.append(node_info)

            logging.info(f"  插入语义 node: {cand['position_names']['position_name']} "
                         f"(帧 {cand['frame_idx']})")

        # 2. 按 frame_index 重新排序
        self.created_nodes.sort(key=lambda n: n['frame_index'])

        # 3. 两阶段重命名，避免目录名冲突
        # 阶段 1: 所有目录 → 临时名
        tmp_mapping = []  # [(node, tmp_dir_name, final_id)]
        for idx, node in enumerate(self.created_nodes):
            old_id = node['position_id']
            new_id = str(idx + self.start_id)
            tmp_name = f"_tmp_rename_{idx}"

            old_dir = self.output_dir / old_id
            tmp_dir = self.output_dir / tmp_name

            if old_dir.exists():
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                    logging.warning(f"  阶段1: 清理残留临时目录 {tmp_name}")
                old_dir.rename(tmp_dir)
                logging.debug(f"  重命名阶段1: {old_id} → {tmp_name}")

            # 更新 distance_estimator 特征映射
            if old_id in self.distance_estimator.node_features:
                self.distance_estimator.node_features[tmp_name] = \
                    self.distance_estimator.node_features.pop(old_id)
            if old_id in self.distance_estimator.node_frames:
                self.distance_estimator.node_frames[tmp_name] = \
                    self.distance_estimator.node_frames.pop(old_id)

            tmp_mapping.append((node, tmp_name, new_id))

        # 阶段 2: 临时名 → 最终编号
        for node, tmp_name, new_id in tmp_mapping:
            tmp_dir = self.output_dir / tmp_name
            new_dir = self.output_dir / new_id

            if tmp_dir.exists():
                if new_dir.exists():
                    shutil.rmtree(new_dir)
                    logging.warning(f"  阶段2: 清理残留目录 {new_id}")
                tmp_dir.rename(new_dir)
                logging.debug(f"  重命名阶段2: {tmp_name} → {new_id}")

            node['node_dir'] = str(new_dir)
            node['position_id'] = new_id

            # 更新 node_position_info.json 中的 position_id
            info_file = new_dir / "node_position_info.json"
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                info['self_position']['position_id'] = new_id
                with open(info_file, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)

            # 更新 distance_estimator 特征映射（临时名 → 最终编号）
            if tmp_name in self.distance_estimator.node_features:
                self.distance_estimator.node_features[new_id] = \
                    self.distance_estimator.node_features.pop(tmp_name)
            if tmp_name in self.distance_estimator.node_frames:
                self.distance_estimator.node_frames[new_id] = \
                    self.distance_estimator.node_frames.pop(tmp_name)

        self.current_id = len(self.created_nodes) + self.start_id

        logging.info(f"重编号完成: {len(self.created_nodes)} 个 node, "
                     f"ID 范围 {self.start_id} - {self.current_id - 1}")
        for node in self.created_nodes:
            logging.info(f"  node {node['position_id']}: {node['position_name']} "
                         f"(帧 {node['frame_index']})")

    def _renumber_after_dedup(self):
        """去重后重新编号，确保 node ID 连续"""
        import json as _json

        if not self.created_nodes:
            return

        # 按 frame_index 排序
        self.created_nodes.sort(key=lambda n: n['frame_index'])

        tmp_mapping = []
        for idx, node in enumerate(self.created_nodes):
            old_id = node['position_id']
            new_id = str(idx + self.start_id)
            tmp_name = f"_tmp_dedup_{idx}"

            old_dir = self.output_dir / old_id
            tmp_dir = self.output_dir / tmp_name

            if old_dir.exists():
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                old_dir.rename(tmp_dir)

            if old_id in self.distance_estimator.node_features:
                self.distance_estimator.node_features[tmp_name] = \
                    self.distance_estimator.node_features.pop(old_id)
            if old_id in self.distance_estimator.node_frames:
                self.distance_estimator.node_frames[tmp_name] = \
                    self.distance_estimator.node_frames.pop(old_id)

            tmp_mapping.append((node, tmp_name, new_id))

        for node, tmp_name, new_id in tmp_mapping:
            tmp_dir = self.output_dir / tmp_name
            new_dir = self.output_dir / new_id

            if tmp_dir.exists():
                if new_dir.exists():
                    shutil.rmtree(new_dir)
                tmp_dir.rename(new_dir)

            node['node_dir'] = str(new_dir)
            node['position_id'] = new_id

            info_file = new_dir / "node_position_info.json"
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = _json.load(f)
                info['self_position']['position_id'] = new_id
                with open(info_file, 'w', encoding='utf-8') as f:
                    _json.dump(info, f, ensure_ascii=False, indent=2)

            if tmp_name in self.distance_estimator.node_features:
                self.distance_estimator.node_features[new_id] = \
                    self.distance_estimator.node_features.pop(tmp_name)
            if tmp_name in self.distance_estimator.node_frames:
                self.distance_estimator.node_frames[new_id] = \
                    self.distance_estimator.node_frames.pop(tmp_name)

        self.current_id = len(self.created_nodes) + self.start_id

        node_count = len(self.created_nodes)
        final_id = self.current_id - 1
        logging.info(f"去重后重编号完成: {node_count} 个 node, ID {self.start_id}-{final_id}")
        for node in self.created_nodes:
            aliases = []
            info_file = Path(node['node_dir']) / "node_position_info.json"
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = _json.load(f)
                aliases = info.get('self_position', {}).get('aliases', [])
            alias_str = f" (别名: {aliases})" if aliases else ""
            logging.info(f"  node {node['position_id']}: {node['position_name']}{alias_str}")

    def generate_connections(self):
        """
        v5: 对每个 node 用 generate_next_positions 一次性生成所有连接
        修复: 闭环检测改为可选参数控制
        """
        if len(self.created_nodes) < 2:
            logging.info("Less than 2 nodes, no connections")
            return

        logging.info("=== Generating connections (v5: PointGrounding + DINOv3 match) ===")

        for i, node in enumerate(self.created_nodes):
            # 确定邻居: 相邻节点
            neighbors = []
            if i > 0:
                neighbors.append(self.created_nodes[i - 1])
            if i < len(self.created_nodes) - 1:
                neighbors.append(self.created_nodes[i + 1])

            # 首尾连接检测 (可选, 默认关闭)
            if self._enable_loop_closure and len(self.created_nodes) >= 4:
                if i == 0:
                    last = self.created_nodes[-1]
                    try:
                        feat_first = self.distance_estimator.extract_frame_feature(node['images'])
                        feat_last = self.distance_estimator.extract_frame_feature(last['images'])
                        sim = self.distance_estimator.compute_similarity(feat_first, feat_last)
                        logging.info(f"First-Last similarity: {sim:.3f} "
                                     f"(threshold: {self._loop_closure_threshold})")
                        if sim > self._loop_closure_threshold and last not in neighbors:
                            neighbors.append(last)
                            logging.info(f"Loop closure: node {node['position_id']} <-> "
                                         f"node {last['position_id']}")
                    except Exception as e:
                        logging.warning(f"First-last check failed: {e}")
                elif i == len(self.created_nodes) - 1:
                    first = self.created_nodes[0]
                    try:
                        feat_last = self.distance_estimator.extract_frame_feature(node['images'])
                        feat_first = self.distance_estimator.extract_frame_feature(first['images'])
                        sim = self.distance_estimator.compute_similarity(feat_last, feat_first)
                        if sim > self._loop_closure_threshold and first not in neighbors:
                            neighbors.append(first)
                            logging.info(f"Loop closure: node {node['position_id']} <-> "
                                         f"node {first['position_id']}")
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
        logging.info("Starting automatic mapping v5.2...")

        frames = self.load_input_data()

        logging.info("Phase 1: Creating nodes (with Qwen naming)...")
        for idx, frame in enumerate(frames):
            self.process_frame(idx, frame)

        # Phase 1.3: VPR 去重 (仅对 VPR 生成的 node)
        dedup_merger = NodeDedupMerger(vpr_dedup_threshold=self._vpr_dedup_threshold)
        self.created_nodes, vpr_alias_map = dedup_merger.dedup_by_vpr(
            self.created_nodes, self.distance_estimator, self.output_dir
        )
        if vpr_alias_map:
            dedup_merger.write_aliases(self.output_dir, vpr_alias_map)
            self._renumber_after_dedup()

        # Phase 1.5: 语义增补 — 扫描中间帧，检测门牌/标识/功能区域，插入新 node
        if self._semantic_detection:
            self._run_semantic_detection()
        else:
            logging.info("Semantic detection disabled, skipping Phase 1.5")

        # Phase 1.6: 同帧合并 (对语义增补产生的同 timestamp node)
        self.created_nodes, frame_alias_map = dedup_merger.merge_same_frame_nodes(
            self.created_nodes, self.output_dir
        )
        if frame_alias_map:
            dedup_merger.write_aliases(self.output_dir, frame_alias_map)
            self._renumber_after_dedup()

        # 停掉 namer 的 Qwen 进程，释放显存给 PointGrounder
        logging.info("Phase 1.5b: Stopping namer Qwen to free GPU memory...")
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
            'semantic_detection': self._semantic_detection,
            'loop_closure': self._enable_loop_closure,
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
