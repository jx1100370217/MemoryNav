#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动节点生成器 v2

改进: 支持外部传入位置名称 (position_names)
"""

import os
import sys
import json
import shutil
import logging
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')

from .auto_landmark_namer import AutoLandmarkNamer


class AutoNodeGenerator:
    """自动节点生成器 v2"""

    def __init__(self, output_base_dir: str, use_qwen_naming: bool = False):
        self.output_base_dir = Path(output_base_dir)
        self.namer = AutoLandmarkNamer(use_qwen=False)  # fallback namer
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"AutoNodeGenerator v2, output: {self.output_base_dir}")

    def create_node_directory(self, position_id: str) -> Path:
        node_dir = self.output_base_dir / position_id
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "crops").mkdir(exist_ok=True)
        return node_dir

    def copy_camera_images(self, source_images: Dict[str, str],
                           node_dir: Path, timestamp: str) -> Dict[str, str]:
        copied = {}
        for cam_id, src in source_images.items():
            if not os.path.exists(src):
                continue
            target_name = f"{timestamp}_{cam_id}.jpg"
            target_path = node_dir / target_name
            try:
                shutil.copy2(src, target_path)
                copied[cam_id] = target_name
            except Exception as e:
                logging.error(f"Failed to copy {src}: {e}")
        return copied

    def create_node_position_info(self, position_id: str, timestamp: str,
                                  camera_files: Dict[str, str],
                                  next_positions: Optional[List[Dict]] = None,
                                  position_names: Optional[Dict[str, str]] = None) -> Dict:
        # 使用外部提供的名称或 fallback
        if position_names:
            name_cn = position_names.get('position_name', f'自动节点_{position_id}')
            name_en = position_names.get('position_name_eng', f'auto_node_{position_id}')
        else:
            names = self.namer.generate_self_position_names(position_id)
            name_cn = names['position_name']
            name_en = names['position_name_eng']

        self_position = {
            "position_id": position_id,
            "position_name": name_cn,
            "position_name_eng": name_en,
        }

        for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            if cam_id in camera_files:
                self_position[cam_id] = camera_files[cam_id]

        return {
            "self_position": self_position,
            "next_positions": next_positions or [],
        }

    def save_node_position_info(self, node_dir: Path, node_info: Dict):
        with open(node_dir / "node_position_info.json", 'w', encoding='utf-8') as f:
            json.dump(node_info, f, ensure_ascii=False, indent=2)

    def create_node(self, position_id: str, timestamp: str,
                    source_images: Dict[str, str],
                    next_positions: Optional[List[Dict]] = None,
                    position_names: Optional[Dict[str, str]] = None) -> Path:
        node_dir = self.create_node_directory(position_id)
        logging.info(f"Creating node {position_id} in {node_dir}")

        camera_files = self.copy_camera_images(source_images, node_dir, timestamp)
        node_info = self.create_node_position_info(
            position_id, timestamp, camera_files, next_positions, position_names
        )
        self.save_node_position_info(node_dir, node_info)

        logging.info(f"Node {position_id} created successfully")
        return node_dir

    def update_node_connections(self, position_id: str, next_positions: List[Dict]):
        info_file = self.output_base_dir / position_id / "node_position_info.json"
        if not info_file.exists():
            logging.error(f"Node info not found: {info_file}")
            return
        with open(info_file, 'r', encoding='utf-8') as f:
            node_info = json.load(f)
        node_info["next_positions"] = next_positions
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(node_info, f, ensure_ascii=False, indent=2)
        logging.info(f"Updated connections for node {position_id}")

    def get_node_info(self, position_id: str) -> Optional[Dict]:
        info_file = self.output_base_dir / position_id / "node_position_info.json"
        if not info_file.exists():
            return None
        with open(info_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_created_nodes(self) -> List[str]:
        nodes = []
        for item in self.output_base_dir.iterdir():
            if item.is_dir() and (item / "node_position_info.json").exists():
                nodes.append(item.name)
        return sorted(nodes)

    def get_statistics(self) -> Dict:
        nodes = self.list_created_nodes()
        total_conn = sum(
            len(self.get_node_info(n).get("next_positions", []))
            for n in nodes if self.get_node_info(n)
        )
        return {
            'total_nodes': len(nodes),
            'total_connections': total_conn,
            'node_ids': nodes,
            'output_directory': str(self.output_base_dir),
        }
