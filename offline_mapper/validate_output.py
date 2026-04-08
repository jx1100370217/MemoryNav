#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
输出格式验证器

验证自动建图生成的数据是否符合 merged_labeled_data 格式规范。

检查项目：
- 目录结构完整性
- JSON 字段完整性
- crop 文件存在性
- 坐标范围正确性
- 文件命名规范
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import re


class OutputValidator:
    """
    输出格式验证器
    """
    
    def __init__(self):
        """初始化验证器"""
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_nodes': 0,
            'total_connections': 0,
            'total_crops': 0,
            'validated_nodes': []
        }
    
    def add_error(self, message: str):
        """添加错误信息"""
        self.errors.append(message)
        logging.error(message)
    
    def add_warning(self, message: str):
        """添加警告信息"""
        self.warnings.append(message)
        logging.warning(message)
    
    def validate_directory_structure(self, node_dir: Path, position_id: str) -> bool:
        """
        验证节点目录结构
        
        Args:
            node_dir: 节点目录路径
            position_id: 位置ID
            
        Returns:
            是否验证通过
        """
        valid = True
        
        # 检查目录存在
        if not node_dir.exists():
            self.add_error(f"Node directory not found: {node_dir}")
            return False
        
        if not node_dir.is_dir():
            self.add_error(f"Not a directory: {node_dir}")
            return False
        
        # 检查必需文件
        required_files = ['node_position_info.json']
        for file_name in required_files:
            file_path = node_dir / file_name
            if not file_path.exists():
                self.add_error(f"Required file missing: {file_path}")
                valid = False
        
        # 检查 crops 子目录
        crops_dir = node_dir / "crops"
        if not crops_dir.exists():
            self.add_warning(f"Crops directory not found: {crops_dir}")
        elif not crops_dir.is_dir():
            self.add_error(f"Crops is not a directory: {crops_dir}")
            valid = False
        
        return valid
    
    def validate_node_position_info(self, info_file: Path, position_id: str) -> Tuple[bool, Dict]:
        """
        验证 node_position_info.json 文件
        
        Args:
            info_file: JSON 文件路径
            position_id: 位置ID
            
        Returns:
            (是否验证通过, JSON数据)
        """
        valid = True
        
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        except json.JSONDecodeError as e:
            self.add_error(f"Invalid JSON format in {info_file}: {e}")
            return False, {}
        
        except Exception as e:
            self.add_error(f"Failed to read {info_file}: {e}")
            return False, {}
        
        # 检查顶级字段
        required_top_fields = ['self_position', 'next_positions']
        for field in required_top_fields:
            if field not in data:
                self.add_error(f"Missing top-level field '{field}' in {info_file}")
                valid = False
        
        # 验证 self_position
        if 'self_position' in data:
            self_pos = data['self_position']
            
            # 必需字段
            required_self_fields = [
                'position_id', 'position_name', 'position_name_eng'
            ]
            
            for field in required_self_fields:
                if field not in self_pos:
                    self.add_error(f"Missing self_position field '{field}' in {info_file}")
                    valid = False
            
            # 检查 position_id 匹配
            if self_pos.get('position_id') != position_id:
                self.add_error(f"Position ID mismatch in {info_file}: expected {position_id}, got {self_pos.get('position_id')}")
                valid = False
            
            # 检查相机文件字段
            camera_fields = ['camera_1', 'camera_2', 'camera_3', 'camera_4']
            for cam_field in camera_fields:
                if cam_field in self_pos:
                    cam_file = self_pos[cam_field]
                    cam_path = info_file.parent / cam_file
                    if not cam_path.exists():
                        self.add_error(f"Camera file not found: {cam_path}")
                        valid = False
        
        # 验证 next_positions
        if 'next_positions' in data:
            next_positions = data['next_positions']
            
            if not isinstance(next_positions, list):
                self.add_error(f"next_positions should be a list in {info_file}")
                valid = False
            else:
                self.stats['total_connections'] += len(next_positions)
                
                for i, next_pos in enumerate(next_positions):
                    if not self.validate_next_position(next_pos, info_file, i):
                        valid = False
        
        return valid, data
    
    def validate_next_position(self, next_pos: Dict, info_file: Path, index: int) -> bool:
        """
        验证 next_positions 中的单个连接
        
        Args:
            next_pos: 连接信息字典
            info_file: JSON 文件路径
            index: 连接索引
            
        Returns:
            是否验证通过
        """
        valid = True
        
        # 必需字段
        required_fields = [
            'position_id', 'position_name', 'camera_name',
            'landmark_name', 'big_box', 'mid_box', 'small_box',
            'crop_image_paths', 'position_name_eng', 'landmark_name_eng'
        ]
        
        for field in required_fields:
            if field not in next_pos:
                self.add_error(f"Missing next_position[{index}] field '{field}' in {info_file}")
                valid = False
        
        # 验证 position_id 格式
        if 'position_id' in next_pos:
            pos_id = next_pos['position_id']
            if not re.match(r'\d+', pos_id):
                self.add_warning(f"Unusual position_id format: {pos_id} in {info_file}")
        
        # 验证 camera_name
        if 'camera_name' in next_pos:
            cam_name = next_pos['camera_name']
            if not re.match(r'camera_[1-4]$', cam_name):
                self.add_error(f"Invalid camera_name: {cam_name} in {info_file}")
                valid = False
        
        # 验证坐标框格式
        box_fields = ['big_box', 'mid_box', 'small_box']
        for box_field in box_fields:
            if box_field in next_pos:
                if not self.validate_bbox_format(next_pos[box_field]):
                    self.add_error(f"Invalid {box_field} format in {info_file}: {next_pos[box_field]}")
                    valid = False
        
        # 验证 crop_image_paths
        if 'crop_image_paths' in next_pos:
            crop_paths = next_pos['crop_image_paths']
            if not isinstance(crop_paths, dict):
                self.add_error(f"crop_image_paths should be a dict in {info_file}")
                valid = False
            else:
                expected_sizes = ['big', 'mid', 'small']
                for size in expected_sizes:
                    if size in crop_paths:
                        crop_file = crop_paths[size]
                        crop_path = info_file.parent / crop_file
                        if not crop_path.exists():
                            self.add_error(f"Crop file not found: {crop_path}")
                            valid = False
                        else:
                            self.stats['total_crops'] += 1
        
        return valid
    
    def validate_bbox_format(self, bbox_str: str) -> bool:
        """
        验证边界框格式
        
        Args:
            bbox_str: 边界框字符串 "x1,y1,x2,y2"
            
        Returns:
            是否格式正确
        """
        if not bbox_str:  # 允许空字符串
            return True
        
        try:
            parts = bbox_str.split(',')
            if len(parts) != 4:
                return False
            
            coords = [float(part) for part in parts]
            
            # 检查坐标范围 [0, 1]
            for coord in coords:
                if coord < 0.0 or coord > 1.0:
                    return False
            
            # 检查 x1 < x2, y1 < y2
            x1, y1, x2, y2 = coords
            if x1 >= x2 or y1 >= y2:
                return False
            
            return True
            
        except (ValueError, IndexError):
            return False
    
    def validate_crop_filename(self, crop_file: str, expected_pattern: str = None) -> bool:
        """
        验证裁剪文件命名格式
        
        Args:
            crop_file: 裁剪文件名
            expected_pattern: 期望的命名模式
            
        Returns:
            是否符合命名规范
        """
        # 基本格式: {timestamp}_camera_{N}__{next_idx}__{size}__{px}_{py}_{pw}_{ph}.jpg
        pattern = r'^\d+_camera_\d+__\d+__(big|mid|small)__\d+_\d+_\d+_\d+\.jpg$'
        
        if not re.match(pattern, os.path.basename(crop_file)):
            self.add_warning(f"Crop file naming may not follow convention: {crop_file}")
            return False
        
        return True
    
    def validate_single_node(self, node_dir: Path) -> bool:
        """
        验证单个节点
        
        Args:
            node_dir: 节点目录路径
            
        Returns:
            是否验证通过
        """
        position_id = node_dir.name
        
        logging.info(f"Validating node: {position_id}")
        
        # 验证目录结构
        if not self.validate_directory_structure(node_dir, position_id):
            return False
        
        # 验证 JSON 文件
        info_file = node_dir / "node_position_info.json"
        valid, data = self.validate_node_position_info(info_file, position_id)
        
        if valid:
            self.stats['validated_nodes'].append(position_id)
        
        return valid
    
    def validate_output_directory(self, output_dir: str) -> Dict:
        """
        验证整个输出目录
        
        Args:
            output_dir: 输出目录路径
            
        Returns:
            验证结果字典
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            self.add_error(f"Output directory not found: {output_path}")
            return self.get_result()
        
        # 查找所有节点目录
        node_dirs = []
        for item in output_path.iterdir():
            if item.is_dir() and re.match(r'\d+', item.name):
                node_dirs.append(item)
        
        if not node_dirs:
            self.add_error(f"No valid node directories found in {output_path}")
            return self.get_result()
        
        self.stats['total_nodes'] = len(node_dirs)
        
        # 验证每个节点
        all_valid = True
        for node_dir in sorted(node_dirs):
            if not self.validate_single_node(node_dir):
                all_valid = False
        
        # 生成总结
        logging.info(f"Validation completed: {len(self.stats['validated_nodes'])}/{self.stats['total_nodes']} nodes passed")
        
        return self.get_result()
    
    def get_result(self) -> Dict:
        """
        获取验证结果
        
        Returns:
            验证结果字典
        """
        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'statistics': self.stats
        }


def validate_output_format(output_dir: str, log_level: str = "INFO") -> Dict:
    """
    验证输出格式的便利函数
    
    Args:
        output_dir: 输出目录路径
        log_level: 日志级别
        
    Returns:
        验证结果字典
    """
    # 设置日志
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.basicConfig(level=numeric_level)
    
    # 执行验证
    validator = OutputValidator()
    result = validator.validate_output_directory(output_dir)
    
    return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="验证自动建图输出格式")
    parser.add_argument('output_dir', help='输出目录路径')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 验证输出
    result = validate_output_format(args.output_dir, args.log_level)
    
    # 打印结果
    print("\n" + "="*50)
    print("验证结果 / Validation Results")
    print("="*50)
    
    stats = result['statistics']
    print(f"📊 总计: {stats['total_nodes']} 节点, {stats['total_connections']} 连接, {stats['total_crops']} 裁剪图像")
    print(f"✅ 通过: {len(stats['validated_nodes'])}/{stats['total_nodes']} 节点")
    
    if result['valid']:
        print("🎉 所有验证通过!")
    else:
        print(f"❌ 发现 {len(result['errors'])} 个错误")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result['warnings']:
        print(f"⚠️  {len(result['warnings'])} 个警告:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    sys.exit(main())