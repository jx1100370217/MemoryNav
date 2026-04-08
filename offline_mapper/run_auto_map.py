#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动建图入口脚本 v2

使用方式:
    python offline_mapper/run_auto_map.py \
        --input_dir memory_test_data \
        --output_dir offline_mapper/merged_labeled_data \
        --start_id 1 \
        --vpr_config deploy/vpr_config.yaml \
        --use_qwen_naming
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')

from offline_mapper.offline_mapper_core import AutoMapperCore


def setup_logging(log_level: str = "INFO") -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('auto_mapping.log', encoding='utf-8')
        ]
    )


def validate_paths(args) -> None:
    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    vpr_path = Path(args.vpr_config)
    if not vpr_path.exists():
        raise FileNotFoundError(f"VPR config not found: {vpr_path}")
    image_files = list(input_path.glob("*_camera_*.jpg"))
    if not image_files:
        raise ValueError(f"No camera images in {input_path}")
    logging.info(f"Found {len(image_files)} camera images")
    import shutil as _shutil
    if Path(args.output_dir).exists():
        _shutil.rmtree(args.output_dir)
        logging.info(f"Cleaned output directory: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


def print_results(stats: dict):
    print("\n" + "=" * 50)
    print("自动建图完成! / Auto Mapping Completed!")
    print("=" * 50)
    print(f"📍 创建节点数量: {stats.get('total_nodes', 0)}")
    print(f"🔗 生成连接数量: {stats.get('total_connections', 0)}")
    print(f"📁 输出目录: {stats.get('output_directory', 'N/A')}")
    print(f"🆔 节点ID范围: {stats.get('start_id', 'N/A')} -> {stats.get('final_id', 'N/A')}")
    if stats.get('node_ids'):
        print(f"📋 节点列表: {', '.join(str(x) for x in stats['node_ids'])}")
    print(f"\n建图参数:")
    print(f"  相似度阈值: {stats.get('similarity_threshold', 'N/A')}")
    print(f"  最小帧间隔: {stats.get('min_frame_interval', 'N/A')}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="MemoryNav 自动建图工具 v2")

    parser.add_argument('--input_dir', type=str, default='memory_test_data')
    parser.add_argument('--output_dir', type=str, default='offline_mapper/merged_labeled_data')
    parser.add_argument('--vpr_config', type=str, default='deploy/vpr_config.yaml')
    parser.add_argument('--start_id', type=int, default=1)
    parser.add_argument('--similarity_threshold', type=float, default=0.525)
    parser.add_argument('--min_frame_interval', type=int, default=1)
    parser.add_argument('--use_qwen_naming', default=True)
    parser.add_argument('--qwen_gpu', type=str, default='1')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--no_semantic', action='store_true', help='关闭语义增补(Phase 1.5)')
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    try:
        print("\n    ====================================================")
        print("    MemoryNav Auto Mapper v2")
        print("    自动建图模块 - 从图像序列生成导航图结构")
        print("    ====================================================\n")

        setup_logging(args.log_level)
        validate_paths(args)

        if args.dry_run:
            print("✅ 验证通过, dry run 完成")
            return 0

        print("配置参数:")
        print(f"  输入目录: {args.input_dir}")
        print(f"  输出目录: {args.output_dir}")
        print(f"  VPR配置: {args.vpr_config}")
        print(f"  起始ID: {args.start_id}")
        print(f"  相似度阈值: {args.similarity_threshold}")
        print(f"  最小帧间隔: {args.min_frame_interval}")
        print(f"  使用Qwen命名: {args.use_qwen_naming}")
        print(f"  Qwen GPU: {args.qwen_gpu}")
        print()

        start_time = time.time()

        mapper = AutoMapperCore(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            vpr_config_path=args.vpr_config,
            start_id=args.start_id,
            similarity_threshold=args.similarity_threshold,
            min_frame_interval=args.min_frame_interval,
            use_qwen_naming=args.use_qwen_naming,
            qwen_gpu=args.qwen_gpu,
            semantic_detection=not args.no_semantic,
        )

        print("🚀 开始自动建图...")
        stats = mapper.run_auto_mapping()
        mapper.cleanup()

        elapsed = time.time() - start_time
        print_results(stats)
        print(f"⏱️  总耗时: {elapsed:.2f} 秒")

        # 验证
        print("\n🔍 运行输出验证...")
        try:
            from offline_mapper.validate_output import validate_output_format
            result = validate_output_format(args.output_dir)
            if result['valid']:
                print("✅ 输出格式验证通过!")
            else:
                print("❌ 验证失败:")
                for err in result['errors']:
                    print(f"  - {err}")
        except ImportError:
            print("⚠️  验证模块未找到")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        return 130
    except Exception as e:
        logging.error(f"Auto mapping failed: {e}")
        logging.exception("Details:")
        print(f"\n❌ 建图失败: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
