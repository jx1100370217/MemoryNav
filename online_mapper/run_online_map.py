#!/usr/bin/env python3
"""CLI entry point for online_mapper"""
import argparse, logging, sys, os
sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')

from online_mapper.config import OnlineMapperConfig
from online_mapper.core.online_mapper_core import OnlineMapperCore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="memory_test_data")
    p.add_argument("--output", default="online_mapper/output/merged_labeled_data")
    p.add_argument("--vpr_config", default="deploy/vpr_config.yaml")
    p.add_argument("--no_depth", action="store_true")
    p.add_argument("--no_grounding_dino", action="store_true")
    p.add_argument("--pointer_backend", default="qwen",
                   choices=["qwen", "gdino", "geom", "molmo", "gsam2"],
                   help="target 方向打点器后端")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    cfg = OnlineMapperConfig(
        input_dir=args.input,
        output_dir=args.output,
        vpr_config_path=args.vpr_config,
        enable_depth=not args.no_depth,
        enable_grounding_dino=not args.no_grounding_dino,
        pointer_backend=args.pointer_backend,
    )
    core = OnlineMapperCore(cfg)
    core.run()
    print("\n=== Online Mapping Done ===")
    print(f"Output: {args.output}")
    print(f"Metrics: {core.metrics}")


if __name__ == "__main__":
    main()
