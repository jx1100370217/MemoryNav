#!/usr/bin/env python3
"""CLI entry point for online_mapper"""
import argparse, logging, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from online_mapper.config import OnlineMapperConfig
from online_mapper.core.online_mapper_core import OnlineMapperCore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="memory_test_data2")
    p.add_argument("--output", default="online_mapper/output/merged_labeled_data")
    p.add_argument("--vpr_config", default="deploy/vpr_config.yaml")
    p.add_argument("--no_depth", action="store_true")
    p.add_argument("--no_grounding_dino", action="store_true")
    p.add_argument("--plate_scan_every_n", type=int, default=None,
                   help="plate_scan 跳帧 (每 N 帧扫一次, 占总耗时 80%, 调大显著加速). 默认用 config 值.")
    p.add_argument("--dataset_kind", default="indoor",
                   choices=["indoor", "outdoor", "mixed", "auto"],
                   help="area_bootstrapper 输入. indoor/outdoor 单类型时不做 plate 分类")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    cfg_kwargs = dict(
        input_dir=args.input,
        output_dir=args.output,
        vpr_config_path=args.vpr_config,
        enable_depth=not args.no_depth,
        enable_grounding_dino=not args.no_grounding_dino,
        dataset_kind=args.dataset_kind,
    )
    if args.plate_scan_every_n is not None:
        cfg_kwargs["plate_scan_every_n_frames"] = args.plate_scan_every_n
    cfg = OnlineMapperConfig(**cfg_kwargs)
    core = OnlineMapperCore(cfg)
    core.run()
    print("\n=== Online Mapping Done ===")
    print(f"Output: {args.output}")
    print(f"Metrics: {core.metrics}")


if __name__ == "__main__":
    main()
