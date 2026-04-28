#!/usr/bin/env python3
"""Synthesized loop-closure test.

Feeds the memory_test_data sequence forward then a reversed tail
to force a revisit. Verifies that the LoopCloser fires geometric
verification on real images.
"""
import os, sys, glob, shutil, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from online_mapper.config import OnlineMapperConfig
from online_mapper.core.online_mapper_core import OnlineMapperCore

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

SRC = "memory_test_data"
SYNTH = "/tmp/memory_test_synth"

# Copy source frames + reversed tail (last 15 frames in reverse) to synth dir
def build_synth():
    if os.path.exists(SYNTH):
        shutil.rmtree(SYNTH)
    os.makedirs(SYNTH)
    files = sorted(glob.glob(f"{SRC}/*_camera_*.jpg"))
    timestamps = sorted({os.path.basename(f).split('_')[0] for f in files})
    print(f"Source frames: {len(timestamps)}")

    next_ts = int(timestamps[-1]) + 1000
    # Forward
    for ts in timestamps:
        for cam in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            src = f"{SRC}/{ts}_{cam}.jpg"
            if os.path.exists(src):
                shutil.copy2(src, f"{SYNTH}/{ts}_{cam}.jpg")
    # Reverse tail
    tail = list(reversed(timestamps[-15:]))
    for ts in tail:
        new_ts = next_ts; next_ts += 1
        for cam in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            src = f"{SRC}/{ts}_{cam}.jpg"
            if os.path.exists(src):
                shutil.copy2(src, f"{SYNTH}/{new_ts}_{cam}.jpg")
    print(f"Synth dir: {len(glob.glob(f'{SYNTH}/*_camera_1.jpg'))} unique timestamps")


build_synth()

# NOTE: 不能 disable qwen + door_plate, 否则 category_clf 100% reject KF →
# self.node_features 为空 → LoopCloser 永远没匹配候选. 此测需要至少一个
# 信号源 (plate/scene) 才能产生 KF node.
cfg = OnlineMapperConfig(
    input_dir=SYNTH,
    output_dir="online_mapper/output/synth_loop_test/merged_labeled_data",
    enable_qwen_naming=True,
    enable_door_plate_detection=True,
    enable_real_connections=False,    # ConnectionBuilder 不影响 LoopCloser, 跳过省时
)
core = OnlineMapperCore(cfg)
core.run()
print("=== SYNTH LOOP TEST ===")
print(f"n_nodes={core.metrics['n_nodes']}")
print(f"n_loop_closures={core.metrics['n_loop_closures']}")
print(f"loop_threshold_used={core.metrics.get('loop_threshold_used')}")
assert core.metrics['n_loop_closures'] >= 1, \
    f"Loop closure did NOT fire on synthesized revisit (got {core.metrics['n_loop_closures']})"
print("PASS: loop closure fired on synthesized revisit")
