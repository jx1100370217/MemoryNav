"""Baseline regression test: rerun the full pipeline on memory_test_data2 and
verify that the algorithm-relevant fingerprint (stable_hash) stays the same.

Skipped by default — needs GPU + Qwen vLLM (port 8199) + ~3min runtime.
Set ONLINE_MAPPER_REGRESSION=1 to enable.

Usage:
  ONLINE_MAPPER_REGRESSION=1 pytest online_mapper/tests/test_baseline_regression.py -s
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.mark.skipif(
    not os.environ.get("ONLINE_MAPPER_REGRESSION"),
    reason="set ONLINE_MAPPER_REGRESSION=1 to run (needs GPU + Qwen vLLM, ~3min)",
)
def test_baseline_stable_hash(tmp_path):
    """E2E: run pipeline on memory_test_data2, compare stable_hash to baseline."""
    from online_mapper.config import OnlineMapperConfig
    from online_mapper.core.online_mapper_core import OnlineMapperCore
    from online_mapper.tests.fingerprint import compute

    cfg = OnlineMapperConfig(
        input_dir=str(PROJECT_ROOT / "memory_test_data2"),
        output_dir=str(tmp_path / "merged_labeled_data"),
        vpr_config_path=str(PROJECT_ROOT / "deploy/vpr_config.yaml"),
    )
    core = OnlineMapperCore(cfg)
    core.run()

    fp = compute(tmp_path)
    baseline = json.loads(
        (Path(__file__).parent / "baseline_fingerprint.json").read_text())
    bh = baseline["stable_hash"]
    nh = fp["stable_hash"]
    assert nh == bh, (
        "stable_hash regression!\n"
        f"  baseline = {bh}\n  new      = {nh}\n"
        f"Run: python -m online_mapper.tests.fingerprint "
        f"online_mapper/tests/baseline_fingerprint.json {tmp_path}"
    )


@pytest.mark.skipif(
    not os.environ.get("ONLINE_MAPPER_REGRESSION"),
    reason="set ONLINE_MAPPER_REGRESSION=1 to run",
)
def test_finalize_emits_visualizations(tmp_path):
    """Finalize 自动调 viz/visualize, 至少产出 pose_graph.png + keyframe_timeline.png."""
    from online_mapper.config import OnlineMapperConfig
    from online_mapper.core.online_mapper_core import OnlineMapperCore

    cfg = OnlineMapperConfig(
        input_dir=str(PROJECT_ROOT / "memory_test_data2"),
        output_dir=str(tmp_path / "merged_labeled_data"),
        vpr_config_path=str(PROJECT_ROOT / "deploy/vpr_config.yaml"),
    )
    core = OnlineMapperCore(cfg)
    core.run()

    expected = ["pose_graph.png", "occupancy.png", "keyframe_timeline.png"]
    found = [p for p in expected if (tmp_path / p).exists()]
    assert len(found) >= 2, f"expected >=2 viz outputs, found {found}"
