"""Unit tests for online_mapper (minimal, no GPU required where possible)"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_pose_graph_optimize():
    from online_mapper.geometry.pose_graph import PoseGraph
    pg = PoseGraph()
    pg.add_node("a", 0, 0, 0)
    pg.add_node("b", 1, 0, 0)
    pg.add_node("c", 2, 0, 0)
    pg.add_edge("a", "b", 1.0, 0, 0)
    pg.add_edge("b", "c", 1.0, 0, 0)
    pg.optimize(20)
    assert abs(pg.nodes["a"].x) < 1e-3
    assert abs(pg.nodes["c"].x - 2.0) < 1e-2


def test_occupancy_integrate():
    from online_mapper.geometry.occupancy import OccupancyGrid
    import numpy as np
    g = OccupancyGrid(size=80, resolution=0.2)
    row = np.ones(32, dtype=np.float32) * 2.0
    ig = g.integrate(0.0, 0.0, 0.0, row)
    s = g.stats()
    assert s["free"] > 0
    assert ig >= 0.0


def test_keyframe_multi_trigger():
    from online_mapper.topology.keyframe_selector import KeyframeSelector
    from online_mapper.config import OnlineMapperConfig
    cfg = OnlineMapperConfig()
    ks = KeyframeSelector(cfg)
    ks.reset(0)
    # advance past min gap
    t, r = ks.should_trigger(10, 0.9)
    assert not t
    ks.update_motion(2.0, 0, 0)
    t, r = ks.should_trigger(15, 0.9)
    assert t and r == "acc_trans"


def test_graph_neighbors():
    from online_mapper.topology.graph import TopoGraph, TopoNode
    g = TopoGraph()
    for nid in ["1", "2", "3", "4"]:
        g.add_node(TopoNode(nid, "t", 0, {}))
    g.add_edge("1", "2")
    g.add_edge("1", "3")
    g.add_edge("1", "4")
    assert len(g.neighbors("1")) == 3


def test_semantic_dedup_cyclic_cosine():
    from online_mapper.semantics.semantic_dedup import cyclic_cosine
    import numpy as np
    cams = ['camera_1', 'camera_2', 'camera_3', 'camera_4']
    a = {c: np.ones(8) / np.sqrt(8) for c in cams}
    b = {c: np.ones(8) / np.sqrt(8) for c in cams}
    assert abs(cyclic_cosine(a, b) - 1.0) < 1e-5


def test_scene_graph_save(tmp_path):
    from online_mapper.semantics.scene_graph import SceneGraph, SceneObject
    sg = SceneGraph()
    sg.add_node("1", room="office", objects=[SceneObject("chair", [0, 0, 1, 1], 0.9, "camera_1")])
    p = tmp_path / "sg.json"
    sg.save(str(p))
    assert p.exists()


if __name__ == "__main__":
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    test_pose_graph_optimize(); print("pose_graph ok")
    test_occupancy_integrate(); print("occupancy ok")
    test_keyframe_multi_trigger(); print("keyframe ok")
    test_graph_neighbors(); print("graph ok")
    test_semantic_dedup_cyclic_cosine(); print("dedup ok")
    test_scene_graph_save(tmp); print("scene_graph ok")
    print("ALL TESTS PASSED")
