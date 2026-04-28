"""online_mapper 行为 fingerprint, 重构前后比对.

用法:
  python -m online_mapper.tests.fingerprint <output_dir>          # 打印 fingerprint
  python -m online_mapper.tests.fingerprint <baseline> <new>      # diff, 仅 stable 项 fail

stable_hash: 算法语义关键项 (节点数/名/类别/edges/scene_graph/metrics), 重构必须保持
soft 项: Hungarian 抖动 (next_camera_names) + Qwen 输出抖动 (next_landmark_names), 仅参考
"""
from __future__ import annotations
import json, hashlib, sys
from pathlib import Path


def normalize(o):
    if isinstance(o, dict): return {k: normalize(o[k]) for k in sorted(o.keys())}
    if isinstance(o, list): return [normalize(x) for x in o]
    if isinstance(o, set): return sorted(normalize(x) for x in o)
    return o


def shash(o) -> str:
    return hashlib.sha256(
        json.dumps(normalize(o), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]


def compute(root: Path) -> dict:
    fp = {}
    m = json.loads((root / "metrics.json").read_text())
    fp["metrics_stable"] = {k: v for k, v in m.items() if k != "runtime_s"}

    fp["scene_graph"] = json.loads((root / "scene_graph.json").read_text())

    pg = json.loads((root / "pose_graph.json").read_text())
    fp["pose_graph"] = {
        "n_nodes": len(pg.get("nodes", [])),
        "n_edges": len(pg.get("edges", [])),
        "node_ids": sorted([n["id"] for n in pg.get("nodes", [])]),
        "edge_kinds": sorted([(e["a"], e["b"], e.get("kind", "odom")) for e in pg.get("edges", [])]),
    }

    nodes_stable, nodes_soft = {}, {}
    for nd in sorted((root / "merged_labeled_data").iterdir()):
        if not nd.is_dir(): continue
        info = json.loads((nd / "node_position_info.json").read_text())
        sp = info.get("self_position", {})
        nps = info.get("next_positions", [])
        nodes_stable[nd.name] = {
            "position_name": sp.get("position_name"),
            "position_name_eng": sp.get("position_name_eng"),
            "category": sp.get("category"),
            "organization": sp.get("organization", ""),
            "nearby_plates": sorted(sp.get("nearby_plates", [])),
            "n_cameras": sum(1 for k in sp if k.startswith("camera_")),
            "n_next_positions": len(nps),
            "next_position_ids": sorted([n.get("position_id") for n in nps]),
        }
        nodes_soft[nd.name] = {
            "next_camera_names": sorted([n.get("camera_name") for n in nps]),
            # Qwen identify_landmark 输出非确定性 (vLLM 多次推理可能不同字)
            "next_landmark_names": sorted([n.get("landmark_name") for n in nps]),
        }
    fp["nodes"] = nodes_stable
    fp["nodes_soft"] = nodes_soft

    pv_path = root / "plate_voter_dump.json"
    if pv_path.exists():
        fp["plate_voter_keys"] = sorted(json.loads(pv_path.read_text()).keys())

    log = {"n_lines": 0, "n_keyframes": 0, "kf_reasons": {}, "rejected": 0, "accepted_by_cat": {}}
    log_path = root / "online_mapping_log.jsonl"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                e = json.loads(line)
                log["n_lines"] += 1
                if e.get("keyframe"):
                    log["n_keyframes"] += 1
                    r = e.get("reason", "")
                    log["kf_reasons"][r] = log["kf_reasons"].get(r, 0) + 1
                cd = e.get("category_decision") or {}
                cat = cd.get("category", "")
                if cat == "reject":
                    log["rejected"] += 1
                elif cat:
                    log["accepted_by_cat"][cat] = log["accepted_by_cat"].get(cat, 0) + 1
    fp["log_summary"] = log

    fp["stable_hash"] = shash({k: v for k, v in fp.items() if k != "nodes_soft"})
    fp["soft_hash"] = shash(fp["nodes_soft"])
    return fp


def diff(baseline: dict, new: dict) -> int:
    """返回 0 表示 stable_hash 一致, 1 表示有破坏性变化."""
    print(f"baseline stable_hash : {baseline['stable_hash']}")
    print(f"new      stable_hash : {new['stable_hash']}")
    print(f"baseline soft_hash   : {baseline['soft_hash']}")
    print(f"new      soft_hash   : {new['soft_hash']}")
    if baseline["stable_hash"] == new["stable_hash"]:
        print("OK: stable_hash matches (refactor preserved algorithm behavior)")
        if baseline["soft_hash"] != new["soft_hash"]:
            print("note: soft_hash differs (Hungarian / Qwen output non-determinism, expected)")
        return 0
    print("FAIL: stable_hash differs. Per-field diff:")
    keys = set(baseline.keys()) | set(new.keys())
    for k in sorted(keys):
        if k in ("stable_hash", "soft_hash", "nodes_soft"): continue
        a, b = baseline.get(k), new.get(k)
        if a != b:
            print(f"  [{k}]")
            sa = json.dumps(a, ensure_ascii=False, indent=2, sort_keys=True).splitlines()
            sb = json.dumps(b, ensure_ascii=False, indent=2, sort_keys=True).splitlines()
            import difflib
            for line in list(difflib.unified_diff(sa, sb, lineterm="", n=2))[:80]:
                print("    " + line)
    return 1


def main(argv):
    if len(argv) == 2:
        print(json.dumps(compute(Path(argv[1])), ensure_ascii=False, indent=2))
        return 0
    if len(argv) == 3:
        a, b = Path(argv[1]), Path(argv[2])
        # 接受 fingerprint.json 文件 或 输出根目录
        ja = json.loads(a.read_text()) if a.is_file() else compute(a)
        jb = json.loads(b.read_text()) if b.is_file() else compute(b)
        return diff(ja, jb)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
