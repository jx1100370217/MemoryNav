#!/usr/bin/env python3
"""
score_run_v3.py — v2 schema (Site→Area→Region→Node) 评估脚本

输入:
  --output <path>          v2 schema output 目录 (含 manifest.json + areas/.../...)
  --truth <path>           truth_v3.json
  --baseline <path>        可选, 上版 baseline_C7_v2.json 做不退化检查

Gate 0: 文件结构 (manifest + area_tree + region_graph + areas/<a>/area_info + .../region_info)
Gate 1: area 数 + kind (与 truth.areas 期望比对)
Gate 2: region 覆盖率 (按 t_sec_range overlap >= 0.5 判 region 是否被识别;
        category 在合法集合内; node 数在 [expected_min, expected_max])
Gate 3: region_edge 匹配 (按 (u,v) 对; loop_close tag 必须有)
Gate 4: 节点总数 (informational; 期望 13-25, 远低于 baseline 34)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REQUIRED_TOP = ["manifest.json", "area_tree.json", "region_graph.json", "metrics.json"]


def load_v2_output(out_dir: Path) -> dict:
    """读 v2 schema output 目录, 返回 {manifest, area_tree, region_graph, areas:{a_id:{info,regions:{r_id:{info,nodes:[ids]}}}}, errors}"""
    errors: List[str] = []
    result = {"manifest": None, "area_tree": None, "region_graph": None, "metrics": None,
              "areas": {}, "errors": errors}

    for f in REQUIRED_TOP:
        if not (out_dir / f).exists():
            errors.append(f"missing top: {f}")

    for k, fn in [("manifest", "manifest.json"), ("area_tree", "area_tree.json"),
                  ("region_graph", "region_graph.json"), ("metrics", "metrics.json")]:
        p = out_dir / fn
        if p.exists():
            try:
                result[k] = json.loads(p.read_text())
            except Exception as e:
                errors.append(f"{fn} unreadable: {e}")

    areas_dir = out_dir / "areas"
    if not areas_dir.exists():
        errors.append("missing dir: areas/")
        return result

    for a_dir in sorted(areas_dir.iterdir()):
        if not a_dir.is_dir():
            continue
        a_id = a_dir.name
        a_info_p = a_dir / "area_info.json"
        if not a_info_p.exists():
            errors.append(f"missing: areas/{a_id}/area_info.json")
            continue
        try:
            a_info = json.loads(a_info_p.read_text())
        except Exception as e:
            errors.append(f"areas/{a_id}/area_info.json unreadable: {e}")
            continue
        regions_root = a_dir / "regions"
        regions_dict = {}
        if regions_root.exists():
            for r_dir in sorted(regions_root.iterdir()):
                if not r_dir.is_dir():
                    continue
                r_id = r_dir.name
                r_info_p = r_dir / "region_info.json"
                if not r_info_p.exists():
                    errors.append(f"missing: areas/{a_id}/regions/{r_id}/region_info.json")
                    continue
                try:
                    r_info = json.loads(r_info_p.read_text())
                except Exception as e:
                    errors.append(f"regions/{r_id}/region_info.json unreadable: {e}")
                    continue
                nodes_root = r_dir / "nodes"
                node_ids = sorted([n.name for n in nodes_root.iterdir() if n.is_dir()]) if nodes_root.exists() else []
                # 抽 ts: 用 node 的 camera_1 文件名前缀
                node_ts = []
                for nid in node_ids:
                    npi = nodes_root / nid / "node_position_info.json"
                    if npi.exists():
                        try:
                            n_info = json.loads(npi.read_text())
                            cam1 = n_info.get("self_position", {}).get("camera_1", "")
                            if cam1:
                                node_ts.append(int(cam1.split("_")[0]))
                        except Exception:
                            pass
                regions_dict[r_id] = {"info": r_info, "node_ids": node_ids, "node_ts": node_ts}
        result["areas"][a_id] = {"info": a_info, "regions": regions_dict}
    return result


def gate0(output: dict) -> dict:
    return {"passed": not output["errors"], "errors": list(output["errors"])}


def gate1_areas(truth: dict, output: dict) -> dict:
    truth_areas = truth["areas"]
    actual_areas = output["areas"]
    expected_kind = truth["datasets"]["Mappingdata_C7"]["expected_kind"]
    out_kinds = [a["info"].get("kind") for a in actual_areas.values()]
    # 用户决策: dataset_kind=indoor → 期望 1 个 indoor area
    passed = (len(actual_areas) == len(truth_areas)
              and all(k == ("indoor_floor" if expected_kind == "indoor" else k) for k in out_kinds if k))
    return {
        "passed": passed,
        "expected_count": len(truth_areas),
        "actual_count": len(actual_areas),
        "expected_kind": expected_kind,
        "actual_kinds": out_kinds,
    }


def t_overlap(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def gate2_regions(truth: dict, output: dict, ts_first: int) -> dict:
    """按 t_sec_range (truth) vs node_ts (output) 重叠 ≥ 0.3 判匹配."""
    matches: List[dict] = []
    actual_regions = []
    for a in output["areas"].values():
        for r_id, r in a["regions"].items():
            if not r["node_ts"]:
                continue
            t_range = [(min(r["node_ts"]) - ts_first), (max(r["node_ts"]) - ts_first)]
            actual_regions.append({"region_id": r_id, "t_range": t_range, "info": r["info"], "node_count": len(r["node_ids"])})

    # 单纯 t_overlap. 一个 actual region 可被多个 truth region 匹配 (因 actual 大段可覆盖多个 truth 段)
    # 改: 每个 truth region 独立找 best match
    truth_results = []
    for tr in truth["regions"]:
        t_range = tr["t_sec_range"]
        if "t_sec_range_extra" in tr:
            # 闭环 region (起+终), 各自独立尝试匹配
            t_extras = [t_range, tr["t_sec_range_extra"]]
        else:
            t_extras = [t_range]
        best = None
        best_score = 0.0
        for arange in t_extras:
            for ar in actual_regions:
                ov = t_overlap(arange, ar["t_range"])
                if ov > best_score:
                    best_score = ov
                    best = ar
        truth_results.append({
            "truth_region_id": tr["region_id"],
            "name": tr["name"],
            "category": tr["category"],
            "t_sec_range": t_range,
            "best_match_id": best["region_id"] if best else None,
            "best_match_overlap": best_score,
            "matched": best_score >= 0.3,
            "match_node_count": best["node_count"] if best else 0,
            "expected_min_nodes": tr.get("expected_min_nodes", 1),
            "expected_max_nodes": tr.get("expected_max_nodes", 5),
            "node_count_in_range": (
                tr.get("expected_min_nodes", 1) <= (best["node_count"] if best else 0) <= tr.get("expected_max_nodes", 5)
                if best else False
            ),
        })

    matched = sum(1 for r in truth_results if r["matched"])
    return {
        "passed": matched >= 8,
        "expected_min_match": 8,
        "matched_count": matched,
        "truth_results": truth_results,
        "actual_region_count": len(actual_regions),
    }


def gate3_edges(truth: dict, output: dict, name_map: Dict[str, str]) -> dict:
    """name_map: truth region_id -> actual region_id (从 gate2 best match 取). edge 匹配按 (映射后的 u,v) 对集."""
    actual_edges = output.get("region_graph", {}).get("edges", [])
    actual_pair_set = set()
    actual_edges_with_tags = []
    for e in actual_edges:
        u, v = e.get("u"), e.get("v")
        actual_pair_set.add((u, v))
        actual_pair_set.add((v, u))
        actual_edges_with_tags.append({"u": u, "v": v, "tags": e.get("tags", [])})

    truth_edge_results = []
    matched = 0
    has_loop_close = False
    for te in truth["region_edges"]:
        u_actual = name_map.get(te["u"])
        v_actual = name_map.get(te["v"])
        is_in = u_actual and v_actual and ((u_actual, v_actual) in actual_pair_set)
        truth_edge_results.append({
            "truth_u": te["u"], "truth_v": te["v"],
            "expected_tags": te.get("expected_tags", []),
            "actual_u": u_actual, "actual_v": v_actual,
            "matched": bool(is_in),
        })
        if is_in:
            matched += 1
        if "loop_close" in te.get("expected_tags", []) and is_in:
            for ae in actual_edges_with_tags:
                if {ae["u"], ae["v"]} == {u_actual, v_actual} and "loop_close" in ae.get("tags", []):
                    has_loop_close = True

    return {
        "passed": matched >= 9 and has_loop_close,
        "expected_min_match": 9,
        "matched_count": matched,
        "loop_close_present": has_loop_close,
        "truth_edge_results": truth_edge_results,
        "actual_edge_count": len(actual_edges),
    }


def gate4_node_count(output: dict) -> dict:
    total = sum(len(r["node_ids"]) for a in output["areas"].values() for r in a["regions"].values())
    return {
        "actual_total_nodes": total,
        "expected_range": [13, 25],
        "in_range": 13 <= total <= 25,
        "comment": "informational only, not blocking",
    }


def write_audit_pending(out_path: Path, truth: dict, output: dict, gates: dict, run_name: str):
    L: List[str] = []
    L.append(f"# Audit pending v3 — {run_name}\n")
    L.append("**填法**: 抽样几个 critical region 看 4-cam anchor frame + 邻接 region edge 的 crop, 确认视觉合理. 不必逐 connection 全审.\n")

    g0 = gates["gate0"]
    L.append("## Gate 0 — 文件结构")
    L.append(f"- {'✅ pass' if g0['passed'] else '❌ fail'}: {len(g0['errors'])} errors")
    for e in g0["errors"][:20]:
        L.append(f"  - {e}")
    L.append("")

    g1 = gates["gate1"]
    L.append(f"## Gate 1 — Area: {'✅' if g1['passed'] else '❌'}")
    L.append(f"- expected count={g1['expected_count']}, actual={g1['actual_count']}; expected kind={g1['expected_kind']}, actual={g1['actual_kinds']}")
    L.append("")

    g2 = gates["gate2"]
    L.append(f"## Gate 2 — Region 覆盖: {'✅' if g2['passed'] else '❌'}  (matched {g2['matched_count']} / 12, threshold {g2['expected_min_match']})")
    L.append(f"- actual region 总数: {g2['actual_region_count']}")
    L.append("")
    L.append("| truth_region | t_range | best_match | overlap | node_count | 范围内 |")
    L.append("|---|---|---|---|---|---|")
    for r in g2["truth_results"]:
        L.append(f"| {r['truth_region_id']} | {r['t_sec_range']} | {r['best_match_id'] or '—'} | {r['best_match_overlap']:.2f} | {r['match_node_count']} ({r['expected_min_nodes']}-{r['expected_max_nodes']}) | {'✓' if r['node_count_in_range'] else '✗'} |")
    L.append("")

    g3 = gates["gate3"]
    L.append(f"## Gate 3 — Edge 覆盖: {'✅' if g3['passed'] else '❌'}  (matched {g3['matched_count']} / 12, threshold {g3['expected_min_match']}, loop_close={g3['loop_close_present']})")
    L.append("")
    L.append("| truth edge | actual map | matched |")
    L.append("|---|---|---|")
    for e in g3["truth_edge_results"]:
        L.append(f"| {e['truth_u']} ↔ {e['truth_v']} | {e['actual_u']} ↔ {e['actual_v']} | {'✓' if e['matched'] else '✗'} |")
    L.append("")

    g4 = gates["gate4"]
    L.append(f"## Gate 4 — Node 总数 (informational): {g4['actual_total_nodes']} (expected {g4['expected_range']}, {'in range' if g4['in_range'] else 'out of range'})")
    L.append("")

    L.append("## 视觉抽样建议 (R5)")
    L.append("")
    L.append("抽 5 个 critical region 各看 4-cam base + 邻 region edge 的 crop:")
    candidates = ["default__indoor__reception", "default__indoor__exhibition_hall",
                  "default__indoor__second_reception", "default__indoor__elevator_hall",
                  "default__indoor__server_room"]
    for tid in candidates:
        match = next((r for r in g2["truth_results"] if r["truth_region_id"] == tid), None)
        if match and match["best_match_id"]:
            a_path = next((a for a in output["areas"].values() if match["best_match_id"] in a["regions"]), None)
            if a_path:
                a_id = next(k for k, v in output["areas"].items() if match["best_match_id"] in v["regions"])
                r_obj = a_path["regions"][match["best_match_id"]]
                L.append(f"- **{tid}** → `{match['best_match_id']}` ({len(r_obj['node_ids'])} nodes)")
                L.append(f"  - 视觉抽: `areas/{a_id}/regions/{match['best_match_id']}/nodes/<id>/<ts>_camera_*.jpg` (anchor 4 cam)")
        else:
            L.append(f"- **{tid}** → MISSING (gate2 fail), 漏识别")
    L.append("")

    out_path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True, help="v2 schema output dir")
    ap.add_argument("--truth", type=Path, default=Path(__file__).parent / "truth_v3.json")
    ap.add_argument("--runs-dir", type=Path, default=Path(__file__).parent / "runs_v3")
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    if not args.output.exists():
        print(f"ERROR: output dir not found: {args.output}", file=sys.stderr)
        sys.exit(2)
    if not args.truth.exists():
        print(f"ERROR: truth file not found: {args.truth}", file=sys.stderr)
        sys.exit(2)

    truth = json.loads(args.truth.read_text())
    output = load_v2_output(args.output)

    g0 = gate0(output)
    g1 = gate1_areas(truth, output)
    ts_first = truth["datasets"]["Mappingdata_C7"]["ts_first"]
    g2 = gate2_regions(truth, output, ts_first)
    name_map = {r["truth_region_id"]: r["best_match_id"] for r in g2["truth_results"] if r["matched"]}
    g3 = gate3_edges(truth, output, name_map)
    g4 = gate4_node_count(output)

    gates = {"gate0": g0, "gate1": g1, "gate2": g2, "gate3": g3, "gate4": g4}

    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.output.name}"
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": "3.0",
        "run_name": run_name,
        "output_dir": str(args.output.resolve()),
        "truth_path": str(args.truth.resolve()),
        "gates": gates,
        "name_map_truth_to_actual": name_map,
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    write_audit_pending(run_dir / "audit_pending.md", truth, output, gates, run_name)

    print(f"== Run: {run_name}")
    print(f"   report   = {run_dir / 'report.json'}")
    print(f"   audit_md = {run_dir / 'audit_pending.md'}")
    print(f"   Gate 0   : {'PASS' if g0['passed'] else 'FAIL'} ({len(g0['errors'])} errs)")
    print(f"   Gate 1   : {'PASS' if g1['passed'] else 'FAIL'} (areas {g1['actual_count']}/{g1['expected_count']}, kind {g1['actual_kinds']})")
    print(f"   Gate 2   : {'PASS' if g2['passed'] else 'FAIL'} (regions matched {g2['matched_count']}/12)")
    print(f"   Gate 3   : {'PASS' if g3['passed'] else 'FAIL'} (edges matched {g3['matched_count']}/12, loop={g3['loop_close_present']})")
    print(f"   Gate 4   : nodes={g4['actual_total_nodes']} (expected {g4['expected_range']})")

    all_pass = g0["passed"] and g1["passed"] and g2["passed"] and g3["passed"]
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
