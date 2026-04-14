"""在线建图结果可视化

会产出以下 PNG:
  - pose_graph.png          2D 拓扑 + 位姿图, 节点标注类别/名称
  - occupancy.png           占据栅格热图 + 机器人轨迹
  - keyframe_timeline.png   帧序列里的 keyframe/loop-closure 决策时间线
  - scene_overview.txt      scene_graph.json 树状摘要 (纯文本便于 log)

调用点: MappingSession.finalize() 或者作为独立脚本:
  python -m online_mapper.output.visualize <output_root>
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# 对齐 NodeCategory.value (lowercase) 及实际可能出现的类别
_CATEGORY_COLOR = {
    "shop": "#e74c3c",
    "room": "#3498db",
    "junction": "#f39c12",
    "landmark": "#27ae60",
    "landmark_facility": "#2ecc71",
    "corridor": "#9b59b6",
    "function_area": "#16a085",
    "reject": "#7f8c8d",
    "unknown": "#95a5a6",
}


def _set_cn_font():
    # 尝试选一个常见中文字体
    for fam in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Source Han Sans SC",
                "SimHei", "DejaVu Sans"]:
        try:
            plt.rcParams['font.sans-serif'] = [fam]
            plt.rcParams['axes.unicode_minus'] = False
            return fam
        except Exception:
            continue
    return None


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def render_pose_graph(output_root: str, pose_graph: dict, mapper=None) -> str:
    """画 2D 位姿图 + 最终拓扑边 (topo.edges, 非 pose_graph odom 链)"""
    path = os.path.join(output_root, "pose_graph.png")
    if not pose_graph or not pose_graph.get("nodes"):
        return ""

    fig, ax = plt.subplots(figsize=(10, 8))
    nodes = pose_graph["nodes"]
    pg_edges = pose_graph.get("edges", [])
    pos = {n["id"]: (n["x"], n["y"]) for n in nodes}

    names: Dict[str, str] = {}
    cats: Dict[str, str] = {}
    topo_edges = []
    if mapper is not None:
        for nid, node in mapper.topo.nodes.items():
            names[nid] = getattr(node, "position_name", None) or node.landmark_name or f"n{nid}"
            cats[nid] = (getattr(node, "category", "") or "unknown").lower()
        topo_edges = [(a, b) for (a, b) in mapper.topo.edges]

    pg_edge_set = {(min(e['a'], e['b']), max(e['a'], e['b'])): e.get('kind', 'odom')
                   for e in pg_edges}
    topo_edge_set = {(min(a, b), max(a, b)) for (a, b) in topo_edges}
    all_edges = pg_edge_set.keys() | topo_edge_set

    def edge_style(ek):
        kind = pg_edge_set.get(ek)
        if kind == 'loop':
            return '#c0392b', '--', 1.4, 0.8
        if kind == 'odom':
            return '#2c3e50', '-', 1.6, 0.8
        # 仅 topo, 无 pose_graph 对应
        return '#7f8c8d', '-', 1.0, 0.6

    for (a, b) in all_edges:
        if a not in pos or b not in pos:
            continue
        color, ls, lw, alpha = edge_style((a, b))
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=color, linestyle=ls, lw=lw, alpha=alpha, zorder=1)

    for nid, (x, y) in pos.items():
        cat = cats.get(nid, "unknown")
        c = _CATEGORY_COLOR.get(cat, "#95a5a6")
        ax.scatter(x, y, s=230, c=c, edgecolors='black', lw=1, zorder=3)
        label = f"{nid}:{names.get(nid, '?')[:8]}"
        ax.annotate(label, (x, y), xytext=(6, 6), textcoords='offset points',
                    fontsize=9, zorder=4)

    ax.set_title(f"Online-Mapping Pose Graph "
                 f"({len(nodes)} nodes, {len(all_edges)} edges)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    cats_used = sorted({cats.get(nid, "unknown") for nid in pos})
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_CATEGORY_COLOR.get(c, "#95a5a6"),
                          markersize=10, label=c, markeredgecolor='black')
               for c in cats_used]
    handles.append(plt.Line2D([0], [0], color='#2c3e50', lw=1.6, label='odom'))
    handles.append(plt.Line2D([0], [0], color='#c0392b', lw=1.4, ls='--', label='loop'))
    handles.append(plt.Line2D([0], [0], color='#7f8c8d', lw=1.0, label='topo-only'))
    ax.legend(handles=handles, loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def render_occupancy(output_root: str, mapper) -> str:
    """画占据栅格 + 机器人轨迹"""
    if mapper is None or not hasattr(mapper, "occ"):
        return ""
    path = os.path.join(output_root, "occupancy.png")
    try:
        occ = mapper.occ
        grid = getattr(occ, "grid", None)
        if grid is None:
            return ""
        # 不同实现的栅格: 可能是 -log odds, 或 0/1
        arr = np.asarray(grid, dtype=float)
        # 用 0/0.5/1 着色: UNKNOWN(-1)→0.5, FREE(0)→1, OCC(1)→0 (黑色=占据)
        viz = np.full_like(arr, 0.5, dtype=float)
        viz[arr == 0] = 1.0
        viz[arr == 1] = 0.0
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(viz, origin='lower', cmap='gray',
                        vmin=0, vmax=1, interpolation='nearest')

        # 机器人轨迹 (用 occ 自己的 world_to_cell)
        logs = _load_jsonl(Path(output_root) / "online_mapping_log.jsonl")
        xs = [l["robot_pose"][0] for l in logs if "robot_pose" in l]
        ys = [l["robot_pose"][1] for l in logs if "robot_pose" in l]
        if xs and ys:
            xi, yi = [], []
            for x, y in zip(xs, ys):
                cx, cy = occ.world_to_cell(x, y)
                xi.append(cx); yi.append(cy)
            ax.plot(xi, yi, '-', color='red', lw=1.2, alpha=0.85, label='robot traj')
            ax.scatter(xi[0], yi[0], c='lime', s=80, label='start', zorder=5,
                       edgecolors='black')
            ax.scatter(xi[-1], yi[-1], c='orange', s=80, label='end', zorder=5,
                       edgecolors='black')
            ax.legend(fontsize=8)

        ax.set_title(f"Occupancy Grid (res={getattr(occ,'res','?')} m, "
                     f"size={getattr(occ,'size','?')})")
        fig.colorbar(im, ax=ax, fraction=0.04)
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close(fig)
        return path
    except Exception as e:
        import logging; logging.getLogger(__name__).warning(f"occupancy render fail: {e}")
        return ""


def render_keyframe_timeline(output_root: str) -> str:
    """画 keyframe/loop_closure 决策时间线"""
    logs = _load_jsonl(Path(output_root) / "online_mapping_log.jsonl")
    if not logs:
        return ""
    path = os.path.join(output_root, "keyframe_timeline.png")
    xs = [l["frame_idx"] for l in logs]
    sims = [l.get("vpr_sim_to_last") or 0 for l in logs]
    gains = [l.get("info_gain") or 0 for l in logs]
    kfs = [l["frame_idx"] for l in logs if l.get("keyframe")]
    rejects = [l["frame_idx"] for l in logs
               if (l.get("category_decision") or {}).get("category", "").lower() == "reject"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(xs, sims, '-', color='#3498db', label='VPR sim to last KF')
    for k in kfs:
        axes[0].axvline(k, color='#e74c3c', alpha=0.3, lw=0.6)
    for r in rejects:
        axes[0].axvline(r, color='#7f8c8d', alpha=0.5, ls=':', lw=0.8)
    axes[0].set_ylabel("VPR similarity")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"Keyframe timeline ({len(kfs)} triggered, "
                      f"{len(rejects)} rejected by category)")

    axes[1].plot(xs, gains, '-', color='#27ae60', label='info gain')
    axes[1].set_ylabel("info gain"); axes[1].set_xlabel("frame_idx")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def render_scene_overview(output_root: str, scene_graph: dict) -> str:
    """scene_graph.json 的树状摘要"""
    if not scene_graph:
        return ""
    path = os.path.join(output_root, "scene_overview.txt")
    nodes_map = scene_graph.get('nodes', scene_graph.get('scene_nodes', {})) or {}
    lines = []
    lines.append(f"SceneGraph summary ({len(nodes_map)} nodes)")
    lines.append("=" * 60)

    floors = scene_graph.get('floors', {})
    for floor_id, rooms in floors.items():
        lines.append(f"Floor: {floor_id}")
        for room_name, node_ids in rooms.items():
            lines.append(f"  Room: {room_name} ({len(node_ids)} nodes)")
            for nid in node_ids:
                n = nodes_map.get(nid) or {}
                objs = n.get('objects', []) if isinstance(n, dict) else []
                counts = {}
                for o in objs:
                    lab = o.get('label') if isinstance(o, dict) else None
                    if lab:
                        counts[lab] = counts.get(lab, 0) + 1
                top_objs = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
                lines.append(f"    [{nid}] top_objs={top_objs}")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    return path


def render_mapping_visuals(output_root: str, mapper=None) -> Dict[str, str]:
    """主入口: 产出所有可视化, 返回路径字典"""
    _set_cn_font()
    output_root = str(output_root)

    results = {}
    pg = _load_json(Path(output_root) / "pose_graph.json")
    sg = _load_json(Path(output_root) / "scene_graph.json")

    if pg:
        p = render_pose_graph(output_root, pg, mapper)
        if p: results["pose_graph"] = p
    p = render_occupancy(output_root, mapper)
    if p: results["occupancy"] = p
    p = render_keyframe_timeline(output_root)
    if p: results["keyframe_timeline"] = p
    if sg:
        p = render_scene_overview(output_root, sg)
        if p: results["scene_overview"] = p

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m online_mapper.output.visualize <output_root>")
        sys.exit(1)
    out = render_mapping_visuals(sys.argv[1])
    print(json.dumps(out, indent=2, ensure_ascii=False))
