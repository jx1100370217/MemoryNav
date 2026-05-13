"""V2Writer — 输出 §3 v2 schema (Site→Area→Region→Node 四级目录).

调用顺序 (在 OnlineMapperCore._finalize 末段):
  1. ConnectionBuilder 已写完 next_positions 到 cfg.output_dir/<id>/  (v1 临时位置)
  2. region_clusterer.cluster() 给出 List[RegionInfo]
  3. area_bootstrapper.bootstrap() 给出 List[AreaInfo]
  4. V2Writer.finalize_v2() 接管:
     - 移动节点目录 v1/<id>/ → v2/areas/<a>/regions/<r>/nodes/<id>/
     - 改写 node_position_info.json 加 region_id/area_id 字段
     - 写 manifest.json / area_tree.json / region_graph.json
     - 各 area_info.json / region_info.json / region_feature.npy
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.0"


class V2Writer:
    def __init__(self, output_root: Path):
        """
        Args:
            output_root: cfg.output_dir 的父目录 (节点 v1 临时位置在 output_root/<id>/).
                         v2 输出: areas/, manifest.json, area_tree.json, region_graph.json,
                         scene_graph.json, pose_graph.json, online_mapping_log.jsonl, metrics.json
                         全在 output_root 下.
        """
        self.root = Path(output_root)

    def finalize_v2(self,
                    nodes_v1_dir: Path,
                    regions: List,        # List[RegionInfo]
                    areas: List,          # List[AreaInfo]
                    pose_graph,
                    nodes_dict: Dict,
                    site_id: str = "default") -> Dict[str, str]:
        """
        Args:
            nodes_v1_dir: 当前节点平铺目录 (e.g. cfg.output_dir = output/merged_labeled_data/)
            regions: List[RegionInfo] (region_clusterer 输出)
            areas: List[AreaInfo] (area_bootstrapper 输出)
            pose_graph: pose_graph 实例 (取 node x/y)
            nodes_dict: Dict[node_id -> TopoNode] (取 ts 等)
        Returns:
            paths dict: {"manifest": ..., "n_nodes_moved": ..., ...}
        """
        # 1. 建 area / region 反查
        node_to_region: Dict[str, str] = {}
        node_to_area: Dict[str, str] = {}
        region_to_area: Dict[str, str] = {}
        for a in areas:
            for r_id in a.region_ids:
                region_to_area[r_id] = a.area_id
        for r in regions:
            a_id = region_to_area.get(r.region_id, "default__indoor")
            for nid in r.node_ids:
                node_to_region[nid] = r.region_id
                node_to_area[nid] = a_id

        # 2. 移动节点目录: v1 → v2
        node_index: Dict[str, str] = {}
        n_moved = 0
        for nid in node_to_region.keys():
            r_id = node_to_region[nid]
            a_id = node_to_area[nid]
            v1_path = nodes_v1_dir / nid
            v2_rel = Path("areas") / a_id / "regions" / r_id / "nodes" / nid
            v2_path = self.root / v2_rel
            v2_path.parent.mkdir(parents=True, exist_ok=True)
            if v1_path.exists() and v1_path.is_dir():
                if v2_path.exists():
                    shutil.rmtree(v2_path)
                shutil.move(str(v1_path), str(v2_path))
                n_moved += 1
                self._patch_node_position_info(v2_path, region_id=r_id, area_id=a_id,
                                                area_kind=self._area_kind(a_id, areas))
            elif v2_path.exists():
                # 二次跑: 已在 v2 位置, 仅 patch json
                self._patch_node_position_info(v2_path, region_id=r_id, area_id=a_id,
                                                area_kind=self._area_kind(a_id, areas))
            else:
                logger.warning(f"[V2Writer] node {nid} dir missing in both v1 and v2")
                continue
            node_index[nid] = str(v2_rel).replace("\\", "/")

        # v1_dir 残留 (没归 region 的节点) 暂留, 不删
        if nodes_v1_dir.exists():
            leftover = [d.name for d in nodes_v1_dir.iterdir() if d.is_dir()]
            if leftover:
                logger.info(f"[V2Writer] {len(leftover)} leftover nodes in v1 (not in any region): {leftover[:5]}...")

        # 3. 写各 area_info.json
        for a in areas:
            a_dir = self.root / "areas" / a.area_id
            a_dir.mkdir(parents=True, exist_ok=True)
            (a_dir / "area_info.json").write_text(
                json.dumps(self._area_info_dict(a), ensure_ascii=False, indent=2),
                encoding="utf-8")

        # 4. 写各 region_info.json + region_feature.npy
        for r in regions:
            a_id = region_to_area.get(r.region_id, "default__indoor")
            r_dir = self.root / "areas" / a_id / "regions" / r.region_id
            r_dir.mkdir(parents=True, exist_ok=True)
            (r_dir / "region_info.json").write_text(
                json.dumps(self._region_info_dict(r, a_id), ensure_ascii=False, indent=2),
                encoding="utf-8")
            if r.region_feature is not None:
                np.save(r_dir / "region_feature.npy", r.region_feature)

        # 5. 写 area_tree.json
        area_tree = self._build_area_tree(areas, site_id)
        (self.root / "area_tree.json").write_text(
            json.dumps(area_tree, ensure_ascii=False, indent=2), encoding="utf-8")

        # 6. 写 region_graph.json (基于节点级 next_positions 推导)
        region_graph = self._derive_region_graph(regions, region_to_area, node_to_region, areas)
        (self.root / "region_graph.json").write_text(
            json.dumps(region_graph, ensure_ascii=False, indent=2), encoding="utf-8")

        # 7. 写 manifest.json
        manifest = {
            "version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "schema": "memorynav-merged-labeled-data",
            "sites": [site_id],
            "areas": [a.area_id for a in areas],
            "regions": [r.region_id for r in regions],
            "node_index": node_index,
            "node_to_region": node_to_region,
            "node_to_area": node_to_area,
            "region_to_area": region_to_area,
            "stats": {
                "n_sites": 1,
                "n_areas": len(areas),
                "n_regions": len(regions),
                "n_nodes": len(node_to_region),
                "n_region_edges": len(region_graph["edges"]),
                "n_indoor_areas": sum(1 for a in areas if a.kind == "indoor_floor"),
                "n_outdoor_areas": sum(1 for a in areas if a.kind == "outdoor_zone"),
                "n_transition_areas": sum(1 for a in areas if a.kind == "transition"),
            },
        }
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "manifest": str(self.root / "manifest.json"),
            "area_tree": str(self.root / "area_tree.json"),
            "region_graph": str(self.root / "region_graph.json"),
            "n_areas": len(areas),
            "n_regions": len(regions),
            "n_nodes_moved": n_moved,
            "n_region_edges": len(region_graph["edges"]),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _area_kind(area_id: str, areas) -> str:
        for a in areas:
            if a.area_id == area_id:
                return a.kind
        return "indoor_floor"

    @staticmethod
    def _patch_node_position_info(node_dir: Path, region_id: str, area_id: str, area_kind: str):
        info_path = node_dir / "node_position_info.json"
        if not info_path.exists():
            return
        try:
            info = json.loads(info_path.read_text())
        except Exception as e:
            logger.warning(f"[V2Writer] {info_path} unreadable: {e}")
            return
        sp = info.setdefault("self_position", {})
        sp["region_id"] = region_id
        sp["area_id"] = area_id
        sp["area_kind"] = area_kind
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _area_info_dict(a) -> dict:
        return {
            "area_id": a.area_id,
            "name": a.name,
            "name_eng": a.name_eng,
            "kind": a.kind,
            "site_id": a.site_id,
            "parent_area_id": a.parent_area_id,
            "building": a.building,
            "building_eng": a.building.title() if a.building else None,
            "floor": a.floor,
            "elevation_m": a.elevation_m,
            "outdoor_type": a.outdoor_type,
            "bbox_2d": a.bbox_2d,
            "description": a.description,
            "regions": list(a.region_ids),
            "color_hex": a.color_hex,
            "tags": list(a.tags),
        }

    @staticmethod
    def _region_info_dict(r, area_id: str) -> dict:
        return {
            "region_id": r.region_id,
            "name": r.dominant_plate or r.region_id.split("__")[-1],
            "name_eng": r.region_id.split("__")[-1],
            "area_id": area_id,
            "category": r.dominant_category or "function_area",
            "description": f"region with dominant plate '{r.dominant_plate or '-'}', {len(r.node_ids)} nodes",
            "node_ids": list(r.node_ids),
            "anchor_node_id": r.anchor_node_id,
            "color_hex": None,
            "tags": [r.dominant_category] if r.dominant_category else [],
        }

    @staticmethod
    def _build_area_tree(areas, site_id: str) -> dict:
        children = []
        for a in areas:
            children.append({
                "area_id": a.area_id,
                "kind": a.kind,
                "children": [],
            })
        return {
            "version": SCHEMA_VERSION,
            "tree": [{
                "area_id": site_id,
                "kind": "site",
                "children": children,
            }],
        }

    def _derive_region_graph(self, regions, region_to_area, node_to_region, areas) -> dict:
        """从节点级 next_positions 推导跨 region 的边. 不依赖人工标注."""
        edges = []
        seen_pairs = set()
        for r in regions:
            for nid in r.node_ids:
                a_id = region_to_area.get(r.region_id)
                node_dir = self.root / "areas" / a_id / "regions" / r.region_id / "nodes" / nid
                info_path = node_dir / "node_position_info.json"
                if not info_path.exists():
                    continue
                try:
                    info = json.loads(info_path.read_text())
                except Exception:
                    continue
                for ni, np_ in enumerate(info.get("next_positions", [])):
                    target_node = str(np_.get("position_id") or "")
                    target_region = node_to_region.get(target_node)
                    if not target_region or target_region == r.region_id:
                        continue
                    pair_key = tuple(sorted([r.region_id, target_region]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    target_area = region_to_area.get(target_region)
                    tags = []
                    if a_id != target_area:
                        # 跨 area 自动 tag
                        kind_u = self._area_kind(a_id, areas)
                        kind_v = self._area_kind(target_area, areas)
                        if {kind_u, kind_v} & {"transition"}:
                            tags.append("transition")
                        if kind_u == "indoor_floor" and kind_v == "outdoor_zone":
                            tags.append("indoor_outdoor")
                        elif kind_u == "outdoor_zone" and kind_v == "indoor_floor":
                            tags.append("outdoor_indoor")
                        tags.append("cross_area")
                    edges.append({
                        "u": r.region_id,
                        "v": target_region,
                        "direction": "bidirectional",
                        "exit_node_id": nid,
                        "exit_camera": np_.get("camera_name"),
                        "exit_next_idx": ni,
                        "enter_node_id": target_node,
                        "enter_camera": None,
                        "enter_next_idx": None,
                        "edge_weight": 1.0,
                        "confidence": 1.0,
                        "tags": tags,
                    })
        # 检测闭环 (起+终 node 物理相同位置 → tag=loop_close)
        # 简化: 若边 (u,v) 存在 且 u 内最早 node 与 v 内最晚 node 时序相距大 → loop
        # 暂不做自动 loop_close 检测, 留给后续手动审计
        return {
            "version": SCHEMA_VERSION,
            "directed": False,
            "edges": edges,
        }
