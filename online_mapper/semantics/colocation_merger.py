"""ColocationMerger — merge nodes that are physically co-located.

Two nodes are considered co-located if ANY of:
  * frame_idx gap < FRAME_GAP_THRESHOLD (e.g. 5)
  * VPR cyclic-cosine similarity >= VPR_SIM_THRESHOLD (e.g. 0.70)
  * spatial pose distance <= SPATIAL_DIST_THRESHOLD (meters)

When merging, the surviving node is chosen by category priority:
  SHOP > ROOM_NAMED > ROOM_NUMBERED > FUNCTION_AREA > LANDMARK_FACILITY
  > JUNCTION_CROSS > JUNCTION_T > REJECT

The merged node's name combines both, e.g. "DEEPROUTE.AI前台" (priority
name + secondary name as suffix).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np

from online_mapper.semantics.node_category import NodeCategory

logger = logging.getLogger(__name__)


# Higher rank = higher priority (kept as anchor in merge)
# 调整后: 功能区/房间作为节点的"语义角色"优先级最高,
# SHOP 仅作品牌标识 (organization) 附加, 不应抢占 anchor 类别.
_CATEGORY_RANK = {
    NodeCategory.ROOM_NAMED.value:        7,
    NodeCategory.ROOM_NUMBERED.value:     6,
    NodeCategory.FUNCTION_AREA.value:     5,
    NodeCategory.LANDMARK_FACILITY.value: 4,
    NodeCategory.SHOP.value:              3,
    NodeCategory.JUNCTION_CROSS.value:    2,
    NodeCategory.JUNCTION_T.value:        1,
    NodeCategory.REJECT.value:            0,
}


@dataclass
class MergeStats:
    pairs_examined: int = 0
    merges: int = 0
    by_reason: Dict[str, int] = None
    aliases: Dict[str, str] = None

    def __post_init__(self):
        if self.by_reason is None:
            self.by_reason = {"frame+spatial": 0, "vpr": 0}
        if self.aliases is None:
            self.aliases = {}


class ColocationMerger:
    FRAME_GAP_THRESHOLD = 8       # 帧间隔上限 (与 spatial AND'd)
    VPR_SIM_THRESHOLD = 0.85       # VPR 相似度门槛 (强信号, 单独触发合并)
    SPATIAL_DIST_THRESHOLD = 0.5   # meters (与 frame_gap AND'd)

    def __init__(self,
                 frame_gap: int = None,
                 vpr_sim: float = None,
                 spatial_dist: float = None):
        if frame_gap is not None:
            self.FRAME_GAP_THRESHOLD = frame_gap
        if vpr_sim is not None:
            self.VPR_SIM_THRESHOLD = vpr_sim
        if spatial_dist is not None:
            self.SPATIAL_DIST_THRESHOLD = spatial_dist
        self.stats = MergeStats()

    @staticmethod
    def _cyclic_cosine(fa: Dict[str, np.ndarray], fb: Dict[str, np.ndarray]) -> float:
        cams = ['camera_1', 'camera_2', 'camera_3', 'camera_4']
        if not all(c in fa and c in fb for c in cams):
            return 0.0
        best = -1.0
        for shift in range(4):
            sims = []
            for i, c in enumerate(cams):
                cb = cams[(i + shift) % 4]
                a = fa[c]; b = fb[cb]
                s = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                sims.append(s)
            best = max(best, sum(sims) / 4.0)
        return best

    def _should_merge(self, node_a, node_b, feats_a, feats_b,
                      pose_a, pose_b) -> Tuple[bool, Optional[str]]:
        # 强信号: VPR 相似度极高 → 合并
        sim = 0.0
        if feats_a and feats_b:
            sim = self._cyclic_cosine(feats_a, feats_b)
            if sim >= self.VPR_SIM_THRESHOLD:
                return True, "vpr"
        # 强信号: 帧间隔小 + 空间距离小 (AND, 防止 VO 不准导致误合并)
        frame_gap = abs(node_a.frame_idx - node_b.frame_idx)
        if pose_a and pose_b and frame_gap <= self.FRAME_GAP_THRESHOLD:
            dx = pose_a[0] - pose_b[0]; dy = pose_a[1] - pose_b[1]
            dist = float(np.hypot(dx, dy))
            if dist <= self.SPATIAL_DIST_THRESHOLD:
                return True, "frame+spatial"
        return False, None

    @staticmethod
    def _rank(node) -> int:
        cat = getattr(node, "category", None) or NodeCategory.REJECT.value
        return _CATEGORY_RANK.get(cat, 0)

    @staticmethod
    def _combined_name(anchor, other) -> Tuple[str, str]:
        """结构化合并 anchor + other 的命名 (新方案).

        通过 NodeName.merge_names 完成 category / organization /
        nearby_plates 的融合, 写回 anchor.name_struct, 然后 render
        display_cn / display_en. 不再做字符串拼接.
        """
        from online_mapper.semantics.node_naming import NodeName, merge_names
        a = getattr(anchor, "name_struct", None)
        b = getattr(other, "name_struct", None)
        # 兼容: 没有 name_struct 时回退到 anchor 原 position_name
        if a is None and b is None:
            cn = getattr(anchor, "position_name", "") or ""
            return cn, getattr(anchor, "position_name_eng", "") or cn
        if a is None:
            a = NodeName(category=getattr(anchor, "position_name", "") or "",
                         category_en=getattr(anchor, "position_name_eng", "") or "")
            anchor.name_struct = a
        if b is None:
            b = NodeName(category=getattr(other, "position_name", "") or "",
                         category_en=getattr(other, "position_name_eng", "") or "")
        merge_names(a, b)
        return a.display_cn(), a.display_en()

    def merge(self,
              nodes: Dict,           # node_id -> TopoNode
              node_features: Dict,
              pose_graph,
              ) -> Dict[str, str]:
        """Returns alias_map: {merged_node_id -> anchor_node_id}.

        Caller is responsible for actually deleting merged nodes from
        topology / pose_graph / scene_graph and re-pointing edges.
        """
        ids = list(nodes.keys())
        n = len(ids)
        alias: Dict[str, str] = {}
        # Iterate by frame_idx ascending, merge greedily
        ordered = sorted(ids, key=lambda i: nodes[i].frame_idx)
        active: List[str] = []
        for nid in ordered:
            if nid in alias:
                continue
            node = nodes[nid]
            feats = node_features.get(nid)
            pose = None
            if pose_graph and nid in pose_graph.nodes:
                pn = pose_graph.nodes[nid]
                pose = (pn.x, pn.y)
            merged_into = None
            for other_id in active:
                if other_id in alias:
                    continue
                self.stats.pairs_examined += 1
                other = nodes[other_id]
                feats_o = node_features.get(other_id)
                pose_o = None
                if pose_graph and other_id in pose_graph.nodes:
                    pn = pose_graph.nodes[other_id]
                    pose_o = (pn.x, pn.y)
                ok, reason = self._should_merge(
                    node, other, feats, feats_o, pose, pose_o)
                if not ok:
                    continue
                # Determine anchor by rank
                if self._rank(node) > self._rank(other):
                    anchor, sub = node, other
                    anchor_id, sub_id = nid, other_id
                else:
                    anchor, sub = other, node
                    anchor_id, sub_id = other_id, nid
                # Combine names on anchor
                new_cn, new_en = self._combined_name(anchor, sub)
                anchor.position_name = new_cn
                anchor.position_name_eng = new_en
                # 锚 frame_idx 取两者较小的 (代表机器人首次到达该位置的时刻)
                # 这样 trajectory 时间序排序时, 合并节点反映"首次发现"
                try:
                    anchor.frame_idx = min(anchor.frame_idx, sub.frame_idx)
                except Exception:
                    pass
                alias[sub_id] = anchor_id
                self.stats.merges += 1
                self.stats.by_reason[reason] += 1
                self.stats.aliases[sub_id] = anchor_id
                logger.info(
                    f"[ColocMerge] {sub_id}({getattr(sub,'position_name','')}) "
                    f"-> {anchor_id}({new_cn}) reason={reason}")
                merged_into = anchor_id
                break
            if merged_into is None:
                active.append(nid)
        return alias
