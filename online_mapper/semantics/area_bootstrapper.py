"""AreaBootstrapper — 根据 dataset_kind 与 plate 关键词把 regions 归类到 areas (R3 重构).

输入:
  cfg.dataset_kind: "indoor" | "outdoor" | "mixed" | "auto"
  regions: List[RegionInfo] (来自 RegionClusterer)
  nodes: Dict[node_id -> TopoNode]
输出:
  List[AreaInfo] (每个 area 含 area_id/kind/regions/...)

策略:
  - dataset_kind == "indoor"  → 单 area, kind=indoor_floor, 全部 regions 归属
  - dataset_kind == "outdoor" → 单 area, kind=outdoor_zone
  - dataset_kind == "mixed" / "auto" → plate 关键词分组:
      "A 座/B 栋/X 楼/前台/电梯/会议室/卫生间" → indoor_floor area
      "广场/园区路/停车场/路段/园区/中庭" → outdoor_zone area
      "大堂/lobby/出入口/玻璃门" → transition area
      未匹配 → 默认 indoor_floor (兜底)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AreaInfo:
    area_id: str
    name: str
    name_eng: str
    kind: str  # indoor_floor | outdoor_zone | transition
    region_ids: List[str] = field(default_factory=list)
    parent_area_id: Optional[str] = None
    site_id: str = "default"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    color_hex: Optional[str] = None
    # —— 室内专用 ——
    building: Optional[str] = None
    floor: Optional[str] = None
    elevation_m: Optional[float] = None
    # —— 室外专用 ——
    outdoor_type: Optional[str] = None
    bbox_2d: Optional[List[float]] = None


# Plate 关键词 → area kind 映射 (仅在 mixed/auto 模式下使用)
_INDOOR_KEYWORDS = (
    "座", "栋", "楼", "前台", "接待", "会议室", "卫生间", "洗手间", "茶水间",
    "电梯", "楼梯", "母婴", "强电", "弱电", "复印", "打印", "档案", "办公",
    "展厅", "休闲", "deeproute", "deepro", "i have", "exhioh", "hioh"
)
_OUTDOOR_KEYWORDS = (
    "广场", "园区路", "园区", "停车", "路段", "中庭", "花园", "庭院", "户外",
    "outdoor", "plaza", "parking", "garden", "yard"
)
_TRANSITION_KEYWORDS = (
    "大堂", "玻璃门", "门厅", "lobby", "entrance", "出入口"
)


class AreaBootstrapper:
    def __init__(self, dataset_kind: str = "indoor", site_id: str = "default"):
        if dataset_kind not in ("indoor", "outdoor", "mixed", "auto"):
            logger.warning(f"unknown dataset_kind={dataset_kind!r}, fallback to 'indoor'")
            dataset_kind = "indoor"
        self.dataset_kind = dataset_kind
        self.site_id = site_id

    def bootstrap(self, regions, nodes: Dict) -> List[AreaInfo]:
        """
        Args:
            regions: List[RegionInfo]
        Returns:
            List[AreaInfo]; 每个 area 的 region_ids 是拥有的 region 列表
        """
        if not regions:
            return []
        if self.dataset_kind in ("indoor", "outdoor"):
            kind = "indoor_floor" if self.dataset_kind == "indoor" else "outdoor_zone"
            area = AreaInfo(
                area_id=f"{self.site_id}__{self.dataset_kind}",
                name=("默认楼层" if kind == "indoor_floor" else "默认室外区"),
                name_eng=(f"Default {kind.replace('_', ' ').title()}"),
                kind=kind,
                site_id=self.site_id,
                region_ids=[r.region_id for r in regions],
                description=f"自动 bootstrap 单 {kind} (dataset_kind={self.dataset_kind})",
                tags=[self.dataset_kind],
            )
            if kind == "indoor_floor":
                area.building = "default"
                area.floor = "F1"
            return [area]

        # mixed / auto: plate 关键词分组
        kind_buckets: Dict[str, List[str]] = {"indoor_floor": [], "outdoor_zone": [], "transition": []}
        for r in regions:
            kind = self._classify_region(r, nodes)
            kind_buckets[kind].append(r.region_id)

        areas: List[AreaInfo] = []
        for kind, region_ids in kind_buckets.items():
            if not region_ids:
                continue
            a = AreaInfo(
                area_id=f"{self.site_id}__{kind}",
                name=self._kind_name_cn(kind),
                name_eng=kind.replace("_", " ").title(),
                kind=kind,
                site_id=self.site_id,
                region_ids=region_ids,
                description=f"自动 bootstrap (dataset_kind={self.dataset_kind})",
                tags=[kind],
            )
            if kind == "indoor_floor":
                a.building = "default"; a.floor = "F1"
            areas.append(a)
        return areas

    def _classify_region(self, region, nodes: Dict) -> str:
        """根据 region 主导 plate / category 关键词判 indoor / outdoor / transition."""
        keywords = []
        if region.dominant_plate:
            keywords.append(region.dominant_plate.lower())
        if region.dominant_category:
            keywords.append(region.dominant_category.lower())
        for nid in region.node_ids:
            n = nodes.get(nid)
            if n:
                pn = (getattr(n, "position_name", "") or "").lower()
                if pn:
                    keywords.append(pn)
        joined = " ".join(keywords)
        # transition 优先 (兼具 indoor + outdoor)
        if any(kw in joined for kw in _TRANSITION_KEYWORDS):
            return "transition"
        in_score = sum(1 for kw in _INDOOR_KEYWORDS if kw in joined)
        out_score = sum(1 for kw in _OUTDOOR_KEYWORDS if kw in joined)
        if out_score > in_score:
            return "outdoor_zone"
        return "indoor_floor"  # 默认 indoor

    @staticmethod
    def _kind_name_cn(kind: str) -> str:
        return {
            "indoor_floor": "室内区",
            "outdoor_zone": "室外区",
            "transition": "室内外过渡区",
        }.get(kind, kind)
