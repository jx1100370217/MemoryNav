"""结构化节点命名 (替代字符串拼接).

设计目标:
1. 把 node 的"名字"从一个 string 升级为结构化字段:
     category       (主类型: 前台/强电井/关爱室/...)
     organization   (主关联实体: DEEPROUTE.AI/NEUMANN/...)
     nearby_plates  (同 node 看到的其他门牌, 仅作语境)
     nearby_landmarks (显著物体)
     instance_suffix (全局重名后的 _2/_3)
2. display_name 由 (category, organization, suffix) 拼接, 用 · 分隔避免
   "DEEPROUTE.AI前台" 这种粘连.
3. 同一节点 4 相机看到多个候选门牌时, 通过 select_organization 选 1 个
   作主标识, 其余进 nearby_plates.
4. 全局唯一性以 (category, organization) 元组为键, 重复加 _N 后缀.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

# 品牌/专有名词模式 (Latin 大写起头, 含数字/点/连字符等)
_BRAND_RE = re.compile(r"^[A-Z][A-Za-z0-9\.\- &']{2,30}$")

# semantic role priority: 高的更"主": 房间名/功能区 > 路口 > 设施 > 店面 > REJECT
# 在 coloc merge 中决定 category 由谁主导 (organization 仍然单独追踪)
_SEMANTIC_RANK = {
    "room_named":         70,
    "room_numbered":      65,
    "function_area":      60,
    "landmark_facility":  50,
    "junction_cross":     30,
    "junction_t":         25,
    "shop":               20,   # 单独的 shop 节点优先级低于功能区, 鼓励作为 organization 附加
    "reject":              0,
}


@dataclass
class NodeName:
    """节点命名的结构化表示, 可序列化."""
    category: str = ""              # 中文 canonical category, 来自 NodeCategoryClassifier
    category_en: str = ""
    organization: str = ""          # 主关联实体 (品牌或门牌主名), 保留原文
    nearby_plates: List[str] = field(default_factory=list)
    nearby_landmarks: List[str] = field(default_factory=list)
    instance_suffix: str = ""       # 全局重名后缀

    # ------------------------------------------------------------------
    def display_cn(self) -> str:
        if not self.category and not self.organization:
            return f"未命名{self.instance_suffix}"
        if self.organization and self.category and self.organization != self.category:
            base = f"{self.category}·{self.organization}"
        else:
            base = self.category or self.organization
        return f"{base}{self.instance_suffix}"

    def display_en(self) -> str:
        if not self.category_en and not self.organization:
            return f"Unnamed{self.instance_suffix}"
        if self.organization and self.category_en and self.organization != self.category_en:
            base = f"{self.category_en} · {self.organization}"
        else:
            base = self.category_en or self.organization
        return f"{base}{self.instance_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "display_name":     self.display_cn(),
            "display_name_eng": self.display_en(),
            "category":         self.category,
            "category_eng":     self.category_en,
            "organization":     self.organization,
            "nearby_plates":    list(self.nearby_plates),
            "nearby_landmarks": list(self.nearby_landmarks),
            "instance_suffix":  self.instance_suffix,
        }

    def dedup_key(self) -> Tuple[str, str]:
        """全局重名检测键 — 同 category + 同 organization 算重复."""
        return (self.category, self.organization)


# ==========================================================================
def is_brand_like(text: str) -> bool:
    """Latin 专有名词 / 品牌名 (DEEPROUTE.AI, NEUMANN). 中文/通用词不算."""
    if not text:
        return False
    return bool(_BRAND_RE.match(text.strip()))


def select_organization(
    plate_observations: List[Dict[str, Any]],
    category: str = "",
) -> Tuple[Optional[str], List[str]]:
    """从一组同节点的 plate 观测中选 1 个作 organization, 其余进 nearby_plates.

    Args:
        plate_observations: list of dict, 每项至少含 keys:
            text  (str)        : OCR/Qwen 文本
            camera (str)       : "camera_1" / "camera_2" / ...
            area  (float)      : 像素 bbox 面积
            votes (int)        : voter 中累计票数
        category: 当前节点 category, 用于将来扩展(语义匹配 boost)

    Returns:
        (organization_text_or_None, other_plate_texts_unique)
    """
    if not plate_observations:
        return None, []

    def score(o: Dict[str, Any]) -> float:
        text = (o.get("text") or "").strip()
        s = 0.0
        # brand-like 强加分 (公司名/品牌通常是节点的"主标识")
        if is_brand_like(text):
            s += 100
        # 前向相机优先 (camera_1 通常是当前所在房间)
        if o.get("camera") == "camera_1":
            s += 30
        # bbox 面积 (近的优于远的)
        s += min(20.0, (o.get("area", 0) / 50000.0) * 20.0)
        # 投票次数 (持续观测优于偶发)
        s += min(20.0, o.get("votes", 1) * 4.0)
        return s

    ranked = sorted(plate_observations, key=score, reverse=True)
    org = (ranked[0].get("text") or "").strip() or None
    nearby: List[str] = []
    seen = set([org])
    for o in ranked[1:]:
        t = (o.get("text") or "").strip()
        if t and t not in seen:
            nearby.append(t)
            seen.add(t)
    return org, nearby


# ==========================================================================
def merge_names(anchor: NodeName, other: NodeName) -> None:
    """把 other 的命名信息融入 anchor (in-place).

    规则:
    1. category 取 _SEMANTIC_RANK 更高的一方 (功能区/房间 > 店面)
    2. organization 优先选 brand-like 的; 都 brand-like 时保留 anchor 的
    3. nearby_plates / nearby_landmarks 取并集 (去重, 排除当前 organization)
    """
    if other is None:
        return
    a_rank = _SEMANTIC_RANK.get(anchor.category, 0) if anchor.category else 0
    o_rank = _SEMANTIC_RANK.get(other.category, 0) if other.category else 0
    if o_rank > a_rank:
        anchor.category = other.category
        anchor.category_en = other.category_en

    # organization 选取
    cands = [c for c in (anchor.organization, other.organization) if c]
    brands = [c for c in cands if is_brand_like(c)]
    if brands:
        anchor.organization = brands[0]
    elif cands:
        anchor.organization = cands[0]

    # 合并 nearby
    seen_p = set([anchor.organization]) if anchor.organization else set()
    seen_p.update(anchor.nearby_plates)
    for src in (other.nearby_plates,
                [other.organization] if other.organization else []):
        for t in src:
            if t and t not in seen_p:
                anchor.nearby_plates.append(t)
                seen_p.add(t)

    seen_l = set(anchor.nearby_landmarks)
    for t in other.nearby_landmarks:
        if t and t not in seen_l:
            anchor.nearby_landmarks.append(t)
            seen_l.add(t)


# ==========================================================================
def resolve_global_uniqueness(name_structs: Dict[str, NodeName]) -> None:
    """对 {node_id -> NodeName} 全局去重: 同 (category, organization) 加 _N (in-place).

    后缀按 node_id 升序分配 (调用方应已按 frame_idx 排序传入).
    """
    by_key: Dict[Tuple[str, str], List[str]] = {}
    for nid, name in name_structs.items():
        if not name.category and not name.organization:
            continue
        by_key.setdefault(name.dedup_key(), []).append(nid)

    for key, ids in by_key.items():
        if len(ids) < 2:
            continue
        for i, nid in enumerate(ids, start=1):
            name_structs[nid].instance_suffix = f"_{i}"
