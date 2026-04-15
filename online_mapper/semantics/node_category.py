"""NodeCategoryClassifier — whitelist/reject node candidates by category.

Goal: only build a node when it represents one of the high-quality
categories the user wants in the final topology graph:

  1. Path structure: junction (cross / T)
  2. Identified function area: numbered room, named meeting room,
     printer area, reception, mother-baby room, tea pantry, etc.
  3. Storefront: brand sign / shop name
  4. Landmark facility: elevator, stairs, restroom, electrical closet, etc.

Rejected: bare corridors, decorative walls (绿植墙/sofa/chairs), random
mid-trajectory positions without any identifying feature.

Decision is made from:
  - The plate/text claims that have already passed multi-frame voting
    + second-pass Qwen verification
  - The Qwen describe_scene candidate (only if it matches a whitelist
    keyword AND scene-verify passed)
  - The junction kind detected from 4-camera depth
  - The GD landmark labels (treated as weak signal: chair / plant / sofa
    push toward REJECT, door / sign push toward ACCEPT)
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Set


class NodeCategory(Enum):
    JUNCTION_CROSS = "junction_cross"
    JUNCTION_T = "junction_t"
    ROOM_NUMBERED = "room_numbered"        # 101 / 10号 / A301 etc.
    ROOM_NAMED = "room_named"               # 纽布里茨会议室 / 母婴室
    FUNCTION_AREA = "function_area"         # 打印区 / 前台 / 茶水间
    SHOP = "shop"                           # storefront with brand sign
    LANDMARK_FACILITY = "landmark_facility" # 电梯 / 楼梯 / 卫生间 / 强电井
    BUILDING_LANDMARK = "building_landmark" # A座 / B座入口 / 13号楼 / D栋 — 园区楼栋标识
    REJECT = "reject"


# ---------------------------------------------------------------------------
# WHITELIST: whitelist patterns + canonical Chinese name
# ---------------------------------------------------------------------------

# Function-area whitelist: keyword (substring match) -> canonical name
FUNCTION_AREA_WHITELIST = {
    "打印":   "打印区",
    "复印":   "打印区",
    "打印区": "打印区",
    "前台":   "前台",
    "接待":   "前台",
    "reception": "前台",
    "茶水":   "茶水间",
    "茶歇":   "茶水间",
    "水吧":   "茶水间",
    "母婴":   "母婴室",
    "哺乳":   "母婴室",
    "关爱":   "关爱室",
    "关爱室": "关爱室",
    "care":   "关爱室",
    "母婴室": "母婴室",
    "休息":   "休息区",
    "lounge": "休息区",
    "休闲":   "休息区",
    "零食":   "零食区",
    "snack":  "零食区",
    "工位":   "工位区",
    "工位区": "工位区",
    # 中文规范名自匹配 (Qwen describe_scene 直接返回中文 canonical)
    # 注意: 不收 '办公区' / '办公' — 办公区太泛化, 不适合作导航节点
    "休息区": "休息区",
    "零食区": "零食区",
    "前台区": "前台",
    # 园区/楼宇公共区 (campus-level lobbies and service points)
    "快递柜":   "快递柜区",
    "快递柜区": "快递柜区",
    "外卖柜":   "外卖柜区",
    "外卖寄存": "外卖柜区",
    "智能取餐柜": "外卖柜区",
    "储物柜":   "储物柜区",
    "储物柜区": "储物柜区",
    "locker":   "储物柜区",
}

# Landmark facility whitelist
# 只保留真正有导航价值的公共设施; 电井/配电间/消防 之类的机房不建节点
LANDMARK_FACILITY_WHITELIST = {
    "电梯":   "电梯口",
    "elevator": "电梯口",
    "lift":   "电梯口",
    "楼梯":   "楼梯口",
    "stair":  "楼梯口",
    "卫生间": "卫生间",
    "洗手间": "卫生间",
    "厕所":   "卫生间",
    "toilet": "卫生间",
    "wc":     "卫生间",
    # 园区可见的服务/住宿设施 (整栋建筑级地标, 比楼栋标识更具体)
    "酒店":   "酒店",
    "hotel":  "酒店",
    "前台大堂": "大堂",     # 仅当 plate 明确写"前台大堂"才命中, 不收单独的"大堂"
}

# Building-landmark patterns: 园区中 'X座' / 'X座入口' / 'X号楼' / 'X栋'
# 不同于其它白名单, 这里 canonical 名称就是原文本身(每栋楼是唯一的).
# 见 match_building_landmark.
BUILDING_LANDMARK_PATTERNS = [
    re.compile(r"^[A-Za-z]座(入口|大堂)?$"),       # A座 / B座入口 / C座大堂
    re.compile(r"^[0-9]{1,3}号楼(入口|大堂)?$"),    # 13号楼 / 1号楼入口
    re.compile(r"^[A-Za-z]栋(入口|大堂)?$"),       # D栋 / D栋入口
    re.compile(r"^[0-9]{1,3}栋(入口|大堂)?$"),      # 5栋
]

# Meeting room indicator
MEETING_ROOM_KEYWORDS = ["会议室", "meeting", "conference"]

# Canonical CN -> EN translation map (used when assigning position_name_eng)
CN_EN_MAP = {
    "打印区":     "Printing Area",
    "前台":       "Reception",
    "茶水间":     "Tea Room",
    "母婴室":     "Mother-Baby Room",
    "关爱室":     "Care Room",
    "休息区":     "Lounge",
    "零食区":     "Snack Area",
    "工位区":     "Workspace",
    "办公区":     "Office Area",
    "电梯口":     "Elevator",
    "楼梯口":     "Stairs",
    "卫生间":     "Restroom",
    "强电井":     "Electrical Closet",
    "弱电井":     "Network Closet",
    "配电间":     "Power Room",
    "水井间":     "Water Room",
    "消防设施":   "Fire Equipment",
    "十字路口":   "Cross Junction",
    "丁字路口":   "T Junction",
    # 园区/服务点扩展
    "快递柜区":   "Parcel Lockers",
    "外卖柜区":   "Food Lockers",
    "储物柜区":   "Storage Lockers",
    "酒店":       "Hotel",
    "大堂":       "Lobby",
}


def cn_to_en(cn_name: str) -> str:
    """Translate a canonical Chinese name to English. Preserves _N suffix."""
    if not cn_name:
        return ""
    import re
    m = re.match(r"^(.*?)(_\d+)?$", cn_name.strip())
    base = m.group(1) if m else cn_name
    suffix = m.group(2) or ""
    if base in CN_EN_MAP:
        return CN_EN_MAP[base] + suffix
    # Brand / Latin pass-through
    if all((c.isascii() and (c.isalnum() or c in ".-_ &'")) for c in base):
        return base + suffix
    # Fallback: keep CN
    return cn_name

# REJECT-list (decorative / structural-only items)
REJECT_KEYWORDS = {
    "绿植", "植物", "盆栽", "花盆",
    "椅子", "凳子", "沙发", "椅", "chair", "stool", "sofa",
    "桌", "table", "desk",
    "装饰", "画", "海报", "poster", "decoration",
    "墙", "wall",
    "地毯", "carpet", "rug",
    "screen", "monitor", "屏幕",
    "电视", "tv",
    "白板", "whiteboard",
    "门", "door",  # bare "door" without other context
    # 公司文化墙 / 标语类 (常见 hallucination 来源)
    "文化", "标语", "口号", "原则", "slogan", "culture",
    # 通用泛指词 (不算具体 plate)
    "标识", "房间标识", "网络出口",
}

# Numbered-room patterns (recognise legitimate room numbers)
ROOM_NUMBER_PATTERNS = [
    re.compile(r"^([A-Z]?\d{1,4})号?(房间|室|会议室|教室|办公室)?$"),
    re.compile(r"^[A-Z]\d{1,4}$"),
    re.compile(r"^\d{1,4}[号室]$"),
]


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def has_room_number(text: str) -> Optional[str]:
    """Return the matched room-number string or None.

    严格 fullmatch: 防止 '13号楼' / '2号柜' / '11层电梯' 之类的非房间数字串
    被当成 ROOM_NUMBERED. 'X号' 后缀必须为空或房间相关词.
    """
    if not text:
        return None
    t = text.strip()
    # Quick digit check
    if not any(c.isdigit() for c in t):
        return None
    # Pure 3-4 digit room number (101 / 1023)
    if re.fullmatch(r"\d{3,4}", t):
        return t
    # X号 / X号房间 / X号室 / X号会议室 / X号教室 / X号办公室 / X号工作室
    if re.fullmatch(r"[0-9]{1,4}号(房|室|房间|会议室|教室|办公室|工作室)?", t):
        return t
    # Letter+digit (A301 / B12)
    if re.fullmatch(r"[A-Za-z]\d{1,4}", t):
        return t
    return None


def match_building_landmark(text: str) -> Optional[str]:
    """Return the canonical building-landmark text or None.

    匹配 'A座' / 'B座入口' / '13号楼' / 'D栋' 这种园区级楼栋标识.
    canonical 名称就是原文 (每栋楼是独一无二的, 不做归一化).
    """
    if not text:
        return None
    t = text.strip()
    for pat in BUILDING_LANDMARK_PATTERNS:
        if pat.fullmatch(t):
            return t
    return None


def is_meeting_room(text: str) -> bool:
    if not text:
        return False
    t = _norm(text)
    return any(k in t for k in MEETING_ROOM_KEYWORDS)


def match_function_area(text: str) -> Optional[str]:
    if not text:
        return None
    t = _norm(text)
    for kw, canon in FUNCTION_AREA_WHITELIST.items():
        if kw in t:
            return canon
    return None


def match_landmark_facility(text: str) -> Optional[str]:
    if not text:
        return None
    t = _norm(text)
    for kw, canon in LANDMARK_FACILITY_WHITELIST.items():
        if kw in t:
            return canon
    return None


def matches_reject(text: str) -> bool:
    if not text:
        return False
    t = _norm(text)
    return any(kw in t for kw in REJECT_KEYWORDS)


# ---------------------------------------------------------------------------
# JunctionKind (mirrors junction_detector.py output)
# ---------------------------------------------------------------------------
class JunctionKind(Enum):
    CROSS = "cross"          # 4 open directions
    T_JUNCTION = "t"         # 3 open
    CORRIDOR = "corridor"    # 2 opposite open
    DEAD_END = "dead_end"    # <=1 open
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
@dataclass
class CategoryDecision:
    category: NodeCategory
    final_name_cn: str
    final_name_en: str
    reason: str


class NodeCategoryClassifier:
    """Decide whether a node candidate is worth keeping and pick its name.

    The decision tree:
      A. Verified plate text contains a room number  -> ROOM_NUMBERED
      B. Verified plate text matches meeting-room keyword (and is not a
         number)                                       -> ROOM_NAMED
      C. Verified plate text or describe_scene matches function-area
         whitelist                                     -> FUNCTION_AREA
      D. Verified plate text matches landmark-facility whitelist
                                                       -> LANDMARK_FACILITY
      E. Verified plate text is a "shop-like" Latin/CJK proper noun (>=4
         chars, no reject keywords)                   -> SHOP
      F. No semantic, but junction == CROSS / T       -> JUNCTION_*
      G. Otherwise                                    -> REJECT
    """

    def classify(
        self,
        plate_text: Optional[str] = None,
        plate_text_verified: bool = False,
        scene_describe: Optional[str] = None,
        scene_verified: bool = False,
        junction_kind: JunctionKind = JunctionKind.UNKNOWN,
        gd_landmark: Optional[str] = None,
    ) -> CategoryDecision:
        # Pre-pass: explicit reject of GD landmark like "white chair", "plant"
        gd_reject = gd_landmark and matches_reject(gd_landmark)

        # ---- A. Room number (highest priority) ----
        if plate_text_verified and plate_text:
            num = has_room_number(plate_text)
            if num:
                if "号" not in num and "室" not in num and num.isdigit():
                    canon = f"{num}号房间"
                else:
                    canon = num
                return CategoryDecision(
                    NodeCategory.ROOM_NUMBERED, canon,
                    cn_to_en(canon),
                    reason=f"room_number_match plate='{plate_text}'")

        # ---- B. Named meeting room ----
        if plate_text_verified and plate_text and is_meeting_room(plate_text):
            cn = plate_text.strip()
            return CategoryDecision(
                NodeCategory.ROOM_NAMED, cn, cn_to_en(cn),
                reason=f"meeting_room plate='{plate_text}'")

        # ---- C. Function area (from plate OR scene_describe) ----
        if plate_text_verified and plate_text:
            fa = match_function_area(plate_text)
            if fa:
                return CategoryDecision(
                    NodeCategory.FUNCTION_AREA, fa, cn_to_en(fa),
                    reason=f"function_area plate='{plate_text}'")
        if scene_describe and scene_verified:
            fa = match_function_area(scene_describe)
            if fa:
                return CategoryDecision(
                    NodeCategory.FUNCTION_AREA, fa, cn_to_en(fa),
                    reason=f"function_area scene='{scene_describe}'")

        # ---- D. Landmark facility ----
        if plate_text_verified and plate_text:
            lf = match_landmark_facility(plate_text)
            if lf:
                return CategoryDecision(
                    NodeCategory.LANDMARK_FACILITY, lf, cn_to_en(lf),
                    reason=f"landmark_facility plate='{plate_text}'")
        if scene_describe and scene_verified:
            lf = match_landmark_facility(scene_describe)
            if lf:
                return CategoryDecision(
                    NodeCategory.LANDMARK_FACILITY, lf, cn_to_en(lf),
                    reason=f"landmark_facility scene='{scene_describe}'")

        # ---- D2. Generic "X室" rule (any 2-6 char CJK ending with 室 that
        #         is not in REJECT). Matches things like 会客室 / 储物室 /
        #         任何陌生功能室. Treat as ROOM_NAMED. ----
        if plate_text_verified and plate_text:
            t = plate_text.strip()
            if (2 <= len(t) <= 6 and t.endswith("室")
                    and not matches_reject(t)
                    and all('\u4e00' <= c <= '\u9fff' for c in t)):
                return CategoryDecision(
                    NodeCategory.ROOM_NAMED, t, cn_to_en(t) if t in CN_EN_MAP else t,
                    reason=f"generic_X室 plate='{t}'")

        # ---- D3. Building landmark: X座 / X座入口 / X号楼 / X栋 ----
        # 园区级楼栋标识. canonical 即原文 (每栋楼独一无二). 必须放在 SHOP
        # 之前: 'D栋' 含 Latin 字母, 否则会被 E 段 SHOP 误抢.
        if plate_text_verified and plate_text:
            bl = match_building_landmark(plate_text)
            if bl:
                return CategoryDecision(
                    NodeCategory.BUILDING_LANDMARK, bl, bl,
                    reason=f"building_landmark plate='{plate_text}'")

        # ---- E. Shop / proper-noun storefront ----
        if plate_text_verified and plate_text:
            t = plate_text.strip()
            if (re.fullmatch(r"[A-Za-z][A-Za-z0-9\.\- &']{3,30}", t)
                    and not matches_reject(t)):
                return CategoryDecision(
                    NodeCategory.SHOP, t, t,
                    reason=f"shop_brand plate='{t}'")

        # ---- F. Junction-only node ----
        if not gd_reject:
            if junction_kind == JunctionKind.CROSS:
                return CategoryDecision(
                    NodeCategory.JUNCTION_CROSS, "十字路口", "Cross Junction",
                    reason="junction=CROSS")
            if junction_kind == JunctionKind.T_JUNCTION:
                return CategoryDecision(
                    NodeCategory.JUNCTION_T, "丁字路口", "T Junction",
                    reason="junction=T")

        # ---- G. Reject everything else ----
        return CategoryDecision(
            NodeCategory.REJECT, "", "",
            reason=f"no_whitelist_match plate='{plate_text}' "
                   f"scene='{scene_describe}' junction={junction_kind.value} "
                   f"gd='{gd_landmark}' gd_reject={gd_reject}")
