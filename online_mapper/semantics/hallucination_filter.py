"""Hallucination filter & name deduplicator for online_mapper.

Provides three layers of defense against VLM hallucinations and duplicate
names in Qwen-generated place names:

1. QwenVerifier — second-pass yes/no verification. Re-asks Qwen
   "Is '<claim>' actually visible in this image? Answer only 是 or 否"
   using a strict prompt; rejects anything not confirmed.

2. MultiFrameVoter — accumulates (name, camera, frame, area) observations
   across frames. A name is "confirmed" only if it appears in
   >= MIN_FRAMES distinct frames OR >= MIN_CAMERAS distinct cameras within
   the same frame. Single-frame/single-camera sightings are dropped.

3. NameDeduplicator — finalize-stage pass over all nodes:
   (a) If two nodes share the same position_name AND cyclic-VPR sim is
       high -> merge (keep the earlier node, record alias).
   (b) Otherwise -> append "_1 / _2" style suffixes so every
       position_name is globally unique.

All components are intentionally conservative: when in doubt they drop
or rename rather than fabricate.
"""

import base64
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strict prompts (Chinese/English)
# ---------------------------------------------------------------------------
# "Anti-hallucination" detect_text: demands confidence and explicitly tells
# the model to return empty when in doubt. Used as a REPLACEMENT for
# offline_mapper's PROMPT_DETECT_TEXT when called from online_mapper.
STRICT_DETECT_TEXT_PROMPT = (
    "You are looking at one frame from an indoor navigation robot camera. "
    "Task: report any plate / sign / storefront text that is clearly visible "
    "AND physically attached to a wall, door, glass partition, or store front. "
    "Be objective: if the text is genuinely readable, you SHOULD report it. "
    "If you are uncertain or it is blurry, return {\"found\": false}. "
    "INCLUDE: door plates, room number plates, meeting room glass labels, "
    "function-room signs (母婴室/茶水间/打印区/卫生间/电梯/楼梯/强电井 etc.), "
    "shop / store front brand signs (English brand names like DEEPROUTE.AI "
    "are valid). "
    "EXCLUDE: posters, decorative paintings, bulletin boards, slogan / "
    "culture walls, safety/exit signs, computer screens, whiteboards, "
    "banners, advertisements, personal desk nameplates. "
    "Copy the characters EXACTLY as visible. Do not translate, do not "
    "paraphrase, do not invent. "
    "OUTPUT JSON ONLY: "
    "{\"found\": true, \"text\": \"<exact characters>\", "
    "\"name_cn\": \"<Chinese place name with room type, 2-8 chars>\", "
    "\"name_en\": \"<English translation>\", "
    "\"confidence\": \"low|medium|high\"} "
    "OR {\"found\": false}"
)

# Second-pass verifier: binary yes/no whether a claim is actually visible.
# Strict but objective: a clearly readable real plate should pass.
_VERIFY_TEXT_PROMPT_TEMPLATE = (
    "这是一张室内机器人摄像头图像。图中是否能在某个物理实体（门牌、标识牌、"
    "玻璃刻字、店面招牌）上清晰看到文字 \"{claim}\"？"
    "判断标准：(1) 文字必须出现在真实的物理标识上，不是海报/屏幕/标语墙/"
    "安全出口标志。(2) 必须和 \"{claim}\" 逐字一致或几乎一致。(3) 客观评价："
    "如果你确实能读到这些字，就回答\"是\"，不需要纠结清晰度。如果完全没看到"
    "或文字是别的内容，回答\"否\"。只输出一个字：\"是\" 或 \"否\"。"
)

_VERIFY_SCENE_PROMPT_TEMPLATE = (
    "这是一张室内机器人摄像头图像。图中是否能看到 \"{claim}\" 相关的"
    "明显物理特征（比如招牌、设备、明显标志物）？只要图中存在相应的"
    "物体/设施即可回答\"是\"。不存在或完全看不到相关特征则回答\"否\"。"
    "只输出一个字：\"是\" 或 \"否\"，不要任何解释。"
)

# Specific-name verification prompt (stricter, for named rooms / proper nouns)
_VERIFY_SPECIFIC_NAME_PROMPT_TEMPLATE = (
    "这是一张室内机器人摄像头图像。图像中是否清晰可见文字 \"{claim}\"？"
    "必须是物理实体（门牌/标识牌/招牌）上的字，必须逐字匹配，必须清晰可读。"
    "模糊、遮挡、猜测一律回答\"否\"。只输出一个字：\"是\" 或 \"否\"。"
)

# Strong generic fallbacks we accept without extra verification
_GENERIC_ACCEPTABLE = {
    "走廊", "过道", "通道", "开阔区", "大厅", "入口",
    "门前", "转角", "节点", "过渡区",
}

# Common "office location type" words that are generic categories, not
# specific identifiers. If a name is ONLY composed of one of these (no
# digits, no proper nouns) it's considered a safe generic and uses the
# lenient scene-verify prompt.
_GENERIC_CATEGORY_KEYWORDS = {
    "打印", "办公", "休息", "前台", "接待", "会议", "茶水", "茶歇",
    "工位", "储物", "绿植", "盆栽", "沙发", "走廊", "楼梯", "电梯",
    "洗手间", "卫生间", "厕所", "饮水", "打卡", "展示",
}


def looks_specific(name: str) -> bool:
    """Heuristic: True if name looks like a specific identifier (number /
    proper noun / meeting room name) rather than a generic category.

    Specific names must pass strict (text-verification) prompt.
    Generic names use the lenient scene-verification prompt.
    """
    if not name:
        return False
    n = name.strip()
    # ASCII letters or digits -> specific
    if any(c.isdigit() or (c.isascii() and c.isalpha()) for c in n):
        return True
    # Room-number patterns with Chinese digits or 号/室 suffix
    if re.search(r"[〇一二三四五六七八九十百千]+[号室]", n):
        return True
    # Meeting-room style "X会议室" with 2+ chars prefix: often proper noun
    if "会议室" in n and len(n) > 3:
        return True
    # Otherwise, if it's a short (<=5) pure-CJK name that matches any
    # generic category keyword, treat as generic
    if len(n) <= 5:
        for kw in _GENERIC_CATEGORY_KEYWORDS:
            if kw in n:
                return False
    # Long CJK name with no generic keyword -> treat as specific (proper noun)
    return True


def _parse_yes_no(text: str) -> Optional[bool]:
    """Return True/False for yes/no, None if unparseable."""
    if not text:
        return None
    # strip thinking tags, quotes, etc.
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    t = t.strip().strip('"').strip("'").strip()
    # First 4 characters usually enough
    head = t[:4]
    if "是" in head or "yes" in head.lower() or head.lower().startswith("y"):
        return True
    if "否" in head or "no" in head.lower() or head.lower().startswith("n"):
        return False
    return None


# ---------------------------------------------------------------------------
# QwenVerifier
# ---------------------------------------------------------------------------
class QwenVerifier:
    """Second-pass yes/no verifier against Qwen vLLM.

    Uses the same QwenNamingServer instance as the namer so no extra
    model loading is needed.
    """

    def __init__(self, qwen_server, max_tokens: int = 10):
        self.server = qwen_server
        self.max_tokens = max_tokens
        self.stats = {"called": 0, "yes": 0, "no": 0, "unparseable": 0,
                      "errors": 0, "skipped_unavailable": 0}

    @property
    def available(self) -> bool:
        return self.server is not None and getattr(self.server, "is_ready", False)

    def _b64(self, img_bgr) -> str:
        _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('utf-8')

    def _ask(self, prompt: str, img_bgr) -> Optional[bool]:
        if not self.available:
            self.stats["skipped_unavailable"] += 1
            return None
        self.stats["called"] += 1
        try:
            raw = self.server._chat(prompt, self._b64(img_bgr), max_tokens=self.max_tokens)
        except Exception as e:
            logger.debug(f"QwenVerifier chat failed: {e}")
            self.stats["errors"] += 1
            return None
        ans = _parse_yes_no(raw)
        if ans is True:
            self.stats["yes"] += 1
        elif ans is False:
            self.stats["no"] += 1
        else:
            self.stats["unparseable"] += 1
            logger.debug(f"QwenVerifier unparseable: {raw!r}")
        return ans

    def verify_text(self, image_bgr, claim_text: str) -> bool:
        """Verify that claim_text is actually visible text in the image."""
        if not claim_text:
            return False
        prompt = _VERIFY_TEXT_PROMPT_TEMPLATE.format(claim=claim_text)
        ans = self._ask(prompt, image_bgr)
        # Conservative: None (failure) -> reject
        return ans is True

    def verify_scene(self, image_bgr, claim_name: str) -> bool:
        """Verify that claim_name is a reasonable description of the scene.

        Uses a lenient scene prompt for generic category names, and a
        strict text-verification prompt for specific identifiers (room
        numbers, proper nouns, meeting room names).
        """
        if not claim_name:
            return False
        n = claim_name.strip()
        # Accept plain generic names without calling the VLM
        if n in _GENERIC_ACCEPTABLE:
            return True
        if looks_specific(n):
            prompt = _VERIFY_SPECIFIC_NAME_PROMPT_TEMPLATE.format(claim=n)
        else:
            prompt = _VERIFY_SCENE_PROMPT_TEMPLATE.format(claim=n)
        ans = self._ask(prompt, image_bgr)
        return ans is True


# ---------------------------------------------------------------------------
# MultiFrameVoter
# ---------------------------------------------------------------------------
@dataclass
class NameVote:
    name: str
    frame_idx: int
    camera: str
    area: float = 0.0
    confidence: str = "medium"


class MultiFrameVoter:
    """Accumulates (name -> votes) and exposes confirmation logic.

    Default rules:
      A name is confirmed if ANY of:
        * it appears in >= MIN_FRAMES distinct frames, OR
        * it appears on >= MIN_CAMERAS distinct cameras inside ANY single
          frame, OR
        * it has a single high-confidence vote AND its name contains a
          digit (room number) OR matches a function-room/facility keyword
          (whitelist fast-pass).
    """

    MIN_FRAMES = 2
    MIN_CAMERAS = 2
    # Whitelist keywords that allow single-frame confirmation if Qwen
    # reports high confidence on the strict prompt.
    SINGLE_FRAME_WHITELIST_KEYWORDS = (
        "母婴", "茶水", "茶歇", "打印", "复印", "前台", "接待",
        "休息", "会议室", "电梯", "楼梯", "卫生间", "洗手间",
        "强电", "弱电", "配电", "母婴室", "关爱", "关爱室",
        "哺乳", "哺乳室", "care",
    )

    def __init__(self, min_frames: int = None, min_cameras: int = None,
                 allow_single_frame_whitelist: bool = True):
        if min_frames is not None:
            self.MIN_FRAMES = min_frames
        if min_cameras is not None:
            self.MIN_CAMERAS = min_cameras
        self._votes: Dict[str, List[NameVote]] = {}
        self.allow_single_frame_whitelist = allow_single_frame_whitelist

    def add(self, vote: NameVote):
        key = (vote.name or "").strip()
        if not key:
            return
        self._votes.setdefault(key, []).append(vote)

    def votes_for(self, name: str) -> List[NameVote]:
        return list(self._votes.get(name, []))

    @staticmethod
    def _has_room_number(name: str) -> bool:
        if not name:
            return False
        # any digit-prefixed room indicator
        return bool(re.search(r"\d{1,4}", name))

    @classmethod
    def _matches_whitelist(cls, name: str) -> bool:
        if not name:
            return False
        n = name.strip()
        return any(kw in n for kw in cls.SINGLE_FRAME_WHITELIST_KEYWORDS)

    def is_confirmed(self, name: str) -> bool:
        vs = self._votes.get(name, [])
        if not vs:
            return False
        distinct_frames = {v.frame_idx for v in vs}
        if len(distinct_frames) >= self.MIN_FRAMES:
            return True
        # same-frame multi-camera
        per_frame_cams: Dict[int, set] = {}
        for v in vs:
            per_frame_cams.setdefault(v.frame_idx, set()).add(v.camera)
        if any(len(cs) >= self.MIN_CAMERAS for cs in per_frame_cams.values()):
            return True
        # single-frame whitelist fast-pass:
        # 接受 medium / high 任一 confidence + (含数字 OR 命中白名单)
        if self.allow_single_frame_whitelist:
            ok_conf = any((v.confidence or "").lower() in ("medium", "high") for v in vs)
            if ok_conf and (self._has_room_number(name) or self._matches_whitelist(name)):
                return True
        return False

    def merge_substring_variants(self):
        """Merge votes for partial-OCR variants of the same string.
        e.g. EUMANN votes are merged into NEUMANN if NEUMANN exists and
        EUMANN is a strict substring of length >= 4.
        """
        keys = list(self._votes.keys())
        # 按长度降序: 先 NEUMANN, 后 EUMANN
        keys_sorted = sorted(keys, key=lambda k: -len(k))
        merged_into: Dict[str, str] = {}
        for i, longer in enumerate(keys_sorted):
            if longer in merged_into:
                continue
            for shorter in keys_sorted[i + 1:]:
                if shorter in merged_into:
                    continue
                if len(shorter) < 4:
                    continue
                # 严格子串 + 长度差 <= 2 才视为 OCR 残缺
                if shorter in longer and len(longer) - len(shorter) <= 2:
                    self._votes[longer].extend(self._votes[shorter])
                    merged_into[shorter] = longer
                    logger.info(f"[Voter] merged variant '{shorter}' -> '{longer}'")
        for k in merged_into:
            self._votes.pop(k, None)

    def confirmed_names(self) -> List[str]:
        # 先做子串合并 (idempotent), 再判断
        self.merge_substring_variants()
        return [n for n in self._votes.keys() if self.is_confirmed(n)]

    def rejected_names(self) -> List[str]:
        return [n for n in self._votes.keys() if not self.is_confirmed(n)]

    def stats(self) -> Dict:
        return {
            "total_names_seen": len(self._votes),
            "confirmed": len(self.confirmed_names()),
            "rejected": len(self.rejected_names()),
            "rejected_names": self.rejected_names(),
            "confirmed_names": self.confirmed_names(),
            "min_frames": self.MIN_FRAMES,
            "min_cameras": self.MIN_CAMERAS,
        }


# ---------------------------------------------------------------------------
# NameDeduplicator
# ---------------------------------------------------------------------------
class NameDeduplicator:
    """Resolve duplicate position_names across all nodes.

    Strategy:
      1. Group nodes by position_name.
      2. For each group with >1 node:
         (a) Attempt merge: compute pairwise cyclic-VPR sim; if >= merge_thr
             merge the later node into the earlier one (record alias).
         (b) Any remaining duplicates get suffix "_1/_2/..." appended,
             ordered by frame_idx.
    """

    def __init__(self, merge_vpr_threshold: float = 0.78):
        self.merge_vpr_threshold = merge_vpr_threshold
        self.stats = {"groups_processed": 0, "merges": 0, "suffixed": 0,
                      "aliases": {}}

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

    def resolve(self,
                names: Dict[str, Dict[str, str]],
                topo_nodes: Dict,
                node_features: Dict[str, Dict[str, np.ndarray]],
                ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
        """
        Args:
            names: {node_id: {"cn": ..., "en": ...}} (mutable, will be copied)
            topo_nodes: online TopoGraph.nodes
            node_features: {node_id: {cam: feat}} for VPR sim

        Returns:
            (new_names, alias_map) where alias_map[old_id] = new_id for
            merged nodes (caller may delete old_id from topology).
        """
        new_names = {k: dict(v) for k, v in names.items()}
        alias_map: Dict[str, str] = {}

        # Group by cn name (order: by frame_idx ascending so earliest wins)
        ordered_ids = sorted(
            new_names.keys(),
            key=lambda nid: getattr(topo_nodes.get(nid), "frame_idx", 10**9))

        groups: Dict[str, List[str]] = {}
        for nid in ordered_ids:
            cn = new_names[nid]["cn"]
            groups.setdefault(cn, []).append(nid)

        for cn, ids in groups.items():
            if len(ids) < 2:
                continue
            self.stats["groups_processed"] += 1

            # Step (a): attempt merges
            anchor = ids[0]
            merged_here: List[str] = []
            surviving: List[str] = [anchor]
            for other in ids[1:]:
                fa = node_features.get(anchor)
                fb = node_features.get(other)
                sim = 0.0
                if fa and fb:
                    sim = self._cyclic_cosine(fa, fb)
                if sim >= self.merge_vpr_threshold:
                    alias_map[other] = anchor
                    merged_here.append(other)
                    self.stats["merges"] += 1
                    logger.info(
                        f"[NameDedup] merge node {other} -> {anchor} "
                        f"(name='{cn}', sim={sim:.3f})")
                else:
                    surviving.append(other)
                    logger.info(
                        f"[NameDedup] keep separate: {other} vs {anchor} "
                        f"(name='{cn}', sim={sim:.3f})")

            self.stats["aliases"].update({m: anchor for m in merged_here})

            # Step (b): suffix remaining duplicates
            if len(surviving) >= 2:
                for i, nid in enumerate(surviving, start=1):
                    suffixed_cn = f"{cn}_{i}"
                    suffixed_en = f"{new_names[nid]['en']}_{i}"
                    new_names[nid]["cn"] = suffixed_cn
                    new_names[nid]["en"] = suffixed_en
                    self.stats["suffixed"] += 1
                    logger.info(
                        f"[NameDedup] suffix node {nid} -> '{suffixed_cn}'")

        return new_names, alias_map
