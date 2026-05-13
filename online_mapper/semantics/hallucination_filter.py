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
from online_mapper.semantics.semantic_dedup import cyclic_cosine

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
    # 任何一票的 bbox 像素面积下限 (1920x1536 帧, ~0.3%).
    # 用于过滤 Qwen 在远处的小检测上做的文字幻觉 (典型例: '酒店' / 'EXHIOH'
    # 在 30~40 px 的小 bbox 上"读"出文字, 实际牌子并不存在).
    # 任意一帧的 bbox 大于此值即视为有"近距离实证", 可信度足够.
    MIN_MAX_AREA = 8000
    # Whitelist keywords that allow single-frame confirmation if Qwen
    # reports high confidence on the strict prompt.
    SINGLE_FRAME_WHITELIST_KEYWORDS = (
        "母婴", "茶水", "茶歇", "打印", "复印", "前台", "接待",
        "休息", "会议室", "电梯", "楼梯", "卫生间", "洗手间",
        "强电", "弱电", "配电", "母婴室", "关爱", "关爱室",
        "哺乳", "哺乳室", "care",
    )

    # plate 字符数最低门槛 (R1 截断 reject):
    # 中文字符 < N 直接 reject (e.g. "门" 单字 / "C" 这种过短的合法 plate 应靠 building landmark RE 兜底)
    # 英文/数字字符 < N 直接 reject (e.g. "iA"/"CR" OCR 截断)
    MIN_CHINESE_CHARS = 2
    MIN_ENGLISH_CHARS = 4

    def __init__(self, min_frames: int = None, min_cameras: int = None,
                 allow_single_frame_whitelist: bool = True,
                 min_max_area: float = None,
                 min_chinese_chars: int = None,
                 min_english_chars: int = None):
        if min_frames is not None:
            self.MIN_FRAMES = min_frames
        if min_cameras is not None:
            self.MIN_CAMERAS = min_cameras
        if min_max_area is not None:
            self.MIN_MAX_AREA = min_max_area
        if min_chinese_chars is not None:
            self.MIN_CHINESE_CHARS = min_chinese_chars
        if min_english_chars is not None:
            self.MIN_ENGLISH_CHARS = min_english_chars
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

    # BUILDING_LANDMARK plates are observed at outdoor distance — bbox area
    # is naturally smaller than indoor door plates, so threshold is relaxed.
    # Pattern is a local copy to avoid circular import with node_category.
    _BUILDING_LANDMARK_MIN_AREA = 1200
    _BUILDING_LANDMARK_RE = re.compile(
        r"^([A-Za-z]座(入口|大堂)?|[A-Za-z]栋(入口|大堂)?|\d+号楼|\d+栋)$"
    )

    @classmethod
    def _is_building_landmark_name(cls, name: str) -> bool:
        return bool(cls._BUILDING_LANDMARK_RE.match((name or "").strip()))

    # BUILDING_LANDMARK needs stronger OCR evidence: observed D座/D栋 hallucinations
    # against glass facades pass 2-cam + 2-vote. Real ones (A/B/C/H 座/入口) reach
    # ≥ 4 votes (min observed: B座 = 4).
    _BUILDING_LANDMARK_MIN_VOTES = 4

    @classmethod
    def _has_valid_plate_format(cls, name: str) -> bool:
        """plate 格式合法性: 字符数门槛 + OCR 截断模式 reject.

        building landmark (A 座/D 栋) 由 _BUILDING_LANDMARK_RE 单独判, 不走此门槛.

        Reject 三类:
          1. 中文字符 < MIN_CHINESE_CHARS (e.g. '门' 单字)
          2. 英文/数字字符 < MIN_ENGLISH_CHARS (e.g. 'CR')
          3. OCR 截断典型模式: 混大小写 + (数字 OR 点) + 短串 (< 10 字)
             (e.g. 'iA.3TI' = 'DEEPROUTE.AI' 的截断错认; 'PROUTE.AI'-class 通过 LCS 合并; 此规则兜 LCS 无法处理的零碎模式)
        """
        if cls._is_building_landmark_name(name):
            return True
        s = (name or "").strip()
        if not s:
            return False
        cn_chars = sum(1 for c in s if "一" <= c <= "鿿")
        en_chars = sum(1 for c in s if c.isalnum() and not ("一" <= c <= "鿿"))
        if cn_chars > 0 and cn_chars < cls.MIN_CHINESE_CHARS:
            return False
        if cn_chars == 0 and en_chars > 0 and en_chars < cls.MIN_ENGLISH_CHARS:
            return False
        # OCR 截断模式: 纯英文+混大小写+(数字 OR 点)+短串
        if cn_chars == 0:
            has_lower = any(c.islower() for c in s)
            has_upper = any(c.isupper() for c in s)
            has_digit = any(c.isdigit() for c in s)
            has_dot = "." in s
            if has_lower and has_upper and (has_digit or has_dot) and len(s) < 10:
                return False
        return True

    def is_confirmed(self, name: str) -> bool:
        # R1: plate 格式校验 (字符数 + OCR 截断模式) 优先 reject
        if not self._has_valid_plate_format(name):
            return False
        vs = self._votes.get(name, [])
        if not vs:
            return False
        max_area = max((v.area or 0.0) for v in vs)
        is_bl = self._is_building_landmark_name(name)
        if is_bl and len(vs) < self._BUILDING_LANDMARK_MIN_VOTES:
            return False
        area_thresh = self._BUILDING_LANDMARK_MIN_AREA if is_bl else self.MIN_MAX_AREA
        meets_area = max_area >= area_thresh

        distinct_frames = {v.frame_idx for v in vs}
        if len(distinct_frames) >= self.MIN_FRAMES and meets_area:
            return True
        # same-frame multi-camera
        per_frame_cams: Dict[int, set] = {}
        for v in vs:
            per_frame_cams.setdefault(v.frame_idx, set()).add(v.camera)
        if any(len(cs) >= self.MIN_CAMERAS for cs in per_frame_cams.values()) and meets_area:
            return True
        # Single-frame whitelist fast-pass: accept medium/high confidence +
        # (contains digits OR matches whitelist). BUILDING_LANDMARK with digits
        # (X号楼/X栋) is excluded to avoid OCR hallucinations of distant
        # buildings passing on a single frame; requires distinct_frames>=2
        # or distinct_cameras>=2.
        if self.allow_single_frame_whitelist and not is_bl:
            ok_conf = any((v.confidence or "").lower() in ("medium", "high") for v in vs)
            if ok_conf and (self._has_room_number(name) or self._matches_whitelist(name)):
                return True
        return False

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """经典 Levenshtein 编辑距离 (短串场景, O(len_a*len_b) 即可)."""
        la, lb = len(a), len(b)
        if la == 0: return lb
        if lb == 0: return la
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            cur = [i] + [0] * lb
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            prev = cur
        return prev[lb]

    def merge_substring_variants(self):
        """Merge votes for OCR variants of the same string.

        两类合并:
        1. 严格子串 + 长度差 <= 2 (e.g. EUMANN -> NEUMANN)
        2. 全大写 Latin 串 + edit_distance <= 2 + 长度 >= 4
           (e.g. HIOF/KHIOH/HIOH/EXHIOH 这种园区招牌的 OCR 抖动)

        合并方向: 票数多的吸收票数少的; 同票时取长度更长的(更"完整"的串).
        """
        import re as _re
        keys = list(self._votes.keys())
        # 第 1 阶段: 严格子串合并
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
                if shorter in longer and len(longer) - len(shorter) <= 2:
                    self._votes[longer].extend(self._votes[shorter])
                    merged_into[shorter] = longer
                    logger.info(f"[Voter] merged variant '{shorter}' -> '{longer}'")
        for k in merged_into:
            self._votes.pop(k, None)

        # 第 2 阶段: 全大写 Latin OCR 模糊聚类
        # 仅作用于 4-8 字符全大写英文串 (招牌典型形态), 中文/混合不动
        is_caps = lambda s: bool(_re.fullmatch(r"[A-Z]{4,8}", s))
        latin_keys = [k for k in self._votes.keys() if is_caps(k)]
        if len(latin_keys) >= 2:
            # 按票数降序作 anchor (票多的吃票少的)
            latin_keys.sort(key=lambda k: (-len(self._votes[k]), -len(k)))
            absorbed: Dict[str, str] = {}
            for i, anchor in enumerate(latin_keys):
                if anchor in absorbed:
                    continue
                for cand in latin_keys[i + 1:]:
                    if cand in absorbed:
                        continue
                    if abs(len(anchor) - len(cand)) > 2:
                        continue
                    d = self._edit_distance(anchor, cand)
                    # 距离阈值: 长度 4 -> 1, 长度 >=5 -> 2
                    max_d = 1 if min(len(anchor), len(cand)) <= 4 else 2
                    if d <= max_d:
                        self._votes[anchor].extend(self._votes[cand])
                        absorbed[cand] = anchor
                        logger.info(f"[Voter] OCR-fuzzy merged '{cand}' -> '{anchor}' "
                                    f"(edit_distance={d})")
            for k in absorbed:
                self._votes.pop(k, None)

        # 第 3 阶段: LCS 截断/前后缀变体合并 (R1 修法)
        # 处理 OCR 截断 prefix/suffix/middle 截断, e.g.
        #   "DEEPROUTE.AI" / "DEEPRO" / "ROUTE.AI" / "PROUTE.AI" / "iA.3TI" / "TE.AI" 应合并
        #   "FINGER CROXX" / "FING CR" / "GER CROXX" / "ANGER CROXX" 应合并
        # 规则: 若两 plate 的 LCS 覆盖率 cov_min ≥ 0.7 (短串大部分被长串覆盖)
        #       且 cov_max ≥ 0.4 (有显著公共子序列), 视为同源 → 票多者作 anchor 吸收
        keys = sorted(self._votes.keys(), key=lambda k: (-len(self._votes[k]), -len(k)))
        absorbed2: Dict[str, str] = {}
        for i, anchor in enumerate(keys):
            if anchor in absorbed2:
                continue
            if len(anchor) < 4:
                continue
            for cand in keys[i + 1:]:
                if cand in absorbed2 or cand == anchor:
                    continue
                if len(cand) < 4:
                    continue
                lcs = self._lcs_len(anchor, cand)
                if lcs < 3:
                    continue
                cov_max = lcs / max(len(anchor), len(cand))
                cov_min = lcs / min(len(anchor), len(cand))
                if cov_min >= 0.7 and cov_max >= 0.4:
                    self._votes[anchor].extend(self._votes[cand])
                    absorbed2[cand] = anchor
                    logger.info(f"[Voter] LCS-truncation merged '{cand}' -> '{anchor}' "
                                f"(lcs={lcs}, cov_min={cov_min:.2f}, cov_max={cov_max:.2f})")
        for k in absorbed2:
            self._votes.pop(k, None)

    @staticmethod
    def _lcs_len(a: str, b: str) -> int:
        """最长公共子序列长度 (用于 OCR 截断/中间断的合并判定)."""
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return 0
        prev = [0] * (lb + 1)
        for i in range(1, la + 1):
            cur = [0] * (lb + 1)
            for j in range(1, lb + 1):
                if a[i - 1] == b[j - 1]:
                    cur[j] = prev[j - 1] + 1
                else:
                    cur[j] = max(prev[j], cur[j - 1])
            prev = cur
        return prev[lb]

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

        # Group by structured (category, organization) when name_struct exists
        ordered_ids = sorted(
            new_names.keys(),
            key=lambda nid: getattr(topo_nodes.get(nid), "frame_idx", 10**9))

        def _key_for(nid):
            node = topo_nodes.get(nid)
            ns = getattr(node, "name_struct", None) if node else None
            if ns is not None and (ns.category or ns.organization):
                return ns.dedup_key()
            return ("", new_names[nid]["cn"])

        groups: Dict[tuple, List[str]] = {}
        for nid in ordered_ids:
            groups.setdefault(_key_for(nid), []).append(nid)

        for key, ids in groups.items():
            cn = key[0] or key[1] if isinstance(key, tuple) else str(key)
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
                    sim = cyclic_cosine(fa, fb)
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
                    node = topo_nodes.get(nid)
                    ns = getattr(node, "name_struct", None) if node else None
                    suffix = f"_{i}"
                    if ns is not None:
                        ns.instance_suffix = suffix
                        suffixed_cn = ns.display_cn()
                        suffixed_en = ns.display_en()
                    else:
                        import re as _re
                        base_cn = _re.sub(r"_\d+$", "", new_names[nid]["cn"])
                        base_en = _re.sub(r"_\d+$", "", new_names[nid]["en"])
                        suffixed_cn = f"{base_cn}{suffix}"
                        suffixed_en = f"{base_en}{suffix}"
                    new_names[nid]["cn"] = suffixed_cn
                    new_names[nid]["en"] = suffixed_en
                    self.stats["suffixed"] += 1
                    logger.info(
                        f"[NameDedup] suffix node {nid} -> '{suffixed_cn}'")

        return new_names, alias_map
