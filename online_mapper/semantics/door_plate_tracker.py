"""Door-plate retrospective frame selection.

跨多帧累积某个语义实体 (门牌/房间名牌) 的检测,
当 finalize 时, 以 bbox 面积最大的那一帧作为该 semantic node 的代表帧。
其位置同步迁移到该帧时的 robot pose。
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PlateObservation:
    frame_idx: int
    timestamp: str
    cameras: Dict[str, str]
    camera: str          # 检测到的相机
    bbox: List[float]    # x1,y1,x2,y2 (像素)
    score: float
    text: str            # OCR / Qwen 识别文本 (key)
    name_cn: str
    name_en: str
    pose: Tuple[float, float, float]  # 当时 robot pose
    area: float = 0.0

    def __post_init__(self):
        if not self.area:
            x1, y1, x2, y2 = self.bbox
            self.area = max(0.0, (x2 - x1) * (y2 - y1))


class DoorPlateTracker:
    """以 text 为键, 累积所有出现的帧"""

    def __init__(self, name_similarity_keys=("text", "name_cn")):
        self._observations: Dict[str, List[PlateObservation]] = {}
        self._sim_keys = name_similarity_keys

    def add(self, obs: PlateObservation):
        key = obs.text or obs.name_cn
        if not key:
            return
        self._observations.setdefault(key, []).append(obs)
        logger.debug(f"[DoorPlate] +{key} area={obs.area:.0f} frame={obs.frame_idx}")

    def all_keys(self) -> List[str]:
        return list(self._observations.keys())

    def best(self, key: str) -> Optional[PlateObservation]:
        obs_list = self._observations.get(key, [])
        if not obs_list:
            return None
        return max(obs_list, key=lambda o: o.area)

    def all_best(self) -> Dict[str, PlateObservation]:
        return {k: self.best(k) for k in self._observations.keys()}

    def __len__(self):
        return len(self._observations)
