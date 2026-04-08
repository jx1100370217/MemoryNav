"""Frontier 检测 + NBV 选择 (日志用)"""
import logging, math
logger = logging.getLogger(__name__)


class FrontierNBV:
    def __init__(self, cfg):
        self.cfg = cfg

    def score_and_pick(self, occupancy, robot_x, robot_y, recent_landmarks: list):
        frontiers = occupancy.find_frontiers()
        if not frontiers:
            return None, []
        scored = []
        for fx, fy in frontiers:
            wx = (fx - occupancy.origin) * occupancy.res
            wy = (fy - occupancy.origin) * occupancy.res
            dist = math.hypot(wx - robot_x, wy - robot_y)
            info = 1.0  # frontier 默认 unit info gain
            cost = dist
            sem = 0.5 if not recent_landmarks else 0.1
            score = (self.cfg.nbv_info_weight * info
                     - self.cfg.nbv_cost_weight * cost
                     + self.cfg.nbv_semantic_weight * sem)
            scored.append({"x": wx, "y": wy, "score": score, "info": info, "cost": cost, "sem": sem})
        scored.sort(key=lambda d: -d["score"])
        return scored[0], scored[:5]
