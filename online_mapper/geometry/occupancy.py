"""轻量 2D 占据栅格"""
import numpy as np, logging
logger = logging.getLogger(__name__)

FREE, UNKNOWN, OCC = 0, -1, 1


class OccupancyGrid:
    def __init__(self, size: int = 200, resolution: float = 0.2):
        self.size = size
        self.res = resolution
        self.grid = np.full((size, size), UNKNOWN, dtype=np.int8)
        self.origin = size // 2  # robot starts at center

    def world_to_cell(self, x, y):
        cx = int(self.origin + x / self.res)
        cy = int(self.origin + y / self.res)
        return cx, cy

    def integrate(self, robot_x, robot_y, robot_theta, depth_row: np.ndarray, fov_rad: float = 1.2):
        """用一行深度 (W,) 沿 FOV 投射射线"""
        W = depth_row.shape[0]
        prev_free = int(np.sum(self.grid == FREE))
        for i, d in enumerate(depth_row):
            ang = robot_theta + (i / W - 0.5) * fov_rad
            ex = robot_x + d * np.cos(ang)
            ey = robot_y + d * np.sin(ang)
            # raycast: free cells along
            steps = max(1, int(d / self.res))
            for s in range(steps):
                fx = robot_x + (s / steps) * d * np.cos(ang)
                fy = robot_y + (s / steps) * d * np.sin(ang)
                cx, cy = self.world_to_cell(fx, fy)
                if 0 <= cx < self.size and 0 <= cy < self.size:
                    if self.grid[cy, cx] == UNKNOWN:
                        self.grid[cy, cx] = FREE
            cx, cy = self.world_to_cell(ex, ey)
            if 0 <= cx < self.size and 0 <= cy < self.size:
                self.grid[cy, cx] = OCC
        new_free = int(np.sum(self.grid == FREE))
        info_gain = (new_free - prev_free) / max(1, self.size * self.size)
        return info_gain

    def find_frontiers(self):
        """frontier = free cell adjacent to unknown cell"""
        frontiers = []
        free_mask = (self.grid == FREE)
        H, W = self.grid.shape
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if free_mask[y, x]:
                    nb = self.grid[y - 1:y + 2, x - 1:x + 2]
                    if (nb == UNKNOWN).any():
                        frontiers.append((x, y))
        return frontiers

    def stats(self):
        return {
            "free": int(np.sum(self.grid == FREE)),
            "occ": int(np.sum(self.grid == OCC)),
            "unknown": int(np.sum(self.grid == UNKNOWN)),
        }
