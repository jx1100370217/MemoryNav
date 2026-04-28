"""轻量 2D 占据栅格 (top-down)

支持两种集成方式:
- integrate(robot_pose, depth_row, fov)         : 1D 深度行 ray-cast (legacy / DA-V2)
- integrate_pointcloud(points_camera, robot_pose, conf) : VGGT dense 点云直填 (阶段 3)
"""
import numpy as np, logging
logger = logging.getLogger(__name__)

FREE, UNKNOWN, OCC = 0, -1, 1


class OccupancyGrid:
    def __init__(self, size: int = 200, resolution: float = 0.2):
        self.size = size
        self.res = resolution
        self.grid = np.full((size, size), UNKNOWN, dtype=np.int8)
        self.origin = size // 2  # robot starts at center

    # ------------------------------------------------------------------
    def world_to_cell(self, x, y):
        cx = int(self.origin + x / self.res)
        cy = int(self.origin + y / self.res)
        return cx, cy

    # ------------------------------------------------------------------
    def integrate(self, robot_x, robot_y, robot_theta, depth_row: np.ndarray, fov_rad: float = 1.2):
        """1D depth row ray-cast (legacy 路径)."""
        W = depth_row.shape[0]
        prev_free = int(np.sum(self.grid == FREE))
        for i, d in enumerate(depth_row):
            ang = robot_theta + (i / W - 0.5) * fov_rad
            ex = robot_x + d * np.cos(ang)
            ey = robot_y + d * np.sin(ang)
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
        return (new_free - prev_free) / max(1, self.size * self.size)

    # ------------------------------------------------------------------
    def integrate_pointcloud(
        self,
        points_camera: np.ndarray,
        robot_x: float, robot_y: float, robot_theta: float,
        conf: np.ndarray = None,
        conf_thresh: float = 1.0,  # VGGT depth_conf 用 expp1 激活, 输出 >=1, 取 1.0 = 不过滤最低
        z_min: float = 0.05, z_max: float = 10.0,      # 相机前方距离范围 (VGGT 自洽米)
        height_min: float = -1.5, height_max: float = 1.5,  # 障碍高度窗
        sample_n: int = 6000,
    ) -> float:
        """用 dense 相机系点云填占据栅格.

        Args:
            points_camera: HxWx3, camera frame (x-right, y-down, z-forward), 米
            robot_x, robot_y, robot_theta: mapper 全局机器人位姿
            conf: HxW depth 置信度, None 则全收
            conf_thresh: 置信度阈值
            z_min/z_max: 沿光轴有效距离, 排除噪点和远场
            height_min/height_max: 障碍高度范围 (相机 y 轴, y-down 故 y_cam<0 是天花板方向)
                我们用 -y_cam 作为高度: height_min=-0.6 表示允许地面以下 0.6m,
                height_max=1.6 表示天花板下 1.6m 内的物体都算障碍
            sample_n: 稀疏采样点数 (控制开销)

        Returns:
            info_gain: 新标 free 单元占总单元的比例
        """
        if points_camera is None or points_camera.size == 0:
            return 0.0
        prev_free = int(np.sum(self.grid == FREE))

        pts = points_camera.reshape(-1, 3).astype(np.float32)
        # camera frame: X=x_right, Y=y_down (height = -Y), Z=z_forward
        x_c = pts[:, 0]
        y_c = pts[:, 1]
        z_c = pts[:, 2]

        valid = (z_c > z_min) & (z_c < z_max)
        height = -y_c
        valid &= (height > height_min) & (height < height_max)
        if conf is not None:
            valid &= (conf.reshape(-1) > conf_thresh)
        pts = pts[valid]
        if pts.shape[0] < 10:
            return 0.0

        # 稀疏采样
        if pts.shape[0] > sample_n:
            idx = np.random.choice(pts.shape[0], sample_n, replace=False)
            pts = pts[idx]

        # camera frame -> robot local (forward = z_c, left = -x_c)
        forward = pts[:, 2]
        left = -pts[:, 0]

        # robot local -> mapper world (绕 z 旋转 robot_theta + 平移)
        cos_t, sin_t = np.cos(robot_theta), np.sin(robot_theta)
        gx = robot_x + cos_t * forward - sin_t * left
        gy = robot_y + sin_t * forward + cos_t * left

        # 标 OCC: 点本身
        cx = (self.origin + gx / self.res).astype(np.int32)
        cy = (self.origin + gy / self.res).astype(np.int32)
        in_bounds = (cx >= 0) & (cx < self.size) & (cy >= 0) & (cy < self.size)
        cx_occ, cy_occ = cx[in_bounds], cy[in_bounds]
        self.grid[cy_occ, cx_occ] = OCC

        # 标 FREE: 沿 robot 到每个 OCC 点的连线 (向量化等步长采样, 替代 Bresenham).
        # 一次性算所有 (t, point) 对的索引, 比原 for-t 循环快 30-60x (5000 pts × 60 steps).
        rcx = self.origin + robot_x / self.res
        rcy = self.origin + robot_y / self.res
        dx = cx_occ - rcx
        dy = cy_occ - rcy
        dist_cells = np.sqrt(dx * dx + dy * dy)
        max_steps = int(min(60, dist_cells.max() if dist_cells.size else 0))
        if max_steps >= 2:
            ts = np.linspace(0.05, 0.92, max_steps)  # 不到达 OCC 自身
            # broadcast: ts (T,) × dx (N,) → (T, N) 网格, 一次性 cast 到 int.
            fx_grid = (rcx + ts[:, None] * dx[None, :]).astype(np.int32).ravel()
            fy_grid = (rcy + ts[:, None] * dy[None, :]).astype(np.int32).ravel()
            ok = ((fx_grid >= 0) & (fx_grid < self.size)
                  & (fy_grid >= 0) & (fy_grid < self.size))
            fx_grid = fx_grid[ok]
            fy_grid = fy_grid[ok]
            # 仅 unknown -> free (一次性查询 + 写入, 等价于按 t 顺序逐次写)
            cur = self.grid[fy_grid, fx_grid]
            unk_mask = (cur == UNKNOWN)
            self.grid[fy_grid[unk_mask], fx_grid[unk_mask]] = FREE

        # 重新覆盖 OCC (free 步骤可能误覆盖, 这里再写一遍确保)
        self.grid[cy_occ, cx_occ] = OCC

        new_free = int(np.sum(self.grid == FREE))
        return (new_free - prev_free) / max(1, self.size * self.size)

    # ------------------------------------------------------------------
    def find_frontiers(self):
        """Frontier = FREE 单元且 3x3 邻接含 UNKNOWN. 用 maximum_filter 一次过,
        替换原 200x200 nested for (40000 次 Python 循环 → 1 次 numpy + 1 次 scipy)."""
        from scipy.ndimage import maximum_filter
        free_mask = (self.grid == FREE)
        unknown_mask = (self.grid == UNKNOWN).astype(np.uint8)
        # 3x3 max filter: 任一邻居是 UNKNOWN 则该单元 neighbor_unknown=True
        neighbor_unknown = maximum_filter(unknown_mask, size=3) > 0
        frontier_mask = free_mask & neighbor_unknown
        # 排除边界 (与原 for y in range(1, H-1) 等价)
        frontier_mask[0, :] = False
        frontier_mask[-1, :] = False
        frontier_mask[:, 0] = False
        frontier_mask[:, -1] = False
        # np.where 输出 row-major (y 升序, 同 y 内 x 升序), 与原 nested loop 一致
        ys, xs = np.where(frontier_mask)
        return list(zip(xs.tolist(), ys.tolist()))

    def stats(self):
        return {
            "free": int(np.sum(self.grid == FREE)),
            "occ": int(np.sum(self.grid == OCC)),
            "unknown": int(np.sum(self.grid == UNKNOWN)),
        }
