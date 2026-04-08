"""单目视觉里程计

支持两种 backend:
- "orb"  : ORB + EssentialMatrix + recoverPose (原 MonoVO, 阶段 2 之前的默认)
- "vggt" : 复用 VGGTDepthEstimator 已缓存的位姿 (无额外推理), 阶段 2 起的新默认

接口契约 (两者一致):
    .estimate(bgr_image, depth_map=None) -> (dtrans_m, drot_rad)
"""
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ======================================================================
class MonoVO:
    """ORB 单目 VO (legacy, 保留可切回)。"""

    def __init__(self, focal: float = 700.0, pp=None):
        self.focal = focal
        self.pp = pp
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_gray = None
        self.last_dtrans = 0.5
        self.last_drot = 0.0

    def estimate(self, bgr_image, depth_map=None):
        if bgr_image is None:
            return self.last_dtrans, self.last_drot

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        pp = self.pp or (w / 2.0, h / 2.0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0

        try:
            kp1, des1 = self.orb.detectAndCompute(self.prev_gray, None)
            kp2, des2 = self.orb.detectAndCompute(gray, None)
            if des1 is None or des2 is None or len(kp1) < 30 or len(kp2) < 30:
                self.prev_gray = gray
                return self._fallback()

            matches = self.bf.match(des1, des2)
            if len(matches) < 30:
                self.prev_gray = gray
                return self._fallback()
            matches = sorted(matches, key=lambda m: m.distance)[:300]
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(
                pts1, pts2, focal=self.focal, pp=pp,
                method=cv2.RANSAC, prob=0.999, threshold=1.5)
            if E is None or E.shape != (3, 3):
                self.prev_gray = gray
                return self._fallback()

            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=self.focal, pp=pp, mask=mask)

            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
            forward_dir = tz
            lateral_dir = tx
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            yaw = np.arctan2(-R[2, 0], sy) if sy > 1e-6 else 0.0

            if depth_map is not None and depth_map.size > 0:
                d = np.asarray(depth_map, dtype=np.float32)
                hh, ww = d.shape
                roi = d[hh // 5: 4 * hh // 5, ww // 5: 4 * ww // 5]
                roi = roi[(roi > 0.1) & (roi < 50.0)]
                scale = float(np.median(roi)) * 0.05 if roi.size > 100 else 0.5
            else:
                scale = 0.5

            dtrans = float(np.hypot(forward_dir, lateral_dir) * scale)
            drot = float(yaw)
            dtrans = max(0.0, min(dtrans, 3.0))
            drot = max(-1.0, min(drot, 1.0))

            self.prev_gray = gray
            self.last_dtrans = dtrans
            self.last_drot = drot
            return dtrans, drot
        except Exception as e:
            logger.debug(f"MonoVO failed: {e}; fallback")
            self.prev_gray = gray
            return self._fallback()

    def _fallback(self):
        return self.last_dtrans, self.last_drot


# ======================================================================
class VGGTVisualOdometry:
    """复用 VGGTDepthEstimator 缓存的位姿计算相对运动 (零额外推理)。

    工作原理:
    - VGGT 每次推理输出窗口内所有帧的 extrinsics (world-to-camera)
    - 同一次推理里, last_extri 与 prev_extri 在同一坐标系下, 可直接做相对变换
    - 取相机在世界系的位置变化求 dtrans, 从相对旋转矩阵提取 yaw 求 drot

    Coordinate convention (VGGT/OpenCV):
        extrinsics: 3x4, [R|T], world-to-camera (X_cam = R*X_world + T)
        相机在世界系的位置: C = -R^T @ T
        camera frame: x-right, y-down, z-forward
        yaw 取绕 y 轴 (top-down view 下的航向)
    """

    def __init__(self, depth_estimator):
        self.depth_estimator = depth_estimator
        self.last_dtrans = 0.0
        self.last_drot = 0.0
        # sanity clamp
        self.max_dtrans = 5.0
        self.max_drot = 1.5

    def estimate(self, bgr_image, depth_map=None):
        de = self.depth_estimator
        if de is None or not getattr(de, "available", False):
            return self._fallback()
        last_extri = getattr(de, "last_extri", None)
        prev_extri = getattr(de, "prev_extri", None)
        if last_extri is None or prev_extri is None:
            # 滑窗只有 1 帧 (首帧或刚重置), 没有相对运动可算
            return 0.0, 0.0
        try:
            R_curr = np.asarray(last_extri[:3, :3], dtype=np.float64)
            T_curr = np.asarray(last_extri[:3, 3], dtype=np.float64)
            R_prev = np.asarray(prev_extri[:3, :3], dtype=np.float64)
            T_prev = np.asarray(prev_extri[:3, 3], dtype=np.float64)

            # 相机在世界系的位置
            C_curr = -R_curr.T @ T_curr
            C_prev = -R_prev.T @ T_prev
            dtrans = float(np.linalg.norm(C_curr - C_prev))

            # cam_curr_from_cam_prev = R_curr @ R_prev.T  (作用于相机系向量)
            R_rel = R_curr @ R_prev.T
            # yaw 绕 y 轴: atan2(R[0,2], R[2,2])
            drot = float(np.arctan2(R_rel[0, 2], R_rel[2, 2]))

            # sanity clamp
            dtrans = max(0.0, min(dtrans, self.max_dtrans))
            drot = max(-self.max_drot, min(drot, self.max_drot))

            self.last_dtrans = dtrans
            self.last_drot = drot
            return dtrans, drot
        except Exception as e:
            logger.warning(f"VGGTVisualOdometry failed: {e}", exc_info=True)
            return self._fallback()

    def _fallback(self):
        return self.last_dtrans, self.last_drot


# ======================================================================
def build_visual_odometry(cfg, depth_estimator=None):
    """根据 cfg.vo_backend 构建 VO.

    "vggt" 需要 depth_estimator 是 VGGTDepthEstimator (复用其缓存位姿).
    """
    backend = getattr(cfg, "vo_backend", "vggt")
    if backend == "vggt":
        # 仅当 depth backend 也是 vggt 且实例可用时启用
        if depth_estimator is not None and hasattr(depth_estimator, "last_extri"):
            return VGGTVisualOdometry(depth_estimator)
        logger.warning("vo_backend=vggt 但 depth_estimator 不是 VGGT, 回退到 ORB MonoVO")
    return MonoVO()
