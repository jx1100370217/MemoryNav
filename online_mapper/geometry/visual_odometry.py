"""轻量单目 VO — ORB + Essential matrix + recoverPose

替换 OnlineMapperCore 的 _proxy_motion 常速代理。

策略:
1. ORB 特征点检测 + BFMatcher
2. RANSAC EssentialMatrix
3. recoverPose -> R, t (t 是单位方向)
4. 用 Depth 中位数估计 metric scale (相对深度 -> 米)
5. 转换到 2D (x, theta) — 取 t.x 作为侧向, t.z 作为前向

失败回退: 上次速度 (常速 fallback), 不再是 0.5/0.02 hardcode
"""
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class MonoVO:
    def __init__(self, focal: float = 700.0, pp=None):
        self.focal = focal
        self.pp = pp  # principal point; None -> use image center
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_gray = None
        self.last_dtrans = 0.5
        self.last_drot = 0.0

    def estimate(self, bgr_image, depth_map=None):
        """
        Args:
            bgr_image: BGR ndarray
            depth_map: optional depth ndarray (relative or metric)
        Returns:
            (dtrans_m, drot_rad)
        """
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

            # 2D motion: assume camera_1 is forward facing
            # t is unit direction (camera frame: x right, y down, z forward)
            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
            forward_dir = tz  # signed
            lateral_dir = tx

            # rotation about Y axis (yaw)
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            yaw = np.arctan2(-R[2, 0], sy) if sy > 1e-6 else 0.0

            # scale via depth median
            if depth_map is not None and depth_map.size > 0:
                d = np.asarray(depth_map, dtype=np.float32)
                # 取中央 60% 区域中位数, 排除 0/inf
                hh, ww = d.shape
                roi = d[hh // 5: 4 * hh // 5, ww // 5: 4 * ww // 5]
                roi = roi[(roi > 0.1) & (roi < 50.0)]
                if roi.size > 100:
                    scale = float(np.median(roi)) * 0.05  # 经验缩放
                else:
                    scale = 0.5
            else:
                scale = 0.5

            # 输出: dtrans 沿前向, 取绝对值; 若 forward_dir<0 (倒退), 仍记为正位移
            dtrans = abs(forward_dir) * scale
            # 对侧向也考虑一点
            dtrans = float(np.hypot(forward_dir, lateral_dir) * scale)
            drot = float(yaw)

            # sanity clamp
            dtrans = max(0.0, min(dtrans, 3.0))
            drot = max(-1.0, min(drot, 1.0))

            self.prev_gray = gray
            self.last_dtrans = dtrans
            self.last_drot = drot
            return dtrans, drot
        except Exception as e:
            logger.debug(f"VO failed: {e}; fallback")
            self.prev_gray = gray
            return self._fallback()

    def _fallback(self):
        return self.last_dtrans, self.last_drot
