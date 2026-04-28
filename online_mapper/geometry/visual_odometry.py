"""单目视觉里程计 (VGGT-based, ORB MonoVO 已删除).

backend:
- "vggt"  : 复用 VGGTDepthEstimator 已缓存的位姿 (无额外推理) — 默认且唯一
- 其他    : 返回 None, 主流程走默认常速代理 (0.5 m/帧, 0.02 rad/帧)

历史 ORB-based MonoVO 因 scale = median(depth)*0.05 是无标定根据的魔法系数,
已于 refactor 阶段 5.2 移除. 真要 ORB fallback 需要重新引入 + 标定.

接口契约: .estimate(bgr_image, depth_map=None) -> (dtrans_m, drot_rad)
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


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
    """根据 cfg.vo_backend 构建 VO. 只接受 'vggt'; 其他 backend 返回 None
    让 OnlineMapperCore._vo_motion 走默认常速代理 (0.5 m/帧, 0.02 rad/帧).

    "vggt" 需要 depth_estimator 是 VGGTDepthEstimator (复用其缓存位姿).
    """
    backend = getattr(cfg, "vo_backend", "vggt")
    if backend != "vggt":
        logger.warning(f"vo_backend={backend!r} not supported (ORB MonoVO removed); "
                       f"VO disabled, using constant motion proxy")
        return None
    if depth_estimator is None or not hasattr(depth_estimator, "last_extri"):
        logger.warning("vo_backend=vggt but depth_estimator is not VGGT-based; "
                       "VO disabled, using constant motion proxy")
        return None
    return VGGTVisualOdometry(depth_estimator)
