"""单目深度估计

支持两种 backend:
- "da_v2"  : Depth-Anything-V2-Small (相对深度 + 伪 metric, 阶段 1 之前的默认)
- "vggt"   : VGGT-1B 滑窗 (metric, 阶段 1 起的新默认)

通过 build_depth_estimator(cfg) 工厂选择, 二者均实现:
    .available : bool
    .estimate(bgr_image) -> np.ndarray HxW float (米)
"""
import logging, numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
class DepthEstimator:
    """Depth-Anything-V2-Small (HF cached / 本地 pretrained 路径)。"""

    def __init__(self, model_id="pretrained/depth-anything-v2-small-hf", device="cuda:0"):
        self.device = device
        self.model_id = model_id
        self._pipe = None
        self.available = False
        try:
            from transformers import pipeline
            import torch
            self._pipe = pipeline("depth-estimation", model=model_id, device=device)
            self.available = True
            logger.info(f"DepthEstimator(DA-V2) loaded {model_id} on {device}")
        except Exception as e:
            logger.warning(f"DepthEstimator(DA-V2) unavailable, fallback uniform: {e}")

    def estimate(self, bgr_image) -> np.ndarray:
        if not self.available:
            h, w = bgr_image.shape[:2]
            return np.ones((h, w), dtype=np.float32)
        try:
            from PIL import Image
            import cv2
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            out = self._pipe(pil)
            depth = np.array(out["depth"], dtype=np.float32)
            d_min, d_max = float(depth.min()), float(depth.max())
            if d_max - d_min < 1e-6:
                return np.ones_like(depth) * 1.0
            depth = (depth - d_min) / (d_max - d_min)
            depth = 0.5 + depth * 4.5  # 伪 metric
            return depth
        except Exception as e:
            logger.warning(f"DA-V2 estimate failed: {e}")
            h, w = bgr_image.shape[:2]
            return np.ones((h, w), dtype=np.float32)


# ======================================================================
class VGGTDepthEstimator:
    """VGGT-1B 滑窗深度估计 (metric)。

    内部维护 BGR ring buffer, 每次 estimate() 把当前帧入栈并跑 VGGT,
    返回最新帧的 depth (HxW float, 米)。

    复用同一个 VGGTBackend 单例 — 后续阶段 2 (VO 替换) / 阶段 3 (point map → occupancy)
    可以通过 .last_pose / .last_world_points 属性零成本拿到。
    """

    def __init__(self, model_path="pretrained/vggt-1b/model.pt",
                 device="cuda:0", window_size=4, dtype="bf16"):
        self.device = device
        self.window_size = window_size
        self._sw = None
        self.available = False
        # 缓存最新一帧的几何输出, 供 VO/Occupancy 旁路读取
        self.last_depth = None
        self.last_depth_conf = None
        self.last_world_points = None
        self.last_extri = None
        self.last_intri = None
        try:
            from online_mapper.geometry.vggt_backend import VGGTBackend, VGGTSlidingWindow
            be = VGGTBackend.get(model_path=model_path, device=device, dtype=dtype)
            if be.available:
                self._sw = VGGTSlidingWindow(be, window_size=window_size)
                self.available = True
                logger.info(f"DepthEstimator(VGGT) ready, window={window_size}")
            else:
                logger.warning("VGGTBackend reported unavailable")
        except Exception as e:
            logger.warning(f"DepthEstimator(VGGT) unavailable: {e}", exc_info=True)

    def estimate(self, bgr_image) -> np.ndarray:
        if not self.available or bgr_image is None:
            h, w = (bgr_image.shape[:2] if bgr_image is not None else (1, 1))
            return np.ones((h, w), dtype=np.float32)
        try:
            out = self._sw.push_and_infer(bgr_image)
            self.last_depth = out["depth"]
            self.last_depth_conf = out["depth_conf"]
            self.last_world_points = out["world_points"]
            self.last_extri = out["extri"]
            self.last_intri = out["intri"]
            # VGGT 输出尺寸为 518x(高), 缩回原图尺寸供下游使用
            import cv2
            h, w = bgr_image.shape[:2]
            depth = cv2.resize(out["depth"].astype(np.float32),
                               (w, h), interpolation=cv2.INTER_LINEAR)
            return depth
        except Exception as e:
            logger.warning(f"VGGT estimate failed: {e}", exc_info=True)
            h, w = bgr_image.shape[:2]
            return np.ones((h, w), dtype=np.float32)


# ======================================================================
def build_depth_estimator(cfg):
    """根据 cfg.depth_backend 构建合适的 depth estimator。

    Args:
        cfg: OnlineMapperConfig

    Returns:
        DepthEstimator | VGGTDepthEstimator
    """
    backend = getattr(cfg, "depth_backend", "vggt")
    if backend == "vggt":
        return VGGTDepthEstimator(
            model_path=getattr(cfg, "vggt_model_path", "pretrained/vggt-1b/model.pt"),
            device=cfg.depth_device,
            window_size=getattr(cfg, "vggt_window_size", 4),
            dtype=getattr(cfg, "vggt_dtype", "bf16"),
        )
    return DepthEstimator(cfg.depth_model_id, cfg.depth_device)
