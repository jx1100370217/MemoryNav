"""VGGT-1B 推理后端封装

提供单例懒加载的 VGGT 模型, 支持:
1. 时序滑窗深度估计 (供 DepthEstimator 替换 Depth-Anything-V2)
2. 相机位姿输出 (供 VO 替换 MonoVO, 阶段 2 启用)
3. dense world point map (供 OccupancyGrid 直接填充, 阶段 3 启用)

接入位置: online_mapper/geometry/vggt_backend.py
依赖: third_party/vggt_space (源码), pretrained/vggt-1b/model.pt (权重)
"""
from __future__ import annotations
import logging, os, sys
from collections import deque
from typing import List, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

# 把 third_party/vggt_space 加到 sys.path (相对项目根)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VGGT_SRC = os.path.join(_PROJECT_ROOT, "third_party", "vggt_space")
if _VGGT_SRC not in sys.path:
    sys.path.insert(0, _VGGT_SRC)


class VGGTBackend:
    """VGGT-1B 推理后端 (进程内单例)。

    用法:
        be = VGGTBackend.get(model_path="pretrained/vggt-1b/model.pt", device="cuda:0")
        out = be.infer_bgr_list([bgr1, bgr2, bgr3])
        # out: {"depth": [H,W float meters], "extri": [3,4], "intri": [3,3],
        #       "world_points": [H,W,3], "depth_conf": [H,W]} 每项是长度为 N 的 list
    """

    _instance: Optional["VGGTBackend"] = None

    @classmethod
    def get(cls, model_path: str, device: str = "cuda:0",
            dtype: str = "bf16") -> "VGGTBackend":
        if cls._instance is None:
            cls._instance = cls(model_path, device, dtype)
        return cls._instance

    def __init__(self, model_path: str, device: str = "cuda:0", dtype: str = "bf16"):
        self.model_path = model_path
        self.device = device
        self._dtype_str = dtype
        self.model = None
        self.available = False
        self._load_fn = None
        self._pose_decode = None
        self._torch = None
        self._torch_dtype = None

        try:
            import torch
            from vggt.models.vggt import VGGT
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            self._torch = torch
            self._torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
            self._pose_decode = pose_encoding_to_extri_intri

            logger.info(f"VGGTBackend: loading {model_path} -> {device} ({dtype})")
            m = VGGT()
            sd = torch.load(model_path, map_location="cpu", weights_only=True)
            m.load_state_dict(sd, strict=True)
            self.model = m.to(device).eval()
            self.available = True
            logger.info(f"VGGTBackend ready ({sum(p.numel() for p in m.parameters())/1e6:.0f}M params)")
        except Exception as e:
            logger.warning(f"VGGTBackend unavailable: {e}", exc_info=True)

    # ------------------------------------------------------------------
    def _preprocess_bgr(self, bgr_list: List[np.ndarray]):
        """BGR uint8 列表 → tensor [N,3,H,W] in [0,1]。

        参考 vggt.utils.load_fn.load_and_preprocess_images:
        - 宽统一 518, 高按比例缩放到 14 的倍数
        - 高 > 518 则中心裁剪
        - 不同尺寸则白色 padding
        """
        import cv2
        torch = self._torch
        tensors = []
        shapes = set()
        target_w = 518
        for bgr in bgr_list:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            new_h = max(14, round(H * (target_w / W) / 14) * 14)
            rgb_r = cv2.resize(rgb, (target_w, new_h), interpolation=cv2.INTER_CUBIC)
            t = torch.from_numpy(rgb_r).float().permute(2, 0, 1) / 255.0  # [3,H,W]
            if new_h > 518:
                s = (new_h - 518) // 2
                t = t[:, s:s + 518, :]
            shapes.add(tuple(t.shape[1:]))
            tensors.append(t)

        if len(shapes) > 1:
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)
            padded = []
            for t in tensors:
                ph = max_h - t.shape[1]
                pw = max_w - t.shape[2]
                if ph or pw:
                    t = torch.nn.functional.pad(
                        t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2),
                        mode="constant", value=1.0)
                padded.append(t)
            tensors = padded

        return torch.stack(tensors).to(self.device)  # [N,3,H,W]

    # ------------------------------------------------------------------
    def infer_bgr_list(self, bgr_list: List[np.ndarray]) -> Dict[str, list]:
        """对 N 帧 BGR 图像运行 VGGT, 返回所有几何输出 (按帧拆分为 list)."""
        assert self.available, "VGGTBackend not available"
        torch = self._torch
        imgs = self._preprocess_bgr(bgr_list)  # [N,3,H,W]
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=self._torch_dtype):
                pred = self.model(imgs)

        depth = pred["depth"].float().cpu()        # [1,N,H,W,1]
        depth_conf = pred["depth_conf"].float().cpu()  # [1,N,H,W]
        wp = pred["world_points"].float().cpu()    # [1,N,H,W,3]
        extri, intri = self._pose_decode(pred["pose_enc"].float(),
                                         imgs.shape[-2:])  # [1,N,3,4], [1,N,3,3]
        extri = extri.cpu()
        intri = intri.cpu()

        N = imgs.shape[0]
        return {
            "depth":         [depth[0, i, ..., 0].numpy() for i in range(N)],
            "depth_conf":    [depth_conf[0, i].numpy() for i in range(N)],
            "world_points":  [wp[0, i].numpy() for i in range(N)],
            "extri":         [extri[0, i].numpy() for i in range(N)],
            "intri":         [intri[0, i].numpy() for i in range(N)],
        }


# ======================================================================
class VGGTSlidingWindow:
    """对 VGGTBackend 的薄包装, 维护一个 BGR 滑窗供时序推理.

    适配 online_mapper 的流式架构:
    - push(bgr) 把当前帧入栈, 自动触发 VGGT 推理
    - 提供 last_*() 接口取最新帧的 depth / pose / world points
    """

    def __init__(self, backend: VGGTBackend, window_size: int = 4):
        self.backend = backend
        self.window_size = max(1, window_size)
        self._buf: deque = deque(maxlen=self.window_size)
        self._last_out: Optional[Dict[str, list]] = None

    @property
    def available(self) -> bool:
        return self.backend.available

    def push_and_infer(self, bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """把当前帧加入滑窗并运行 VGGT, 返回最新帧 + 上一帧的几何输出.

        关键: VGGT 每次推理位姿都规范化到窗口首帧, 所以 last 与 prev
        必须同源 (来自同一次 forward) 才能直接做相对运动计算.
        """
        if bgr is None:
            return {}
        self._buf.append(bgr.copy())
        out = self.backend.infer_bgr_list(list(self._buf))
        self._last_out = out
        N = len(out["depth"])
        last = N - 1
        prev = N - 2 if N >= 2 else None
        ret = {
            "depth":        out["depth"][last],
            "depth_conf":   out["depth_conf"][last],
            "world_points": out["world_points"][last],
            "extri":        out["extri"][last],
            "intri":        out["intri"][last],
            "prev_extri":   out["extri"][prev] if prev is not None else None,
            "prev_intri":   out["intri"][prev] if prev is not None else None,
        }
        return ret

    def infer_stateless(self, bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """单帧推理 (不入栈, 不带 buffer 上下文).

        给 junction_detector 这种"只要绝对深度中位数"的旁路用 — 速度优先,
        VGGT 单帧依然给出合理 metric depth.
        """
        if bgr is None or not self.available:
            return {}
        out = self.backend.infer_bgr_list([bgr])
        last = 0
        return {
            "depth":        out["depth"][last],
            "depth_conf":   out["depth_conf"][last],
            "world_points": out["world_points"][last],
            "extri":        out["extri"][last],
            "intri":        out["intri"][last],
        }
