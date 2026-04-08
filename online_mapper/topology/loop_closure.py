"""全局 VPR 闭环检测 + 几何验证

改进:
- auto-tune threshold: 根据观测到的 sim 分布动态调整 (mean+2*std)
- 几何验证: 对 top-k 候选用 ORB 特征匹配 + RANSAC, 内点 >= min_inliers 才接受
"""
import logging
import numpy as np
import cv2
logger = logging.getLogger(__name__)


class LoopCloser:
    def __init__(self, cfg, distance_estimator):
        self.cfg = cfg
        self.estimator = distance_estimator
        self._sim_history: list = []
        self._auto_thresh = None
        self._orb = cv2.ORB_create(nfeatures=800)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def update_sim(self, sim: float):
        if sim is not None and sim > 0:
            self._sim_history.append(float(sim))
            if len(self._sim_history) >= 5:
                arr = np.asarray(self._sim_history[-200:])
                mu = float(arr.mean()); sd = float(arr.std() + 1e-6)
                # robust threshold: mean + 2 sigma (top 2.5%), but bounded
                self._auto_thresh = float(min(0.92, max(0.65, mu + 2.0 * sd)))

    def current_threshold(self) -> float:
        # 取 cfg / auto 的较小者 (更宽容)
        cfg_thr = getattr(self.cfg, "loop_closure_vpr_threshold", 0.78)
        if self._auto_thresh is None:
            return cfg_thr
        return min(cfg_thr, self._auto_thresh)

    def detect(self, current_node_id, current_features, all_nodes,
               frame_idx_of, current_frame_idx):
        thr = self.current_threshold()
        candidates = []
        for nid, feats in all_nodes.items():
            if nid == current_node_id:
                continue
            f_old = frame_idx_of.get(nid, -1)
            if abs(current_frame_idx - f_old) < self.cfg.loop_closure_min_gap:
                continue
            sim = self._cyclic_cosine(current_features, feats)
            self.update_sim(sim)
            if sim >= thr:
                candidates.append((nid, float(sim)))
        candidates.sort(key=lambda x: -x[1])
        return candidates[: self.cfg.loop_closure_top_k]

    def geometric_verify(self, img_a_path: str, img_b_path: str,
                         min_inliers: int = 15) -> bool:
        """ORB + RANSAC 内点数验证"""
        try:
            ia = cv2.imread(img_a_path); ib = cv2.imread(img_b_path)
            if ia is None or ib is None:
                return False
            ga = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)
            ka, da = self._orb.detectAndCompute(ga, None)
            kb, db = self._orb.detectAndCompute(gb, None)
            if da is None or db is None or len(ka) < 30 or len(kb) < 30:
                return False
            matches = self._bf.match(da, db)
            if len(matches) < min_inliers:
                return False
            pts1 = np.float32([ka[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kb[m.trainIdx].pt for m in matches])
            _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
            if mask is None:
                return False
            inliers = int(mask.sum())
            return inliers >= min_inliers
        except Exception as e:
            logger.debug(f"geom verify failed: {e}")
            return False

    def _cyclic_cosine(self, fa, fb):
        cams = ['camera_1', 'camera_2', 'camera_3', 'camera_4']
        if not all(c in fa and c in fb for c in cams):
            return 0.0
        best = -1.0
        for shift in range(4):
            sims = []
            for i, c in enumerate(cams):
                cb = cams[(i + shift) % 4]
                a = fa[c]; b = fb[cb]
                s = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                sims.append(s)
            best = max(best, sum(sims) / 4.0)
        return best
