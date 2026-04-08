"""ConnectionBuilder — 包装 offline_mapper 的 AutoSubImageExtractor

子类化 AutoSubImageExtractor 重写 generate_next_positions, 在 Hungarian
匹配后增加相似度阈值过滤 (默认 0.40), 防止线性走廊场景出现 garbage 匹配。

不修改 offline_mapper, 仅 import / 子类化。
"""
import os, sys, logging, cv2, numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')

from offline_mapper.auto_sub_image_extractor import AutoSubImageExtractor


class ThresholdedSubImageExtractor(AutoSubImageExtractor):
    """AutoSubImageExtractor + Hungarian 匹配相似度阈值过滤"""

    SIM_THRESHOLD = 0.40

    def __init__(self, sim_threshold: float = None, **kwargs):
        super().__init__(**kwargs)
        if sim_threshold is not None:
            self.SIM_THRESHOLD = sim_threshold
        logger.info(f"[ThresholdedSubImageExtractor] sim_threshold={self.SIM_THRESHOLD}")

    def generate_next_positions(self, node_info, neighbor_nodes,
                                 all_frames=None, qwen_namer=None):
        position_id = node_info['position_id']
        timestamp = node_info['timestamp']
        node_dir = node_info['node_dir']
        images = node_info['images']
        my_frame_idx = node_info.get('frame_index')

        logger.info(f"=== [Thresholded] node {position_id} (sim>={self.SIM_THRESHOLD}) ===")

        # ---- Step 1: 4 cameras 并行打点 + Y 修正 ----
        camera_points = {}

        def _predict_one(cam_id, cam_path):
            cam_img = cv2.imread(cam_path)
            if cam_img is None:
                return cam_id, None
            r = self._grounder.predict(cam_img, self.POINT_PROMPT)
            if r['success'] and r['point']:
                h, w = cam_img.shape[:2]
                cx = int(r['point'][0] * w); cy = int(r['point'][1] * h)
                return cam_id, (cx, cy, r.get('confidence', 0.0), h, w)
            return cam_id, None

        cam_tasks = [(c, images[c]) for c in
                     ['camera_1', 'camera_2', 'camera_3', 'camera_4']
                     if images.get(c)]
        with ThreadPoolExecutor(max_workers=max(1, len(cam_tasks))) as ex:
            futures = {ex.submit(_predict_one, c, p): c for c, p in cam_tasks}
            for fu in as_completed(futures):
                cam_id, res = fu.result()
                if res is not None:
                    cx, cy, conf, h, w = res
                    cx, cy = self._fix_point_y(cx, cy, h, w)
                    camera_points[cam_id] = (cx, cy, conf)

        if not camera_points:
            logger.warning(f"  node {position_id}: no point on any camera")
            return []

        # ---- Step 2: crop CLS feat ----
        cam_crop_features, cam_crop_cache = {}, {}
        for cam_id, (cx, cy, _) in camera_points.items():
            cam_img = cv2.imread(images[cam_id])
            if cam_img is None:
                continue
            crop_img, _ = self._make_square_crop(cam_img, cx, cy, scale=1.0)
            if crop_img.size == 0:
                continue
            cam_crop_features[cam_id] = self._cls_feature(crop_img)
            cam_crop_cache[cam_id] = (cam_img, cx, cy)

        # ---- Step 3: neighbor corridor / fallback features ----
        use_corridor = (all_frames is not None and my_frame_idx is not None)
        neighbor_features = {}
        for nb in neighbor_nodes:
            nb_id = nb['position_id']
            feats = None
            nb_fi = nb.get('frame_index')
            if use_corridor and nb_fi is not None:
                feats = self._compute_corridor_features(
                    all_frames, my_frame_idx, nb_fi, nb_id)
            if feats is None:
                feats = self._compute_neighbor_full_features(nb)
            if feats:
                neighbor_features[nb_id] = feats

        cam_ids = list(cam_crop_features.keys())
        nb_ids = list(neighbor_features.keys())
        if not cam_ids or not nb_ids:
            return []

        # ---- Step 4: similarity matrix + Hungarian ----
        sim_matrix = np.zeros((len(cam_ids), len(nb_ids)))
        for i, cam_id in enumerate(cam_ids):
            for j, nb_id in enumerate(nb_ids):
                feats = neighbor_features.get(nb_id, [])
                if feats:
                    sim_matrix[i][j] = max(
                        self._cos_sim(cam_crop_features[cam_id], f) for f in feats)

        # ---- Step 4.5: 几何方向先验 (cam_1=front, cam_2=right, cam_3=rear, cam_4=left) ----
        # 修复纯视觉匹配在线性走廊+相似 landmark 场景下的方向错配.
        my_pose = node_info.get("pose")
        if my_pose is not None:
            import math as _math
            cam_angles = {
                "camera_1": 0.0,
                "camera_2": -_math.pi / 2,
                "camera_3":  _math.pi,
                "camera_4":  _math.pi / 2,
            }
            def _wrap(a):
                return _math.atan2(_math.sin(a), _math.cos(a))
            mx, my, mth = my_pose
            geo_bonus = np.zeros_like(sim_matrix)
            for i, cam_id in enumerate(cam_ids):
                ca = cam_angles.get(cam_id, 0.0)
                for j, nb_id in enumerate(nb_ids):
                    nb_obj = next((n for n in neighbor_nodes if n["position_id"] == nb_id), None)
                    nbp = nb_obj.get("pose") if nb_obj else None
                    if nbp is None:
                        continue
                    nx, ny, _ = nbp
                    dx, dy = nx - mx, ny - my
                    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                        continue
                    world_ang = _math.atan2(dy, dx)
                    robot_ang = _wrap(world_ang - mth)  # neighbor 相对机器人朝向的角度
                    diff = _wrap(robot_ang - ca)
                    score = _math.cos(diff)             # ∈ [-1, 1]
                    geo_bonus[i][j] = score
                    if score < -0.3:
                        # 相机和 neighbor 方向夹角 > ~108°, 几乎不可能看到, 强力惩罚
                        sim_matrix[i][j] -= 1.0
            # 几何融合: visual_sim + α * angular_cos
            ALPHA = 0.6
            sim_matrix = sim_matrix + ALPHA * geo_bonus
            for i, cam_id in enumerate(cam_ids):
                for j, nb_id in enumerate(nb_ids):
                    logger.info(f"  geo[{cam_id}->{nb_id}] cos={geo_bonus[i][j]:+.2f} "
                                f"final_sim={sim_matrix[i][j]:+.3f}")

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)

        matches = []
        for r, c in zip(row_ind, col_ind):
            sim = float(sim_matrix[r][c])
            cam_id = cam_ids[r]; nb_id = nb_ids[c]
            if sim < self.SIM_THRESHOLD:
                logger.info(f"  DROP {cam_id}->{nb_id} sim={sim:.3f} < {self.SIM_THRESHOLD}")
                continue
            matches.append((cam_id, nb_id, sim))
            logger.info(f"  KEEP {cam_id}->{nb_id} sim={sim:.3f}")

        # ---- Step 5: save crops + return next_positions ----
        next_positions = []
        for cam_id, nb_id, sim in matches:
            cam_img, cx, cy = cam_crop_cache[cam_id]
            crop_paths, norm_boxes, big_crop_img = self._save_crops(
                cam_img, (cx, cy), node_dir, timestamp, cam_id, nb_id)
            if not crop_paths:
                continue

            landmark_cn, landmark_en = " ", " "
            if qwen_namer and big_crop_img is not None:
                try:
                    lm_cn, lm_en = qwen_namer.qwen_identify_landmark(big_crop_img)
                    if lm_cn and lm_cn != " ":
                        landmark_cn, landmark_en = lm_cn, lm_en
                except Exception:
                    pass

            nb_info = next((n for n in neighbor_nodes if n['position_id'] == nb_id), None)
            pos_name = nb_info.get('position_name', f'node_{nb_id}') if nb_info else f'node_{nb_id}'
            pos_name_en = nb_info.get('position_name_eng', f'node_{nb_id}') if nb_info else f'node_{nb_id}'

            next_positions.append({
                "position_id": nb_id,
                "position_name": pos_name,
                "camera_name": cam_id,
                "landmark_name": landmark_cn,
                "big_box": norm_boxes.get('big', ''),
                "mid_box": norm_boxes.get('mid', ''),
                "small_box": norm_boxes.get('small', ''),
                "pixel_box": "",
                "crop_image_path": crop_paths.get('big', ''),
                "crop_image_paths": crop_paths,
                "position_name_eng": pos_name_en,
                "landmark_name_eng": landmark_en,
                "_match_sim": sim,
            })

        logger.info(f"  node {position_id}: {len(next_positions)} kept connections")
        return next_positions


class ConnectionBuilder:
    """Online TopoNode -> offline_mapper-format dict, drives ThresholdedSubImageExtractor"""

    def __init__(self, sim_threshold: float = 0.40, device: str = "cuda:0",
                 qwen_gpu: str = "1", namer=None):
        self.sim_threshold = sim_threshold
        self._device = device
        self._qwen_gpu = qwen_gpu
        self._namer = namer
        self._extractor: Optional[ThresholdedSubImageExtractor] = None

    def _ensure(self):
        if self._extractor is None:
            self._extractor = ThresholdedSubImageExtractor(
                sim_threshold=self.sim_threshold,
                device=self._device, qwen_gpu=self._qwen_gpu)

    @staticmethod
    def topo_node_to_dict(node, node_dir: Path, pose_graph=None) -> Dict:
        d = {
            "position_id": node.node_id,
            "position_name": getattr(node, "position_name", "") or f"node_{node.node_id}",
            "position_name_eng": getattr(node, "position_name_eng", "") or f"node_{node.node_id}",
            "timestamp": node.timestamp,
            "node_dir": str(node_dir),
            "images": dict(node.cameras),
            "frame_index": node.frame_idx,
        }
        if pose_graph is not None and node.node_id in pose_graph.nodes:
            pn = pose_graph.nodes[node.node_id]
            d["pose"] = (pn.x, pn.y, pn.theta)
        return d

    @staticmethod
    def frame_to_dict(frame: Dict) -> Dict:
        return {
            "timestamp": frame["timestamp"],
            "frame_index": frame["frame_idx"],
            "images": dict(frame["cameras"]),
        }

    def build_for_node(self, node, neighbors: List, node_dir: Path,
                       neighbor_dirs: Dict[str, Path],
                       all_frames: List[Dict],
                       pose_graph=None) -> List[Dict]:
        self._ensure()
        node_info = self.topo_node_to_dict(node, node_dir, pose_graph=pose_graph)
        nb_dicts = [self.topo_node_to_dict(nb, neighbor_dirs[nb.node_id], pose_graph=pose_graph)
                    for nb in neighbors]
        all_meta = [self.frame_to_dict(f) for f in all_frames]
        try:
            return self._extractor.generate_next_positions(
                node_info=node_info, neighbor_nodes=nb_dicts,
                all_frames=all_meta, qwen_namer=self._namer)
        except Exception as e:
            logger.warning(f"[ConnectionBuilder] node {node.node_id} failed: {e}")
            import traceback; traceback.print_exc()
            return []

    def cleanup(self):
        if self._extractor:
            try:
                self._extractor.cleanup()
            except Exception:
                pass
            self._extractor = None
