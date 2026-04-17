"""ConnectionBuilder — 包装 AutoSubImageExtractor

子类化 AutoSubImageExtractor 重写 generate_next_positions, 在 Hungarian
匹配后增加相似度阈值过滤 (默认 0.40), 防止线性走廊场景出现 garbage 匹配。
"""
import os, sys, logging, cv2, numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

from online_mapper.topology.auto_sub_image_extractor import AutoSubImageExtractor


class ThresholdedSubImageExtractor(AutoSubImageExtractor):
    """AutoSubImageExtractor + Hungarian 匹配相似度阈值过滤"""

    SIM_THRESHOLD = 0.40

    # ---- 几何 crop 兜底 (Round 7) ----
    # 约定: 世界/机器人坐标 x=前, y=左, theta=CCW (atan2 风格).
    # 柱面图 FOV=180°, pixel_norm_to_angle: (0.5-x_norm)*π, 画面左 theta_h>0.
    # 相机方位角来自 memory_nav/coord_transform.py _DEFAULT_AZIMUTHS
    # (从 params.yaml T_ic 计算, 逆时针正).
    HFOV_DEG = 180.0
    USE_GEOMETRIC_FALLBACK = True
    # |cx/w - 0.5| <= FALLBACK_CENTER_TOL 判为 Qwen 居中 fallback, 触发几何替换.
    # Qwen 真实点通常偏离中心 5% 以上, 居中 fallback 精确落在 0.500.
    FALLBACK_CENTER_TOL = 0.03
    _CAM_AZIMUTH_DEG = {
        "camera_1":   39.42,
        "camera_2":  -35.84,
        "camera_3": -142.04,
        "camera_4":  143.52,
    }

    def __init__(self, sim_threshold: float = None, **kwargs):
        super().__init__(**kwargs)
        if sim_threshold is not None:
            self.SIM_THRESHOLD = sim_threshold
        logger.info(f"[ThresholdedSubImageExtractor] sim_threshold={self.SIM_THRESHOLD} "
                    f"geo_fallback={self.USE_GEOMETRIC_FALLBACK} hfov={self.HFOV_DEG} "
                    f"center_tol={self.FALLBACK_CENTER_TOL}")

    def _project_target_to_camera(self, self_pose, target_pose, cam_id,
                                    img_w, img_h):
        """位姿投影: target 在给定 camera 柱面图上的像素 (cx, cy).

        返回 None 如果目标落在相机 FOV 之外或 cam_id 未知.
        """
        import math
        if cam_id not in self._CAM_AZIMUTH_DEG:
            return None
        mx, my, mth = self_pose[0], self_pose[1], self_pose[2]
        nx, ny = target_pose[0], target_pose[1]
        dx, dy = nx - mx, ny - my
        if dx * dx + dy * dy < 1e-8:
            return None
        world_ang = math.atan2(dy, dx)
        robot_ang = math.atan2(math.sin(world_ang - mth),
                                math.cos(world_ang - mth))
        cam_az = math.radians(self._CAM_AZIMUTH_DEG[cam_id])
        theta_h = math.atan2(math.sin(robot_ang - cam_az),
                              math.cos(robot_ang - cam_az))
        half = math.radians(self.HFOV_DEG) / 2.0
        if abs(theta_h) >= half:
            return None
        x_norm = 0.5 - theta_h / math.radians(self.HFOV_DEG)
        x_norm = max(0.02, min(0.98, x_norm))
        cx = int(x_norm * img_w)
        cy = int(self.TARGET_Y_PCT * img_h)
        return (cx, cy, theta_h)

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

        # Qwen 全相机失败 (空旷广场无通道锚点, 如 D 栋保安亭): 退回图像中心,
        # 让后续 Hungarian + geo override 照常工作, 避免节点 0 连接成孤岛.
        if not camera_points:
            logger.warning(f"  node {position_id}: qwen failed all cams, "
                            f"fallback to image center")
            for cam_id, cam_path in cam_tasks:
                cam_img = cv2.imread(cam_path)
                if cam_img is None:
                    continue
                h, w = cam_img.shape[:2]
                camera_points[cam_id] = (w // 2, int(h * self.TARGET_Y_PCT), 0.0)
        if not camera_points:
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

        # ---- Step 4.5: 几何方向先验 ----
        # 使用权威 camera azimuth (memory_nav/coord_transform._DEFAULT_AZIMUTHS,
        # 由 params.yaml T_ic 推出): cam1=+39.42°, cam2=-35.84°,
        # cam3=-142.04°, cam4=+143.52° (逆时针正, y 轴向左).
        # 修复纯视觉匹配在线性走廊+相似 landmark 场景下的方向错配.
        my_pose = node_info.get("pose")
        if my_pose is not None:
            import math as _math
            cam_angles = {k: _math.radians(v)
                           for k, v in self._CAM_AZIMUTH_DEG.items()}
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
            # α 不能太大: VO/pose_graph 的 theta 在室内场景 (小转弯, 短轨迹) 经常漂移
            # 几百度, 如果 α 过大会让"几何看似对齐但视觉完全对不上"的相机胜过
            # "视觉显著最好但几何方向偏"的真正对应相机.
            # 实测: α=0.6 时, 前台→关爱室被误判成 camera_1 (pose 说正对, 但图像
            # 是前台柜台); α=0.2 时正确落到 camera_2 (走廊视角, 视觉相似度 0.82).
            # cos<-0.3 的 -1 硬惩罚保留, 防止背向相机混进匹配.
            ALPHA = 0.2
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

        # ---- Step 4.5: Qwen 居中 fallback 时用几何投影兜底 (Round 7) ----
        # Qwen "通道正中间位置" 在室内走廊锚得住, 用户确认 test2 结果很好.
        # 室外广场/大空间 prompt 失锚, Qwen 退化为图像正中心 cx≈0.500.
        # 策略: 仅当 |cx/w - 0.5| <= FALLBACK_CENTER_TOL 时判定为居中 fallback,
        # 用 self→target pose 投影替换; 其他情况保留 Qwen 点.
        geo_overrides = {}
        if self.USE_GEOMETRIC_FALLBACK and my_pose is not None:
            for cam_id, nb_id, sim in matches:
                if cam_id not in cam_crop_cache:
                    continue
                cam_img, old_cx, old_cy = cam_crop_cache[cam_id]
                h, w = cam_img.shape[:2]
                cx_norm = old_cx / float(w) if w else 0.5
                if abs(cx_norm - 0.5) > self.FALLBACK_CENTER_TOL:
                    logger.info(f"  GEO[{cam_id}->{nb_id}] qwen cx={cx_norm:.3f} "
                                f"not centered, keep qwen")
                    continue
                nb_obj = next((n for n in neighbor_nodes
                                if n["position_id"] == nb_id), None)
                nbp = nb_obj.get("pose") if nb_obj else None
                if nbp is None:
                    continue
                proj = self._project_target_to_camera(my_pose, nbp, cam_id, w, h)
                if proj is None:
                    logger.info(f"  GEO[{cam_id}->{nb_id}] out of FOV, keep qwen "
                                f"center ({old_cx},{old_cy})")
                    continue
                new_cx, new_cy, theta_h = proj
                geo_overrides[cam_id] = (new_cx, new_cy)
                logger.info(f"  GEO[{cam_id}->{nb_id}] qwen-centered (cx={cx_norm:.3f}) "
                            f"-> geo theta_h={theta_h:+.3f}rad ({theta_h*57.30:+.1f}°) "
                            f"new=({new_cx},{new_cy})")

        # ---- Step 5: save crops + return next_positions ----
        next_positions = []
        for cam_id, nb_id, sim in matches:
            cam_img, old_cx, old_cy = cam_crop_cache[cam_id]
            cx, cy = geo_overrides.get(cam_id, (old_cx, old_cy))
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
    """Online TopoNode -> merged_labeled_data-format dict, drives ThresholdedSubImageExtractor"""

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
