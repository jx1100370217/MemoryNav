#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动子图提取器 v10 — PointGrounding + 走廊中间帧匹配 + 匈牙利 + Y坐标修正

v9 问题: 用 crop/全图 CLS 特征匹配 camera→邻居节点全图, 但 camera 朝向邻居时
  看到的是两节点之间的走廊/通道, 不是邻居节点本身的场景, 导致匹配错误

v10 修复: 用走廊中间帧(intermediate frames)作为匹配目标
  - 机器人从 NodeA 走到 NodeB 的中间帧就在走廊里拍摄
  - NodeB 的 camera 朝向 NodeA 时, 看到的走廊与中间帧高度重叠
  - 匹配 crop 特征 vs 中间帧特征, 语义上完全正确
"""

import os, sys, cv2, torch, numpy as np, logging, base64
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')

from memory_nav.qwen35_point_grounder import Qwen35PointGrounder

logger = logging.getLogger(__name__)


class AutoSubImageExtractor:
    TARGET_W_RATIO = 0.27
    TARGET_H_RATIO = 0.34
    MID_SHRINK = 0.87
    SMALL_SHRINK = 0.75

    POINT_PROMPT = "通道正中间位置"

    # Y 坐标限制 (人工标注规律: Y ≈ 47%~50%)
    MAX_Y_PCT = 0.52    # Y 超过此比例则修正
    TARGET_Y_PCT = 0.48  # 修正目标 Y

    # 走廊中间帧匹配参数
    MAX_CORRIDOR_FRAMES = 30   # 超过此数量的中间帧则 fallback (可能不是直接走廊)
    CORRIDOR_SAMPLE_COUNT = 3  # 采样中间帧数量

    def __init__(self, device="cuda:0", qwen_gpu="1"):
        self.device = device
        self.qwen_gpu = qwen_gpu

        self._grounder = Qwen35PointGrounder(gpu=qwen_gpu)
        self._grounder.start()

        from memory_nav.sub_image_matcher import DINOv3Strategy
        self._dinov3 = DINOv3Strategy(device=device)
        self._dinov3.preload()

        logger.info("AutoSubImageExtractor v10.1 (parallel cameras + corridor-frame match + Hungarian + Y-fix) initialized")

    # ------------------------------------------------------------------
    # CLS 特征
    # ------------------------------------------------------------------
    def _cls_feature(self, img):
        t, h, w = self._dinov3._prepare_image(img, target_size=518)
        with torch.no_grad():
            f = self._dinov3._model.forward_features(t)
        return torch.nn.functional.normalize(f[:, 0, :].squeeze(0), dim=-1)

    def _cos_sim(self, feat_a, feat_b):
        return torch.cosine_similarity(feat_a.unsqueeze(0), feat_b.unsqueeze(0)).item()

    # ------------------------------------------------------------------
    # Y 坐标修正
    # ------------------------------------------------------------------
    def _fix_point_y(self, cx, cy, img_h, img_w):
        y_pct = cy / img_h
        if y_pct > self.MAX_Y_PCT:
            old_cy = cy
            cy = int(img_h * self.TARGET_Y_PCT)
            logger.info(f"    Y-fix: {old_cy}({y_pct*100:.1f}%) -> {cy}({self.TARGET_Y_PCT*100:.1f}%)")
        return cx, cy

    # ------------------------------------------------------------------
    # 正方形 Crop
    # ------------------------------------------------------------------
    def _make_square_crop(self, img, cx, cy, scale=1.0):
        h, w = img.shape[:2]
        side = int(min(w * self.TARGET_W_RATIO, h * self.TARGET_H_RATIO) * scale)
        x1 = max(0, min(w - side, cx - side // 2))
        y1 = max(0, min(h - side, cy - side // 2))
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        return img[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _save_crops(self, img, center, output_dir, timestamp, camera_name, next_idx):
        h, w = img.shape[:2]
        os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)
        cam_num = camera_name.split('_')[1]
        crop_paths, norm_boxes = {}, {}
        big_crop_img = None
        cx, cy = center
        logger.info(f"  Saving crops: center=({cx},{cy}) = ({cx/w*100:.1f}%,{cy/h*100:.1f}%)")

        for size, scale in [('big', 1.0), ('mid', self.MID_SHRINK), ('small', self.SMALL_SHRINK)]:
            cropped, (x1, y1, x2, y2) = self._make_square_crop(img, cx, cy, scale)
            if cropped.size == 0:
                continue
            fname = f"{timestamp}_camera_{cam_num}__{next_idx}__{size}__{x1}_{y1}_{x2-x1}_{y2-y1}.jpg"
            cv2.imwrite(os.path.join(output_dir, "crops", fname), cropped)
            crop_paths[size] = f"crops/{fname}"
            norm_boxes[size] = f"{x1/w:.6f},{y1/h:.6f},{x2/w:.6f},{y2/h:.6f}"
            if size == 'big':
                big_crop_img = cropped
            logger.info(f"    {size}: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}")

        return crop_paths, norm_boxes, big_crop_img

    # ------------------------------------------------------------------
    # 走廊中间帧特征计算
    # ------------------------------------------------------------------
    def _compute_corridor_features(self, all_frames, my_frame_idx, nb_frame_idx, nb_id):
        """
        计算两个节点之间走廊中间帧的 CLS 特征.
        返回特征列表, 如果无法计算则返回 None.
        """
        f_start = min(my_frame_idx, nb_frame_idx)
        f_end = max(my_frame_idx, nb_frame_idx)

        mid_indices = list(range(f_start + 1, f_end))

        if not mid_indices:
            logger.info(f"  Neighbor {nb_id}: no intermediate frames (adjacent), fallback")
            return None

        if len(mid_indices) > self.MAX_CORRIDOR_FRAMES:
            logger.info(f"  Neighbor {nb_id}: too many intermediate frames "
                       f"({len(mid_indices)}), probably not direct corridor, fallback")
            return None

        # 采样: 均匀取 CORRIDOR_SAMPLE_COUNT 帧
        if len(mid_indices) > self.CORRIDOR_SAMPLE_COUNT:
            n = len(mid_indices)
            step = n / (self.CORRIDOR_SAMPLE_COUNT + 1)
            sampled = [mid_indices[int(step * (i + 1))] for i in range(self.CORRIDOR_SAMPLE_COUNT)]
            mid_indices = sampled

        logger.info(f"  Neighbor {nb_id}: using {len(mid_indices)} corridor frames "
                   f"(frame indices: {mid_indices})")

        corridor_feats = []
        for idx in mid_indices:
            frame = all_frames[idx]
            for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
                cam_path = frame['images'].get(cam_id)
                if cam_path:
                    img = cv2.imread(cam_path)
                    if img is not None:
                        corridor_feats.append(self._cls_feature(img))

        logger.info(f"  Neighbor {nb_id}: {len(corridor_feats)} corridor features computed")
        return corridor_feats if corridor_feats else None

    def _compute_neighbor_full_features(self, neighbor_node):
        """Fallback: 用邻居节点自身的 4 张全图特征"""
        nb_id = neighbor_node['position_id']
        feats = []
        for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            nb_path = neighbor_node['images'].get(cam_id)
            if nb_path:
                nb_img = cv2.imread(nb_path)
                if nb_img is not None:
                    feats.append(self._cls_feature(nb_img))
        if feats:
            logger.info(f"  Neighbor {nb_id}: using {len(feats)} full image features (fallback)")
        return feats

    # ------------------------------------------------------------------
    # 核心: 生成一个 node 的所有 next_positions
    # ------------------------------------------------------------------
    def generate_next_positions(self, node_info, neighbor_nodes, all_frames=None, qwen_namer=None):
        position_id = node_info['position_id']
        timestamp = node_info['timestamp']
        node_dir = node_info['node_dir']
        images = node_info['images']
        my_frame_idx = node_info.get('frame_index')

        logger.info(f"=== Generating next_positions for node {position_id} ===")

        # ---- Step 1: 在每个相机上并行打点 + Y修正 (v10.1: 4 cameras 并发) ----
        camera_points = {}

        def _predict_one_camera(cam_id, cam_path):
            cam_img = cv2.imread(cam_path)
            if cam_img is None:
                return cam_id, None
            result = self._grounder.predict(cam_img, self.POINT_PROMPT)
            if result['success'] and result['point']:
                h, w = cam_img.shape[:2]
                cx = int(result['point'][0] * w)
                cy = int(result['point'][1] * h)
                conf = result.get('confidence', 0.0)
                return cam_id, (cx, cy, conf, h, w)
            return cam_id, None

        cam_tasks = [(cid, images[cid]) for cid in ['camera_1', 'camera_2', 'camera_3', 'camera_4']
                     if images.get(cid)]

        with ThreadPoolExecutor(max_workers=len(cam_tasks)) as executor:
            futures = {executor.submit(_predict_one_camera, cid, cp): cid for cid, cp in cam_tasks}
            for future in as_completed(futures):
                cam_id, result = future.result()
                if result is not None:
                    cx, cy, conf, h, w = result
                    logger.info(f"  {cam_id}: raw point=({cx},{cy}) = ({cx/w*100:.1f}%,{cy/h*100:.1f}%) conf={conf:.3f}")
                    cx, cy = self._fix_point_y(cx, cy, h, w)
                    camera_points[cam_id] = (cx, cy, conf)
                else:
                    logger.info(f"  {cam_id}: no point (target not found)")

        if not camera_points:
            logger.warning(f"  Node {position_id}: no camera has valid corridor point!")
            return []

        # ---- Step 2: 裁 crop + 提取 crop CLS 特征 ----
        cam_crop_features = {}
        cam_crop_cache = {}
        for cam_id in camera_points:
            cam_img = cv2.imread(images[cam_id])
            if cam_img is None:
                continue
            cx, cy, conf = camera_points[cam_id]
            crop_img, crop_box = self._make_square_crop(cam_img, cx, cy, scale=1.0)
            if crop_img.size == 0:
                continue
            cam_crop_features[cam_id] = self._cls_feature(crop_img)
            cam_crop_cache[cam_id] = (cam_img, cx, cy)
            logger.info(f"  {cam_id}: crop feature extracted, box={crop_box}")

        # ---- Step 3: 计算每个邻居对应的走廊特征 ----
        # v10 核心改动: 优先用中间帧特征, 而不是邻居节点自身的全图特征
        # 原理: camera 朝向邻居时看到的是走廊, 走廊中间帧与之高度重叠
        use_corridor = (all_frames is not None and my_frame_idx is not None)

        neighbor_features = {}
        for nb in neighbor_nodes:
            nb_id = nb['position_id']
            nb_frame_idx = nb.get('frame_index')

            feats = None
            if use_corridor and nb_frame_idx is not None:
                feats = self._compute_corridor_features(
                    all_frames, my_frame_idx, nb_frame_idx, nb_id)

            if feats is None:
                feats = self._compute_neighbor_full_features(nb)

            if feats:
                neighbor_features[nb_id] = feats

        # ---- Step 4: 匈牙利算法匹配 (crop特征 vs 走廊中间帧特征) ----
        cam_ids = list(cam_crop_features.keys())
        nb_ids = list(neighbor_features.keys())

        if not cam_ids or not nb_ids:
            return []

        sim_matrix = np.zeros((len(cam_ids), len(nb_ids)))
        for i, cam_id in enumerate(cam_ids):
            for j, nb_id in enumerate(nb_ids):
                feats = neighbor_features.get(nb_id, [])
                if feats:
                    best_sim = max(
                        self._cos_sim(cam_crop_features[cam_id], f)
                        for f in feats
                    )
                    sim_matrix[i][j] = best_sim

        match_target = "corridor frames" if use_corridor else "neighbor full images"
        logger.info(f"  Similarity matrix (crop features vs {match_target}):")
        for i, cam_id in enumerate(cam_ids):
            row = ", ".join(f"{nb_ids[j]}={sim_matrix[i][j]:.3f}" for j in range(len(nb_ids)))
            logger.info(f"    {cam_id}: [{row}]")

        from scipy.optimize import linear_sum_assignment
        cost_matrix = -sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        for r, c in zip(row_ind, col_ind):
            cam_id = cam_ids[r]
            nb_id = nb_ids[c]
            sim = sim_matrix[r][c]
            matches.append((cam_id, nb_id, sim))
            logger.info(f"  Hungarian: {cam_id} -> node {nb_id} (sim={sim:.3f})")

        # ---- Step 5: 保存 crop 并生成 next_positions ----
        next_positions = []
        for cam_id, nb_id, sim in matches:
            if cam_id not in cam_crop_cache:
                continue
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
            pos_name = nb_info.get('position_name', f'auto_{nb_id}') if nb_info else f'auto_{nb_id}'
            pos_name_en = nb_info.get('position_name_eng', f'auto_{nb_id}') if nb_info else f'auto_{nb_id}'

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
            })

        logger.info(f"  Node {position_id}: generated {len(next_positions)} connections")
        return next_positions

    # ------------------------------------------------------------------
    # 兼容旧接口
    # ------------------------------------------------------------------
    def extract_connection(self, current_node_images, next_node_images,
                           current_timestamp, next_position_id, output_dir,
                           intermediate_frames=None, qwen_namer=None):
        logger.warning("extract_connection called (legacy)")
        for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            cam_path = current_node_images.get(cam_id)
            if not cam_path:
                continue
            cam_img = cv2.imread(cam_path)
            if cam_img is None:
                continue
            result = self._grounder.predict(cam_img, self.POINT_PROMPT)
            if result['success'] and result['point']:
                h, w = cam_img.shape[:2]
                cx = int(result['point'][0] * w)
                cy = int(result['point'][1] * h)
                cx, cy = self._fix_point_y(cx, cy, h, w)
                next_idx = next_position_id.split('_')[0] if '_' in next_position_id else next_position_id
                crop_paths, norm_boxes, _ = self._save_crops(
                    cam_img, (cx, cy), output_dir, current_timestamp, cam_id, next_idx)
                if crop_paths:
                    return {
                        "position_id": next_position_id,
                        "position_name": f"auto_{next_idx}",
                        "camera_name": cam_id,
                        "landmark_name": " ",
                        "big_box": norm_boxes.get('big', ''),
                        "mid_box": norm_boxes.get('mid', ''),
                        "small_box": norm_boxes.get('small', ''),
                        "pixel_box": "",
                        "crop_image_path": crop_paths.get('big', ''),
                        "crop_image_paths": crop_paths,
                        "position_name_eng": f"auto_{next_idx}",
                        "landmark_name_eng": " ",
                    }
        return None

    def cleanup(self):
        if self._grounder:
            self._grounder.stop()
