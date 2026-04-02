#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语义节点检测器 v2.2

v2.2 改进:
1. 只检测前方 camera (camera_1, camera_2)，避免同帧走廊两侧不同门牌创建重复 node
2. 同帧前方 camera 检测到多个不同名称时，用 bbox 面积选最近的
3. 场景分类也只用前方 camera
4. 保留 v2.1 的黑名单、交叉验证、YOLO 辅助
"""

import os
import sys
import cv2
import logging
import base64
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')

logger = logging.getLogger(__name__)

# 无导航意义的标识黑名单
USELESS_SIGNS = {
    '安全出口', '安全出口标识', '消防', '消防标识', '禁止吸烟',
    '门牌/房间标识', '房间标识', '标识', '标识牌', '走廊', '办公区',
    '通道', '过道', '走廊通道', '办公工位区', '工位区',
    '未知', '未知位置', '无',
    '安全通道', '疏散指示', '灭火器', '消火栓',
}

# YOLO COCO 类别 → 功能区映射
YOLO_AREA_MAPPING = {
    57: ('沙发区', 'Sofa Area'),         # couch
    72: ('茶水间', 'Pantry'),             # refrigerator
    62: ('会议室区', 'Meeting Room'),     # tv/monitor
    60: ('餐饮区', 'Dining Area'),        # dining table
}

YOLO_LANDMARK_CLASSES = {57, 72, 62, 60, 56}  # couch, fridge, tv, dining table, chair


class SemanticNodeDetector:
    """语义节点检测器 v2.2"""

    SAMPLE_INTERVAL = 2
    MIN_CONFIRM_FRAMES = 1          # v2.1: room_sign 类降为 1 帧即可
    MIN_CONFIRM_CROSS_VALIDATED = 1  # 交叉验证(同帧多camera): 1 帧即可
    GROUP_GAP_THRESHOLD = 10

    CAMERA_PRIORITY = ['camera_1', 'camera_2']  # v2.2: 只检测前方camera
    CAMERA_PRIORITY_ALL = ['camera_2', 'camera_4', 'camera_1', 'camera_3']  # 完整列表供YOLO等用

    def __init__(self, yolo_model=None):
        self._yolo = yolo_model
        self._yolo_loaded = yolo_model is not None
        logger.info(f"SemanticNodeDetector v2.2 initialized (YOLO: {self._yolo_loaded})")

    def _load_yolo(self):
        if self._yolo_loaded:
            return
        try:
            from ultralytics import YOLO
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "pretrained", "yolov8n.pt"
            )
            if os.path.exists(model_path):
                self._yolo = YOLO(model_path)
                self._yolo_loaded = True
                logger.info(f"[SemanticDetector] YOLO loaded: {model_path}")
        except Exception as e:
            logger.warning(f"[SemanticDetector] YOLO load failed: {e}")

    def _yolo_detect_objects(self, image_path: str) -> Dict[int, float]:
        if not self._yolo_loaded or self._yolo is None:
            return {}
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {}
            h, w = img.shape[:2]
            total_area = h * w
            results = self._yolo.predict(img, device='cuda:0', conf=0.35, verbose=False)
            detections = {}
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    if cls_id not in YOLO_LANDMARK_CLASSES:
                        continue
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    area_ratio = (x2 - x1) * (y2 - y1) / total_area
                    if cls_id not in detections or area_ratio > detections[cls_id]:
                        detections[cls_id] = area_ratio
            return detections
        except Exception as e:
            logger.warning(f"  YOLO detect failed: {e}")
            return {}

    def _yolo_scan_frame(self, frame_data: Dict) -> Optional[Tuple[str, str]]:
        if not self._yolo_loaded:
            return None
        merged = {}
        images = frame_data.get('images', {})
        for cam_id in self.CAMERA_PRIORITY_ALL:  # YOLO用全部camera
            cam_path = images.get(cam_id)
            if not cam_path:
                continue
            dets = self._yolo_detect_objects(cam_path)
            for cls_id, area in dets.items():
                if cls_id not in merged or area > merged[cls_id]:
                    merged[cls_id] = area
        if not merged:
            return None
        best_cls = max(merged, key=merged.get)
        if best_cls in YOLO_AREA_MAPPING and merged[best_cls] > 0.02:
            name_cn, name_en = YOLO_AREA_MAPPING[best_cls]
            logger.info(f"    YOLO: 检测到 {name_cn} (class={best_cls}, area={merged[best_cls]:.3f})")
            return (name_cn, name_en)
        return None

    def _encode_image_b64(self, image_path: str) -> Optional[str]:
        if not os.path.exists(image_path):
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('utf-8')

    def _is_useless(self, name_cn: str) -> bool:
        if not name_cn or not name_cn.strip():
            return True
        name = name_cn.strip()
        if name in USELESS_SIGNS:
            return True
        for useless in USELESS_SIGNS:
            if not useless:
                continue  # 跳过空字符串
            if useless in name and len(name) <= len(useless) + 2:
                return True
        return False


    def _pick_closest_detection(self, detections: List[Dict], qwen_server, frame_data: Dict) -> List[Dict]:
        """从同帧多个不同名称的检测中，选最靠近机器人的那个

        v2.2: 判断标准 — bbox 面积最大 = 离机器人最近
        如果 bbox 不可用，优先选 camera_1 的检测
        """
        if len(detections) <= 1:
            return detections

        # 按名称去重，每个名称保留一个代表
        by_name = {}
        for det in detections:
            name = det['name_cn']
            if name not in by_name:
                by_name[name] = det

        if len(by_name) <= 1:
            return detections  # 同名不需要筛选

        # 尝试用 locate_sign_bbox 获取 bbox 面积
        best_det = None
        best_area = -1.0

        for name, det in by_name.items():
            cam_id = det['camera_id']
            cam_path = frame_data.get('images', {}).get(cam_id)
            if not cam_path:
                continue

            b64 = self._encode_image_b64(cam_path)
            if not b64:
                continue

            try:
                bbox_result = qwen_server.locate_sign_bbox(b64)
                if bbox_result.get('has_bbox'):
                    bbox = bbox_result.get('bbox', {})
                    if isinstance(bbox, dict):
                        area = abs(bbox.get('x2', 0) - bbox.get('x1', 0)) * \
                               abs(bbox.get('y2', 0) - bbox.get('y1', 0))
                        logger.info(f"    {cam_id} [{name}]: bbox area = {area:.4f}")
                        if area > best_area:
                            best_area = area
                            best_det = det
            except Exception as e:
                logger.warning(f"    {cam_id} [{name}]: locate_sign_bbox 失败 - {e}")

        if best_det:
            logger.info(f"    选择最近: {best_det['name_cn']} (area={best_area:.4f})")
            return [best_det]

        # fallback: 优先 camera_1
        for det in detections:
            if det['camera_id'] == 'camera_1':
                return [det]

        return [detections[0]]

    def _detect_on_frame(self, frame_data: Dict, qwen_server) -> List[Dict]:
        """对单帧的所有 camera 执行语义检测

        v2.1 改进: 返回所有 camera 的检测结果(带 camera_id)，不再同帧去重
        后续由 _cluster_detections 做交叉验证
        """
        images = frame_data.get('images', {})
        detections = []

        # === 阶段 1: Qwen 文字/标识识别 (所有 camera) ===
        for cam_id in self.CAMERA_PRIORITY:
            cam_path = images.get(cam_id)
            if not cam_path or not os.path.exists(cam_path):
                continue

            b64 = self._encode_image_b64(cam_path)
            if not b64:
                continue

            try:
                result = qwen_server.detect_landmark_sign(b64)
                if result.get('status') == 'ok' and result.get('found', False):
                    name_cn = result.get('name_cn', '').strip()
                    name_en = result.get('name_en', '').strip()
                    category = result.get('category', 'area')

                    if self._is_useless(name_cn):
                        logger.info(f"    {cam_id}: Qwen 返回 [{name_cn}] → 黑名单过滤")
                        continue

                    logger.info(f"    {cam_id}: Qwen 检测到 [{category}] {name_cn} ({name_en})")
                    detections.append({
                        'name_cn': name_cn, 'name_en': name_en,
                        'category': category, 'camera_id': cam_id,
                        'source': f'qwen_{cam_id}',
                    })
            except Exception as e:
                logger.warning(f"    {cam_id}: Qwen 异常 - {e}")

        # v2.2: 如果前方camera检测到多个不同名称，选最靠近机器人的
        if detections:
            unique_names = set(d['name_cn'] for d in detections)
            if len(unique_names) > 1:
                detections = self._pick_closest_detection(detections, qwen_server, frame_data)
                logger.info(f"    同帧多名称 → 选最近: {detections[0]['name_cn']}")
            return detections

        # === 阶段 2: 场景分类 (只在 Qwen 全部无结果时) ===
        for cam_id in ['camera_1', 'camera_2']:  # v2.2: 只用前方camera做场景分类
            cam_path = images.get(cam_id)
            if not cam_path or not os.path.exists(cam_path):
                continue
            b64 = self._encode_image_b64(cam_path)
            if not b64:
                continue
            try:
                result = qwen_server.classify_scene(b64)
                if result.get('status') == 'ok' and result.get('type') == 'area':
                    name_cn = result.get('name_cn', '').strip()
                    name_en = result.get('name_en', '').strip()
                    if not self._is_useless(name_cn):
                        logger.info(f"    {cam_id}: 场景分类 → {name_cn} ({name_en})")
                        detections.append({
                            'name_cn': name_cn, 'name_en': name_en,
                            'category': 'area', 'camera_id': cam_id,
                            'source': f'scene_{cam_id}',
                        })
                        break  # 场景分类有结果就不继续
            except Exception as e:
                logger.warning(f"    {cam_id}: 场景分类异常 - {e}")
            break

        # === 阶段 3: YOLO 辅助补充 ===
        if not detections:
            yolo_result = self._yolo_scan_frame(frame_data)
            if yolo_result:
                yolo_cn, yolo_en = yolo_result
                if not self._is_useless(yolo_cn):
                    detections.append({
                        'name_cn': yolo_cn, 'name_en': yolo_en,
                        'category': 'area', 'camera_id': 'yolo',
                        'source': 'yolo',
                    })

        return detections

    def _cluster_detections(self, frame_detections: List[Tuple[int, List[Dict], Dict]],
                            existing_names: Set[str]) -> List[Dict]:
        """将帧级检测结果聚类去重

        v2.1: 同帧多 camera 交叉验证
        - 同帧 2+ camera 检测到同名 → cross_validated=True → 1 帧即可
        - 单 camera 检测 → 需要 MIN_CONFIRM_FRAMES 帧
        """
        if not frame_detections:
            return []

        # 统计每个 name_cn 在哪些帧出现、是否交叉验证
        name_info = defaultdict(lambda: {
            'frames': [],           # (frame_idx, frame_data, best_det)
            'cross_validated': False,
        })

        for frame_idx, dets_in_frame, frame_data in frame_detections:
            # 按 name_cn 分组，统计同帧同名出现次数
            name_counts = defaultdict(list)
            for det in dets_in_frame:
                name_counts[det['name_cn']].append(det)

            for name_cn, det_list in name_counts.items():
                # 选最佳 det (优先 room_sign，其次名称更长)
                best = det_list[0]
                for d in det_list[1:]:
                    if d['category'] == 'room_sign' and best['category'] != 'room_sign':
                        best = d
                    elif len(d.get('name_en', '')) > len(best.get('name_en', '')):
                        best = d

                info = name_info[name_cn]
                info['frames'].append((frame_idx, frame_data, best))

                # 同帧 2+ camera 检测到 → 交叉验证
                unique_cameras = set(d['camera_id'] for d in det_list)
                if len(unique_cameras) >= 2:
                    info['cross_validated'] = True
                    logger.info(f"    [{name_cn}] 帧 {frame_idx}: {len(unique_cameras)} 个 camera 交叉验证 ✓")

        # 生成候选
        candidates = []
        for name_cn, info in name_info.items():
            frames = info['frames']
            cross_validated = info['cross_validated']

            # 按位置拆分 (间隔大的视为不同 node)
            frames.sort(key=lambda x: x[0])
            groups = [[frames[0]]]
            for i in range(1, len(frames)):
                if frames[i][0] - frames[i - 1][0] > self.GROUP_GAP_THRESHOLD:
                    groups.append([frames[i]])
                else:
                    groups[-1].append(frames[i])

            for group in groups:
                n_frames = len(group)
                min_required = self.MIN_CONFIRM_CROSS_VALIDATED if cross_validated else self.MIN_CONFIRM_FRAMES

                if n_frames < min_required:
                    cv_tag = " (交叉验证)" if cross_validated else ""
                    logger.info(f"  [{name_cn}] {n_frames} 帧{cv_tag}，不足 {min_required} 帧，丢弃")
                    continue

                if name_cn in existing_names:
                    logger.info(f"  [{name_cn}] 已存在同名 node，跳过")
                    continue

                # 取居中帧
                mid_idx = len(group) // 2
                rep_frame_idx, rep_frame_data, rep_det = group[mid_idx]

                cv_tag = " [交叉验证]" if cross_validated else ""
                logger.info(f"  [{name_cn}] 确认 {n_frames} 帧{cv_tag} "
                             f"(source: {rep_det.get('source', '?')})，选取帧 {rep_frame_idx}")

                candidates.append({
                    'frame_idx': rep_frame_idx,
                    'timestamp': rep_frame_data['timestamp'],
                    'images': rep_frame_data['images'],
                    'position_names': {
                        'position_name': name_cn,
                        'position_name_eng': rep_det.get('name_en', name_cn),
                    },
                    'category': rep_det.get('category', 'area'),
                    'confirm_count': n_frames,
                    'cross_validated': cross_validated,
                    'source': rep_det.get('source', 'unknown'),
                })

        return candidates

    def scan_gap(self, all_frames: List[Dict],
                 start_frame_idx: int, end_frame_idx: int,
                 qwen_server,
                 existing_names: Set[str]) -> List[Dict]:
        """扫描两个 node 之间的中间帧"""
        self._load_yolo()

        mid_indices = list(range(start_frame_idx + 1, end_frame_idx))
        if not mid_indices:
            logger.info(f"  帧 {start_frame_idx}->{end_frame_idx}: 无中间帧，跳过")
            return []

        if len(mid_indices) > 3:
            sampled = mid_indices[::self.SAMPLE_INTERVAL]
        else:
            sampled = mid_indices

        logger.info(f"  帧 {start_frame_idx}->{end_frame_idx}: "
                     f"{len(mid_indices)} 个中间帧，采样 {len(sampled)} 帧")

        # 逐帧检测 (返回所有 camera 的检测结果)
        frame_detections = []  # (frame_idx, [detections], frame_data)
        for idx in sampled:
            if idx >= len(all_frames):
                continue
            frame = all_frames[idx]
            logger.info(f"  扫描帧 {idx} (ts: {frame['timestamp']})...")

            dets = self._detect_on_frame(frame, qwen_server)
            if dets:
                frame_detections.append((idx, dets, frame))

        if not frame_detections:
            logger.info(f"  帧 {start_frame_idx}->{end_frame_idx}: 未检测到有意义的语义标识")
            return []

        candidates = self._cluster_detections(frame_detections, existing_names)

        logger.info(f"  帧 {start_frame_idx}->{end_frame_idx}: "
                     f"发现 {len(candidates)} 个语义 node 候选")
        return candidates
