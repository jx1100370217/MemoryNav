#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语义节点检测器 v3 — 纯字符识别

功能: 扫描相邻 node 之间的中间帧，用 Qwen3.5 识别门牌/标识上的文字
      (汉字、英文、数字)，将有意义的文字作为新的语义 node 插入。

不识别: 物体(沙发、椅子等)、场景类型
只识别: 墙面/门牌/标识上的可读文字

v3 变更 (相对 v2.1):
- 去掉 YOLO 物体检测
- 去掉场景分类 (classify_scene)
- 去掉旧的 detect_landmark_sign 调用
- 统一使用 QwenNamingServer.detect_text() 做字符识别
- 同帧多 camera 交叉验证逻辑保留
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
    '走廊', '办公区', '通道', '过道', '走廊通道', '办公工位区', '工位区',
    '未知', '未知位置', '无',
    '安全通道', '疏散指示', '灭火器', '消火栓',
    '标识', '标识牌', '门牌', '房间标识',
}


class SemanticNodeDetector:
    """语义节点检测器 v3 — 纯字符识别"""

    SAMPLE_INTERVAL = 2
    MIN_CONFIRM_FRAMES = 1          # 单 camera 最少确认帧数
    MIN_CONFIRM_CROSS_VALIDATED = 1  # 交叉验证(同帧多camera): 1 帧即可
    GROUP_GAP_THRESHOLD = 10

    CAMERA_PRIORITY = ['camera_2', 'camera_4', 'camera_1', 'camera_3']

    def __init__(self):
        logger.info("SemanticNodeDetector v3 initialized (text-only mode)")

    def _encode_image_b64(self, image_path: str) -> Optional[str]:
        """将图片编码为 base64"""
        if not os.path.exists(image_path):
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('utf-8')

    def _is_useless(self, name_cn: str) -> bool:
        """检查名称是否在黑名单中"""
        if not name_cn or not name_cn.strip():
            return True
        name = name_cn.strip()
        if name in USELESS_SIGNS:
            return True
        for useless in USELESS_SIGNS:
            if not useless:
                continue
            if useless in name and len(name) <= len(useless) + 2:
                return True
        return False

    def _detect_on_frame(self, frame_data: Dict, qwen_server) -> List[Dict]:
        """对单帧的所有 camera 执行字符识别

        调用 qwen_server.detect_text() 识别门牌/标识上的文字
        """
        images = frame_data.get('images', {})
        detections = []

        for cam_id in self.CAMERA_PRIORITY:
            cam_path = images.get(cam_id)
            if not cam_path or not os.path.exists(cam_path):
                continue

            b64 = self._encode_image_b64(cam_path)
            if not b64:
                continue

            try:
                result = qwen_server.detect_text(b64)
                if result.get('status') == 'ok' and result.get('found', False):
                    text = result.get('text', '').strip()
                    name_cn = result.get('name_cn', '').strip()
                    name_en = result.get('name_en', '').strip()

                    # 用 name_cn 做黑名单过滤，如果 name_cn 为空则用 text
                    check_name = name_cn if name_cn else text
                    if self._is_useless(check_name):
                        logger.info(f"    {cam_id}: 识别到文字 [{text}] → 黑名单过滤")
                        continue

                    logger.info(f"    {cam_id}: 识别到文字 [{text}] → {name_cn} ({name_en})")
                    detections.append({
                        'name_cn': name_cn if name_cn else text,
                        'name_en': name_en if name_en else text,
                        'text': text,
                        'camera_id': cam_id,
                        'source': f'text_{cam_id}',
                    })
            except Exception as e:
                logger.warning(f"    {cam_id}: Qwen detect_text 异常 - {e}")

        return detections

    def _cluster_detections(self, frame_detections: List[Tuple[int, List[Dict], Dict]],
                            existing_names: Set[str]) -> List[Dict]:
        """将帧级检测结果聚类去重

        同帧多 camera 交叉验证:
        - 同帧 2+ camera 检测到同名 → cross_validated=True → 1 帧即可
        - 单 camera 检测 → 需要 MIN_CONFIRM_FRAMES 帧
        """
        if not frame_detections:
            return []

        # 统计每个 name_cn 在哪些帧出现、是否交叉验证
        name_info = defaultdict(lambda: {
            'frames': [],
            'cross_validated': False,
        })

        for frame_idx, dets_in_frame, frame_data in frame_detections:
            # 按 name_cn 分组
            name_counts = defaultdict(list)
            for det in dets_in_frame:
                name_counts[det['name_cn']].append(det)

            for name_cn, det_list in name_counts.items():
                # 选最佳 det (名称更长的优先)
                best = det_list[0]
                for d in det_list[1:]:
                    if len(d.get('name_en', '')) > len(best.get('name_en', '')):
                        best = d

                info = name_info[name_cn]
                info['frames'].append((frame_idx, frame_data, best))

                # 同帧 2+ camera 检测到 → 交叉验证
                unique_cameras = set(d['camera_id'] for d in det_list)
                if len(unique_cameras) >= 2:
                    info['cross_validated'] = True
                    logger.info(f"    [{name_cn}] 帧 {frame_idx}: "
                                f"{len(unique_cameras)} 个 camera 交叉验证 ✓")

        # 生成候选
        candidates = []
        for name_cn, info in name_info.items():
            frames = info['frames']
            cross_validated = info['cross_validated']

            # 按位置拆分 (间隔大的视为不同位置)
            frames.sort(key=lambda x: x[0])
            groups = [[frames[0]]]
            for i in range(1, len(frames)):
                if frames[i][0] - frames[i - 1][0] > self.GROUP_GAP_THRESHOLD:
                    groups.append([frames[i]])
                else:
                    groups[-1].append(frames[i])

            for group in groups:
                n_frames = len(group)
                min_required = (self.MIN_CONFIRM_CROSS_VALIDATED
                                if cross_validated
                                else self.MIN_CONFIRM_FRAMES)

                if n_frames < min_required:
                    cv_tag = " (交叉验证)" if cross_validated else ""
                    logger.info(f"  [{name_cn}] {n_frames} 帧{cv_tag}，"
                                f"不足 {min_required} 帧，丢弃")
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
                    'confirm_count': n_frames,
                    'cross_validated': cross_validated,
                    'source': rep_det.get('source', 'unknown'),
                })

        return candidates

    def scan_gap(self, all_frames: List[Dict],
                 start_frame_idx: int, end_frame_idx: int,
                 qwen_server,
                 existing_names: Set[str]) -> List[Dict]:
        """扫描两个 node 之间的中间帧，识别文字标识"""

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

        # 逐帧检测
        frame_detections = []
        for idx in sampled:
            if idx >= len(all_frames):
                continue
            frame = all_frames[idx]
            logger.info(f"  扫描帧 {idx} (ts: {frame['timestamp']})...")

            dets = self._detect_on_frame(frame, qwen_server)
            if dets:
                frame_detections.append((idx, dets, frame))

        if not frame_detections:
            logger.info(f"  帧 {start_frame_idx}->{end_frame_idx}: 未检测到文字标识")
            return []

        candidates = self._cluster_detections(frame_detections, existing_names)

        logger.info(f"  帧 {start_frame_idx}->{end_frame_idx}: "
                    f"发现 {len(candidates)} 个语义 node 候选")
        return candidates
