#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语义节点检测器 v3.1 — 纯字符识别 (严格过滤版)

功能: 扫描相邻 node 之间的中间帧，用 Qwen3.5 识别门牌/标识上的文字
      (汉字、英文、数字)，将有意义的文字作为新的语义 node 插入。

不识别: 物体(沙发、椅子等)、场景类型
只识别: 墙面/门牌/标识上的可读文字

v3.1 修复 (相对 v3):
- 提高 MIN_CONFIRM_FRAMES 到 2: 单帧检测不可靠，至少 2 帧确认
- 新增名称质量校验: 过滤单字、纯数字、人名等无导航意义的识别结果
- 扩充黑名单: 覆盖更多常见误识别
- 新增长度限制: name_cn 必须 >= 2 字符
"""

import os
import sys
import re
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
    # 安全/消防相关
    '安全出口', '安全出口标识', '消防', '消防标识', '禁止吸烟',
    '安全通道', '疏散指示', '灭火器', '消火栓', '紧急出口',
    # 太泛的通用名
    '走廊', '办公区', '通道', '过道', '走廊通道', '办公工位区', '工位区',
    '未知', '未知位置', '无', '标识', '标识牌', '门牌', '房间标识',
    '开放空间', '办公室', '工位',
    # 非地点类
    '企业文化', '公告栏', '宣传栏', '展示墙', '文化墙',
    '中国航天', '中国制造',
}

# 明显不是地点名的模式 (正则)
USELESS_PATTERNS = [
    r'^.$',              # 单字 (准、难、错、的、为...)
    r'^[\d.]+$',         # 纯数字 (但 "101室" 是可以的)
    r'^[a-zA-Z]{1,3}$',  # 1-3 字母缩写 (JDS 等噪声)
    r'^(我的|你的|他的|她的|它的)',  # 代词开头
    r'^(再|又|也|都|就|才)',        # 副词开头的碎片
]


def _is_name_quality_ok(name_cn: str) -> bool:
    """检查名称是否有导航意义

    合格: "关爱室", "101室", "103-2006室", "会议室A", "茶水间", "强电井"
    不合格: "准", "难", "JDS", "林明", "我的", "叉号", "再学习"
    """
    if not name_cn or len(name_cn.strip()) < 2:
        return False

    name = name_cn.strip()

    # 黑名单
    if name in USELESS_SIGNS:
        return False
    for useless in USELESS_SIGNS:
        if useless in name and len(name) <= len(useless) + 2:
            return False

    # 正则过滤
    for pattern in USELESS_PATTERNS:
        if re.match(pattern, name):
            return False

    # 必须包含"地点特征词"或"房间号格式"才算有效
    # 地点特征词: 室、间、区、厅、房、井、台、梯、所、处、站、库、廊、堂、馆
    location_keywords_suffix = ['室', '间', '区', '厅', '房', '井', '台', '梯',
                         '所', '处', '站', '库', '廊', '堂', '馆']
    # 用词尾匹配: '关爱室'→匹配'室', '库尔'→不匹配'库'
    has_location_keyword = any(name.endswith(kw) for kw in location_keywords_suffix)
    # 也匹配含'中心'的 (如'服务中心')
    has_location_keyword = has_location_keyword or '中心' in name

    # 房间号格式: 数字+室, 或纯数字+字母(如 A101)
    room_number_pattern = re.match(r'^[A-Za-z]?\d+[-\d]*[室号]?$', name)

    if has_location_keyword or room_number_pattern:
        return True

    # 都不满足 → 拒绝 (宁可漏掉也不要垃圾)
    logger.info(f"    名称质量校验不通过: [{name}] (无地点特征词)")
    return False


class SemanticNodeDetector:
    """语义节点检测器 v3.1 — 纯字符识别 (严格过滤)"""

    SAMPLE_INTERVAL = 2
    MIN_CONFIRM_FRAMES = 2          # v3.1: 至少 2 帧确认，避免幻觉
    MIN_CONFIRM_CROSS_VALIDATED = 1  # 交叉验证(同帧多camera): 1 帧即可
    GROUP_GAP_THRESHOLD = 10

    CAMERA_PRIORITY = ['camera_2', 'camera_4', 'camera_1', 'camera_3']

    def __init__(self):
        logger.info("SemanticNodeDetector v3.1 initialized (text-only, strict filter)")

    def _encode_image_b64(self, image_path: str) -> Optional[str]:
        """将图片编码为 base64"""
        if not os.path.exists(image_path):
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('utf-8')

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

                    # 名称质量校验
                    check_name = name_cn if name_cn else text
                    if not _is_name_quality_ok(check_name):
                        logger.info(f"    {cam_id}: 识别到 [{text}] → 质量过滤")
                        continue

                    logger.info(f"    {cam_id}: 识别到文字 [{text}] → {name_cn} ({name_en})")
                    detections.append({
                        'name_cn': name_cn if name_cn else text,
                        'name_en': name_en if name_en else text,
                        'text': text,
                        'camera_id': cam_id,
                        'source': f'text_{cam_id}',
                    })
                else:
                    logger.debug(f"    {cam_id}: 未检测到文字")
            except Exception as e:
                logger.warning(f"    {cam_id}: Qwen detect_text 异常 - {e}")

        return detections

    def _cluster_detections(self, frame_detections: List[Tuple[int, List[Dict], Dict]],
                            existing_names: Set[str]) -> List[Dict]:
        """将帧级检测结果聚类去重

        同帧多 camera 交叉验证:
        - 同帧 2+ camera 检测到同名 → cross_validated=True → 1 帧即可
        - 单 camera 检测 → 需要 MIN_CONFIRM_FRAMES 帧 (>=2)
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
