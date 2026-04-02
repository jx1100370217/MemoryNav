#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语义节点检测器 v3.2 — 纯字符识别 (平衡过滤版)

功能: 扫描相邻 node 之间的中间帧，用 Qwen3.5 识别门牌/标识上的文字
      (汉字、英文、数字)，将有意义的文字作为新的语义 node 插入。

v3.2 改进 (相对 v3.1):
- 采样间隔从 2 降为 1: 不跳帧，避免漏掉关键门牌
- 质量过滤放宽: >=4字中文名直接通过 (多帧确认兜底防幻觉)
- 支持"XX会议室"格式 (Qwen prompt 引导组合输出)
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
    '开放空间', '办公室', '工位', '会议室',
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

    三级通过:
    1. 包含地点特征词后缀 (室/间/区/厅等) → 直接通过
    2. 匹配房间号格式 (101, A302等) → 直接通过
    3. >=4个中文字符的名称 → 通过 (多帧确认兜底)

    合格: "关爱室", "101室", "纽曼会议室", "10号会议室", "摩尔会议室"
    不合格: "准", "难", "JDS", "林明", "我的", "叉号", "再学习", "会议室"(太泛)
    """
    if not name_cn or len(name_cn.strip()) < 2:
        return False

    name = name_cn.strip()

    # 黑名单精确匹配
    if name in USELESS_SIGNS:
        return False
    # 黑名单包含匹配 (只对短名称生效，避免误杀"纽曼会议室"这种)
    for useless in USELESS_SIGNS:
        if not useless:
            continue
        if useless in name and len(name) <= len(useless) + 2:
            return True if name != useless else False  # 长一点的就放过

    # 正则过滤
    for pattern in USELESS_PATTERNS:
        if re.match(pattern, name):
            return False

    # === 通过条件 ===

    # 条件1: 地点特征词后缀
    location_suffixes = ['室', '间', '区', '厅', '房', '井', '台', '梯',
                         '所', '处', '站', '廊', '堂', '馆']
    if any(name.endswith(kw) for kw in location_suffixes):
        return True

    # 条件2: 包含"中心"
    if '中心' in name:
        return True

    # 条件3: 房间号格式 (数字开头或字母+数字)
    if re.match(r'^[A-Za-z]?\d+[-\d]*[室号]?$', name):
        return True

    # 条件4: 包含"号"+"房间类型词" (如 "10号会议室")
    if '号' in name and any(kw in name for kw in ['会议', '办公', '培训', '接待']):
        return True

    # 条件5: >=4 中文字符 → 放行 (依赖多帧确认防幻觉)
    chinese_chars = len([c for c in name if '\u4e00' <= c <= '\u9fff'])
    if chinese_chars >= 4:
        return True

    # 2-3 字中文且没有地点特征词 → 拒绝 (如 "林明"、"库尔"、"叉号")
    logger.info(f"    名称质量校验不通过: [{name}] (短名称无地点特征词)")
    return False


class SemanticNodeDetector:
    """语义节点检测器 v3.2 — 纯字符识别 (平衡过滤)"""

    SAMPLE_INTERVAL = 1             # v3.2: 不跳帧，逐帧扫描
    MIN_CONFIRM_FRAMES = 2          # 至少 2 帧确认
    MIN_CONFIRM_CROSS_VALIDATED = 1  # 交叉验证: 1 帧即可
    GROUP_GAP_THRESHOLD = 10

    CAMERA_PRIORITY = ['camera_2', 'camera_4', 'camera_1', 'camera_3']

    def __init__(self):
        logger.info("SemanticNodeDetector v3.2 initialized (text-only, balanced filter)")

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
        """对单帧的所有 camera 执行字符识别"""
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

                    check_name = name_cn if name_cn else text
                    if not _is_name_quality_ok(check_name):
                        logger.info(f"    {cam_id}: 识别到 [{text}] → name_cn=[{name_cn}] 质量过滤")
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
        """将帧级检测结果聚类去重"""
        if not frame_detections:
            return []

        name_info = defaultdict(lambda: {
            'frames': [],
            'cross_validated': False,
        })

        for frame_idx, dets_in_frame, frame_data in frame_detections:
            name_counts = defaultdict(list)
            for det in dets_in_frame:
                name_counts[det['name_cn']].append(det)

            for name_cn, det_list in name_counts.items():
                best = det_list[0]
                for d in det_list[1:]:
                    if len(d.get('name_en', '')) > len(best.get('name_en', '')):
                        best = d

                info = name_info[name_cn]
                info['frames'].append((frame_idx, frame_data, best))

                unique_cameras = set(d['camera_id'] for d in det_list)
                if len(unique_cameras) >= 2:
                    info['cross_validated'] = True
                    logger.info(f"    [{name_cn}] 帧 {frame_idx}: "
                                f"{len(unique_cameras)} 个 camera 交叉验证 ✓")

        candidates = []
        for name_cn, info in name_info.items():
            frames = info['frames']
            cross_validated = info['cross_validated']

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

        # v3.2: 全量扫描，不跳帧
        sampled = mid_indices

        logger.info(f"  帧 {start_frame_idx}->{end_frame_idx}: "
                    f"{len(mid_indices)} 个中间帧，扫描 {len(sampled)} 帧")

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
