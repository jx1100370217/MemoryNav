#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 工具函数模块

包含图像编解码、动作转换等通用工具函数。
"""

import io
import base64
import logging
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def decode_base64_image(base64_data: str) -> Optional[np.ndarray]:
    """
    解码base64图像数据

    Args:
        base64_data: base64编码的图像数据

    Returns:
        numpy array (H, W, 3) uint8，失败返回None
    """
    try:
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return np.array(image)
    except Exception as e:
        logger.error(f"解码base64图像失败: {e}")
        return None


def decode_base64_depth(base64_data: str) -> Optional[np.ndarray]:
    """
    解码base64深度图数据

    Args:
        base64_data: base64编码的深度图数据

    Returns:
        numpy array (H, W) float32，失败返回None
    """
    try:
        depth_bytes = base64.b64decode(base64_data)
        depth_image = Image.open(io.BytesIO(depth_bytes))
        depth_array = np.array(depth_image, dtype=np.float32)
        return depth_array
    except Exception as e:
        logger.error(f"解码base64深度图失败: {e}")
        return None


def encode_numpy_to_base64(array: np.ndarray) -> Optional[str]:
    """
    将numpy数组编码为base64

    Args:
        array: numpy array

    Returns:
        base64编码的字符串，失败返回None
    """
    try:
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"编码numpy数组失败: {e}")
        return None


def convert_output_action_to_robot_action(output_action: List[int]) -> Tuple[List[List[float]], str]:
    """
    将离散动作序列转换为机器人控制命令

    Args:
        output_action: 离散动作列表 [0=STOP, 1=前进, 2=左转, 3=右转, 5=后退]

    Returns:
        (action_list, task_status) 元组
        - action_list: [[x, y, yaw]] 格式的机器人命令
        - task_status: "end" 或 "executing"
    """
    import math

    STEP_SIZE = 0.25
    TURN_ANGLE = math.pi / 24

    forward_count = 0
    left_turn_count = 0
    right_turn_count = 0
    has_stop = False

    for action in output_action:
        if action == 0:
            has_stop = True
        elif action == 1:
            forward_count += 1
        elif action == 2:
            left_turn_count += 1
        elif action == 3:
            right_turn_count += 1

    x = forward_count * STEP_SIZE
    y = 0.0
    yaw = (left_turn_count - right_turn_count) * TURN_ANGLE

    task_status = "end" if has_stop else "executing"
    return [[x, y, yaw]], task_status


def convert_trajectory_to_robot_action(output_trajectory: List[List[float]]) -> List[List[float]]:
    """
    将轨迹增量点列表转换为累积坐标的机器人控制命令格式

    参照 internnav/model/utils/vln_utils.py 中的 reconstruct_xy_from_delta 函数

    输入: 33个点 [[0,0], [dx1, dy1], [dx2, dy2], ...] - 第一个点是起点(0,0)，后续是增量
    输出: 33个点 [[0, 0, 0], [dx1, dy1, 0], [dx1+dx2, dy1+dy2, 0], ...] - 累积坐标

    Args:
        output_trajectory: 轨迹增量点列表

    Returns:
        累积坐标列表 [[x, y, yaw], ...]
    """
    if not output_trajectory or len(output_trajectory) == 0:
        return []

    traj_array = np.array(output_trajectory)

    # 跳过第一个点(起点 0,0)，取后续的增量值
    delta_xy = traj_array[1:, :2] if traj_array.shape[0] > 1 else np.zeros((0, 2))

    # 计算累积和
    if len(delta_xy) > 0:
        cumsum_xy = np.cumsum(delta_xy, axis=0)
    else:
        cumsum_xy = np.zeros((0, 2))

    # 构建输出
    action_list = [[0.0, 0.0, 0.0]]  # 起点
    for i in range(len(cumsum_xy)):
        action_list.append([float(cumsum_xy[i, 0]), float(cumsum_xy[i, 1]), 0.0])

    return action_list


def resize_image(image: np.ndarray, target_width: int = 640, target_height: int = 480) -> np.ndarray:
    """
    调整图像尺寸

    Args:
        image: 输入图像
        target_width: 目标宽度
        target_height: 目标高度

    Returns:
        调整后的图像
    """
    import cv2
    if image.shape[1] != target_width or image.shape[0] != target_height:
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return image


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    """
    归一化特征向量

    Args:
        feature: 特征向量

    Returns:
        归一化后的特征向量
    """
    norm = np.linalg.norm(feature)
    if norm > 1e-8:
        return feature / norm
    return feature
