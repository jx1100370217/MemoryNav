#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鱼眼去畸变 - 柱面投影（纯 Python 实现）

移植自 cam/tools/fisheye_undist_cpu.h，使用 numpy + cv2.remap。
每个相机的 remap 表只需计算一次，之后每帧直接 cv2.remap 即可。

用法:
    undistorter = FisheyeUndistorter.from_yaml("cam/params.yaml")
    perspective_img = undistorter.undistort(fisheye_img, "camera_1")
    perspective_dict = undistorter.undistort_batch(camera_images)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 默认路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "cam", "params.yaml")

DEG_TO_RAD = np.pi / 180.0


class FisheyeUndistort:
    """
    单相机鱼眼去畸变（移植自 fisheye_undist_cpu.h）

    初始化时生成 remap 表，之后每帧只做 cv2.remap。

    Args:
        intrinsics: [xi, fx, fy, cx, cy]
        dist_coeffs: [k1, k2, p1, p2]
        fov: 输出视场角（度）
        width: 输出宽度
        height: 输出高度
        pitch_up: 视角向上偏移角度（度），正数向上
    """

    def __init__(self, intrinsics: List[float], dist_coeffs: List[float],
                 fov: int = 180, width: int = 640, height: int = 480,
                 pitch_up: float = 0.0):
        self.fov = fov
        self.width = width
        self.height = height

        xi, fx, fy, cx, cy = intrinsics
        k1, k2, p1, p2 = dist_coeffs

        focal_gen = width / (fov * DEG_TO_RAD)
        theta = pitch_up * DEG_TO_RAD
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # 像素坐标网格
        xs = np.arange(width, dtype=np.float64)
        ys = np.arange(height, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)

        # 柱面投影：像素 → 3D 方向
        nx = np.tan((xx - width / 2) / focal_gen) * focal_gen
        scale = np.sqrt(nx ** 2 + focal_gen ** 2) / focal_gen
        ny = (yy - height / 2) * scale

        # 绕 x 轴旋转 pitch_up（视角向上偏移）
        obj_x = nx
        obj_y = ny * cos_t - focal_gen * sin_t
        obj_z = ny * sin_t + focal_gen * cos_t

        # 归一化到单位球
        norm = np.sqrt(obj_x ** 2 + obj_y ** 2 + obj_z ** 2)
        xs_n = obj_x / norm
        ys_n = obj_y / norm
        zs_n = obj_z / norm

        # 鱼眼模型投影
        denom = zs_n + xi
        xu_x = xs_n / denom
        xu_y = ys_n / denom

        r2 = xu_x ** 2 + xu_y ** 2
        r4 = r2 ** 2

        xd_x = (xu_x * (1 + k1 * r2 + k2 * r4)
                 + 2 * p1 * xu_x * xu_y
                 + p2 * (r2 + 2 * xu_x ** 2))
        xd_y = (xu_y * (1 + k1 * r2 + k2 * r4)
                 + p1 * (r2 + 2 * xu_y ** 2)
                 + 2 * p2 * xu_x * xu_y)

        self.xmap = (fx * xd_x + cx).astype(np.float32)
        self.ymap = (fy * xd_y + cy).astype(np.float32)

    def remap(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.xmap, self.ymap, cv2.INTER_LINEAR)


class FisheyeUndistorter:
    """
    多相机鱼眼去畸变管理器

    从 cam/params.yaml 加载所有相机内参，为每个相机预计算 remap 表。
    """

    def __init__(self, cameras: Dict[str, FisheyeUndistort] = None):
        self.cameras: Dict[str, FisheyeUndistort] = cameras or {}

    @classmethod
    def from_yaml(cls, yaml_path: str = None, fov: int = 180, width: int = 1920,
                  height: int = 1536, pitch_up: float = 15.0) -> 'FisheyeUndistorter':
        """
        从 params.yaml 创建

        Args:
            yaml_path: 相机参数文件路径（None 则使用默认 cam/params.yaml）
            fov: 输出视场角（度）
            width: 输出宽度
            height: 输出高度
            pitch_up: 视角向上偏移角度（度）
        """
        import yaml

        yaml_path = yaml_path or _DEFAULT_CONFIG
        if not os.path.exists(yaml_path):
            logger.warning(f"[FisheyeUndistorter] 参数文件不存在: {yaml_path}")
            return cls()

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        cameras = {}
        for cam in config.get('cameras', []):
            name = cam.get('name', '')
            intrinsics = cam.get('intrinsics')
            dist_coeffs = cam.get('distortion_coeffs')
            if not intrinsics or not dist_coeffs:
                continue
            if len(intrinsics) != 5 or len(dist_coeffs) != 4:
                continue

            cameras[name] = FisheyeUndistort(
                intrinsics=intrinsics, dist_coeffs=dist_coeffs,
                fov=fov, width=width, height=height, pitch_up=pitch_up,
            )

        logger.info(f"[FisheyeUndistorter] 已加载 {len(cameras)} 个相机: "
                    f"{list(cameras.keys())}, fov={fov}, {width}x{height}, pitch_up={pitch_up}")
        return cls(cameras)

    def undistort(self, image: np.ndarray, camera_id: str) -> np.ndarray:
        """对鱼眼图像去畸变，若无该相机参数则返回原图"""
        if camera_id not in self.cameras:
            return image
        return self.cameras[camera_id].remap(image)

    def undistort_batch(self, camera_images):
        """批量去畸变 — 并行处理4相机 (cv2.remap 释放 GIL)"""
        from concurrent.futures import ThreadPoolExecutor
        def _undist(item):
            cam_id, img = item
            return cam_id, self.undistort(img, cam_id)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(_undist, camera_images.items())
            return {cam_id: img for cam_id, img in results}

    def is_ready(self) -> bool:
        return len(self.cameras) > 0
