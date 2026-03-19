#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.5-9B 打点模块 - 作为 MemoryNav 的兜底模型方案

通过子进程启动 qwen35_grounding_server.py (运行在 qwen3 conda 环境)，
提供 predict(image, landmark_name) 接口返回归一化坐标。

用法:
    grounder = Qwen35PointGrounder()
    grounder.start()  # 启动子进程
    result = grounder.predict(camera_image, "电梯")
    # result = {"point": [0.45, 0.62], "confidence": 0.8, ...}
"""

import json
import logging
import subprocess
import time
import base64
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 子进程脚本路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_SCRIPT = os.path.join(_SCRIPT_DIR, "qwen35_grounding_server.py")

# conda 环境名
CONDA_ENV = "qwen3"

# 默认 GPU (供子进程使用)
DEFAULT_GPU = "1"


class Qwen35PointGrounder:
    """
    Qwen3.5-9B 打点器
    
    通过子进程模式运行 Qwen3.5，避免 transformers 版本冲突。
    子进程运行在 qwen3 conda 环境 (transformers 5.x)。
    """

    def __init__(self, gpu: str = DEFAULT_GPU, timeout: float = 120.0):
        """
        Args:
            gpu: CUDA_VISIBLE_DEVICES 设置
            timeout: 启动超时时间 (秒)
        """
        self.gpu = gpu
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self._process is not None and self._process.poll() is None

    def start(self):
        """启动 Qwen3.5 子进程"""
        if self.is_ready:
            logger.info("[Qwen35] 子进程已在运行")
            return

        # 构建启动命令
        conda_base = os.environ.get("CONDA_PREFIX", "").rsplit("/envs/", 1)[0]
        if not conda_base:
            conda_base = os.path.expanduser("~/miniconda3")
        
        activate_cmd = f"source {conda_base}/etc/profile.d/conda.sh && conda activate {CONDA_ENV}"
        cmd = f"{activate_cmd} && CUDA_VISIBLE_DEVICES={self.gpu} python {_SERVER_SCRIPT}"

        logger.info(f"[Qwen35] 启动子进程: GPU={self.gpu}")

        self._process = subprocess.Popen(
            ["bash", "-c", cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # 行缓冲
        )

        # 等待 ready 信号
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            line = self._process.stdout.readline().strip()
            if not line:
                if self._process.poll() is not None:
                    stderr = self._process.stderr.read()
                    raise RuntimeError(f"[Qwen35] 子进程启动失败: {stderr[:500]}")
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"[Qwen35] 启动输出: {line}")
                continue

            status = msg.get("status", "")
            if status == "ready":
                load_time = msg.get("load_time", "?")
                logger.info(f"[Qwen35] 子进程就绪，模型加载耗时: {load_time}s")
                self._ready = True
                return
            elif status == "loading":
                logger.info(f"[Qwen35] {msg.get('message', 'Loading...')}")
            else:
                logger.debug(f"[Qwen35] 启动消息: {msg}")

        raise TimeoutError(f"[Qwen35] 子进程启动超时 ({self.timeout}s)")

    def stop(self):
        """停止子进程"""
        if self._process is None:
            return

        try:
            self._send({"action": "quit"})
            self._process.wait(timeout=5)
        except Exception:
            self._process.kill()
        finally:
            self._ready = False
            self._process = None
            logger.info("[Qwen35] 子进程已停止")

    def _send(self, request: dict) -> dict:
        """发送请求并接收响应"""
        if not self.is_ready:
            raise RuntimeError("[Qwen35] 子进程未就绪")

        line = json.dumps(request, ensure_ascii=False) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

        # 读取响应
        resp_line = self._process.stdout.readline().strip()
        if not resp_line:
            raise RuntimeError("[Qwen35] 子进程无响应")

        return json.loads(resp_line)

    def predict(self, image: np.ndarray, landmark_name: str) -> Dict:
        """
        对图像执行打点预测

        Args:
            image: OpenCV BGR 图像 (np.ndarray)
            landmark_name: 地标名称 (中文)，如 "电梯"、"打印机"

        Returns:
            {
                "success": bool,
                "point": [x_norm, y_norm] or None,  # 归一化 [0,1] 坐标
                "point_pixel": [px, py] or None,     # 像素坐标
                "confidence": float,
                "raw_response": str,
                "latency": float,
            }
        """
        if not self.is_ready:
            return {
                "success": False,
                "point": None,
                "point_pixel": None,
                "confidence": 0.0,
                "error": "Qwen3.5 子进程未就绪",
            }

        # 编码图像为 base64
        _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(buf).decode('utf-8')

        try:
            resp = self._send({
                "action": "predict",
                "image_b64": image_b64,
                "landmark_name": landmark_name,
            })
        except Exception as e:
            logger.error(f"[Qwen35] 推理失败: {e}")
            return {
                "success": False,
                "point": None,
                "point_pixel": None,
                "confidence": 0.0,
                "error": str(e),
            }

        if resp.get("status") == "ok" and resp.get("point"):
            h, w = image.shape[:2]
            px = int(resp["point"][0] * w)
            py = int(resp["point"][1] * h)
            return {
                "success": True,
                "point": resp["point"],           # [0,1] 归一化
                "point_pixel": [px, py],           # 像素坐标
                "confidence": resp.get("confidence", 0.8),
                "raw_response": resp.get("raw_response", ""),
                "latency": resp.get("latency", 0),
            }
        else:
            return {
                "success": False,
                "point": None,
                "point_pixel": None,
                "confidence": 0.0,
                "raw_response": resp.get("raw_response", ""),
                "error": resp.get("error", "Unknown error"),
                "latency": resp.get("latency", 0),
            }

    def predict_on_camera(self, camera_images: Dict[str, np.ndarray],
                          landmark_name: str,
                          target_camera: str = None) -> Dict:
        """
        在指定相机（或所有相机）上执行打点

        Args:
            camera_images: {"camera_1": img, "camera_2": img, ...}
            landmark_name: 地标名称
            target_camera: 指定相机 (如 "camera_1")，None 则遍历所有相机

        Returns:
            {
                "success": bool,
                "camera_name": str,
                "point": [x_norm, y_norm],
                "point_pixel": [px, py],
                "confidence": float,
                ...
            }
        """
        cameras_to_try = [target_camera] if target_camera else sorted(camera_images.keys())

        best_result = None
        best_confidence = -1.0

        for cam_name in cameras_to_try:
            if cam_name not in camera_images:
                continue

            result = self.predict(camera_images[cam_name], landmark_name)
            result["camera_name"] = cam_name

            if result["success"] and result["confidence"] > best_confidence:
                best_result = result
                best_confidence = result["confidence"]

        if best_result:
            return best_result

        return {
            "success": False,
            "camera_name": cameras_to_try[0] if cameras_to_try else "",
            "point": None,
            "point_pixel": None,
            "confidence": 0.0,
            "error": f"All cameras failed for landmark '{landmark_name}'",
        }

    def __del__(self):
        self.stop()
