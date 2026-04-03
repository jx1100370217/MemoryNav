#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.5-9B 打点模块 v2 - vLLM HTTP (并行) + 子进程 (串行) 双后端

优先使用 vLLM HTTP API (与 auto_landmark_namer 共享同一个 vLLM 服务)，
避免子进程重复加载模型导致 GPU 显存不足。

当 vLLM 服务不可用时，自动回退到子进程模式 (qwen3 conda 环境)。

用法:
    grounder = Qwen35PointGrounder()
    grounder.start()  # 自动检测 vLLM，优先 HTTP
    result = grounder.predict(camera_image, "电梯")
    # result = {"point": [0.45, 0.62], "confidence": 0.8, ...}
"""

import json
import logging
import subprocess
import time
import base64
import os
import re
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# vLLM 默认配置
VLLM_BASE_URL = "http://localhost:8199/v1"
VLLM_MODEL_NAME = "qwen3.5-9b"


# ===== vLLM 模式的 Prompt 和解析 (从 qwen35_grounding_server.py 移植) =====

def _build_existence_prompt(landmark_name: str) -> str:
    return (
        f'Is "{landmark_name}" visible in this image? '
        f'Answer with ONLY one word: yes or no.'
    )

def _build_point_prompt(landmark_name: str) -> str:
    return (
        f'Point to "{landmark_name}" in this image. '
        f'Output ONLY JSON: {{"point": [x, y]}} '
        f'where coordinates are in [0, 1000] range.'
    )

def _parse_coordinates(response: str):
    """从模型输出中解析坐标"""
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    clean = re.sub(r'```\w*\n?', '', clean).strip()

    # 检测"找不到"语义
    not_found_patterns = [
        r'not\s+(?:visible|found|present|seen|detected)',
        r'cannot\s+(?:find|see|detect|locate)',
        r"can't\s+(?:find|see|detect|locate)",
        r'no\s+(?:visible|clear)',
        r'unable\s+to\s+(?:find|locate|detect)',
        r"doesn't\s+appear",
        r'not\s+in\s+(?:the|this)\s+image',
        r'找不到', r'未找到', r'不存在', r'没有找到', r'看不到', r'无法找到',
    ]
    clean_lower = clean.lower()
    for pat in not_found_patterns:
        if re.search(pat, clean_lower):
            return None

    # 尝试提取 JSON
    m = re.search(r'\{[^{}]*\}', clean)
    if m:
        try:
            d = json.loads(m.group())
            if "point_2d" in d and "point" not in d:
                d["point"] = d.pop("point_2d")
            if "bbox_2d" in d and "point" not in d:
                b = d["bbox_2d"]
                if len(b) == 4:
                    d["point"] = [(b[0]+b[2])/2, (b[1]+b[3])/2]
            if "point" in d:
                if d["point"] is None:
                    return None
                return d
        except json.JSONDecodeError:
            pass

    # regex fallback
    m = re.search(r'point(?:_2d)?\D*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', clean)
    if m:
        return {"point": [float(m.group(1)), float(m.group(2))]}

    return None


class Qwen35PointGrounder:
    """
    Qwen3.5-9B 打点器 — 双后端 (vLLM HTTP / 子进程)
    
    优先使用 vLLM HTTP API，避免重复加载模型占用 GPU 显存。
    vLLM 不可用时回退到子进程模式 (qwen3 conda 环境)。
    """

    def __init__(self, gpu: str = DEFAULT_GPU, timeout: float = 120.0,
                 vllm_url: str = None, vllm_model: str = None):
        """
        Args:
            gpu: CUDA_VISIBLE_DEVICES 设置 (仅子进程模式使用)
            timeout: 启动超时时间 (秒)
            vllm_url: vLLM 服务地址，默认 http://localhost:8199/v1
            vllm_model: vLLM 模型名，默认 qwen3.5-9b
        """
        self.gpu = gpu
        self.timeout = timeout
        self._vllm_url = vllm_url or VLLM_BASE_URL
        self._vllm_model = vllm_model or VLLM_MODEL_NAME
        self._process: Optional[subprocess.Popen] = None
        self._ready = False
        self._backend = None  # "vllm" or "subprocess"
        self._session = None  # requests.Session for vLLM

    @property
    def is_ready(self) -> bool:
        if self._backend == "vllm":
            return self._ready
        elif self._backend == "subprocess":
            return self._ready and self._process is not None and self._process.poll() is None
        return False

    def _try_vllm(self) -> bool:
        """尝试连接 vLLM 服务"""
        try:
            import requests
            resp = requests.get(f"{self._vllm_url}/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                model_ids = [m.get("id", "") for m in models]
                logger.info(f"[Qwen35] vLLM 服务可用, models={model_ids}")
                return True
        except Exception as e:
            logger.debug(f"[Qwen35] vLLM 服务不可用: {e}")
        return False

    def start(self):
        """启动打点器：优先 vLLM HTTP，回退子进程"""
        if self.is_ready:
            logger.info(f"[Qwen35] 已就绪 (backend={self._backend})")
            return

        # 优先尝试 vLLM
        if self._try_vllm():
            self._backend = "vllm"
            self._ready = True
            logger.info(f"[Qwen35] 使用 vLLM HTTP 后端: {self._vllm_url}")
            return

        # 回退到子进程
        logger.info("[Qwen35] vLLM 不可用，回退到子进程模式")
        self._start_subprocess()

    def _start_subprocess(self):
        """启动 Qwen3.5 子进程 (原有逻辑)"""
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
            bufsize=1,
        )

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
                self._backend = "subprocess"
                self._ready = True
                return
            elif status == "loading":
                logger.info(f"[Qwen35] {msg.get('message', 'Loading...')}")
            else:
                logger.debug(f"[Qwen35] 启动消息: {msg}")

        raise TimeoutError(f"[Qwen35] 子进程启动超时 ({self.timeout}s)")

    def stop(self):
        """停止打点器"""
        if self._backend == "subprocess" and self._process is not None:
            try:
                self._send_subprocess({"action": "quit"})
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            finally:
                self._process = None
        
        if self._session:
            self._session.close()
            self._session = None

        self._ready = False
        self._backend = None
        logger.info("[Qwen35] 已停止")

    # ------------------------------------------------------------------
    # 子进程模式通信
    # ------------------------------------------------------------------
    def _send_subprocess(self, request: dict) -> dict:
        if not self._process or self._process.poll() is not None:
            raise RuntimeError("[Qwen35] 子进程未就绪")

        line = json.dumps(request, ensure_ascii=False) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

        resp_line = self._process.stdout.readline().strip()
        if not resp_line:
            raise RuntimeError("[Qwen35] 子进程无响应")

        return json.loads(resp_line)

    # ------------------------------------------------------------------
    # vLLM HTTP 模式
    # ------------------------------------------------------------------
    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _vllm_chat(self, prompt: str, image_b64: str, max_tokens: int = 64) -> str:
        """调用 vLLM chat completion API"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        payload = {
            "model": self._vllm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        session = self._get_session()
        resp = session.post(
            f"{self._vllm_url}/chat/completions",
            json=payload,
            timeout=30
        )
        resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()

    def _predict_vllm(self, image: np.ndarray, landmark_name: str) -> Dict:
        """vLLM HTTP 模式的打点推理 (两步: 存在性检测 + 打点)"""
        # 编码图像
        _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(buf).decode('utf-8')

        h, w = image.shape[:2]
        t0 = time.time()

        # 第一步: 存在性检测
        exist_prompt = _build_existence_prompt(landmark_name)
        try:
            exist_resp = self._vllm_chat(exist_prompt, image_b64, max_tokens=8)
        except Exception as e:
            logger.error(f"[Qwen35-vLLM] 存在性检测失败: {e}")
            return self._fail_result(str(e))

        exist_clean = re.sub(r'<think>.*?</think>', '', exist_resp, flags=re.DOTALL).strip().lower()
        exist_clean = re.sub(r'[^a-z]', '', exist_clean)

        if not exist_clean.startswith('yes'):
            latency = time.time() - t0
            return {
                "success": False,
                "point": None,
                "point_pixel": None,
                "confidence": 0.0,
                "raw_response": f"[existence_check] {exist_resp}",
                "latency": round(latency, 3),
                "error": f"Target '{landmark_name}' not found (model: {exist_resp.strip()})",
            }

        # 第二步: 打点
        point_prompt = _build_point_prompt(landmark_name)
        try:
            point_resp = self._vllm_chat(point_prompt, image_b64, max_tokens=64)
        except Exception as e:
            logger.error(f"[Qwen35-vLLM] 打点推理失败: {e}")
            return self._fail_result(str(e))

        latency = time.time() - t0

        # 解析坐标
        result = _parse_coordinates(point_resp)

        if result and "point" in result:
            px_norm = result["point"][0] / 1000.0
            py_norm = result["point"][1] / 1000.0

            px = int(px_norm * w)
            py = int(py_norm * h)

            return {
                "success": True,
                "point": [round(px_norm, 4), round(py_norm, 4)],
                "point_pixel": [px, py],
                "confidence": 0.8,  # vLLM 模式无 token-level confidence
                "raw_response": point_resp,
                "latency": round(latency, 3),
            }
        else:
            return {
                "success": False,
                "point": None,
                "point_pixel": None,
                "confidence": 0.0,
                "raw_response": point_resp,
                "latency": round(latency, 3),
                "error": "Failed to parse coordinates",
            }

    def _predict_subprocess(self, image: np.ndarray, landmark_name: str) -> Dict:
        """子进程模式的打点推理 (原有逻辑)"""
        _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(buf).decode('utf-8')

        try:
            resp = self._send_subprocess({
                "action": "predict",
                "image_b64": image_b64,
                "landmark_name": landmark_name,
            })
        except Exception as e:
            logger.error(f"[Qwen35] 推理失败: {e}")
            return self._fail_result(str(e))

        if resp.get("status") == "ok" and resp.get("point"):
            h, w = image.shape[:2]
            px = int(resp["point"][0] * w)
            py = int(resp["point"][1] * h)
            return {
                "success": True,
                "point": resp["point"],
                "point_pixel": [px, py],
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

    @staticmethod
    def _fail_result(error: str) -> Dict:
        return {
            "success": False,
            "point": None,
            "point_pixel": None,
            "confidence": 0.0,
            "error": error,
        }

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def predict(self, image: np.ndarray, landmark_name: str) -> Dict:
        """
        对图像执行打点预测 (自动选择后端)

        Args:
            image: OpenCV BGR 图像 (np.ndarray)
            landmark_name: 地标名称 (中文)，如 "电梯"、"打印机"

        Returns:
            {
                "success": bool,
                "point": [x_norm, y_norm] or None,
                "point_pixel": [px, py] or None,
                "confidence": float,
                "raw_response": str,
                "latency": float,
            }
        """
        if not self.is_ready:
            return self._fail_result("Qwen3.5 未就绪")

        if self._backend == "vllm":
            return self._predict_vllm(image, landmark_name)
        else:
            return self._predict_subprocess(image, landmark_name)

    def predict_on_camera(self, camera_images: Dict[str, np.ndarray],
                          landmark_name: str,
                          target_camera: str = None) -> Dict:
        """
        在所有相机上并行执行打点 (v2: vLLM continuous batching 并行)

        Args:
            camera_images: {"camera_1": img, "camera_2": img, ...}
            landmark_name: 地标名称
            target_camera: 指定相机优先 (如 "camera_1")，None 则遍历所有相机

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
        if target_camera:
            cameras_to_try = [target_camera] + [c for c in sorted(camera_images.keys()) if c != target_camera]
        else:
            cameras_to_try = sorted(camera_images.keys())

        valid_cameras = [c for c in cameras_to_try if c in camera_images]
        if not valid_cameras:
            return {
                "success": False,
                "camera_name": "",
                "point": None,
                "point_pixel": None,
                "confidence": 0.0,
                "error": f"No valid cameras for landmark '{landmark_name}'",
            }

        # vLLM 后端: 并行提交所有 camera 请求
        if self._backend == "vllm" and len(valid_cameras) > 1:
            def _predict_one(cam_name):
                result = self.predict(camera_images[cam_name], landmark_name)
                result["camera_name"] = cam_name
                return result

            results = []
            with ThreadPoolExecutor(max_workers=len(valid_cameras)) as executor:
                futures = {executor.submit(_predict_one, c): c for c in valid_cameras}
                for future in as_completed(futures):
                    results.append(future.result())

            # 选最高置信度
            best_result = None
            best_confidence = -1.0
            for r in results:
                if r["success"] and r["confidence"] > best_confidence:
                    best_result = r
                    best_confidence = r["confidence"]

            if best_result:
                logger.info(f"[Qwen35] predict_on_camera 成功(parallel): camera={best_result['camera_name']}, "
                           f"landmark='{landmark_name}', conf={best_confidence:.4f}, "
                           f"tried={valid_cameras}")
                return best_result

            logger.warning(f"[Qwen35] predict_on_camera 全部失败(parallel): landmark='{landmark_name}', "
                          f"tried={valid_cameras}")
            return {
                "success": False,
                "camera_name": valid_cameras[0],
                "point": None,
                "point_pixel": None,
                "confidence": 0.0,
                "error": f"All cameras failed for landmark '{landmark_name}'",
            }

        # 子进程后端: 保持串行 (子进程同一时间只能处理一个请求)
        best_result = None
        best_confidence = -1.0

        for cam_name in valid_cameras:
            result = self.predict(camera_images[cam_name], landmark_name)
            result["camera_name"] = cam_name

            if result["success"] and result["confidence"] > best_confidence:
                best_result = result
                best_confidence = result["confidence"]
                if best_confidence >= 0.5:
                    break

        if best_result:
            logger.info(f"[Qwen35] predict_on_camera 成功: camera={best_result['camera_name']}, "
                       f"landmark='{landmark_name}', conf={best_confidence:.4f}, "
                       f"tried={valid_cameras}")
            return best_result

        logger.warning(f"[Qwen35] predict_on_camera 全部失败: landmark='{landmark_name}', "
                      f"tried={valid_cameras}, target_camera={target_camera}")
        return {
            "success": False,
            "camera_name": valid_cameras[0],
            "point": None,
            "point_pixel": None,
            "confidence": 0.0,
            "error": f"All cameras failed for landmark '{landmark_name}'",
        }

    def __del__(self):
        self.stop()
