#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动地标命名器 v13 — vLLM 推理后端

v13 改进:
- 替换 transformers 子进程为 vLLM OpenAI 兼容 API (HTTP)
- 推理速度提升 2-3x (continuous batching + PagedAttention)
- 需先启动 vLLM 服务: bash deploy/start_qwen_vllm.sh

功能: describe_scene + identify_landmark + detect_text(语义增补用)
打点功能已移至 AutoSubImageExtractor (使用 Qwen35PointGrounder)
"""

import os, sys, logging, json, time, base64, re
from typing import Dict, Tuple, Optional
import cv2, numpy as np
sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')
logger = logging.getLogger(__name__)

# vLLM 服务默认地址
VLLM_BASE_URL = "http://localhost:8199/v1"
VLLM_MODEL_NAME = "qwen3.5-9b"


def _build_image_url(b64: str) -> str:
    """将 base64 图片编码为 data URL"""
    return f"data:image/jpeg;base64,{b64}"


# ===== Prompt 定义 =====

PROMPT_DESCRIBE_SCENE = (
    "Robot indoor camera. Give a SHORT unique Chinese name (2-5 chars) for this location "
    "based on the most specific feature (printer, elevator, front desk, mother-baby room, microwave). "
    "Avoid generic names. Output ONLY JSON: {\"name_cn\":\"...\",\"name_en\":\"...\"}"
)

PROMPT_IDENTIFY_LANDMARK = (
    "Indoor navigation image. Name the SINGLE most prominent landmark visible "
    "(e.g. trash can, white chair, printer, sign, plant, stool). "
    "Output ONLY JSON: {\"name_cn\":\"...\",\"name_en\":\"...\"}"
)

PROMPT_DETECT_TEXT = (
    "This is an indoor navigation robot camera image. "
    "Look ONLY for text that identifies a ROOM or PLACE — specifically: "
    "(1) Door plates or room number signs (e.g. '101', '10', 'A3') "
    "(2) Room name plates on or beside doors (e.g. '关爱室', '母婴室', '茶水间') "
    "(3) Meeting room names on glass walls/doors (e.g. 'NEUMANN', 'MOORE', 'EINSTEIN') "
    "(4) Store or shop name signs above entrances "
    "DO NOT report text from: posters, decorations, paintings, bulletin boards, "
    "company slogans, corporate culture walls, safety/exit/fire signs, computer screens, "
    "whiteboards, banners, advertisements, or personal nameplates on desks. "
    "IMPORTANT: In name_cn, ALWAYS include the room type. Examples: "
    "- See '10' on door plate near meeting room → name_cn: '10号会议室' "
    "- See 'NEUMANN' on meeting room glass → name_cn: '纽曼会议室' "
    "- See 'MOORE' on meeting room glass → name_cn: '摩尔会议室' "
    "- See '关爱室' on door → name_cn: '关爱室' "
    "If you see a room/place identifying sign, output JSON: "
    "{\"found\": true, \"text\": \"exact text on the sign\", "
    "\"name_cn\": \"Chinese place name WITH room type (2-8 chars)\", "
    "\"name_en\": \"English translation\"} "
    "If NO room/place sign is visible, output: {\"found\": false}"
)


def _parse_json(text: str) -> Optional[dict]:
    """从模型输出中提取 JSON"""
    # 去除 <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r'\{[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return None


class QwenNamingServer:
    """通过 vLLM OpenAI API 调用 Qwen3.5-9B"""

    def __init__(self, gpu="1", timeout=30.0,
                 base_url=None, model_name=None):
        self.gpu = gpu
        self.timeout = timeout
        self.base_url = base_url or VLLM_BASE_URL
        self.model_name = model_name or VLLM_MODEL_NAME
        self._ready = False
        self._session = None

    @property
    def is_ready(self):
        return self._ready

    def start(self):
        """检查 vLLM 服务是否可用"""
        import requests
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                timeout=5
            )
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                model_ids = [m.get("id") for m in models]
                if self.model_name in model_ids:
                    self._ready = True
                    logger.info(f"[QwenServer v13] vLLM 服务已连接: {self.base_url}, model={self.model_name}")
                    return
                else:
                    raise RuntimeError(
                        f"vLLM 服务已启动但未找到模型 {self.model_name}, "
                        f"可用模型: {model_ids}"
                    )
        except requests.ConnectionError:
            raise RuntimeError(
                f"无法连接 vLLM 服务 ({self.base_url}). "
                f"请先启动: bash deploy/start_qwen_vllm.sh {self.gpu}"
            )
        except Exception as e:
            raise RuntimeError(f"vLLM 服务检查失败: {e}")

    def stop(self):
        """HTTP 客户端无需 stop"""
        self._ready = False
        if self._session:
            self._session.close()
            self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _chat(self, prompt: str, image_b64: str, max_tokens: int = 80) -> str:
        """调用 vLLM chat completion API"""
        if not self._ready:
            raise RuntimeError("vLLM 服务未就绪")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _build_image_url(image_b64)
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
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        session = self._get_session()
        resp = session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout
        )
        resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()

    def describe_scene(self, b64: str) -> dict:
        try:
            raw = self._chat(PROMPT_DESCRIBE_SCENE, b64, max_tokens=80)
            d = _parse_json(raw)
            if d:
                return {"status": "ok",
                        "name_cn": d.get("name_cn", "未知"),
                        "name_en": d.get("name_en", "unknown")}
            return {"status": "ok", "name_cn": "未知", "name_en": "unknown"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def identify_landmark(self, b64: str) -> dict:
        try:
            raw = self._chat(PROMPT_IDENTIFY_LANDMARK, b64, max_tokens=80)
            d = _parse_json(raw)
            if d:
                return {"status": "ok",
                        "name_cn": d.get("name_cn", " "),
                        "name_en": d.get("name_en", " ")}
            return {"status": "ok", "name_cn": " ", "name_en": " "}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def detect_text(self, b64: str) -> dict:
        try:
            raw = self._chat(PROMPT_DETECT_TEXT, b64, max_tokens=120)
            d = _parse_json(raw)
            if d and d.get("found"):
                return {"status": "ok", "found": True,
                        "text": d.get("text", ""),
                        "name_cn": d.get("name_cn", ""),
                        "name_en": d.get("name_en", "")}
            return {"status": "ok", "found": False}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def __del__(self):
        self.stop()


class AutoLandmarkNamer:
    def __init__(self, use_qwen=False, gpu="1"):
        self.use_qwen = use_qwen
        self._qwen_server = None
        if use_qwen:
            try:
                self._qwen_server = QwenNamingServer(gpu=gpu)
                self._qwen_server.start()
                logger.info("[AutoLandmarkNamer v13] vLLM 后端已连接")
            except Exception as e:
                logger.error(f"[AutoLandmarkNamer] Failed: {e}")
                self._qwen_server = None
                self.use_qwen = False

    def _b64(self, img):
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('utf-8')

    def qwen_describe_scene(self, img):
        if not self._qwen_server or not self._qwen_server.is_ready:
            return "未知位置", "unknown"
        try:
            r = self._qwen_server.describe_scene(self._b64(img))
            if r.get("status") == "ok":
                return r["name_cn"], r["name_en"]
        except Exception as e:
            logger.warning(f"describe_scene: {e}")
        return "未知位置", "unknown"

    def qwen_identify_landmark(self, img):
        if not self._qwen_server or not self._qwen_server.is_ready:
            return " ", " "
        try:
            r = self._qwen_server.identify_landmark(self._b64(img))
            if r.get("status") == "ok":
                return r["name_cn"], r["name_en"]
        except Exception as e:
            logger.warning(f"identify_landmark: {e}")
        return " ", " "

    def generate_position_name(self, position_id, camera_images=None):
        if self.use_qwen and camera_images and self._qwen_server and self._qwen_server.is_ready:
            for ck in ['camera_1', 'camera_2', 'camera_3']:
                cp = camera_images.get(ck)
                if cp and os.path.exists(cp):
                    img = cv2.imread(cp)
                    if img is not None:
                        cn, en = self.qwen_describe_scene(img)
                        if cn not in ("未知位置", "办公区", "走廊", "未知"):
                            return cn, en
        n = int(re.search(r'\d+', position_id).group()) if re.search(r'\d+', position_id) else 0
        tcn = ["走廊节点", "房间入口", "转弯处", "通道中央", "门前区域", "开阔空间", "过道节点", "交叉路口"]
        ten = ["corridor_node", "room_entrance", "corner_area", "passage_center",
               "door_front", "open_space", "hallway_node", "intersection"]
        i = n % len(tcn)
        return f"{tcn[i]}_{n}", f"{ten[i]}_{n}"

    def generate_self_position_names(self, position_id, camera_images=None):
        cn, en = self.generate_position_name(position_id, camera_images)
        return {"position_name": cn, "position_name_eng": en}

    def stop(self):
        if self._qwen_server:
            self._qwen_server.stop()

    def __del__(self):
        self.stop()
