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
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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

    # 绕过系统代理, vLLM 是本地服务
    _NO_PROXY = {"http": None, "https": None}

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
            import requests as _requests
            _s = _requests.Session()
            _s.trust_env = False
            resp = _s.get(
                f"{self.base_url}/models",
                timeout=5,
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
            self._session.trust_env = False  # 不读环境变量代理(all_proxy/http_proxy等)
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

    def pick_best_cam(self, cam_imgs_b64: dict, target_img_b64: str,
                       max_tokens: int = 8) -> Optional[str]:
        """Single-image VQA via grid composition (Qwen 4096 ctx limit = cannot
        fit 5 separate images). Target at top-left, candidate cams labeled 1-4.

        cam_imgs_b64: dict {cam_name -> b64 jpeg}
        target_img_b64: b64 jpeg of target node
        Returns chosen cam_name or None.
        """
        if not self._ready:
            return None
        cam_names = list(cam_imgs_b64.keys())
        if not cam_names:
            return None

        # Decode and build grid
        try:
            import base64 as _b64m, cv2 as _cv2, numpy as _np
            def _decode(b):
                buf = _np.frombuffer(_b64m.b64decode(b), _np.uint8)
                return _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
            tgt = _decode(target_img_b64)
            cams = [(n, _decode(cam_imgs_b64[n])) for n in cam_names]
            if tgt is None or any(img is None for _, img in cams):
                return None
            tile = 256
            cols = 3
            rows = (1 + len(cams) + cols - 1) // cols
            canvas = _np.full((rows * tile, cols * tile, 3), 255, _np.uint8)
            def _place(i, lbl, img):
                r, c = i // cols, i % cols
                resized = _cv2.resize(img, (tile, tile))
                canvas[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = resized
                _cv2.rectangle(canvas, (c*tile, r*tile), (c*tile+tile-1, r*tile+tile-1), (0, 0, 255), 2)
                _cv2.putText(canvas, lbl, (c*tile+10, r*tile+30),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            _place(0, "TARGET", tgt)
            for i, (name, img) in enumerate(cams):
                _place(i + 1, str(i + 1), img)
            _, grid_buf = _cv2.imencode('.jpg', canvas, [_cv2.IMWRITE_JPEG_QUALITY, 88])
            grid_b64 = _b64m.b64encode(grid_buf).decode('utf-8')
        except Exception as e:
            logger.warning(f"pick_best_cam grid build failed: {e}")
            return None

        prompt_text = (
            f"图中 TARGET 是目标场景,数字 1-{len(cams)} 是机器人 {len(cams)} 个相机的当前视角. "
            f"哪个数字(1-{len(cams)})的图与 TARGET 里的建筑/地标/场景最相似? "
            f"只回答一个数字, 不要解释."
        )
        content = [
            {"type": "image_url", "image_url": {"url": _build_image_url(grid_b64)}},
            {"type": "text", "text": prompt_text},
        ]
        messages = [{"role": "user", "content": content}]
        payload = {
            "model": self.model_name, "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            session = self._get_session()
            resp = session.post(f"{self.base_url}/chat/completions",
                                json=payload, timeout=self.timeout)
            resp.raise_for_status()
            txt = resp.json()["choices"][0]["message"]["content"].strip()
            logger.info(f"pick_best_cam raw={txt!r}")
            for ch in txt:
                if ch.isdigit():
                    idx = int(ch) - 1
                    if 0 <= idx < len(cam_names):
                        return cam_names[idx]
                    break
        except Exception as e:
            logger.warning(f"pick_best_cam request failed: {e}")
        return None

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

    def vqa_pick_cam(self, self_cam_imgs: dict, target_img) -> Optional[str]:
        """Ask Qwen VLM which cam image matches target scene best.

        self_cam_imgs: dict {cam_name -> BGR numpy}
        target_img: BGR numpy of target node
        Returns cam_name or None.
        """
        if not self.use_qwen or not self._qwen_server or not self._qwen_server.is_ready:
            return None
        try:
            cam_b64 = {n: self._b64(img) for n, img in self_cam_imgs.items() if img is not None}
            if not cam_b64:
                return None
            tgt_b64 = self._b64(target_img)
            return self._qwen_server.pick_best_cam(cam_b64, tgt_b64)
        except Exception as e:
            logger.warning(f"vqa_pick_cam: {e}")
            return None

    def stop(self):
        if self._qwen_server:
            self._qwen_server.stop()

    def __del__(self):
        self.stop()
