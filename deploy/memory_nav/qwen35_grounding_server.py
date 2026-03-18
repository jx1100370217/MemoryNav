#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.5-9B 打点推理子进程服务

运行在 qwen3 conda 环境 (transformers 5.x)，通过 stdin/stdout JSON 行协议通信。
主进程通过 subprocess 启动本脚本，发送图像和 landmark_name，返回归一化坐标。

协议:
  请求: {"action": "predict", "image_b64": "...", "landmark_name": "电梯"}
  响应: {"status": "ok", "point": [x, y], "confidence": 0.85, "raw_response": "..."}

  请求: {"action": "ping"}
  响应: {"status": "ready"}

  请求: {"action": "quit"}
  (进程退出)
"""

import json
import sys
import time
import re
import base64
import io
import os

import torch
from PIL import Image


MODEL_PATH = os.path.expanduser("~/Disk/models/Qwen3.5-9B")
MODEL = None
PROCESSOR = None


def load_model():
    global MODEL, PROCESSOR
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(json.dumps({"status": "loading", "message": "Loading Qwen3.5-9B..."}), flush=True)
    t0 = time.time()

    PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    MODEL = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    MODEL.eval()

    elapsed = time.time() - t0
    print(json.dumps({"status": "ready", "load_time": round(elapsed, 1)}), flush=True)


def build_prompt(landmark_name: str) -> str:
    """构建打点 prompt，直接使用中文 landmark_name，要求输出置信度"""
    return (
        f'Point to "{landmark_name}" in this image. '
        f'Output ONLY JSON: {{"point": [x, y], "confidence": c}} '
        f'where x,y are coordinates in [0, 1000] range, '
        f'and c is your confidence between 0.0 and 1.0 '
        f'(1.0 = certain the object is there, 0.0 = not found).'
    )


def parse_coordinates(response: str):
    """从模型输出中解析坐标"""
    # 清理 thinking 标签
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    clean = re.sub(r'```\w*\n?', '', clean).strip()

    # 尝试提取 JSON
    m = re.search(r'\{[^{}]*\}', clean)
    if m:
        try:
            d = json.loads(m.group())
            # 兼容多种格式
            if "point_2d" in d and "point" not in d:
                d["point"] = d.pop("point_2d")
            if "point" in d:
                # 确保 confidence 字段存在
                if "confidence" not in d:
                    d["confidence"] = 0.5
                return d
        except json.JSONDecodeError:
            pass

    # regex fallback
    m = re.search(r'point(?:_2d)?\D*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', clean)
    if m:
        return {"point": [float(m.group(1)), float(m.group(2))]}

    # 数字 fallback
    nums = re.findall(r'\d+(?:\.\d+)?', clean)
    if len(nums) >= 2:
        return {"point": [float(nums[0]), float(nums[1])]}

    return None


def predict(image_b64: str, landmark_name: str):
    """执行推理"""
    # 解码图像
    img_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 限制图像大小以减少 token
    max_side = 960
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    prompt = build_prompt(landmark_name)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    text = PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = PROCESSOR(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(MODEL.device)

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        gen_ids = MODEL.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
        )
    gen_trimmed = gen_ids[:, inputs.input_ids.shape[1]:]
    response = PROCESSOR.batch_decode(gen_trimmed, skip_special_tokens=True)[0]
    latency = time.time() - t0

    # 解析坐标
    result = parse_coordinates(response)

    if result and "point" in result:
        # 归一化到 [0, 1]
        px_norm = result["point"][0] / 1000.0
        py_norm = result["point"][1] / 1000.0

        # 使用模型输出的置信度
        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "status": "ok",
            "point": [round(px_norm, 4), round(py_norm, 4)],
            "point_1000": result["point"],
            "confidence": round(confidence, 2),
            "raw_response": response,
            "latency": round(latency, 3),
        }
    else:
        return {
            "status": "failed",
            "point": None,
            "confidence": 0.0,
            "raw_response": response,
            "latency": round(latency, 3),
            "error": "Failed to parse coordinates",
        }


def main():
    load_model()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"status": "error", "error": f"Invalid JSON: {e}"}), flush=True)
            continue

        action = req.get("action", "")

        if action == "ping":
            print(json.dumps({"status": "ready"}), flush=True)

        elif action == "quit":
            print(json.dumps({"status": "bye"}), flush=True)
            break

        elif action == "predict":
            image_b64 = req.get("image_b64", "")
            landmark_name = req.get("landmark_name", "")

            if not image_b64 or not landmark_name:
                print(json.dumps({
                    "status": "error",
                    "error": "Missing image_b64 or landmark_name"
                }), flush=True)
                continue

            try:
                result = predict(image_b64, landmark_name)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as e:
                print(json.dumps({
                    "status": "error",
                    "error": str(e)
                }), flush=True)

        else:
            print(json.dumps({
                "status": "error",
                "error": f"Unknown action: {action}"
            }), flush=True)


if __name__ == "__main__":
    main()
