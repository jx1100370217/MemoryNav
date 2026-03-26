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


def build_existence_prompt(landmark_name: str) -> str:
    """构建存在性检测 prompt (第一步: 判断目标是否在图中)"""
    return (
        f'Is "{landmark_name}" visible in this image? '
        f'Answer with ONLY one word: yes or no.'
    )


def build_point_prompt(landmark_name: str) -> str:
    """构建打点 prompt (第二步: 仅在确认目标存在后调用)"""
    return (
        f'Point to "{landmark_name}" in this image. '
        f'Output ONLY JSON: {{"point": [x, y]}} '
        f'where coordinates are in [0, 1000] range.'
    )


def parse_coordinates(response: str):
    """从模型输出中解析坐标，支持模型回答"找不到目标"的情况"""
    # 清理 thinking 标签
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    clean = re.sub(r'```\w*\n?', '', clean).strip()

    # 检测"找不到"语义 (模型可能用自然语言回答而非 JSON)
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
            # 兼容多种格式
            if "point_2d" in d and "point" not in d:
                d["point"] = d.pop("point_2d")
            if "bbox_2d" in d and "point" not in d:
                # bbox → 取中心点
                b = d["bbox_2d"]
                if len(b) == 4:
                    d["point"] = [(b[0]+b[2])/2, (b[1]+b[3])/2]
            # 检查 point 是否为 null (模型明确表示找不到)
            if "point" in d:
                if d["point"] is None:
                    return None
                return d
        except json.JSONDecodeError:
            pass

    # regex fallback - 仅匹配明确的 point 格式
    m = re.search(r'point(?:_2d)?\D*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', clean)
    if m:
        return {"point": [float(m.group(1)), float(m.group(2))]}

    # [已移除] 数字 fallback — 这是导致"无论有没有目标都打点"的核心 BUG
    # 原逻辑: 只要响应里有2个数字就当坐标，即使模型在说 "I can't find it"
    # 现在: 如果没有明确的 point 格式，直接返回 None

    return None


def compute_token_confidence(scores, generated_ids):
    """
    从 generate 输出的 scores 计算 token 级别置信度

    Args:
        scores: tuple of (num_tokens,) tensors, each shape (1, vocab_size) — logits
        generated_ids: tensor of generated token ids, shape (num_tokens,)

    Returns:
        float: 平均 token 概率，范围 [0, 1]
    """
    import torch.nn.functional as F

    if not scores or len(scores) == 0:
        return 0.5

    probs_list = []
    for i, score in enumerate(scores):
        if i >= len(generated_ids):
            break
        # score: (1, vocab_size) → softmax → 取对应 token 的概率
        prob = F.softmax(score[0], dim=-1)
        token_prob = prob[generated_ids[i]].item()
        probs_list.append(token_prob)

    if not probs_list:
        return 0.5

    # 平均概率作为置信度
    avg_prob = sum(probs_list) / len(probs_list)
    return round(avg_prob, 4)


def _run_inference(image, prompt, max_new_tokens=64):
    """底层推理函数，返回 (response_text, token_confidence, latency)"""
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
        gen_output = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    gen_ids = gen_output.sequences
    gen_trimmed = gen_ids[:, inputs.input_ids.shape[1]:]
    response = PROCESSOR.batch_decode(gen_trimmed, skip_special_tokens=True)[0]
    latency = time.time() - t0

    token_confidence = compute_token_confidence(gen_output.scores, gen_trimmed[0])
    return response, token_confidence, latency


def check_existence(image, landmark_name: str):
    """
    第一步: 判断目标是否在图像中存在
    
    Returns:
        (exists: bool, confidence: float, response: str, latency: float)
    """
    prompt = build_existence_prompt(landmark_name)
    response, confidence, latency = _run_inference(image, prompt, max_new_tokens=8)
    
    # 解析 yes/no
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip().lower()
    # 去掉标点
    clean = re.sub(r'[^a-z]', '', clean)
    
    exists = clean.startswith('yes')
    return exists, confidence, response, latency


def predict(image_b64: str, landmark_name: str):
    """
    两步推理:
      1. 先问模型目标是否存在 (存在性检测)
      2. 只有确认存在时才执行打点
    """
    # 解码图像
    img_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 限制图像大小以减少 token
    max_side = 960
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # ===== 第一步: 存在性检测 =====
    exists, exist_conf, exist_response, exist_latency = check_existence(image, landmark_name)
    
    if not exists:
        return {
            "status": "not_found",
            "point": None,
            "confidence": 0.0,
            "raw_response": f"[existence_check] {exist_response}",
            "latency": round(exist_latency, 3),
            "error": f"Target '{landmark_name}' not found in image (model answered: {exist_response.strip()})",
        }

    # ===== 第二步: 打点 (仅在目标存在时执行) =====
    prompt = build_point_prompt(landmark_name)
    response, token_confidence, point_latency = _run_inference(image, prompt)

    total_latency = exist_latency + point_latency

    # 解析坐标
    result = parse_coordinates(response)

    if result and "point" in result:
        # 归一化到 [0, 1]
        px_norm = result["point"][0] / 1000.0
        py_norm = result["point"][1] / 1000.0

        return {
            "status": "ok",
            "point": [round(px_norm, 4), round(py_norm, 4)],
            "confidence": token_confidence,
            "raw_response": response,
            "latency": round(total_latency, 3),
        }
    else:
        return {
            "status": "failed",
            "point": None,
            "confidence": 0.0,
            "raw_response": response,
            "latency": round(total_latency, 3),
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
