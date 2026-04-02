#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动地标命名器 v12 — 命名 + 字符识别

功能: describe_scene + identify_landmark + detect_text(语义增补用)
打点功能已移至 AutoSubImageExtractor (使用 Qwen35PointGrounder)
"""

import os, sys, logging, json, subprocess, time, base64, re
from typing import Dict, Tuple, Optional
import cv2, numpy as np
sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')
logger = logging.getLogger(__name__)

_QWEN_SERVER = r'''#!/usr/bin/env python3
import json, sys, time, re, base64, io, os, torch
from PIL import Image

MODEL_PATH = os.path.expanduser("~/Disk/models/Qwen3.5-9B")
MODEL = None; PROCESSOR = None

def load_model():
    global MODEL, PROCESSOR
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(json.dumps({"status":"loading","message":"Loading Qwen3.5-9B..."}), flush=True)
    t0 = time.time()
    PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    MODEL = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa")
    MODEL.eval()
    print(json.dumps({"status":"ready","load_time":round(time.time()-t0,1)}), flush=True)

def infer(image, prompt, max_tokens=80):
    msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
    text = PROCESSOR.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = PROCESSOR(text=[text], images=[image], padding=True, return_tensors="pt").to(MODEL.device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen = MODEL.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    return PROCESSOR.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

def clean(t): return re.sub(r"<think>.*?</think>","",t,flags=re.DOTALL).strip()

def decode_img(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    w,h = img.size
    if max(w,h)>960: s=960/max(w,h); img=img.resize((int(w*s),int(h*s)),Image.LANCZOS)
    return img

def parse_json(text):
    m = re.search(r'\{[^{}]*\}', text)
    if m:
        try: return json.loads(m.group())
        except: pass
    return None

load_model()

for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try: req = json.loads(line)
    except: print(json.dumps({"status":"error","error":"bad json"}),flush=True); continue

    action = req.get("action","")
    if action == "quit":
        print(json.dumps({"status":"bye"}),flush=True); break

    elif action == "describe_scene":
        try:
            img = decode_img(req["image_b64"])
            prompt = ("Robot indoor camera. Give a SHORT unique Chinese name (2-5 chars) for this location "
                      "based on the most specific feature (printer, elevator, front desk, mother-baby room, microwave). "
                      "Avoid generic names. Output ONLY JSON: {\"name_cn\":\"...\",\"name_en\":\"...\"}")
            d = parse_json(clean(infer(img,prompt)))
            if d: print(json.dumps({"status":"ok","name_cn":d.get("name_cn","未知"),"name_en":d.get("name_en","unknown")},ensure_ascii=False),flush=True)
            else: print(json.dumps({"status":"ok","name_cn":"未知","name_en":"unknown"}),flush=True)
        except Exception as e: print(json.dumps({"status":"error","error":str(e)}),flush=True)

    elif action == "identify_landmark":
        try:
            img = decode_img(req["image_b64"])
            prompt = ("Indoor navigation image. Name the SINGLE most prominent landmark visible "
                      "(e.g. trash can, white chair, printer, sign, plant, stool). "
                      "Output ONLY JSON: {\"name_cn\":\"...\",\"name_en\":\"...\"}")
            d = parse_json(clean(infer(img,prompt)))
            if d: print(json.dumps({"status":"ok","name_cn":d.get("name_cn","方向标记"),"name_en":d.get("name_en","marker")},ensure_ascii=False),flush=True)
            else: print(json.dumps({"status":"ok","name_cn":"方向标记","name_en":"marker"}),flush=True)
        except Exception as e: print(json.dumps({"status":"error","error":str(e)}),flush=True)

    elif action == "detect_text":
        try:
            img = decode_img(req["image_b64"])
            prompt = ("Look at this indoor image carefully. "
                      "Is there any visible text on signs, door plates, wall labels, room numbers, or nameplates? "
                      "Text includes Chinese characters, English words, and numbers/digits. "
                      "Ignore: exit signs, fire safety signs, no-smoking signs, evacuation signs, safety channel signs. "
                      "If you see meaningful text (e.g. room name, room number, area label), output JSON: "
                      "{\"found\": true, \"text\": \"the exact text you see\", "
                      "\"name_cn\": \"a short Chinese name based on the text (2-6 chars)\", "
                      "\"name_en\": \"English translation\"} "
                      "If no meaningful text is visible, output: {\"found\": false}")
            raw = clean(infer(img, prompt, max_tokens=120))
            d = parse_json(raw)
            if d and d.get("found"):
                print(json.dumps({"status":"ok","found":True,
                    "text":d.get("text",""),
                    "name_cn":d.get("name_cn",""),
                    "name_en":d.get("name_en","")},ensure_ascii=False),flush=True)
            else:
                print(json.dumps({"status":"ok","found":False}),flush=True)
        except Exception as e:
            print(json.dumps({"status":"error","error":str(e)}),flush=True)

    else:
        print(json.dumps({"status":"error","error":f"unknown:{action}"}),flush=True)
'''


class QwenNamingServer:
    CONDA_ENV = "qwen3"
    def __init__(self, gpu="1", timeout=120.0):
        self.gpu=gpu; self.timeout=timeout; self._process=None; self._ready=False

    @property
    def is_ready(self):
        return self._ready and self._process is not None and self._process.poll() is None

    def start(self):
        if self.is_ready: return
        script_path = "/tmp/_qwen_naming_v12.py"
        with open(script_path,'w') as f: f.write(_QWEN_SERVER)
        conda_base = os.environ.get("CONDA_PREFIX","").rsplit("/envs/",1)[0] or os.path.expanduser("~/miniconda3")
        cmd = (f"source {conda_base}/etc/profile.d/conda.sh && conda activate {self.CONDA_ENV} && "
               f"CUDA_VISIBLE_DEVICES={self.gpu} python {script_path}")
        logger.info(f"[QwenServer v12] Starting GPU={self.gpu}")
        self._process = subprocess.Popen(["bash","-c",cmd],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        t0 = time.time()
        while time.time()-t0 < self.timeout:
            line = self._process.stdout.readline().strip()
            if not line:
                if self._process.poll() is not None: raise RuntimeError(self._process.stderr.read()[:500])
                continue
            try: msg = json.loads(line)
            except: continue
            if msg.get("status")=="ready":
                logger.info(f"[QwenServer v12] Ready, {msg.get('load_time')}s"); self._ready=True; return
            elif msg.get("status")=="loading": logger.info(f"[QwenServer] {msg.get('message','...')}")
        raise TimeoutError(f"Timeout ({self.timeout}s)")

    def stop(self):
        if not self._process: return
        try: self._send({"action":"quit"}); self._process.wait(timeout=5)
        except: self._process.kill()
        finally: self._ready=False; self._process=None

    def _send(self, req):
        if not self.is_ready: raise RuntimeError("Not ready")
        self._process.stdin.write(json.dumps(req,ensure_ascii=False)+"\n"); self._process.stdin.flush()
        line = self._process.stdout.readline().strip()
        if not line: raise RuntimeError("No response")
        return json.loads(line)

    def describe_scene(self,b64): return self._send({"action":"describe_scene","image_b64":b64})
    def identify_landmark(self,b64): return self._send({"action":"identify_landmark","image_b64":b64})
    def detect_text(self,b64): return self._send({"action":"detect_text","image_b64":b64})
    def __del__(self): self.stop()


class AutoLandmarkNamer:
    def __init__(self, use_qwen=False, gpu="1"):
        self.use_qwen=use_qwen; self._qwen_server=None
        if use_qwen:
            try:
                self._qwen_server=QwenNamingServer(gpu=gpu); self._qwen_server.start()
                logger.info("[AutoLandmarkNamer v12] Qwen3.5 naming started")
            except Exception as e:
                logger.error(f"[AutoLandmarkNamer] Failed: {e}"); self._qwen_server=None; self.use_qwen=False

    def _b64(self,img):
        _,buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,85])
        return base64.b64encode(buf).decode('utf-8')

    def qwen_describe_scene(self,img):
        if not self._qwen_server or not self._qwen_server.is_ready: return "未知位置","unknown"
        try:
            r=self._qwen_server.describe_scene(self._b64(img))
            if r.get("status")=="ok": return r["name_cn"],r["name_en"]
        except Exception as e: logger.warning(f"describe_scene: {e}")
        return "未知位置","unknown"

    def qwen_identify_landmark(self,img):
        if not self._qwen_server or not self._qwen_server.is_ready: return "方向标记","direction_marker"
        try:
            r=self._qwen_server.identify_landmark(self._b64(img))
            if r.get("status")=="ok": return r["name_cn"],r["name_en"]
        except Exception as e: logger.warning(f"identify_landmark: {e}")
        return "方向标记","direction_marker"

    def generate_position_name(self, position_id, camera_images=None):
        if self.use_qwen and camera_images and self._qwen_server and self._qwen_server.is_ready:
            for ck in ['camera_1','camera_2','camera_3']:
                cp=camera_images.get(ck)
                if cp and os.path.exists(cp):
                    img=cv2.imread(cp)
                    if img is not None:
                        cn,en=self.qwen_describe_scene(img)
                        if cn not in ("未知位置","办公区","走廊","未知"): return cn,en
        n=int(re.search(r'\d+',position_id).group()) if re.search(r'\d+',position_id) else 0
        tcn=["走廊节点","房间入口","转弯处","通道中央","门前区域","开阔空间","过道节点","交叉路口"]
        ten=["corridor_node","room_entrance","corner_area","passage_center","door_front","open_space","hallway_node","intersection"]
        i=n%len(tcn); return f"{tcn[i]}_{n}",f"{ten[i]}_{n}"

    def generate_self_position_names(self, position_id, camera_images=None):
        cn,en=self.generate_position_name(position_id,camera_images)
        return {"position_name":cn,"position_name_eng":en}

    def stop(self):
        if self._qwen_server: self._qwen_server.stop()
    def __del__(self): self.stop()
