#!/usr/bin/env python3
"""建图模式 + 导航模式 端到端测试客户端.

建图流程:
  1. 连接 ws://localhost:9528
  2. 发 start_mapping
  3. 遍历 memory_test_data 下所有时间戳, 每个时间戳的 4 cam 作为一帧发给服务器
  4. 发 stop_mapping -> 收 summary 打印
  5. 发 reset 切回 nav 冒烟

用法:
  /home/ubuntu/miniconda3/envs/internvla/bin/python test_mapping_client.py [--data DIR] [--max N] [--port P]
"""
from __future__ import annotations
import argparse, base64, glob, json, os, sys, time

import websocket


def send(ws, msg, print_key_only: bool = True):
    data_out = json.dumps(msg, ensure_ascii=False)
    ws.send(data_out)
    raw = ws.recv()
    resp = json.loads(raw)
    return resp


def load_timestamps(data_dir: str):
    files = glob.glob(os.path.join(data_dir, "*_camera_*.jpg"))
    tss = sorted({os.path.basename(f).split('_')[0] for f in files})
    return tss


def encode_b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def build_frame(data_dir: str, ts: str):
    imgs = {}
    for cam in ('camera_1', 'camera_2', 'camera_3', 'camera_4'):
        p = os.path.join(data_dir, f"{ts}_{cam}.jpg")
        if os.path.exists(p):
            imgs[cam] = encode_b64(p)
    # front_1 也顺带 (nav 模式会用)
    front = os.path.join(data_dir, f"{ts}_front_1.jpg")
    if os.path.exists(front):
        imgs['front_1'] = encode_b64(front)
    return imgs


def run_mapping(ws, data_dir: str, max_frames: int | None):
    print(f"\n=== MAPPING MODE ===")

    resp = send(ws, {"command": "start_mapping"})
    print(f"  start_mapping: {resp.get('status')} mode={resp.get('mode')}")
    print(f"    output_dir: {resp.get('output_dir')}")
    if resp.get('status') != 'success':
        print(f"    FATAL: {resp.get('message')}")
        return resp

    tss = load_timestamps(data_dir)
    if max_frames:
        tss = tss[:max_frames]
    print(f"  feeding {len(tss)} frames from {data_dir}")

    t0 = time.time()
    for i, ts in enumerate(tss):
        imgs = build_frame(data_dir, ts)
        if len(imgs) < 4:
            print(f"  [{i}] skip ts={ts} (only {list(imgs)})")
            continue
        req = {
            "id": "TEST_MAPPER",
            "pts": int(ts),
            "task": None,
            "images": imgs,
        }
        t_s = time.time()
        resp = send(ws, req)
        dt = time.time() - t_s
        log = resp.get('log', {}) or {}
        mapping = resp.get('mapping', {}) or {}
        flag_kf = "KF" if log.get('keyframe') else "  "
        reason = log.get('reason', '')
        cat = (log.get('category_decision') or {}).get('category', '')
        sim = log.get('vpr_sim_to_last')
        sim_s = f"{sim:.3f}" if isinstance(sim, (int, float)) else " n/a "
        ig = log.get('info_gain') or 0.0
        nodes = mapping.get('n_nodes', 0)
        lc = mapping.get('n_loop_closures', 0)
        print(f"  [{i:3d}] ts={ts} {flag_kf} sim={sim_s} ig={ig:.4f} "
              f"cat={cat:<8} nodes={nodes:2d} loops={lc} "
              f"dt={dt*1000:.0f}ms reason={reason}")

    print(f"  feed elapsed: {time.time()-t0:.1f}s")

    resp = send(ws, {"command": "stop_mapping"})
    print(f"\n  stop_mapping: {resp.get('status')}")
    if resp.get('status') == 'success':
        s = resp['summary']
        print(f"    output_dir: {s.get('output_dir')}")
        print(f"    frames: {s.get('n_frames_processed')}")
        print(f"    finalize: {s.get('finalize_seconds')}s")
        m = s.get('metrics', {}) or {}
        print(f"    metrics: n_nodes={m.get('n_nodes')} n_edges={m.get('n_edges')} "
              f"n_loops={m.get('n_loop_closures')} n_plates={m.get('n_door_plates')} "
              f"keyframes_triggered={m.get('n_keyframes_triggered')} "
              f"merges={m.get('n_semantic_merges')}")
        print(f"    artifacts: {json.dumps(s.get('artifacts',{}), ensure_ascii=False, indent=6)}")
        print(f"    visuals: {s.get('visualizations')}")
    return resp


def run_nav_smoke(ws, data_dir: str):
    """Nav 模式冒烟: 只发几帧, 确认响应为 nav action shape"""
    print("\n=== NAV SMOKE ===")
    # 先 reset
    resp = send(ws, {"command": "reset"})
    print(f"  reset: {resp.get('status')}")

    tss = load_timestamps(data_dir)[:3]
    for i, ts in enumerate(tss):
        imgs = build_frame(data_dir, ts)
        req = {
            "id": "TEST_NAV",
            "pts": int(ts),
            "task": "去前台",
            "images": imgs,
        }
        resp = send(ws, req)
        act = resp.get('action')
        ts_s = resp.get('task_status')
        mode = resp.get('mode', 'nav')
        msg = resp.get('message', '')[:80]
        print(f"  [{i}] ts={ts} status={resp.get('status')} task_status={ts_s} "
              f"mode={mode} action_shape={'ok' if isinstance(act, list) else 'BAD'} "
              f"msg={msg}")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='memory_test_data')
    ap.add_argument('--max', type=int, default=0, help='限制帧数, 0=全部')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=9528)
    ap.add_argument('--mode', choices=['both', 'mapping', 'nav'], default='both')
    args = ap.parse_args()

    url = f"ws://{args.host}:{args.port}"
    print(f"connecting {url} ...")
    ws = websocket.create_connection(url, timeout=600)
    print("connected")

    try:
        if args.mode in ('both', 'mapping'):
            run_mapping(ws, args.data, args.max or None)
        if args.mode in ('both', 'nav'):
            run_nav_smoke(ws, args.data)
    finally:
        ws.close()
        print("closed")


if __name__ == "__main__":
    main()
