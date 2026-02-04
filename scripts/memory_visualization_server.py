#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 可视化测试服务器 v2.5

功能:
1. 启动前端测试网页（全中文界面）
2. 支持开启/关闭记忆功能
3. 实时显示拓扑图可视化
4. 支持指定测试连续帧目录路径（前置图 + 4张环视图）
5. 集成WebSocket连接进行连续帧导航推理
6. 记忆功能开启时创建关键帧拓扑图
7. Dijkstra最短路径规划
8. 节点合并功能
9. VPR图片上传识别起点（4张环视图）
10. 语义描述匹配检索终点
11. 智能路径规划（结合VPR+语义检索+Dijkstra）

使用方式:
    conda activate internvla
    python scripts/memory_visualization_server.py --port 9530

作者: Memory Navigation Team
日期: 2026-01-27
"""

import os
import sys
import json
import time
import base64
import logging
import argparse
import glob as glob_module
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import threading
import re

# 禁用localhost的代理
no_proxy = os.environ.get('no_proxy', os.environ.get('NO_PROXY', ''))
if 'localhost' not in no_proxy:
    localhost_list = 'localhost,127.0.0.1,::1'
    if no_proxy:
        os.environ['no_proxy'] = f"{no_proxy},{localhost_list}"
    else:
        os.environ['no_proxy'] = localhost_list

# 设置项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deploy"))

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from flask import Flask, render_template_string, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask 未安装。请执行: pip install flask flask-cors")

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("websocket-client 未安装。请执行: pip install websocket-client")

# 导入数据库模块
try:
    from deploy.memory_modules.database import TopologyDatabase, get_database
    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False
    print(f"数据库模块不可用: {e}")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# HTML模板 - 中文版本，支持连续帧测试
# =============================================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>记忆导航系统 - 可视化测试 v2.5</title>
    <script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Microsoft YaHei', 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container { max-width: 1800px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        h1 { color: #00d4ff; font-size: 2em; margin-bottom: 5px; }
        .subtitle { color: #888; font-size: 1em; }
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 480px;
            gap: 20px;
        }
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 15px;
        }
        .panel-title {
            color: #00d4ff;
            font-size: 1.2em;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid rgba(0,212,255,0.3);
        }
        #topology-graph {
            width: 100%;
            height: 450px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .control-group { margin-bottom: 15px; }
        .control-label { display: block; color: #aaa; margin-bottom: 6px; font-size: 0.9em; }
        .switch-container { display: flex; align-items: center; gap: 12px; }
        .switch { position: relative; width: 50px; height: 26px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider {
            position: absolute; cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: #444; transition: 0.4s; border-radius: 26px;
        }
        .slider:before {
            position: absolute; content: "";
            height: 20px; width: 20px; left: 3px; bottom: 3px;
            background-color: white; transition: 0.4s; border-radius: 50%;
        }
        input:checked + .slider { background-color: #00d4ff; }
        input:checked + .slider:before { transform: translateX(24px); }
        .status-text { font-size: 0.85em; }
        .status-on { color: #00ff88; }
        .status-off { color: #ff6b6b; }
        .btn {
            padding: 8px 14px; border: none; border-radius: 6px;
            cursor: pointer; font-size: 0.9em; transition: all 0.3s; margin: 2px;
        }
        .btn-primary { background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%); color: white; }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 3px 15px rgba(0,212,255,0.4); }
        .btn-danger { background: linear-gradient(135deg, #ff6b6b 0%, #cc4444 100%); color: white; }
        .btn-success { background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%); color: white; }
        .btn-warning { background: linear-gradient(135deg, #ffcc00 0%, #ff9900 100%); color: #333; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
        .stat-card { background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 1.5em; color: #00d4ff; font-weight: bold; }
        .stat-label { color: #888; font-size: 0.75em; margin-top: 2px; }
        .log-container {
            max-height: 150px; overflow-y: auto;
            background: rgba(0,0,0,0.3); border-radius: 8px;
            padding: 10px; font-family: monospace; font-size: 0.8em;
        }
        .log-entry { padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .log-time { color: #00d4ff; }
        .log-info { color: #00ff88; }
        .log-warn { color: #ffcc00; }
        .log-error { color: #ff6b6b; }
        .legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; font-size: 0.8em; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
        .legend-dot.keyframe { background: #00ff88; }
        .legend-dot.normal { background: #00d4ff; }
        .legend-dot.current { background: #ffcc00; }
        .legend-dot.target { background: #ff6b6b; }
        /* 布局控制按钮组 */
        .layout-controls { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
        .layout-controls .btn { font-size: 0.85em; padding: 6px 12px; }
        input[type="number"], input[type="text"], select {
            background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.2);
            color: white; padding: 8px; border-radius: 6px; width: 100%;
        }
        .input-group { display: flex; gap: 8px; margin-bottom: 8px; }
        .input-group input { flex: 1; }
        .node-detail {
            background: rgba(0,0,0,0.3); border-radius: 8px;
            padding: 12px; margin-top: 10px; display: none;
            max-height: 400px; overflow-y: auto;
        }
        .node-detail.show { display: block; }
        .detail-section { margin-bottom: 12px; }
        .detail-section h4 { color: #00d4ff; font-size: 12px; margin-bottom: 6px; border-bottom: 1px solid rgba(0,212,255,0.3); padding-bottom: 4px; }
        .detail-section p { margin: 4px 0; font-size: 12px; }
        .detail-section strong { color: #aaa; }
        .label-tags { display: flex; flex-wrap: wrap; gap: 4px; }
        .label-tag { background: rgba(0,212,255,0.2); color: #00d4ff; padding: 2px 8px; border-radius: 12px; font-size: 11px; border: 1px solid rgba(0,212,255,0.3); }
        .scene-desc { color: #ccc; font-size: 11px; line-height: 1.5; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px; }
        /* 帧导航控制 */
        .frame-nav {
            display: flex; align-items: center; gap: 8px;
            margin: 10px 0; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px;
        }
        .frame-nav input[type="range"] {
            flex: 1; height: 6px; -webkit-appearance: none;
            background: rgba(0,212,255,0.3); border-radius: 3px;
        }
        .frame-nav input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none; width: 14px; height: 14px;
            background: #00d4ff; border-radius: 50%; cursor: pointer;
        }
        .frame-info { min-width: 80px; text-align: center; color: #00d4ff; font-size: 0.9em; }
        /* 进度条 */
        .progress-bar {
            width: 100%; height: 8px; background: rgba(255,255,255,0.1);
            border-radius: 4px; overflow: hidden; margin: 8px 0;
        }
        .progress-bar-fill { height: 100%; background: linear-gradient(90deg, #00d4ff, #00ff88); transition: width 0.3s; }
        /* 目录输入 */
        .dir-input-group { display: flex; gap: 8px; margin-bottom: 12px; }
        .dir-input-group input { flex: 1; }
        /* 图片预览 */
        .image-preview-container { margin: 10px 0; }
        .current-frame-img {
            width: 100%; max-height: 200px; object-fit: contain;
            border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);
            background: rgba(0,0,0,0.3);
        }
        /* 推理结果 */
        .inference-result {
            background: rgba(0,0,0,0.3); border-radius: 8px;
            padding: 12px; margin-top: 10px;
        }
        .action-display {
            display: flex; gap: 5px; flex-wrap: wrap; margin-top: 8px;
        }
        .action-item {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: white; padding: 4px 10px; border-radius: 4px; font-size: 0.85em;
        }
        .action-item.stop { background: linear-gradient(135deg, #ff6b6b 0%, #cc4444 100%); }
        /* 路径结果 */
        .path-result { background: rgba(0,0,0,0.3); border-radius: 8px; padding: 12px; margin-top: 10px; }
        .path-nodes { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
        .path-node {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.85em;
        }
        .path-node.start { background: linear-gradient(135deg, #ffcc00 0%, #ff9900 100%); color: #333; }
        .path-node.end { background: linear-gradient(135deg, #ff6b6b 0%, #cc4444 100%); }
        /* WebSocket状态 */
        .ws-status { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
        .ws-indicator { width: 10px; height: 10px; border-radius: 50%; }
        .ws-indicator.connected { background: #00ff88; }
        .ws-indicator.disconnected { background: #ff6b6b; }
        .ws-indicator.connecting { background: #ffcc00; animation: blink 1s infinite; }
        @keyframes blink { 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>记忆导航系统</h1>
            <p class="subtitle">v2.5 - 连续帧导航推理 | 美观拓扑图 | 智能布局 | Dijkstra路径规划</p>
        </header>

        <div class="main-grid">
            <!-- 左侧：拓扑图可视化 -->
            <div>
                <div class="panel">
                    <h2 class="panel-title">拓扑图可视化</h2>
                    <div id="topology-graph"></div>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-dot keyframe"></div><span>关键帧</span></div>
                        <div class="legend-item"><div class="legend-dot normal"></div><span>普通节点</span></div>
                        <div class="legend-item"><div class="legend-dot current"></div><span>当前位置</span></div>
                        <div class="legend-item"><div class="legend-dot target"></div><span>目标节点</span></div>
                    </div>
                    <!-- v2.5: 布局控制按钮 -->
                    <div class="layout-controls">
                        <button class="btn btn-primary" onclick="resetLayout('hierarchical')">🏛️ 层次布局</button>
                        <button class="btn btn-primary" onclick="resetLayout('force')">🔄 力导向布局</button>
                        <button class="btn btn-primary" onclick="resetLayout('circular')">⭕ 环形布局</button>
                        <button class="btn btn-warning" onclick="resetLayout('optimal')">✨ 一键优化</button>
                        <button class="btn btn-danger" onclick="network && network.fit()">📍 适应视图</button>
                    </div>
                    <div class="node-detail" id="node-detail">
                        <h4 style="color: #00d4ff; margin-bottom: 8px;">节点详情</h4>
                        <div id="node-detail-content"></div>
                    </div>
                </div>

                <!-- 系统日志 -->
                <div class="panel">
                    <h2 class="panel-title">系统日志</h2>
                    <div class="log-container" id="log-container"></div>
                </div>
            </div>

            <!-- 右侧：控制面板 -->
            <div>
                <!-- WebSocket连接状态 -->
                <div class="panel">
                    <h2 class="panel-title">推理服务状态</h2>
                    <div class="ws-status">
                        <div class="ws-indicator disconnected" id="ws-indicator"></div>
                        <span id="ws-status-text">检查中...</span>
                    </div>
                    <div style="color: #888; font-size: 0.85em; margin-top: 8px;">
                        <span id="ws-url-display">ws://localhost:9528</span>
                        <br><span style="color: #666;">（服务启动时自动连接）</span>
                    </div>
                </div>

                <!-- 记忆控制 -->
                <div class="panel">
                    <h2 class="panel-title">记忆控制</h2>
                    <div class="control-group">
                        <label class="control-label">记忆功能开关</label>
                        <div class="switch-container">
                            <label class="switch">
                                <input type="checkbox" id="memory-toggle" checked>
                                <span class="slider"></span>
                            </label>
                            <span id="memory-status" class="status-text status-on">已开启</span>
                        </div>
                    </div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="stat-nodes">0</div>
                            <div class="stat-label">节点</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-edges">0</div>
                            <div class="stat-label">边</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-keyframes">0</div>
                            <div class="stat-label">关键帧</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-current">-</div>
                            <div class="stat-label">当前</div>
                        </div>
                    </div>
                    <div class="color-legend" style="margin-top: 10px; padding: 8px; background: #2d2d2d; border-radius: 6px;">
                        <div style="font-size: 12px; color: #888; margin-bottom: 6px;">节点颜色说明：</div>
                        <div style="display: flex; flex-wrap: wrap; gap: 8px; font-size: 11px;">
                            <span style="display: flex; align-items: center; gap: 4px;">
                                <span style="width: 12px; height: 12px; border-radius: 50%; background: #00d4ff;"></span>
                                <span style="color: #aaa;">普通节点</span>
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px;">
                                <span style="width: 12px; height: 12px; border-radius: 50%; background: #00ff88;"></span>
                                <span style="color: #aaa;">关键帧</span>
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px;">
                                <span style="width: 12px; height: 12px; border-radius: 50%; background: #ffcc00;"></span>
                                <span style="color: #aaa;">当前位置</span>
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px;">
                                <span style="width: 12px; height: 12px; border-radius: 50%; background: #ff8800;"></span>
                                <span style="color: #aaa;">路径节点</span>
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px;">
                                <span style="width: 12px; height: 12px; border-radius: 50%; background: #ff6b6b;"></span>
                                <span style="color: #aaa;">目标终点</span>
                            </span>
                        </div>
                    </div>
                    <div style="margin-top: 8px;">
                        <button class="btn btn-primary" id="btn-refresh">刷新</button>
                        <button class="btn btn-danger" id="btn-clear">清空记忆</button>
                    </div>
                </div>

                <!-- 数据库管理面板 v3.0 -->
                <div class="panel">
                    <h2 class="panel-title">🗄️ 数据库管理 (PostgreSQL)</h2>

                    <!-- 数据库统计 -->
                    <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #00d4ff;">数据库状态</span>
                            <span id="db-status" style="color: #00ff88;">● 已连接</span>
                        </div>
                        <div style="display: flex; gap: 20px; margin-top: 8px; font-size: 0.9em;">
                            <span>节点: <strong id="db-node-count">0</strong></span>
                            <span>边: <strong id="db-edge-count">0</strong></span>
                        </div>
                    </div>

                    <!-- 节点操作 -->
                    <div style="margin-bottom: 12px;">
                        <label style="display: block; margin-bottom: 6px; font-size: 0.9em; color: #aaa;">📍 节点操作</label>
                        <div class="input-group">
                            <input type="number" id="db-node-id" placeholder="节点ID" style="width: 30%;">
                            <input type="text" id="db-node-name" placeholder="节点名称" style="width: 70%;">
                        </div>
                        <div class="input-group">
                            <input type="text" id="db-node-desc" placeholder="场景描述（可选）" style="width: 100%;">
                        </div>
                        <div class="input-group">
                            <input type="text" id="db-node-labels" placeholder="语义标签（逗号分隔）" style="width: 100%;">
                        </div>
                        <div style="display: flex; gap: 4px; margin-top: 8px;">
                            <button class="btn btn-success" id="btn-db-create-node" style="flex: 1;">创建</button>
                            <button class="btn btn-primary" id="btn-db-read-node" style="flex: 1;">查询</button>
                            <button class="btn btn-warning" id="btn-db-update-node" style="flex: 1;">更新</button>
                            <button class="btn btn-danger" id="btn-db-delete-node" style="flex: 1;">删除</button>
                        </div>
                    </div>

                    <!-- 边操作 -->
                    <div style="margin-bottom: 12px;">
                        <label style="display: block; margin-bottom: 6px; font-size: 0.9em; color: #aaa;">🔗 边操作</label>
                        <div class="input-group">
                            <input type="number" id="db-edge-source" placeholder="源节点ID" style="width: 50%;">
                            <input type="number" id="db-edge-target" placeholder="目标节点ID" style="width: 50%;">
                        </div>
                        <div style="display: flex; gap: 4px; margin-top: 8px;">
                            <button class="btn btn-success" id="btn-db-create-edge" style="flex: 1;">创建边</button>
                            <button class="btn btn-danger" id="btn-db-delete-edge" style="flex: 1;">删除边</button>
                        </div>
                    </div>

                    <!-- 导入导出 -->
                    <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px;">
                        <label style="display: block; margin-bottom: 6px; font-size: 0.9em; color: #aaa;">📦 数据导入/导出</label>
                        <div style="display: flex; gap: 4px;">
                            <button class="btn btn-primary" id="btn-db-import" style="flex: 1;">从JSON导入</button>
                            <button class="btn btn-primary" id="btn-db-export" style="flex: 1;">导出到JSON</button>
                        </div>
                    </div>

                    <!-- 操作结果显示 -->
                    <div id="db-operation-result" style="display: none; margin-top: 12px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 6px; font-size: 0.85em;">
                        <span id="db-result-text"></span>
                    </div>
                </div>

                <!-- 连续帧测试 -->
                <div class="panel">
                    <h2 class="panel-title">连续帧导航测试</h2>
                    <div class="dir-input-group">
                        <input type="text" id="test-dir-path" placeholder="输入测试数据目录路径...">
                        <button class="btn btn-primary" id="btn-load-dir">加载</button>
                    </div>
                    <div class="input-group">
                        <input type="text" id="nav-instruction" placeholder="导航指令（可选，留空则读取instruction.txt）">
                    </div>
                    <div id="dir-info" style="color: #888; font-size: 0.85em; margin-bottom: 10px;"></div>

                    <!-- 当前帧图片预览 -->
                    <div class="image-preview-container" id="image-preview-container" style="display: none;">
                        <img id="current-frame-img" class="current-frame-img" src="" alt="当前帧">
                    </div>

                    <!-- 帧导航 -->
                    <div class="frame-nav" id="frame-nav" style="display: none;">
                        <button class="btn btn-primary" id="btn-prev-frame">&lt;</button>
                        <input type="range" id="frame-slider" min="0" max="0" value="0">
                        <button class="btn btn-primary" id="btn-next-frame">&gt;</button>
                        <div class="frame-info">
                            <span id="frame-current">0</span> / <span id="frame-total">0</span>
                        </div>
                    </div>

                    <!-- 处理按钮 -->
                    <div style="margin-top: 10px;">
                        <button class="btn btn-success" id="btn-process-frame" style="width: 48%;">处理当前帧</button>
                        <button class="btn btn-warning" id="btn-batch-process" style="width: 48%;">批量推理</button>
                    </div>
                    <!-- v2.5.2: 多场景记忆合并选项 -->
                    <div style="margin-top: 8px; display: flex; align-items: center; gap: 12px;">
                        <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                            <input type="checkbox" id="keep-memory-checkbox" checked style="width: 16px; height: 16px;">
                            <span style="color: #00ff88; font-size: 0.9em;">🧠 保留记忆（多场景合并）</span>
                        </label>
                    </div>
                    <div style="margin-top: 8px;">
                        <button class="btn btn-danger" id="btn-reset-agent" style="width: 100%;">重置Agent</button>
                    </div>

                    <!-- 进度条 -->
                    <div id="batch-progress" style="display: none;">
                        <div class="progress-bar">
                            <div class="progress-bar-fill" id="progress-fill" style="width: 0%;"></div>
                        </div>
                        <div style="text-align: center; font-size: 0.8em; color: #888;">
                            <span id="progress-text">处理中...</span>
                        </div>
                    </div>

                    <!-- 推理结果 -->
                    <div class="inference-result" id="inference-result" style="display: none;">
                        <strong>推理结果:</strong>
                        <div id="inference-result-content"></div>
                    </div>
                </div>

                <!-- 最短路径规划 -->
                <div class="panel">
                    <h2 class="panel-title">最短路径规划</h2>
                    <div class="input-group">
                        <input type="number" id="start-node" placeholder="起始节点">
                        <input type="number" id="target-node" placeholder="目标节点">
                    </div>
                    <button class="btn btn-primary" id="btn-plan-path" style="width: 100%;">查找最短路径</button>
                    <div class="path-result" id="path-result" style="display: none;">
                        <strong>路径:</strong>
                        <div class="path-nodes" id="path-nodes"></div>
                        <p style="margin-top: 8px; color: #888; font-size: 0.85em;">
                            距离: <span id="path-distance">-</span> | 步数: <span id="path-steps">-</span>
                        </p>
                    </div>
                </div>

                <!-- 智能路径规划（VPR+语义） -->
                <div class="panel">
                    <h2 class="panel-title">🧠 智能路径规划</h2>
                    <p style="color: #888; font-size: 0.8em; margin-bottom: 10px;">上传环视图片识别起点，输入目标描述检索终点</p>

                    <!-- VPR图片上传 -->
                    <div style="margin-bottom: 12px;">
                        <label style="display: block; margin-bottom: 6px; font-size: 0.9em; color: #aaa;">📷 上传4张环视图片（识别起点）:</label>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
                            <div class="upload-item">
                                <label for="vpr-cam1" style="font-size: 0.75em; color: #666;">camera_1 (前)</label>
                                <input type="file" id="vpr-cam1" accept="image/*" style="font-size: 0.8em; width: 100%;">
                            </div>
                            <div class="upload-item">
                                <label for="vpr-cam2" style="font-size: 0.75em; color: #666;">camera_2 (右)</label>
                                <input type="file" id="vpr-cam2" accept="image/*" style="font-size: 0.8em; width: 100%;">
                            </div>
                            <div class="upload-item">
                                <label for="vpr-cam3" style="font-size: 0.75em; color: #666;">camera_3 (后)</label>
                                <input type="file" id="vpr-cam3" accept="image/*" style="font-size: 0.8em; width: 100%;">
                            </div>
                            <div class="upload-item">
                                <label for="vpr-cam4" style="font-size: 0.75em; color: #666;">camera_4 (左)</label>
                                <input type="file" id="vpr-cam4" accept="image/*" style="font-size: 0.8em; width: 100%;">
                            </div>
                        </div>
                        <div id="vpr-preview" style="display: none; margin-top: 8px; text-align: center;">
                            <span style="color: #00d4ff; font-size: 0.85em;">已选择 <span id="vpr-count">0</span> 张图片</span>
                        </div>
                    </div>

                    <!-- 语义目标描述 -->
                    <div style="margin-bottom: 12px;">
                        <label style="display: block; margin-bottom: 6px; font-size: 0.9em; color: #aaa;">🎯 目标描述（检索终点）:</label>
                        <input type="text" id="goal-query" placeholder="例如：前台、沙发、门口..." style="width: 100%; padding: 8px; background: #2d2d4a; border: 1px solid #444; color: #fff; border-radius: 4px;">
                    </div>

                    <!-- 或者手动指定节点 -->
                    <div style="margin-bottom: 12px;">
                        <label style="display: block; margin-bottom: 6px; font-size: 0.8em; color: #666;">或手动指定节点:</label>
                        <div class="input-group">
                            <input type="number" id="smart-start-node" placeholder="起点节点（可选）" style="width: 48%;">
                            <input type="number" id="smart-goal-node" placeholder="终点节点（可选）" style="width: 48%;">
                        </div>
                    </div>

                    <button class="btn btn-success" id="btn-smart-plan" style="width: 100%; margin-bottom: 8px;">🔍 智能路径规划</button>
                    <button class="btn btn-primary" id="btn-vpr-only" style="width: 48%;">VPR识别起点</button>
                    <button class="btn btn-primary" id="btn-semantic-only" style="width: 48%;">语义检索终点</button>

                    <!-- 智能规划结果 -->
                    <div id="smart-path-result" style="display: none; margin-top: 12px; padding: 10px; background: #1a1a2e; border-radius: 6px;">
                        <div id="smart-start-info" style="margin-bottom: 8px;"></div>
                        <div id="smart-goal-info" style="margin-bottom: 8px;"></div>
                        <strong>规划路径:</strong>
                        <div class="path-nodes" id="smart-path-nodes" style="margin-top: 6px;"></div>
                        <p style="margin-top: 8px; color: #888; font-size: 0.85em;">
                            总距离: <span id="smart-path-distance">-</span> | 经过节点: <span id="smart-path-steps">-</span>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 全局变量
        let network = null;
        let nodes = new vis.DataSet([]);
        let edges = new vis.DataSet([]);
        let currentPath = [];
        let testFrames = [];
        let currentFrameIndex = 0;
        let memoryEnabled = true;

        // 辅助函数
        function setTextContent(id, text) {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        }

        function addLog(type, message) {
            const container = document.getElementById('log-container');
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = '<span class="log-time">[' + time + ']</span> <span class="log-' + type + '">' + message + '</span>';
            container.insertBefore(entry, container.firstChild);
            if (container.children.length > 100) container.removeChild(container.lastChild);
        }

        // v2.5: 美观节点样式配置 (参考美团知识图谱样式)
        const nodeColors = {
            normal: {
                background: 'linear-gradient(135deg, #00d4ff 0%, #0099cc 100%)',
                border: '#0077aa',
                highlight: { background: '#00eeff', border: '#00bbdd' },
                hover: { background: '#00eeff', border: '#00bbdd' }
            },
            keyframe: {
                background: 'linear-gradient(135deg, #00ff88 0%, #00cc66 100%)',
                border: '#00aa55',
                highlight: { background: '#33ffaa', border: '#00dd77' },
                hover: { background: '#33ffaa', border: '#00dd77' }
            },
            current: {
                background: 'linear-gradient(135deg, #ffcc00 0%, #ff9900 100%)',
                border: '#cc7700',
                highlight: { background: '#ffdd33', border: '#ffaa00' },
                hover: { background: '#ffdd33', border: '#ffaa00' }
            },
            path: {
                background: 'linear-gradient(135deg, #ff8800 0%, #cc5500 100%)',
                border: '#aa4400',
                highlight: { background: '#ffaa33', border: '#dd6600' },
                hover: { background: '#ffaa33', border: '#dd6600' }
            }
        };

        // 初始化网络图
        function initNetwork() {
            const container = document.getElementById('topology-graph');
            const data = { nodes: nodes, edges: edges };
            // v2.5: 优化的节点和边样式
            const options = {
                nodes: {
                    shape: 'dot',
                    size: 22,
                    font: {
                        color: '#ffffff',
                        size: 12,
                        face: 'Microsoft YaHei, Arial',
                        strokeWidth: 3,
                        strokeColor: 'rgba(0,0,0,0.7)'
                    },
                    borderWidth: 3,
                    borderWidthSelected: 5,
                    shadow: {
                        enabled: true,
                        color: 'rgba(0,0,0,0.5)',
                        size: 10,
                        x: 3,
                        y: 3
                    },
                    color: {
                        background: '#00d4ff',
                        border: '#0099cc',
                        highlight: { background: '#00eeff', border: '#00bbdd' },
                        hover: { background: '#00eeff', border: '#00bbdd' }
                    }
                },
                edges: {
                    color: {
                        color: 'rgba(100,150,200,0.6)',
                        highlight: '#00d4ff',
                        hover: '#00d4ff'
                    },
                    smooth: {
                        enabled: true,
                        type: 'continuous',
                        roundness: 0.3
                    },
                    width: 2,
                    hoverWidth: 3,
                    selectionWidth: 4,
                    shadow: {
                        enabled: true,
                        color: 'rgba(0,0,0,0.3)',
                        size: 5
                    },
                    arrows: {
                        to: { enabled: false }
                    }
                },
                physics: {
                    enabled: true,
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {
                        gravitationalConstant: -80,
                        centralGravity: 0.015,
                        springConstant: 0.08,
                        springLength: 120,
                        damping: 0.4,
                        avoidOverlap: 0.8
                    },
                    stabilization: {
                        enabled: true,
                        iterations: 200,
                        updateInterval: 25,
                        fit: true
                    },
                    maxVelocity: 50,
                    minVelocity: 0.1
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 150,
                    dragNodes: true,
                    dragView: true,
                    zoomView: true,
                    navigationButtons: false,
                    keyboard: { enabled: true }
                },
                layout: {
                    improvedLayout: true,
                    clusterThreshold: 150
                }
            };
            network = new vis.Network(container, data, options);

            // 稳定后禁用物理引擎，防止持续跳动
            network.on('stabilizationIterationsDone', function() {
                network.setOptions({ physics: { enabled: false } });
                network.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
                addLog('info', '布局稳定完成');
            });

            network.on('click', function(params) {
                if (params.nodes.length > 0) {
                    showNodeDetail(params.nodes[0]);
                    addLog('info', '选中节点: ' + params.nodes[0]);
                }
            });

            // v2.5.1: 拖动节点时不启用物理引擎，只移动被拖动的节点
            // 这样可以防止点击或拖动时布局被打乱
            var isDragging = false;
            var draggedNodeId = null;

            network.on('dragStart', function(params) {
                if (params.nodes.length > 0) {
                    isDragging = true;
                    draggedNodeId = params.nodes[0];
                    // 不启用物理引擎，保持其他节点位置不变
                }
            });

            network.on('dragging', function(params) {
                // 拖动过程中节点位置由vis.js自动处理，无需额外操作
            });

            network.on('dragEnd', function(params) {
                isDragging = false;
                draggedNodeId = null;
                // 不需要做任何操作，节点保持在拖放位置
            });
        }

        // v2.5: 布局重置函数
        function resetLayout(layoutType) {
            if (!network) {
                addLog('warn', '网络图未初始化');
                return;
            }

            addLog('info', '正在应用 ' + layoutType + ' 布局...');

            // 先启用物理引擎
            network.setOptions({ physics: { enabled: true } });

            switch(layoutType) {
                case 'hierarchical':
                    // 层次布局 - 适合有明显层级关系的图
                    network.setOptions({
                        layout: {
                            hierarchical: {
                                enabled: true,
                                direction: 'UD',  // Up-Down
                                sortMethod: 'hubsize',
                                nodeSpacing: 150,
                                treeSpacing: 200,
                                levelSeparation: 150,
                                blockShifting: true,
                                edgeMinimization: true,
                                parentCentralization: true
                            }
                        },
                        physics: { enabled: false }
                    });
                    setTimeout(function() {
                        network.setOptions({ layout: { hierarchical: { enabled: false } } });
                        network.fit({ animation: { duration: 500 } });
                        addLog('info', '层次布局完成');
                    }, 500);
                    break;

                case 'force':
                    // 力导向布局 - 通用布局，节点间相互排斥
                    network.setOptions({
                        layout: { hierarchical: { enabled: false } },
                        physics: {
                            enabled: true,
                            solver: 'forceAtlas2Based',
                            forceAtlas2Based: {
                                gravitationalConstant: -100,
                                centralGravity: 0.02,
                                springConstant: 0.1,
                                springLength: 100,
                                damping: 0.4,
                                avoidOverlap: 0.9
                            },
                            stabilization: { enabled: true, iterations: 300 }
                        }
                    });
                    // v2.5.1: 确保稳定后禁用物理引擎
                    setTimeout(function() {
                        network.setOptions({ physics: { enabled: false } });
                        network.fit({ animation: { duration: 300 } });
                        addLog('info', '力导向布局完成');
                    }, 3000);
                    break;

                case 'circular':
                    // 环形布局 - 手动计算节点位置
                    var allNodes = nodes.get();
                    var nodeCount = allNodes.length;
                    if (nodeCount === 0) {
                        addLog('warn', '没有节点可布局');
                        return;
                    }
                    var radius = Math.max(200, nodeCount * 25);
                    var angleStep = (2 * Math.PI) / nodeCount;
                    var updates = [];
                    allNodes.forEach(function(node, index) {
                        var angle = index * angleStep - Math.PI / 2;
                        updates.push({
                            id: node.id,
                            x: radius * Math.cos(angle),
                            y: radius * Math.sin(angle)
                        });
                    });
                    nodes.update(updates);
                    network.setOptions({
                        layout: { hierarchical: { enabled: false } },
                        physics: { enabled: false }
                    });
                    network.fit({ animation: { duration: 500 } });
                    addLog('info', '环形布局完成');
                    break;

                case 'optimal':
                    // 一键优化 - 先用层次布局确定大致位置，再用力导向微调
                    addLog('info', '一键优化：第一阶段 - 层次预布局...');
                    network.setOptions({
                        layout: {
                            hierarchical: {
                                enabled: true,
                                direction: 'LR',  // Left-Right更美观
                                sortMethod: 'directed',
                                nodeSpacing: 180,
                                treeSpacing: 250,
                                levelSeparation: 200
                            }
                        },
                        physics: { enabled: false }
                    });
                    setTimeout(function() {
                        // 第二阶段：禁用层次布局，启用力导向微调
                        addLog('info', '一键优化：第二阶段 - 力导向微调...');
                        network.setOptions({
                            layout: { hierarchical: { enabled: false } },
                            physics: {
                                enabled: true,
                                solver: 'forceAtlas2Based',
                                forceAtlas2Based: {
                                    gravitationalConstant: -60,
                                    centralGravity: 0.01,
                                    springConstant: 0.05,
                                    springLength: 150,
                                    damping: 0.5,
                                    avoidOverlap: 0.95
                                },
                                stabilization: { enabled: true, iterations: 150 }
                            }
                        });
                        // v2.5.1: 确保稳定后禁用物理引擎，防止点击后布局变乱
                        setTimeout(function() {
                            network.setOptions({ physics: { enabled: false } });
                            network.fit({ animation: { duration: 300 } });
                            addLog('info', '一键优化完成，布局已锁定');
                        }, 2000);
                    }, 800);
                    break;

                default:
                    addLog('warn', '未知布局类型: ' + layoutType);
            }
        }

        function showNodeDetail(nodeId) {
            // 首先尝试从本地缓存的数据中获取节点信息
            var nodeData = null;
            if (lastGraphData && lastGraphData.nodes) {
                nodeData = lastGraphData.nodes.find(n => n.id === nodeId);
            }

            // 如果本地有数据，直接显示
            if (nodeData) {
                displayNodeDetail(nodeData);
                // v2.1: 如果本地缓存已有完整数据（source_timestamps和front_view_embedding），
                // 不再从服务器获取（服务器API不返回这些字段会覆盖掉本地数据）
                if (nodeData.source_timestamps !== undefined || nodeData.front_view_embedding !== undefined) {
                    return;  // 使用本地缓存的完整数据，不再请求服务器
                }
            }

            // 只有在本地没有完整数据时，才从服务器获取
            fetch('/api/node/' + nodeId)
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.node) {
                        displayNodeDetail(data.node);
                    }
                });
        }

        function displayNodeDetail(n) {
            const detail = document.getElementById('node-detail');
            const content = document.getElementById('node-detail-content');

            // 格式化语义标签
            var labelsHtml = '-';
            if (n.semantic_labels && n.semantic_labels.length > 0) {
                labelsHtml = '<div class="label-tags">' +
                    n.semantic_labels.map(l => '<span class="label-tag">' + l + '</span>').join('') +
                    '</div>';
            }

            // 格式化pixel_target
            var pixelTargetHtml = '无';
            if (n.pixel_target && n.pixel_target.length > 0) {
                pixelTargetHtml = '[' + n.pixel_target.map(v => v.toFixed(3)).join(', ') + ']';
            }

            // 格式化前视特征嵌入信息 - v2.3 直接显示512维向量
            var frontViewEmbeddingHtml = '无';
            if (n.front_view_embedding && Array.isArray(n.front_view_embedding)) {
                var emb = n.front_view_embedding;
                // 直接显示完整的512维向量（格式化为4位小数）
                var vectorStr = '[' + emb.map(v => v.toFixed(4)).join(', ') + ']';
                frontViewEmbeddingHtml = '<div style="background:#1a1a2e;padding:8px;border-radius:4px;font-size:10px;margin-top:5px;max-height:200px;overflow-y:auto;">' +
                    '<p style="margin:2px 0;"><strong>维度:</strong> ' + emb.length + '</p>' +
                    '<p style="margin:2px 0;word-break:break-all;line-height:1.4;">' + vectorStr + '</p>' +
                    '</div>';
            } else if (n.has_front_view_feature) {
                frontViewEmbeddingHtml = '<span style="color:#00ff88">有（详情未加载）</span>';
            }

            // 格式化时间戳
            var timestampHtml = '-';
            if (n.created_at || n.timestamp) {
                var ts = n.created_at || n.timestamp;
                var date = new Date(ts * 1000);
                timestampHtml = date.toLocaleString('zh-CN');
            }

            // v2.1: 构建节点来源HTML（使用安全的字符串处理）
            var sourceTimestampsHtml = '暂无来源记录';
            if (n.source_timestamps && n.source_timestamps.length > 0) {
                var sourceItems = [];
                n.source_timestamps.forEach(function(src, idx) {
                    var mergedDate = src.merged_at ? new Date(src.merged_at * 1000).toLocaleString('zh-CN') : '-';
                    var isInitialText = src.is_initial ? '(初始)' : '(合并)';
                    var isInitialColor = src.is_initial ? '#00ff88' : '#ffcc00';
                    var tsValue = String(src.timestamp || '').replace(/[<>&"']/g, '');  // 基本转义
                    var cameraValue = String(src.camera || 'front_1').replace(/[<>&"']/g, '');
                    sourceItems.push(
                        '<div style="padding:4px 0;border-bottom:1px solid #333;">' +
                        '<p><strong>' + (idx + 1) + '. 时间戳:</strong> <code style="color:#00d4ff">' + tsValue + '</code> <span style="color:' + isInitialColor + '">' + isInitialText + '</span></p>' +
                        '<p style="font-size:11px;color:#888"><strong>相机:</strong> ' + cameraValue + ' | <strong>记录时间:</strong> ' + mergedDate + '</p>' +
                        '</div>'
                    );
                });
                sourceTimestampsHtml = '<div style="max-height:150px;overflow-y:auto;">' + sourceItems.join('') + '</div>';
            }

            // 构建详情HTML - v2.1增强版（含前视特征嵌入和节点来源）
            content.innerHTML =
                '<div class="detail-section">' +
                '<h4>📍 基本信息</h4>' +
                '<p><strong>节点ID:</strong> ' + n.id + '</p>' +
                (n.node_name ? '<p><strong>节点名称:</strong> <span style="color:#00ff88">' + n.node_name + '</span></p>' : '') +
                '<p><strong>类型:</strong> ' + (n.is_keyframe ? '<span style="color:#00ff88">关键帧</span>' : '普通节点') + '</p>' +
                '<p><strong>访问次数:</strong> ' + (n.visit_count || 1) + '</p>' +
                '<p><strong>当前位置:</strong> ' + (n.is_current ? '<span style="color:#ffcc00">是</span>' : '否') + '</p>' +
                '<p><strong>创建时间:</strong> ' + timestampHtml + '</p>' +
                '</div>' +
                '<div class="detail-section">' +
                '<h4>📁 节点来源</h4>' +
                '<p><strong>来源数量:</strong> ' + (n.source_timestamps ? n.source_timestamps.length : 0) + '</p>' +
                sourceTimestampsHtml +
                '</div>' +
                '<div class="detail-section">' +
                '<h4>🏷️ 语义标签</h4>' +
                labelsHtml +
                '</div>' +
                '<div class="detail-section">' +
                '<h4>📝 场景描述</h4>' +
                '<p class="scene-desc">' + (n.scene_description || '暂无描述') + '</p>' +
                '</div>' +
                '<div class="detail-section">' +
                '<h4>🧭 导航信息</h4>' +
                '<p><strong>导航指令:</strong> ' + (n.navigation_instruction || n.instruction_context || '无') + '</p>' +
                '<p><strong>像素目标:</strong> ' + pixelTargetHtml + '</p>' +
                '</div>' +
                '<div class="detail-section">' +
                '<h4>🔮 前视特征嵌入</h4>' +
                frontViewEmbeddingHtml +
                '</div>';

            detail.classList.add('show');
        }

        // 刷新图形
        async function refreshGraph(forceRefresh = false) {
            try {
                const response = await fetch('/api/graph');
                const data = await response.json();
                if (data.success) {
                    updateGraph(data.data, forceRefresh);
                }
            } catch (error) {
                // v2.5.2: 静默处理fetch失败，避免日志刷屏
                console.debug('refreshGraph fetch error:', error.message);
            }
        }

        // 记录上一次的节点和边数量，用于判断是否需要重新布局
        var lastNodeCount = 0;
        var lastEdgeCount = 0;
        var lastGraphData = null;

        function updateGraph(graphData, forceRefresh = false) {
            // 检查是否有新节点或边加入
            var nodeCount = graphData.nodes.length;
            var edgeCount = graphData.edges.length;
            var needsLayout = forceRefresh || (nodeCount !== lastNodeCount) || (edgeCount !== lastEdgeCount);

            // 保存当前数据供节点详情使用
            lastGraphData = graphData;

            // v2.5: 获取节点美观样式
            function getNodeStyle(n, isInPath) {
                let colorConfig;
                let size = 22;
                let borderWidth = 3;

                // 颜色优先级：当前位置 > 路径节点 > 关键帧 > 普通节点
                if (n.is_current) {
                    colorConfig = {
                        background: '#ffcc00',
                        border: '#cc9900',
                        highlight: { background: '#ffdd33', border: '#ddaa00' },
                        hover: { background: '#ffdd33', border: '#ddaa00' }
                    };
                    size = 28;  // 当前位置更大
                    borderWidth = 5;
                } else if (isInPath) {
                    colorConfig = {
                        background: '#ff8800',
                        border: '#cc5500',
                        highlight: { background: '#ffaa33', border: '#dd6600' },
                        hover: { background: '#ffaa33', border: '#dd6600' }
                    };
                    borderWidth = 4;
                } else if (n.is_keyframe) {
                    colorConfig = {
                        background: '#00ff88',
                        border: '#00aa55',
                        highlight: { background: '#33ffaa', border: '#00dd77' },
                        hover: { background: '#33ffaa', border: '#00dd77' }
                    };
                    size = 24;  // 关键帧稍大
                } else {
                    colorConfig = {
                        background: '#00d4ff',
                        border: '#0099cc',
                        highlight: { background: '#00eeff', border: '#00bbdd' },
                        hover: { background: '#00eeff', border: '#00bbdd' }
                    };
                }

                return { color: colorConfig, size: size, borderWidth: borderWidth };
            }

            if (needsLayout) {
                // 有新节点/边加入时，完全刷新图形
                nodes.clear();
                edges.clear();
                graphData.nodes.forEach(n => {
                    let isInPath = currentPath.includes(n.id);
                    let style = getNodeStyle(n, isInPath);

                    // v2.5: 显示节点名称（如果有），限制长度
                    let nodeLabel = n.node_name || String(n.id);
                    if (nodeLabel.length > 10) nodeLabel = nodeLabel.substring(0, 10) + '...';

                    let tooltipText = '🔹 节点 ' + n.id + (n.node_name ? '\\n📛 ' + n.node_name : '') +
                        '\\n📋 类型: ' + (n.is_keyframe ? '关键帧 ⭐' : '普通节点') + (isInPath ? ' [路径]' : '') +
                        '\\n🏷️ 标签: ' + (n.semantic_labels || []).slice(0, 5).join(', ') +
                        (n.scene_description ? '\\n📝 描述: ' + n.scene_description.substring(0, 50) + '...' : '');

                    nodes.add({
                        id: n.id,
                        label: nodeLabel,
                        title: tooltipText,
                        color: style.color,
                        size: style.size,
                        borderWidth: style.borderWidth,
                        shadow: {
                            enabled: true,
                            color: 'rgba(0,0,0,0.5)',
                            size: 10,
                            x: 3,
                            y: 3
                        }
                    });
                });
                graphData.edges.forEach((e, idx) => {
                    // v2.5: 更美观的边样式
                    edges.add({
                        id: idx,
                        from: e.from,
                        to: e.to,
                        title: '连接权重: ' + e.weight.toFixed(3),
                        color: {
                            color: 'rgba(100,150,200,0.5)',
                            highlight: '#00d4ff',
                            hover: 'rgba(0,212,255,0.8)'
                        },
                        width: Math.max(1, Math.min(4, e.weight * 3)),  // 根据权重调整边宽度
                        smooth: { type: 'continuous', roundness: 0.3 }
                    });
                });
                lastNodeCount = nodeCount;
                lastEdgeCount = edgeCount;
                addLog('info', '图形已更新: ' + nodeCount + ' 节点, ' + edgeCount + ' 边');
            } else {
                // 没有新节点/边，只更新节点状态（颜色、边框等）但不改变位置
                graphData.nodes.forEach(n => {
                    let isInPath = currentPath.includes(n.id);
                    let style = getNodeStyle(n, isInPath);

                    var existingNode = nodes.get(n.id);
                    if (existingNode) {
                        nodes.update({
                            id: n.id,
                            color: style.color,
                            size: style.size,
                            borderWidth: style.borderWidth
                        });
                    }
                });
            }

            setTextContent('stat-nodes', graphData.nodes.length);
            setTextContent('stat-edges', graphData.edges.length);
            setTextContent('stat-keyframes', graphData.nodes.filter(n => n.is_keyframe).length);
            setTextContent('stat-current', graphData.current_node !== null ? graphData.current_node : '-');
        }

        // 加载测试目录
        async function loadTestDirectory() {
            const dirPath = document.getElementById('test-dir-path').value.trim();
            if (!dirPath) { addLog('warn', '请输入目录路径'); return; }

            try {
                addLog('info', '正在加载: ' + dirPath);
                const response = await fetch('/api/test/load_directory', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: dirPath })
                });
                const data = await response.json();

                if (data.success) {
                    testFrames = data.frames;
                    currentFrameIndex = 0;
                    document.getElementById('dir-info').textContent = '已加载 ' + testFrames.length + ' 帧';
                    document.getElementById('frame-nav').style.display = 'flex';
                    document.getElementById('frame-slider').max = testFrames.length - 1;
                    setTextContent('frame-total', testFrames.length);
                    if (data.instruction) {
                        document.getElementById('nav-instruction').value = data.instruction;
                    }
                    if (testFrames.length > 0) loadFrame(0);
                    addLog('info', '成功加载 ' + testFrames.length + ' 帧');
                } else {
                    addLog('error', data.message || '加载失败');
                    document.getElementById('dir-info').textContent = data.message || '加载失败';
                }
            } catch (error) {
                addLog('error', '加载失败: ' + error.message);
            }
        }

        // 加载指定帧
        async function loadFrame(index) {
            if (index < 0 || index >= testFrames.length) return;
            currentFrameIndex = index;
            setTextContent('frame-current', index + 1);
            document.getElementById('frame-slider').value = index;

            try {
                const response = await fetch('/api/test/get_frame/' + index);
                const data = await response.json();
                if (data.success && data.image) {
                    document.getElementById('current-frame-img').src = data.image;
                    document.getElementById('image-preview-container').style.display = 'block';
                }
            } catch (error) {
                addLog('error', '加载帧失败: ' + error.message);
            }
        }

        // 处理当前帧
        async function processCurrentFrame() {
            if (testFrames.length === 0) { addLog('warn', '请先加载测试数据'); return; }

            const instruction = document.getElementById('nav-instruction').value.trim();

            try {
                addLog('info', '正在处理第 ' + (currentFrameIndex + 1) + ' 帧...');
                const response = await fetch('/api/frame/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        frame_index: currentFrameIndex,
                        instruction: instruction || null,
                        memory_enabled: memoryEnabled,
                        is_first_frame: currentFrameIndex === 0
                    })
                });
                const data = await response.json();

                if (data.success) {
                    displayInferenceResult(data);
                    if (memoryEnabled) refreshGraph();
                    addLog('info', '推理完成，耗时: ' + (data.inference_time || 0).toFixed(2) + 's');
                } else {
                    addLog('error', data.message || '推理失败');
                }
            } catch (error) {
                addLog('error', '处理失败: ' + error.message);
            }
        }

        // 显示推理结果
        function displayInferenceResult(data) {
            const resultDiv = document.getElementById('inference-result');
            const contentDiv = document.getElementById('inference-result-content');
            resultDiv.style.display = 'block';

            let html = '';
            if (data.output_action) {
                const actionMap = {0: 'STOP', 1: '前进', 2: '左转', 3: '右转', 5: '向下看'};
                html += '<div class="action-display">';
                data.output_action.forEach(a => {
                    const name = actionMap[a] || ('动作' + a);
                    const cls = a === 0 ? 'action-item stop' : 'action-item';
                    html += '<span class="' + cls + '">' + name + '</span>';
                });
                html += '</div>';
            }
            if (data.pixel_target) {
                html += '<p style="margin-top:8px;font-size:0.85em;">像素目标: [' +
                    data.pixel_target[0].toFixed(3) + ', ' + data.pixel_target[1].toFixed(3) + ']</p>';
            }
            if (data.node_id !== undefined) {
                html += '<p style="margin-top:8px;font-size:0.85em;">记忆节点: ' + data.node_id +
                    (data.is_new ? ' (新建)' : ' (复用)') + '</p>';
            }
            contentDiv.innerHTML = html;
        }

        // 批量处理
        async function processBatchFrames() {
            if (testFrames.length === 0) { addLog('warn', '请先加载测试数据'); return; }

            const progressDiv = document.getElementById('batch-progress');
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');
            progressDiv.style.display = 'block';

            // v2.5.2: 获取"保留记忆"复选框状态
            const keepMemory = document.getElementById('keep-memory-checkbox').checked;

            // 先重置Agent（根据keepMemory决定是否保留拓扑图）
            const resetRes = await fetch('/api/agent/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ keep_memory: keepMemory })
            });
            const resetData = await resetRes.json();
            if (keepMemory) {
                addLog('info', '🧠 Agent已重置，记忆已保留（多场景合并模式）');
            } else {
                addLog('info', 'Agent和记忆已重置');
            }

            const instruction = document.getElementById('nav-instruction').value.trim() || 'Walk straight ahead, then turn left and stop at the sofa';

            try {
                // 开始记忆记录 (如果启用了记忆功能)
                if (memoryEnabled) {
                    const startRes = await fetch('/api/memory/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ instruction: instruction })
                    });
                    const startData = await startRes.json();
                    if (startData.success) {
                        addLog('info', '🔴 开始记忆记录: ' + instruction);
                    } else {
                        addLog('warn', '记忆记录启动失败: ' + (startData.error || '未知错误'));
                    }
                }

                addLog('info', '开始批量推理 ' + testFrames.length + ' 帧...');

                for (let i = 0; i < testFrames.length; i++) {
                    await loadFrame(i);

                    const response = await fetch('/api/frame/process', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            frame_index: i,
                            instruction: instruction,
                            memory_enabled: memoryEnabled,
                            is_first_frame: i === 0
                        })
                    });
                    const data = await response.json();

                    const progress = ((i + 1) / testFrames.length * 100).toFixed(1);
                    progressFill.style.width = progress + '%';
                    progressText.textContent = '处理中 ' + (i + 1) + '/' + testFrames.length;

                    if (data.success) {
                        displayInferenceResult(data);
                    }

                    // 短暂延迟避免过快
                    await new Promise(r => setTimeout(r, 100));
                }

                // 停止记忆记录 (如果启用了记忆功能)
                if (memoryEnabled) {
                    const stopRes = await fetch('/api/memory/stop', { method: 'POST' });
                    const stopData = await stopRes.json();
                    if (stopData.success) {
                        addLog('info', '⏹️ 记忆记录已停止');
                        if (stopData.memory_info) {
                            addLog('info', '📊 记忆统计: ' + JSON.stringify(stopData.memory_info));
                        }
                    } else {
                        addLog('warn', '记忆记录停止失败: ' + (stopData.error || '未知错误'));
                    }
                }

                addLog('info', '批量推理完成！');
                if (memoryEnabled) refreshGraph();
                progressText.textContent = '完成！';
                setTimeout(() => { progressDiv.style.display = 'none'; }, 2000);

            } catch (error) {
                addLog('error', '批量处理失败: ' + error.message);
                // 确保在错误时也停止记忆记录
                if (memoryEnabled) {
                    await fetch('/api/memory/stop', { method: 'POST' });
                }
                progressDiv.style.display = 'none';
            }
        }

        // 路径规划
        async function planPath() {
            const startNode = parseInt(document.getElementById('start-node').value);
            const targetNode = parseInt(document.getElementById('target-node').value);

            if (isNaN(startNode) || isNaN(targetNode)) {
                addLog('warn', '请输入有效的节点ID');
                return;
            }

            try {
                const response = await fetch('/api/path/plan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ start: startNode, goal: targetNode })
                });
                const data = await response.json();

                if (data.success && data.path) {
                    currentPath = data.path;
                    displayPath(data);
                    highlightPath(data);
                    addLog('info', '路径: ' + data.path.length + ' 节点, 距离: ' + data.total_distance.toFixed(2));
                } else {
                    addLog('warn', data.message || '未找到路径');
                    document.getElementById('path-result').style.display = 'none';
                }
            } catch (error) {
                addLog('error', '路径规划失败: ' + error.message);
            }
        }

        function displayPath(pathData) {
            const container = document.getElementById('path-result');
            const nodesContainer = document.getElementById('path-nodes');
            nodesContainer.innerHTML = '';
            pathData.path.forEach((nodeId, index) => {
                const span = document.createElement('span');
                span.className = 'path-node';
                if (index === 0) span.classList.add('start');
                else if (index === pathData.path.length - 1) span.classList.add('end');
                span.textContent = nodeId;
                nodesContainer.appendChild(span);
                if (index < pathData.path.length - 1) {
                    const arrow = document.createElement('span');
                    arrow.textContent = ' → ';
                    arrow.style.color = '#666';
                    nodesContainer.appendChild(arrow);
                }
            });
            setTextContent('path-distance', pathData.total_distance.toFixed(2));
            setTextContent('path-steps', pathData.path.length - 1);
            container.style.display = 'block';
        }

        function highlightPath(pathData) {
            // 先重置所有节点的边框
            nodes.forEach(node => nodes.update({ id: node.id, borderWidth: 2 }));
            // 高亮路径节点：起点(黄色) -> 中间节点(橙色) -> 终点(红色)
            pathData.path.forEach((nodeId, index) => {
                let color = '#ff8800';  // 中间路径节点用橙色
                if (index === 0) color = '#ffcc00';  // 起点用黄色
                else if (index === pathData.path.length - 1) color = '#ff6b6b';  // 终点用红色
                nodes.update({ id: nodeId, color: color, borderWidth: 4 });
            });
        }

        // ========== 智能路径规划相关函数 ==========

        // 获取上传的VPR图片数量
        function updateVprPreview() {
            var count = 0;
            ['vpr-cam1', 'vpr-cam2', 'vpr-cam3', 'vpr-cam4'].forEach(function(id) {
                var input = document.getElementById(id);
                if (input && input.files && input.files.length > 0) count++;
            });
            var preview = document.getElementById('vpr-preview');
            var countSpan = document.getElementById('vpr-count');
            if (count > 0) {
                preview.style.display = 'block';
                countSpan.textContent = count;
            } else {
                preview.style.display = 'none';
            }
            return count;
        }

        // 读取文件为base64
        function readFileAsBase64(file) {
            return new Promise(function(resolve, reject) {
                var reader = new FileReader();
                reader.onload = function() { resolve(reader.result); };
                reader.onerror = function() { reject(reader.error); };
                reader.readAsDataURL(file);
            });
        }

        // VPR识别起点
        async function vprIdentify() {
            var formData = new FormData();
            var hasFile = false;

            var camIds = ['vpr-cam1', 'vpr-cam2', 'vpr-cam3', 'vpr-cam4'];
            var camNames = ['camera_1', 'camera_2', 'camera_3', 'camera_4'];

            for (var i = 0; i < camIds.length; i++) {
                var input = document.getElementById(camIds[i]);
                if (input && input.files && input.files.length > 0) {
                    formData.append(camNames[i], input.files[0]);
                    hasFile = true;
                }
            }

            if (!hasFile) {
                addLog('warn', '请先上传至少一张环视图片');
                return null;
            }

            try {
                addLog('info', '正在进行VPR识别...');
                var response = await fetch('/api/vpr/identify', {
                    method: 'POST',
                    body: formData
                });
                var data = await response.json();

                if (data.success) {
                    addLog('info', 'VPR识别成功: 节点' + data.matched_node + ', 相似度: ' + (data.similarity * 100).toFixed(1) + '%');
                    return data;
                } else {
                    addLog('warn', 'VPR识别失败: ' + (data.message || data.error || '未知错误'));
                    return null;
                }
            } catch (error) {
                addLog('error', 'VPR识别错误: ' + error.message);
                return null;
            }
        }

        // 语义检索终点
        async function semanticSearch() {
            var query = document.getElementById('goal-query').value.trim();
            if (!query) {
                addLog('warn', '请输入目标描述');
                return null;
            }

            try {
                addLog('info', '正在进行语义检索: ' + query);
                var response = await fetch('/api/semantic/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                var data = await response.json();

                if (data.success && data.best_match) {
                    var matchId = data.best_match.id || data.best_match.node_id;
                    addLog('info', '语义检索成功: 节点' + matchId);
                    return data;
                } else {
                    addLog('warn', '语义检索失败: ' + (data.message || '未找到匹配节点'));
                    return null;
                }
            } catch (error) {
                addLog('error', '语义检索错误: ' + error.message);
                return null;
            }
        }

        // 智能路径规划
        async function smartPathPlan() {
            var startNode = document.getElementById('smart-start-node').value;
            var goalNode = document.getElementById('smart-goal-node').value;
            var goalQuery = document.getElementById('goal-query').value.trim();

            // 构建FormData
            var formData = new FormData();
            var hasVprImages = false;

            var camIds = ['vpr-cam1', 'vpr-cam2', 'vpr-cam3', 'vpr-cam4'];
            var camNames = ['camera_1', 'camera_2', 'camera_3', 'camera_4'];

            for (var i = 0; i < camIds.length; i++) {
                var input = document.getElementById(camIds[i]);
                if (input && input.files && input.files.length > 0) {
                    formData.append(camNames[i], input.files[0]);
                    hasVprImages = true;
                }
            }

            formData.append('goal_query', goalQuery);
            if (startNode) formData.append('start_node', startNode);
            if (goalNode) formData.append('goal_node', goalNode);

            // 检查是否有足够信息
            if (!hasVprImages && !startNode) {
                addLog('warn', '请上传环视图片或指定起点节点');
                return;
            }
            if (!goalQuery && !goalNode) {
                addLog('warn', '请输入目标描述或指定终点节点');
                return;
            }

            try {
                addLog('info', '正在进行智能路径规划...');
                var response = await fetch('/api/smart_path/plan', {
                    method: 'POST',
                    body: formData
                });
                var data = await response.json();

                if (data.success && data.path) {
                    displaySmartPathResult(data);
                    currentPath = data.path;
                    highlightPath(data);
                    addLog('info', '智能路径规划成功: ' + data.path.length + ' 个节点');
                } else {
                    addLog('warn', '智能路径规划失败: ' + (data.message || data.error || '未知错误'));
                    document.getElementById('smart-path-result').style.display = 'none';
                }
            } catch (error) {
                addLog('error', '智能路径规划错误: ' + error.message);
            }
        }

        // 显示智能路径规划结果
        function displaySmartPathResult(data) {
            var resultDiv = document.getElementById('smart-path-result');
            var startInfo = document.getElementById('smart-start-info');
            var goalInfo = document.getElementById('smart-goal-info');
            var nodesContainer = document.getElementById('smart-path-nodes');

            // 清空并显示起点信息
            startInfo.textContent = '';
            if (data.start_node !== undefined) {
                var startSpan = document.createElement('span');
                startSpan.style.color = '#ffcc00';
                startSpan.textContent = '📍 起点: 节点 ' + data.start_node;
                startInfo.appendChild(startSpan);
            }

            // 清空并显示终点信息
            goalInfo.textContent = '';
            if (data.goal_node !== undefined) {
                var goalSpan = document.createElement('span');
                goalSpan.style.color = '#ff6b6b';
                var goalText = '🎯 终点: 节点 ' + data.goal_node;
                if (data.goal_query) {
                    goalText += ' (匹配: "' + data.goal_query + '")';
                }
                goalSpan.textContent = goalText;
                goalInfo.appendChild(goalSpan);
            }

            // 显示路径节点
            nodesContainer.textContent = '';
            data.path.forEach(function(nodeId, index) {
                var span = document.createElement('span');
                span.className = 'path-node';
                if (index === 0) span.classList.add('start');
                else if (index === data.path.length - 1) span.classList.add('end');
                span.textContent = nodeId;
                nodesContainer.appendChild(span);
                if (index < data.path.length - 1) {
                    var arrow = document.createElement('span');
                    arrow.textContent = ' → ';
                    arrow.style.color = '#666';
                    nodesContainer.appendChild(arrow);
                }
            });

            setTextContent('smart-path-distance', data.total_distance ? data.total_distance.toFixed(2) : '-');
            setTextContent('smart-path-steps', data.path.length);
            resultDiv.style.display = 'block';
        }

        // 定期检查并更新连接状态（前端只监控，不主动连接）
        async function updateConnectionStatus() {
            var indicator = document.getElementById('ws-indicator');
            var statusText = document.getElementById('ws-status-text');
            var urlDisplay = document.getElementById('ws-url-display');

            try {
                var response = await fetch('/api/ws/status');
                var data = await response.json();

                if (data.connected) {
                    indicator.className = 'ws-indicator connected';
                    statusText.textContent = '已连接';
                    if (urlDisplay && data.ws_url) {
                        urlDisplay.textContent = data.ws_url;
                    }
                } else {
                    indicator.className = 'ws-indicator disconnected';
                    statusText.textContent = '未连接';
                }
            } catch (error) {
                indicator.className = 'ws-indicator disconnected';
                statusText.textContent = '状态未知';
                console.error('检查连接状态失败:', error);
            }
        }

        // 事件绑定 - 使用try-catch包裹每个绑定以防止单个失败影响其他
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM已加载，开始初始化...');

            try { initNetwork(); } catch(e) { console.error('initNetwork失败:', e); }
            try { refreshGraph(); } catch(e) { console.error('refreshGraph失败:', e); }

            // 定时刷新拓扑图
            setInterval(function() {
                try { refreshGraph(); } catch(e) { console.error('定时refreshGraph失败:', e); }
            }, 10000);

            // 定时检查连接状态
            updateConnectionStatus();
            setInterval(function() {
                try { updateConnectionStatus(); } catch(e) { console.error('updateConnectionStatus失败:', e); }
            }, 5000);

            addLog('info', '系统已初始化 v2.4 - 支持智能路径规划');

            // 记忆开关
            var memoryToggle = document.getElementById('memory-toggle');
            if (memoryToggle) {
                memoryToggle.addEventListener('change', async function() {
                    memoryEnabled = this.checked;
                    var status = document.getElementById('memory-status');
                    try {
                        await fetch('/api/memory/toggle', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ enabled: memoryEnabled })
                        });
                        if (status) {
                            status.textContent = memoryEnabled ? '已开启' : '已关闭';
                            status.className = memoryEnabled ? 'status-text status-on' : 'status-text status-off';
                        }
                        addLog('info', '记忆功能' + (memoryEnabled ? '已开启' : '已关闭'));
                    } catch(e) { addLog('error', '切换记忆功能失败: ' + e.message); }
                });
            }

            // 刷新按钮
            var btnRefresh = document.getElementById('btn-refresh');
            if (btnRefresh) {
                btnRefresh.onclick = function() { refreshGraph(true); };  // 强制刷新
            }

            // 清空记忆按钮
            var btnClear = document.getElementById('btn-clear');
            if (btnClear) {
                btnClear.onclick = async function() {
                    if (!confirm('确定清空所有记忆？')) return;
                    try {
                        await fetch('/api/memory/clear', { method: 'POST' });
                        addLog('warn', '记忆已清空');
                        refreshGraph();
                    } catch(e) { addLog('error', '清空失败: ' + e.message); }
                };
            }

            // 加载目录按钮
            var btnLoadDir = document.getElementById('btn-load-dir');
            if (btnLoadDir) {
                btnLoadDir.onclick = function() {
                    console.log('加载目录按钮被点击');
                    loadTestDirectory();
                };
            }

            // 处理当前帧按钮
            var btnProcessFrame = document.getElementById('btn-process-frame');
            if (btnProcessFrame) {
                btnProcessFrame.onclick = function() { processCurrentFrame(); };
            }

            // 批量推理按钮
            var btnBatchProcess = document.getElementById('btn-batch-process');
            if (btnBatchProcess) {
                btnBatchProcess.onclick = function() { processBatchFrames(); };
            }

            // 路径规划按钮
            var btnPlanPath = document.getElementById('btn-plan-path');
            if (btnPlanPath) {
                btnPlanPath.onclick = function() { planPath(); };
            }

            // 智能路径规划按钮
            var btnSmartPlan = document.getElementById('btn-smart-plan');
            if (btnSmartPlan) {
                btnSmartPlan.onclick = function() { smartPathPlan(); };
            }

            // VPR识别按钮
            var btnVprOnly = document.getElementById('btn-vpr-only');
            if (btnVprOnly) {
                btnVprOnly.onclick = async function() {
                    var result = await vprIdentify();
                    if (result && result.matched_node !== undefined) {
                        document.getElementById('smart-start-node').value = result.matched_node;
                    }
                };
            }

            // 语义检索按钮
            var btnSemanticOnly = document.getElementById('btn-semantic-only');
            if (btnSemanticOnly) {
                btnSemanticOnly.onclick = async function() {
                    var result = await semanticSearch();
                    if (result && result.best_match) {
                        var nodeId = result.best_match.id || result.best_match.node_id;
                        document.getElementById('smart-goal-node').value = nodeId;
                    }
                };
            }

            // VPR图片上传监听
            ['vpr-cam1', 'vpr-cam2', 'vpr-cam3', 'vpr-cam4'].forEach(function(id) {
                var input = document.getElementById(id);
                if (input) {
                    input.onchange = function() { updateVprPreview(); };
                }
            });

            // 上一帧按钮
            var btnPrevFrame = document.getElementById('btn-prev-frame');
            if (btnPrevFrame) {
                btnPrevFrame.onclick = function() {
                    if (currentFrameIndex > 0) loadFrame(currentFrameIndex - 1);
                };
            }

            // 下一帧按钮
            var btnNextFrame = document.getElementById('btn-next-frame');
            if (btnNextFrame) {
                btnNextFrame.onclick = function() {
                    if (currentFrameIndex < testFrames.length - 1) loadFrame(currentFrameIndex + 1);
                };
            }

            // 帧滑块
            var frameSlider = document.getElementById('frame-slider');
            if (frameSlider) {
                frameSlider.oninput = function() { loadFrame(parseInt(this.value)); };
            }

            // 重置Agent按钮
            var btnResetAgent = document.getElementById('btn-reset-agent');
            if (btnResetAgent) {
                btnResetAgent.onclick = async function() {
                    try {
                        await fetch('/api/agent/reset', { method: 'POST' });
                        addLog('info', 'Agent已重置');
                    } catch(e) { addLog('error', '重置失败: ' + e.message); }
                };
            }

            // ================== 数据库操作按钮 v3.0 ==================

            // 显示数据库操作结果
            function showDbResult(message, isError) {
                var resultDiv = document.getElementById('db-operation-result');
                var resultText = document.getElementById('db-result-text');
                resultDiv.style.display = 'block';
                resultText.textContent = message;
                resultText.style.color = isError ? '#ff6b6b' : '#00ff88';
                setTimeout(function() { resultDiv.style.display = 'none'; }, 5000);
            }

            // 刷新数据库统计
            async function refreshDbStats() {
                try {
                    var response = await fetch('/api/db/stats');
                    var data = await response.json();
                    if (data.success) {
                        document.getElementById('db-node-count').textContent = data.stats.node_count;
                        document.getElementById('db-edge-count').textContent = data.stats.edge_count;
                        document.getElementById('db-status').textContent = '● 已连接';
                        document.getElementById('db-status').style.color = '#00ff88';
                    }
                } catch(e) {
                    document.getElementById('db-status').textContent = '● 连接失败';
                    document.getElementById('db-status').style.color = '#ff6b6b';
                }
            }

            // 创建节点
            document.getElementById('btn-db-create-node').onclick = async function() {
                var nodeId = document.getElementById('db-node-id').value;
                var nodeName = document.getElementById('db-node-name').value;
                var nodeDesc = document.getElementById('db-node-desc').value;
                var nodeLabels = document.getElementById('db-node-labels').value;

                if (!nodeId) { showDbResult('请输入节点ID', true); return; }

                var labelsArray = nodeLabels ? nodeLabels.split(/[,，、]/).map(s => s.trim()).filter(s => s) : [];

                try {
                    var response = await fetch('/api/node', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            node_id: parseInt(nodeId),
                            node_name: nodeName || null,
                            scene_description: nodeDesc || null,
                            semantic_labels: labelsArray
                        })
                    });
                    var data = await response.json();
                    if (data.success) {
                        showDbResult('节点 ' + nodeId + ' 创建成功', false);
                        addLog('info', '创建节点: ' + nodeId);
                        refreshGraph();
                        refreshDbStats();
                    } else {
                        showDbResult('创建失败: ' + (data.error || data.message), true);
                    }
                } catch(e) { showDbResult('创建失败: ' + e.message, true); }
            };

            // 查询节点
            document.getElementById('btn-db-read-node').onclick = async function() {
                var nodeId = document.getElementById('db-node-id').value;
                if (!nodeId) { showDbResult('请输入节点ID', true); return; }

                try {
                    var response = await fetch('/api/node/' + nodeId);
                    var data = await response.json();
                    if (data.success && data.node) {
                        var node = data.node;
                        document.getElementById('db-node-name').value = node.node_name || '';
                        document.getElementById('db-node-desc').value = node.scene_description || '';
                        document.getElementById('db-node-labels').value = (node.semantic_labels || []).join(', ');
                        showDbResult('查询成功: 节点 ' + nodeId, false);
                    } else {
                        showDbResult('节点 ' + nodeId + ' 不存在', true);
                    }
                } catch(e) { showDbResult('查询失败: ' + e.message, true); }
            };

            // 更新节点
            document.getElementById('btn-db-update-node').onclick = async function() {
                var nodeId = document.getElementById('db-node-id').value;
                var nodeName = document.getElementById('db-node-name').value;
                var nodeDesc = document.getElementById('db-node-desc').value;
                var nodeLabels = document.getElementById('db-node-labels').value;

                if (!nodeId) { showDbResult('请输入节点ID', true); return; }

                var labelsArray = nodeLabels ? nodeLabels.split(/[,，、]/).map(s => s.trim()).filter(s => s) : [];

                try {
                    var response = await fetch('/api/node/' + nodeId, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            node_name: nodeName || null,
                            scene_description: nodeDesc || null,
                            semantic_labels: labelsArray
                        })
                    });
                    var data = await response.json();
                    if (data.success) {
                        showDbResult('节点 ' + nodeId + ' 更新成功', false);
                        addLog('info', '更新节点: ' + nodeId);
                        refreshGraph();
                    } else {
                        showDbResult('更新失败: ' + (data.error || data.message), true);
                    }
                } catch(e) { showDbResult('更新失败: ' + e.message, true); }
            };

            // 删除节点
            document.getElementById('btn-db-delete-node').onclick = async function() {
                var nodeId = document.getElementById('db-node-id').value;
                if (!nodeId) { showDbResult('请输入节点ID', true); return; }

                if (!confirm('确定删除节点 ' + nodeId + '？相关的边也会被删除。')) return;

                try {
                    var response = await fetch('/api/node/' + nodeId, { method: 'DELETE' });
                    var data = await response.json();
                    if (data.success) {
                        showDbResult('节点 ' + nodeId + ' 删除成功', false);
                        addLog('warn', '删除节点: ' + nodeId);
                        refreshGraph();
                        refreshDbStats();
                        // 清空输入框
                        document.getElementById('db-node-id').value = '';
                        document.getElementById('db-node-name').value = '';
                        document.getElementById('db-node-desc').value = '';
                        document.getElementById('db-node-labels').value = '';
                    } else {
                        showDbResult('删除失败: ' + (data.error || data.message), true);
                    }
                } catch(e) { showDbResult('删除失败: ' + e.message, true); }
            };

            // 创建边
            document.getElementById('btn-db-create-edge').onclick = async function() {
                var sourceId = document.getElementById('db-edge-source').value;
                var targetId = document.getElementById('db-edge-target').value;

                if (!sourceId || !targetId) { showDbResult('请输入源节点和目标节点ID', true); return; }

                try {
                    var response = await fetch('/api/edge', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            source_node_id: parseInt(sourceId),
                            target_node_id: parseInt(targetId),
                            action: [1],
                            weight: 1.0
                        })
                    });
                    var data = await response.json();
                    if (data.success) {
                        showDbResult('边 ' + sourceId + ' -> ' + targetId + ' 创建成功', false);
                        addLog('info', '创建边: ' + sourceId + ' -> ' + targetId);
                        refreshGraph();
                        refreshDbStats();
                    } else {
                        showDbResult('创建失败: ' + (data.error || data.message), true);
                    }
                } catch(e) { showDbResult('创建失败: ' + e.message, true); }
            };

            // 删除边
            document.getElementById('btn-db-delete-edge').onclick = async function() {
                var sourceId = document.getElementById('db-edge-source').value;
                var targetId = document.getElementById('db-edge-target').value;

                if (!sourceId || !targetId) { showDbResult('请输入源节点和目标节点ID', true); return; }

                if (!confirm('确定删除边 ' + sourceId + ' -> ' + targetId + '？')) return;

                try {
                    var response = await fetch('/api/edge/' + sourceId + '/' + targetId, { method: 'DELETE' });
                    var data = await response.json();
                    if (data.success) {
                        showDbResult('边 ' + sourceId + ' -> ' + targetId + ' 删除成功', false);
                        addLog('warn', '删除边: ' + sourceId + ' -> ' + targetId);
                        refreshGraph();
                        refreshDbStats();
                    } else {
                        showDbResult('删除失败: ' + (data.error || data.message), true);
                    }
                } catch(e) { showDbResult('删除失败: ' + e.message, true); }
            };

            // 从JSON导入
            document.getElementById('btn-db-import').onclick = async function() {
                if (!confirm('从JSON文件导入将覆盖数据库中的现有数据，确定继续？')) return;

                try {
                    var response = await fetch('/api/db/import', { method: 'POST' });
                    var data = await response.json();
                    if (data.success) {
                        showDbResult(data.message, false);
                        addLog('info', '数据导入成功');
                        refreshGraph();
                        refreshDbStats();
                    } else {
                        showDbResult('导入失败: ' + (data.error || data.message), true);
                    }
                } catch(e) { showDbResult('导入失败: ' + e.message, true); }
            };

            // 导出到JSON
            document.getElementById('btn-db-export').onclick = async function() {
                try {
                    var response = await fetch('/api/db/export');
                    var data = await response.json();
                    if (data.success) {
                        showDbResult(data.message, false);
                        addLog('info', '数据导出成功');
                    } else {
                        showDbResult('导出失败: ' + (data.error || data.message), true);
                    }
                } catch(e) { showDbResult('导出失败: ' + e.message, true); }
            };

            // 初始化时刷新数据库统计
            refreshDbStats();

            console.log('所有事件绑定完成');
        });
    </script>
</body>
</html>
'''


# =============================================================================
# WebSocket客户端
# =============================================================================

class InferenceClient:
    """推理服务WebSocket客户端"""

    def __init__(self):
        self.ws = None
        self.ws_url = None
        self.connected = False

    def connect(self, host: str = 'localhost', port: int = 9528) -> Tuple[bool, str]:
        """连接到推理服务

        Returns:
            Tuple[bool, str]: (成功与否, 错误信息)
        """
        if not WEBSOCKET_AVAILABLE:
            logger.error("websocket-client未安装")
            return False, "websocket-client库未安装"

        try:
            self.ws_url = f"ws://{host}:{port}"
            # 使用较短的超时时间，避免长时间等待
            self.ws = websocket.create_connection(self.ws_url, timeout=10)
            self.connected = True
            logger.info(f"已连接到推理服务: {self.ws_url}")
            return True, ""
        except ConnectionRefusedError:
            logger.error(f"连接被拒绝: {host}:{port} - 推理服务可能未启动")
            self.connected = False
            return False, f"连接被拒绝，请确认推理服务已在 {host}:{port} 启动"
        except TimeoutError:
            logger.error(f"连接超时: {host}:{port}")
            self.connected = False
            return False, "连接超时"
        except Exception as e:
            logger.error(f"连接失败: {e}")
            self.connected = False
            return False, str(e)

    def disconnect(self):
        """断开连接"""
        if self.ws:
            self.ws.close()
            self.ws = None
        self.connected = False

    def call_inference(self, instruction: str, rgb_path: str,
                      robot_id: str = "TEST_ROBOT_001",
                      pts: Optional[int] = None,
                      look_down: bool = False,
                      surround_images: Optional[Dict[str, str]] = None) -> Optional[dict]:
        """调用推理接口

        Args:
            instruction: 导航指令
            rgb_path: 前视图路径（front_1）
            robot_id: 机器人ID
            pts: 时间戳
            look_down: 是否向下看
            surround_images: 环视图路径字典 {'camera_1': path, 'camera_2': path, ...}

        Returns:
            推理结果
        """
        if not self.connected or not self.ws:
            logger.error("未连接到推理服务")
            return None

        try:
            if not os.path.exists(rgb_path):
                logger.error(f"图像不存在: {rgb_path}")
                return None

            # 编码front_1图像
            with open(rgb_path, 'rb') as f:
                rgb_base64 = base64.b64encode(f.read()).decode('utf-8')

            if pts is None:
                pts = int(time.time() * 1000)

            # 构建images字典
            images = {'front_1': rgb_base64}

            # 添加环视图（如果提供）
            if surround_images:
                for cam_id, cam_path in surround_images.items():
                    if cam_path and os.path.exists(cam_path):
                        with open(cam_path, 'rb') as f:
                            images[cam_id] = base64.b64encode(f.read()).decode('utf-8')
                        logger.debug(f"添加环视图: {cam_id}")

            data = {
                'id': robot_id,
                'task': instruction,
                'pts': pts,
                'images': images,
                'look_down': look_down
            }

            logger.info(f"发送推理请求: task={instruction}, 图像数={len(images)}")
            self.ws.send(json.dumps(data))
            result = self.ws.recv()
            return json.loads(result)

        except Exception as e:
            logger.error(f"推理调用失败: {e}")
            self.reconnect()
            return None

    def reset_agent(self, keep_memory: bool = False) -> Optional[dict]:
        """
        重置Agent状态
        v2.5.2: 支持keep_memory参数
        - keep_memory=True: 只重置Agent，保留拓扑图记忆
        - keep_memory=False: 重置Agent和记忆
        """
        if not self.connected or not self.ws:
            return None

        try:
            self.ws.send(json.dumps({'command': 'reset', 'keep_memory': keep_memory}))
            result = self.ws.recv()
            return json.loads(result) if result else None
        except Exception as e:
            logger.error(f"重置失败: {e}")
            return None

    def reconnect(self):
        """重新连接"""
        if self.ws_url:
            try:
                self.ws = websocket.create_connection(self.ws_url, timeout=120)
                self.connected = True
                logger.info("重新连接成功")
            except:
                self.connected = False

    def send_command(self, command: str, **kwargs) -> Optional[dict]:
        """发送命令到推理服务

        Args:
            command: 命令名称 (reset, start_memory, stop_memory等)
            **kwargs: 额外参数

        Returns:
            命令执行结果
        """
        if not self.connected or not self.ws:
            logger.error("未连接到推理服务")
            return None

        try:
            data = {'command': command}
            data.update(kwargs)
            logger.info(f"发送命令: {command}, 参数: {kwargs}")
            self.ws.send(json.dumps(data))
            result = self.ws.recv()
            response = json.loads(result) if result else None
            logger.info(f"命令响应: {response}")
            return response
        except Exception as e:
            logger.error(f"命令执行失败: {e}")
            self.reconnect()
            return None

    def start_memory_recording(self, instruction: str) -> Optional[dict]:
        """开始记忆记录

        Args:
            instruction: 原始导航指令，用于关联记忆

        Returns:
            命令执行结果
        """
        return self.send_command('start_memory', original_instruction=instruction)

    def stop_memory_recording(self) -> Optional[dict]:
        """停止记忆记录

        Returns:
            命令执行结果
        """
        return self.send_command('stop_memory')


# =============================================================================
# 服务器类
# =============================================================================

class MemoryVisualizationServer:
    """记忆可视化服务器 v2.4 - 增加VPR检索和语义路径规划"""

    def __init__(self, port: int = 9530):
        self.port = port
        self.memory_enabled = True
        self.topo_map = None
        self.config = None
        self.test_frames = []
        self.test_dir = None
        self.current_instruction = None
        self.inference_client = InferenceClient()

        # VPR特征提取器（用于图片上传识别起点）
        self.feature_extractor = None
        # 已加载的图数据（从semantic_graph.json或推理服务获取）
        self.loaded_graph_data = None

        self._init_memory_system()

        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            CORS(self.app)
            self._setup_routes()
        else:
            self.app = None

    def _init_memory_system(self):
        """初始化记忆系统"""
        logger.info("正在初始化记忆系统...")
        try:
            from memory_modules.config import MemoryNavigationConfig
            from memory_modules.topological_map import TopologicalMapManager

            self.config = MemoryNavigationConfig()
            # 设置默认GPU ID (使用GPU 1)
            if self.config.gpu_id is None:
                self.config.gpu_id = "1"
            self.topo_map = TopologicalMapManager(self.config)
            logger.info("记忆系统初始化成功")

            # 尝试初始化LongCLIP特征提取器（用于图片上传VPR识别）
            try:
                from memory_modules.feature_extraction import LongCLIPFeatureExtractor
                device = f"cuda:{self.config.gpu_id}"
                self.feature_extractor = LongCLIPFeatureExtractor(
                    self.config.longclip_model_path,
                    device=device,
                    feature_dim=self.config.feature_dim
                )
                if self.feature_extractor.is_available:
                    logger.info(f"LongCLIP特征提取器初始化成功 (设备: {device})")
                else:
                    logger.warning("LongCLIP特征提取器不可用，将使用回退方案")
            except Exception as ve:
                logger.warning(f"LongCLIP特征提取器初始化失败: {ve}")
                self.feature_extractor = None

            # 尝试加载已保存的记忆数据
            self._load_saved_memory_data()

        except Exception as e:
            logger.error(f"记忆系统初始化失败: {e}")
            self.topo_map = None
            self.config = None

    def _load_saved_memory_data(self):
        """加载已保存的记忆数据（语义图和VPR索引）

        注意: 此方法加载的pickle文件仅来自系统自身生成的内部数据，
        保存路径由config.memory_save_path指定，不加载任何外部来源的文件。
        """
        if self.topo_map is None or self.config is None:
            return

        save_path = self.config.memory_save_path
        if not os.path.exists(save_path):
            logger.info(f"记忆数据目录不存在: {save_path}")
            return

        try:
            # 导入pickle用于加载系统内部生成的路线数据
            # 安全说明: 仅加载config.memory_save_path目录下系统自动生成的pkl文件
            import pickle
            import networkx as nx

            # 1. 加载语义图
            semantic_graph_path = os.path.join(save_path, 'semantic_graph.json')
            semantic_metadata_path = os.path.join(save_path, 'semantic_metadata.json')

            if os.path.exists(semantic_graph_path):
                with open(semantic_graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)

                # 加载到语义图管理器
                self.topo_map.semantic_graph.semantic_graph = nx.node_link_graph(graph_data)
                logger.info(f"语义图已加载: {len(graph_data.get('nodes', []))} 个节点")

                # 从图数据中提取节点信息到拓扑图
                for node_data in graph_data.get('nodes', []):
                    node_id = node_data.get('id')
                    if node_id is not None:
                        # 创建节点占位符
                        from memory_modules.models import TopologicalNode
                        if node_id not in self.topo_map.nodes:
                            # 创建简单的节点对象
                            node = TopologicalNode(
                                node_id=node_id,
                                visual_feature=np.zeros(self.config.feature_dim),
                                timestamp=time.time()
                            )
                            node.scene_description = node_data.get('description', '')
                            node.semantic_labels = node_data.get('labels', [])
                            self.topo_map.nodes[node_id] = node

                            # 添加到networkx图
                            if self.topo_map.graph is not None:
                                self.topo_map.graph.add_node(node_id)

                # 加载边
                for edge_data in graph_data.get('links', []):
                    source = edge_data.get('source')
                    target = edge_data.get('target')
                    if source is not None and target is not None and self.topo_map.graph is not None:
                        self.topo_map.graph.add_edge(source, target, weight=1.0)

                self.topo_map.next_node_id = max(self.topo_map.nodes.keys(), default=-1) + 1
                logger.info(f"拓扑图节点已加载: {len(self.topo_map.nodes)} 个节点, 下一节点ID: {self.topo_map.next_node_id}")

            # 2. 加载语义元数据
            if os.path.exists(semantic_metadata_path):
                with open(semantic_metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                self.topo_map.semantic_graph.node_metadata = {
                    int(k): v for k, v in metadata.get('node_metadata', {}).items()
                }
                self.topo_map.semantic_graph.label_index = metadata.get('label_index', {})
                self.topo_map.semantic_graph.description_index = {
                    int(k): v for k, v in metadata.get('description_index', {}).items()
                }
                logger.info(f"语义元数据已加载: {len(self.topo_map.semantic_graph.label_index)} 个标签索引")

            # 3. 加载VPR特征索引
            # 查找特征文件 (仅加载系统生成的.npy文件)
            feature_files = [f for f in os.listdir(save_path) if f.endswith('_features.npy')]
            for feature_file in feature_files:
                feature_path = os.path.join(save_path, feature_file)
                try:
                    features = np.load(feature_path)
                    logger.info(f"加载特征文件: {feature_file}, 形状: {features.shape}")

                    # 对应的pkl文件 (系统内部生成的路线数据)
                    route_id = feature_file.replace('_features.npy', '')
                    pkl_path = os.path.join(save_path, f"{route_id}.pkl")

                    node_sequence = None
                    if os.path.exists(pkl_path):
                        # 安全: 仅加载系统自身生成的pickle文件
                        with open(pkl_path, 'rb') as f:
                            route_data = pickle.load(f)
                        node_sequence = route_data.get('node_sequence', [])
                        logger.info(f"加载路线数据: {route_id}, 节点序列长度: {len(node_sequence)}")

                    # 将特征添加到VPR索引
                    # 如果有节点序列，按节点添加
                    if node_sequence and len(node_sequence) == features.shape[0]:
                        for i, node_id in enumerate(node_sequence):
                            feature = features[i]
                            # 添加到VPR索引
                            self.topo_map.vpr.add_feature(
                                feature=feature,
                                node_id=node_id,
                                timestamp=time.time()
                            )
                            # 更新节点特征
                            if node_id in self.topo_map.nodes:
                                self.topo_map.nodes[node_id].visual_feature = feature
                    else:
                        # 没有节点序列，按顺序添加
                        for i, feature in enumerate(features):
                            node_id = i % len(self.topo_map.nodes) if self.topo_map.nodes else i
                            self.topo_map.vpr.add_feature(
                                feature=feature,
                                node_id=node_id,
                                timestamp=time.time()
                            )

                    logger.info(f"VPR索引已重建: {self.topo_map.vpr.index.ntotal} 个特征")

                except Exception as e:
                    logger.warning(f"加载特征文件失败 {feature_file}: {e}")

            logger.info("记忆数据加载完成")

        except Exception as e:
            logger.error(f"加载记忆数据失败: {e}", exc_info=True)

    def _scan_test_directory(self, dir_path: str) -> Tuple[List[Dict], Optional[str]]:
        """扫描测试数据目录

        支持多相机帧格式: {timestamp}_{camera_type}.jpg
        - front_1: 前视图（主相机）
        - camera_1~4: 环视图（4个环视相机）
        同一时间戳的5张图片算作1帧
        """
        frames = []
        instruction = None
        dir_path = Path(dir_path)

        if not dir_path.exists():
            logger.error(f"目录不存在: {dir_path}")
            return frames, instruction

        # 读取instruction.txt
        instruction_file = dir_path / 'instruction.txt'
        if instruction_file.exists():
            instruction = instruction_file.read_text().strip()
            logger.info(f"读取到导航指令: {instruction}")

        # 查找图像文件
        image_files = sorted(dir_path.glob('*.jpg'))
        # 过滤掉look_down图像
        image_files = [f for f in image_files if '_look_down' not in f.name]

        if not image_files:
            # 尝试png格式
            image_files = sorted(dir_path.glob('*.png'))
            image_files = [f for f in image_files if '_look_down' not in f.name]

        if not image_files:
            logger.warning(f"目录中未找到图像文件: {dir_path}")
            return frames, instruction

        # 检测数据格式：是否为多相机格式 {timestamp}_{camera_type}.jpg
        first_file = image_files[0].name
        import re
        multi_camera_pattern = re.compile(r'^(\d+)_(front_1|camera_[1-4])\.(?:jpg|png)$')

        if multi_camera_pattern.match(first_file):
            # 多相机格式：按时间戳分组
            logger.info("检测到多相机数据格式，按时间戳分组...")
            timestamp_groups = {}

            for img_file in image_files:
                match = multi_camera_pattern.match(img_file.name)
                if match:
                    timestamp = match.group(1)
                    camera_type = match.group(2)

                    if timestamp not in timestamp_groups:
                        timestamp_groups[timestamp] = {}
                    timestamp_groups[timestamp][camera_type] = str(img_file)

            # 按时间戳排序，构建帧数据
            for timestamp in sorted(timestamp_groups.keys()):
                cameras = timestamp_groups[timestamp]

                # 必须有front_1
                if 'front_1' not in cameras:
                    logger.warning(f"时间戳 {timestamp} 缺少 front_1 图像，跳过")
                    continue

                frame_data = {
                    'timestamp': timestamp,
                    'path': cameras['front_1'],  # 主图为front_1
                    'name': f"帧_{timestamp}",
                    'front_1': cameras.get('front_1'),
                    'camera_1': cameras.get('camera_1'),
                    'camera_2': cameras.get('camera_2'),
                    'camera_3': cameras.get('camera_3'),
                    'camera_4': cameras.get('camera_4'),
                }
                frames.append(frame_data)

            # 统计相机覆盖情况
            total_cameras = sum(1 for f in frames for k in ['front_1', 'camera_1', 'camera_2', 'camera_3', 'camera_4'] if f.get(k))
            logger.info(f"多相机模式: {len(timestamp_groups)} 个时间戳 -> {len(frames)} 帧 (共 {total_cameras} 张图像)")
        else:
            # 单图像格式：每个文件算一帧
            logger.info("检测到单图像数据格式...")
            for img_file in image_files:
                frames.append({
                    'path': str(img_file),
                    'name': img_file.name,
                    'front_1': str(img_file)
                })
            logger.info(f"单图像模式: 找到 {len(frames)} 帧图像")

        return frames, instruction

    def _regenerate_node_name(self, scene_description: str, semantic_labels: List[str]) -> str:
        """
        回退命名方法（仅在没有存储的node_name时使用）

        简化版：只返回基本场景类型，不做复杂的组合
        """
        if not scene_description and not semantic_labels:
            return "未知位置"

        combined_text = (scene_description or "") + " " + " ".join(semantic_labels or [])

        # 只提取基本场景类型
        scene_types = ['走廊', '办公区', '休息区', '大厅']
        for st in scene_types:
            if st in combined_text:
                return st

        return "位置"

    def _setup_routes(self):
        """设置API路由"""

        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.app.route('/api/graph')
        def get_graph():
            """获取拓扑图数据 - v3.0 只从PostgreSQL数据库读取"""
            try:
                if not DATABASE_AVAILABLE:
                    logger.error("数据库模块不可用，无法获取图数据")
                    return jsonify({'success': False, 'error': '数据库模块不可用'})

                # 从数据库获取图数据
                db = get_database()
                graph_data = db.get_graph_data()

                # 转换为前端格式
                nodes = []
                for node in graph_data.get('nodes', []):
                    nodes.append({
                        'id': node.get('id'),
                        'label': f"N{node.get('id')}",
                        'is_keyframe': node.get('is_keyframe', False),
                        'visit_count': node.get('visit_count', 1),
                        'semantic_labels': node.get('semantic_labels', []),
                        'scene_description': node.get('scene_description', ''),
                        'is_current': False,
                        'node_name': node.get('node_name'),
                        'navigation_instruction': node.get('navigation_instruction'),
                        'pixel_target': node.get('pixel_target'),
                        'created_at': node.get('created_at'),
                        'updated_at': node.get('updated_at'),
                        'has_front_view_feature': node.get('has_front_view_feature', False),
                        'front_view_embedding': node.get('front_view_embedding'),
                        'source_timestamps': node.get('source_timestamps', [])
                    })

                edges = []
                for edge in graph_data.get('edges', []):
                    edges.append({
                        'from': edge.get('source'),
                        'to': edge.get('target'),
                        'weight': edge.get('weight', 1.0)
                    })

                data = {'nodes': nodes, 'edges': edges, 'current_node': None}
                logger.info(f"从数据库加载图数据: {len(nodes)} 节点, {len(edges)} 边")
                return jsonify({'success': True, 'data': data})

            except Exception as e:
                logger.error(f"获取图形数据错误: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/node/<int:node_id>')
        def get_node(node_id):
            """获取指定节点信息 - v3.0 从PostgreSQL数据库读取"""
            try:
                if not DATABASE_AVAILABLE:
                    return jsonify({'success': False, 'error': '数据库模块不可用'})

                db = get_database()
                node = db.get_node(node_id)

                if node is None:
                    return jsonify({'success': False, 'message': '节点未找到'})

                return jsonify({
                    'success': True,
                    'node': {
                        'id': node.get('node_id'),
                        'is_keyframe': node.get('is_keyframe', False),
                        'visit_count': node.get('visit_count', 1),
                        'semantic_labels': node.get('semantic_labels', []),
                        'scene_description': node.get('scene_description', ''),
                        'node_name': node.get('node_name'),
                        'navigation_instruction': node.get('navigation_instruction'),
                        'pixel_target': node.get('pixel_target'),
                        'created_at': node.get('created_at'),
                        'updated_at': node.get('updated_at'),
                        'has_front_view_feature': node.get('front_view_feature') is not None
                    }
                })
            except Exception as e:
                logger.error(f"获取节点错误: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/memory/toggle', methods=['POST'])
        def toggle_memory():
            data = request.json
            self.memory_enabled = data.get('enabled', True)
            logger.info(f"记忆功能: {'开启' if self.memory_enabled else '关闭'}")
            return jsonify({'success': True, 'enabled': self.memory_enabled})

        @self.app.route('/api/memory/clear', methods=['POST'])
        def clear_memory():
            """清空记忆数据 - v3.0 从数据库清空"""
            try:
                # 通过推理服务清空记忆
                if self.inference_client.connected:
                    response = self.inference_client.send_command('clear_memory')
                    if response and response.get('status') == 'success':
                        logger.info("通过推理服务清空记忆成功")

                # 清空数据库
                if DATABASE_AVAILABLE:
                    db = get_database()
                    db.clear_all()
                    logger.info("数据库记忆已清空")
                    return jsonify({'success': True})
                else:
                    return jsonify({'success': False, 'error': '数据库模块不可用'})
            except Exception as e:
                logger.error(f"清空记忆失败: {e}")
                return jsonify({'success': False, 'error': str(e)})

        # ================== CRUD API v3.0 ==================

        @self.app.route('/api/node', methods=['POST'])
        def create_node():
            """创建新节点"""
            try:
                if not DATABASE_AVAILABLE:
                    return jsonify({'success': False, 'error': '数据库模块不可用'})

                data = request.json
                db = get_database()
                node_id = db.add_node(data)
                logger.info(f"创建节点: {node_id}")
                return jsonify({'success': True, 'node_id': node_id})
            except Exception as e:
                logger.error(f"创建节点失败: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/node/<int:node_id>', methods=['PUT'])
        def update_node(node_id):
            """更新节点"""
            try:
                if not DATABASE_AVAILABLE:
                    return jsonify({'success': False, 'error': '数据库模块不可用'})

                data = request.json
                db = get_database()
                success = db.update_node(node_id, data)
                if success:
                    logger.info(f"更新节点: {node_id}")
                    return jsonify({'success': True})
                else:
                    return jsonify({'success': False, 'message': '节点未找到'})
            except Exception as e:
                logger.error(f"更新节点失败: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/node/<int:node_id>', methods=['DELETE'])
        def delete_node(node_id):
            """删除节点"""
            try:
                if not DATABASE_AVAILABLE:
                    return jsonify({'success': False, 'error': '数据库模块不可用'})

                db = get_database()
                success = db.delete_node(node_id)
                if success:
                    logger.info(f"删除节点: {node_id}")
                    return jsonify({'success': True})
                else:
                    return jsonify({'success': False, 'message': '节点未找到'})
            except Exception as e:
                logger.error(f"删除节点失败: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/edge', methods=['POST'])
        def create_edge():
            """创建新边"""
            try:
                if not DATABASE_AVAILABLE:
                    return jsonify({'success': False, 'error': '数据库模块不可用'})

                data = request.json
                db = get_database()
                edge_id = db.add_edge(
                    source_id=data.get('source_node_id'),
                    target_id=data.get('target_node_id'),
                    action=data.get('action', []),
                    weight=data.get('weight', 1.0),
                    description=data.get('description', '')
                )
                logger.info(f"创建边: {data.get('source_node_id')} -> {data.get('target_node_id')}")
                return jsonify({'success': True, 'edge_id': edge_id})
            except Exception as e:
                logger.error(f"创建边失败: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/edge/<int:source_id>/<int:target_id>', methods=['DELETE'])
        def delete_edge(source_id, target_id):
            """删除边"""
            try:
                if not DATABASE_AVAILABLE:
                    return jsonify({'success': False, 'error': '数据库模块不可用'})

                db = get_database()
                success = db.delete_edge(source_id, target_id)
                if success:
                    logger.info(f"删除边: {source_id} -> {target_id}")
                    return jsonify({'success': True})
                else:
                    return jsonify({'success': False, 'message': '边未找到'})
            except Exception as e:
                logger.error(f"删除边失败: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/db/stats', methods=['GET'])
        def get_db_stats():
            """获取数据库统计信息"""
            try:
                if not DATABASE_AVAILABLE:
                    return jsonify({'success': False, 'error': '数据库模块不可用'})

                db = get_database()
                return jsonify({
                    'success': True,
                    'stats': {
                        'node_count': db.get_node_count(),
                        'edge_count': db.get_edge_count()
                    }
                })
            except Exception as e:
                logger.error(f"获取统计信息失败: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/db/import', methods=['POST'])
        def import_from_json_files():
            """从JSON文件导入数据到数据库"""
            try:
                if not DATABASE_AVAILABLE:
                    return jsonify({'success': False, 'error': '数据库模块不可用'})

                memory_data_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "deploy/logs/memory_data"
                )

                graph_path = os.path.join(memory_data_dir, "semantic_graph.json")
                metadata_path = os.path.join(memory_data_dir, "semantic_metadata.json")

                if not os.path.exists(graph_path):
                    return jsonify({'success': False, 'error': 'semantic_graph.json 不存在'})

                with open(graph_path, 'r') as f:
                    graph_json = json.load(f)

                metadata_json = None
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata_json = json.load(f)

                db = get_database()
                db.import_from_json(graph_json, metadata_json)

                return jsonify({
                    'success': True,
                    'message': f'导入完成: {db.get_node_count()} 节点, {db.get_edge_count()} 边'
                })
            except Exception as e:
                logger.error(f"导入失败: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/db/export', methods=['GET'])
        def export_to_json_files():
            """从数据库导出数据到JSON文件"""
            try:
                if not DATABASE_AVAILABLE:
                    return jsonify({'success': False, 'error': '数据库模块不可用'})

                db = get_database()
                graph_json, metadata_json = db.export_to_json()

                memory_data_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "deploy/logs/memory_data"
                )

                graph_path = os.path.join(memory_data_dir, "semantic_graph.json")
                metadata_path = os.path.join(memory_data_dir, "semantic_metadata.json")

                with open(graph_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_json, f, ensure_ascii=False, indent=2)

                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_json, f, ensure_ascii=False, indent=2)

                return jsonify({
                    'success': True,
                    'message': f'导出完成: {len(graph_json["nodes"])} 节点, {len(graph_json["links"])} 边'
                })
            except Exception as e:
                logger.error(f"导出失败: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/ws/connect', methods=['POST'])
        def ws_connect():
            data = request.json
            host = data.get('host', 'localhost')
            port = data.get('port', 9528)

            success, error_msg = self.inference_client.connect(host, port)
            return jsonify({
                'success': success,
                'message': '连接成功' if success else error_msg,
                'ws_url': f'ws://{host}:{port}' if success else None
            })

        @self.app.route('/api/ws/status', methods=['GET'])
        def ws_status():
            """获取WebSocket连接状态"""
            return jsonify({
                'connected': self.inference_client.connected,
                'ws_url': self.inference_client.ws_url
            })

        @self.app.route('/api/test/load_directory', methods=['POST'])
        def load_test_directory():
            try:
                data = request.json
                dir_path = data.get('path', '')

                if not dir_path:
                    return jsonify({'success': False, 'message': '请提供目录路径'})

                self.test_frames, instruction = self._scan_test_directory(dir_path)
                self.test_dir = dir_path
                self.current_instruction = instruction

                if not self.test_frames:
                    return jsonify({'success': False, 'message': '未找到图像文件'})

                return jsonify({
                    'success': True,
                    'frames': [{'index': i, 'name': f['name']} for i, f in enumerate(self.test_frames)],
                    'total': len(self.test_frames),
                    'instruction': instruction
                })

            except Exception as e:
                logger.error(f"加载目录错误: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/test/get_frame/<int:index>')
        def get_frame(index):
            try:
                if index < 0 or index >= len(self.test_frames):
                    return jsonify({'success': False, 'message': '帧索引越界'})

                frame_data = self.test_frames[index]
                img_path = frame_data['path']

                if os.path.exists(img_path):
                    with open(img_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                        ext = os.path.splitext(img_path)[1].lower()
                        mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                        return jsonify({
                            'success': True,
                            'image': f'data:{mime_type};base64,{img_data}',
                            'frame_index': index
                        })

                return jsonify({'success': False, 'message': '图像文件不存在'})

            except Exception as e:
                logger.error(f"获取帧错误: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/frame/process', methods=['POST'])
        def process_frame():
            try:
                data = request.json
                frame_index = data.get('frame_index', 0)
                instruction = data.get('instruction')
                memory_enabled = data.get('memory_enabled', True)
                is_first_frame = data.get('is_first_frame', False)

                if frame_index < 0 or frame_index >= len(self.test_frames):
                    return jsonify({'success': False, 'message': '帧索引越界'})

                frame_data = self.test_frames[frame_index]
                rgb_path = frame_data.get('front_1') or frame_data.get('path')

                # 获取环视图路径（如果有）
                surround_images = {}
                for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
                    if frame_data.get(cam_id):
                        surround_images[cam_id] = frame_data[cam_id]

                # 确定使用的instruction (使用comprehensive_memory_test.py中相同的默认指令)
                use_instruction = instruction or self.current_instruction or "Walk straight ahead, then turn left and stop at the sofa."
                # 第一帧使用真实指令，后续帧传"None"
                send_instruction = use_instruction if is_first_frame else "None"

                logger.info(f"处理帧 {frame_index}: instruction={send_instruction}, "
                           f"front_1={os.path.basename(rgb_path) if rgb_path else None}, "
                           f"环视图数量={len(surround_images)}")

                result = {
                    'success': True,
                    'frame_index': frame_index,
                    'output_action': None,
                    'pixel_target': None,
                    'inference_time': 0,
                    'node_id': None,
                    'is_new': False
                }

                # 如果连接了推理服务，调用推理
                if self.inference_client.connected:
                    start_time = time.time()
                    response = self.inference_client.call_inference(
                        instruction=send_instruction,
                        rgb_path=rgb_path,
                        pts=int(time.time() * 1000),
                        surround_images=surround_images if surround_images else None
                    )
                    elapsed = time.time() - start_time

                    if response and response.get('status') == 'success':
                        result['output_action'] = response.get('output_action')
                        result['pixel_target'] = response.get('pixel_target')
                        result['inference_time'] = response.get('inference_time', elapsed)
                        result['task_status'] = response.get('task_status', 'executing')

                        # 从推理服务获取记忆信息
                        memory_info = response.get('memory_info', {})
                        if memory_info:
                            result['node_id'] = memory_info.get('node_id')
                            result['is_new'] = memory_info.get('is_new_node', False)
                            result['is_keyframe'] = memory_info.get('is_keyframe', False)
                            result['topo_stats'] = memory_info.get('topo_stats', {})
                            logger.info(f"帧 {frame_index} 推理完成: node_id={result['node_id']}, is_new={result['is_new']}, "
                                       f"nodes={result['topo_stats'].get('total_nodes', 0)}, "
                                       f"edges={result['topo_stats'].get('total_edges', 0)}")
                        else:
                            logger.info(f"帧 {frame_index} 推理完成: action={result['output_action']}, status={result['task_status']}")
                    else:
                        error_msg = response.get('message', '未知错误') if response else '无响应'
                        logger.warning(f"帧 {frame_index} 推理失败: {error_msg}")
                else:
                    # 模拟推理结果
                    result['output_action'] = [1, 1, 1]  # 模拟前进
                    result['pixel_target'] = [0.5, 0.5]
                    result['inference_time'] = 0.1
                    logger.info(f"帧 {frame_index} 模拟推理完成")

                    # 本地模式：使用本地topo_map
                    if memory_enabled and self.topo_map is not None and self.config is not None:
                        try:
                            # 生成特征（实际应用中应使用特征提取器）
                            feature = np.random.randn(self.config.feature_dim).astype('float32')
                            feature = feature / np.linalg.norm(feature)

                            # 是否为关键帧（每5帧或检测到STOP）
                            is_keyframe = (frame_index % 5 == 0)
                            if result['output_action'] and 0 in result['output_action']:
                                is_keyframe = True

                            node_id, is_new, _ = self.topo_map.add_observation(
                                visual_feature=feature,
                                surround_features={},
                                action_from_prev=result['output_action'] if frame_index > 0 else None,
                                is_keyframe=is_keyframe,
                                semantic_labels=[f'帧_{frame_index}'],
                                scene_description=f'帧 {frame_index}'
                            )

                            result['node_id'] = node_id
                            result['is_new'] = is_new
                            logger.info(f"记忆节点(本地): {node_id}, 新建: {is_new}, 关键帧: {is_keyframe}")

                        except Exception as e:
                            logger.error(f"添加记忆节点失败: {e}")

                return jsonify(result)

            except Exception as e:
                logger.error(f"处理帧错误: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/agent/reset', methods=['POST'])
        def reset_agent():
            """
            v2.5.2: 支持keep_memory参数
            - keep_memory=True: 只重置Agent，保留拓扑图记忆（用于多场景合并）
            - keep_memory=False: 重置Agent和记忆（默认行为）
            """
            try:
                data = request.json or {}
                keep_memory = data.get('keep_memory', False)

                if self.inference_client.connected:
                    self.inference_client.reset_agent(keep_memory=keep_memory)

                if keep_memory:
                    logger.info("Agent已重置，记忆已保留（多场景合并模式）")
                else:
                    logger.info("Agent和记忆已重置")

                return jsonify({'success': True, 'keep_memory': keep_memory})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/memory/start', methods=['POST'])
        def start_memory():
            """开始记忆记录"""
            try:
                data = request.json or {}
                instruction = data.get('instruction') or self.current_instruction or "Walk straight ahead, then turn left and stop at the sofa."

                if self.inference_client.connected:
                    result = self.inference_client.start_memory_recording(instruction)
                    if result and result.get('status') == 'success':
                        logger.info(f"开始记忆记录: {instruction}")
                        return jsonify({'success': True, 'message': '记忆记录已开始', 'instruction': instruction})
                    else:
                        error_msg = result.get('message', '未知错误') if result else '无响应'
                        return jsonify({'success': False, 'error': error_msg})
                else:
                    logger.warning("未连接推理服务，记忆记录模拟开始")
                    return jsonify({'success': True, 'message': '模拟记忆记录开始', 'instruction': instruction})
            except Exception as e:
                logger.error(f"开始记忆记录失败: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/memory/stop', methods=['POST'])
        def stop_memory():
            """停止记忆记录"""
            try:
                if self.inference_client.connected:
                    result = self.inference_client.stop_memory_recording()
                    if result and result.get('status') == 'success':
                        logger.info("停止记忆记录")
                        memory_info = result.get('memory_info', {})
                        return jsonify({
                            'success': True,
                            'message': '记忆记录已停止',
                            'memory_info': memory_info
                        })
                    else:
                        error_msg = result.get('message', '未知错误') if result else '无响应'
                        return jsonify({'success': False, 'error': error_msg})
                else:
                    logger.warning("未连接推理服务，记忆记录模拟停止")
                    return jsonify({'success': True, 'message': '模拟记忆记录停止'})
            except Exception as e:
                logger.error(f"停止记忆记录失败: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/path/plan', methods=['POST'])
        def plan_path():
            """路径规划API - 使用Dijkstra算法从图数据计算最短路径"""
            try:
                data = request.json
                start_node = data.get('start')
                goal_node = data.get('goal')

                if start_node is None or goal_node is None:
                    return jsonify({'success': False, 'message': '请提供节点ID'})

                logger.info(f"路径规划请求: start={start_node}, goal={goal_node}")

                # 从推理服务获取路径规划（如果连接）
                if self.inference_client.connected:
                    response = self.inference_client.send_command('plan_path', start=start_node, goal=goal_node)
                    if response and response.get('status') == 'success':
                        path_data = response.get('data', {})
                        if path_data.get('path'):
                            logger.info(f"从推理服务获取路径: {path_data['path']}")
                            return jsonify({
                                'success': True,
                                'path': path_data['path'],
                                'total_distance': path_data.get('total_distance', len(path_data['path']) - 1),
                                'total_steps': len(path_data['path'])
                            })

                # 从本地图数据计算路径
                graph_data = self._load_graph_data()
                if graph_data and graph_data.get('nodes') and graph_data.get('edges'):
                    path_result = self._dijkstra_shortest_path(graph_data, start_node, goal_node)
                    if path_result['success']:
                        logger.info(f"本地路径规划成功: {path_result['path']}")
                        return jsonify(path_result)
                    else:
                        return jsonify({'success': False, 'message': path_result.get('message', '未找到路径')})

                # 本地topo_map
                if self.topo_map is not None:
                    result = self.topo_map.plan_shortest_path(start_node, goal_node)
                    if result.success:
                        return jsonify({
                            'success': True,
                            'path': result.path,
                            'total_distance': result.total_distance,
                            'total_steps': result.total_steps
                        })
                    return jsonify({'success': False, 'message': '未找到路径'})

                return jsonify({'success': False, 'message': '无可用图数据'})

            except Exception as e:
                logger.error(f"路径规划错误: {e}", exc_info=True)
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/vpr/identify', methods=['POST'])
        def vpr_identify():
            """VPR位置识别API - 上传环视图片，识别匹配的记忆节点"""
            try:
                # 支持multipart/form-data和JSON两种方式
                if request.content_type and 'multipart/form-data' in request.content_type:
                    images = {}
                    for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
                        if cam_id in request.files:
                            file = request.files[cam_id]
                            img_data = file.read()
                            img = Image.open(BytesIO(img_data))
                            images[cam_id] = np.array(img)
                else:
                    data = request.json
                    images = {}
                    for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
                        if cam_id in data and data[cam_id]:
                            img_b64 = data[cam_id]
                            if ',' in img_b64:
                                img_b64 = img_b64.split(',')[1]
                            img_data = base64.b64decode(img_b64)
                            img = Image.open(BytesIO(img_data))
                            images[cam_id] = np.array(img)

                if not images:
                    return jsonify({'success': False, 'message': '请上传至少一张环视图片'})

                logger.info(f"VPR识别请求: 收到 {len(images)} 张图片")

                # 方法1: 通过推理服务进行VPR识别
                if self.inference_client.connected:
                    # 将图片转为base64发送
                    img_b64_dict = {}
                    for cam_id, img_arr in images.items():
                        img_pil = Image.fromarray(img_arr)
                        buffer = BytesIO()
                        img_pil.save(buffer, format='JPEG')
                        img_b64_dict[cam_id] = base64.b64encode(buffer.getvalue()).decode('utf-8')

                    response = self.inference_client.send_command('vpr_identify', images=img_b64_dict)
                    if response and response.get('status') == 'success':
                        result_data = response.get('data', {})
                        return jsonify({
                            'success': True,
                            'matched_node': result_data.get('matched_node'),
                            'similarity': result_data.get('similarity', 0),
                            'top_matches': result_data.get('top_matches', [])
                        })

                # 方法2: 本地VPR识别（使用LongCLIP + topo_map.vpr）
                if self.feature_extractor is not None and self.topo_map is not None:
                    logger.info("使用本地VPR进行位置识别...")
                    try:
                        # 提取各相机的特征
                        query_features = {}
                        for cam_id, img_arr in images.items():
                            feat = self.feature_extractor.extract_feature(img_arr)
                            query_features[cam_id] = feat

                        # 使用topo_map的VPR进行搜索
                        if query_features and self.topo_map.vpr.get_size() > 0:
                            # 首先尝试多视角搜索
                            results = self.topo_map.vpr.search_multi_view(query_features, k=5)
                            if results:
                                best_match = results[0]
                                top_matches = [
                                    {
                                        'node_id': r.node_id,
                                        'similarity': r.weighted_similarity,
                                        'voting_score': r.voting_score
                                    }
                                    for r in results[:5]
                                ]
                                return jsonify({
                                    'success': True,
                                    'matched_node': best_match.node_id,
                                    'similarity': best_match.weighted_similarity,
                                    'top_matches': top_matches,
                                    'source': 'local_vpr_multi_view'
                                })

                            # 回退: 多视角索引为空，使用主索引搜索（融合多个相机的特征）
                            logger.info("多视角索引为空，使用主索引搜索...")
                            # 将多个相机特征融合为一个特征
                            feat_list = list(query_features.values())
                            if feat_list:
                                fused_feature = np.mean(feat_list, axis=0)
                                search_results = self.topo_map.vpr.search(fused_feature, k=5)
                                if search_results:
                                    best_node_id, best_sim = search_results[0]
                                    top_matches = [
                                        {'node_id': node_id, 'similarity': sim}
                                        for node_id, sim in search_results[:5]
                                    ]
                                    return jsonify({
                                        'success': True,
                                        'matched_node': best_node_id,
                                        'similarity': best_sim,
                                        'top_matches': top_matches,
                                        'source': 'local_vpr_fused'
                                    })

                            return jsonify({'success': False, 'message': 'VPR搜索未找到匹配节点'})
                        else:
                            return jsonify({'success': False, 'message': 'VPR索引为空，请先进行导航构建记忆'})
                    except Exception as local_e:
                        logger.error(f"本地VPR识别失败: {local_e}", exc_info=True)
                        return jsonify({'success': False, 'message': f'本地VPR识别失败: {str(local_e)}'})

                return jsonify({'success': False, 'message': 'VPR功能不可用（推理服务未连接且本地VPR未初始化）'})

            except Exception as e:
                logger.error(f"VPR识别错误: {e}", exc_info=True)
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/semantic/search', methods=['POST'])
        def semantic_search():
            """语义检索API - 根据语言描述匹配记忆节点"""
            try:
                data = request.json
                query = data.get('query', '').strip()

                if not query:
                    return jsonify({'success': False, 'message': '请输入搜索描述'})

                logger.info(f"语义检索请求: query='{query}'")

                # 通过推理服务进行语义检索
                if self.inference_client.connected:
                    response = self.inference_client.send_command('semantic_search', query=query)
                    if response and response.get('status') == 'success':
                        result_data = response.get('data', {})
                        return jsonify({
                            'success': True,
                            'matched_nodes': result_data.get('matched_nodes', []),
                            'best_match': result_data.get('best_match')
                        })

                # 本地语义检索（从图数据）
                graph_data = self._load_graph_data()
                if graph_data and graph_data.get('nodes'):
                    matched_nodes = self._local_semantic_search(graph_data['nodes'], query)
                    if matched_nodes:
                        return jsonify({
                            'success': True,
                            'matched_nodes': matched_nodes,
                            'best_match': matched_nodes[0] if matched_nodes else None
                        })

                return jsonify({'success': False, 'message': '未找到匹配节点'})

            except Exception as e:
                logger.error(f"语义检索错误: {e}", exc_info=True)
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/smart_path/plan', methods=['POST'])
        def smart_path_plan():
            """智能路径规划API - 结合VPR识别起点和语义检索终点"""
            try:
                # 获取起点（VPR识别或指定节点）
                start_node = None
                goal_node = None

                if request.content_type and 'multipart/form-data' in request.content_type:
                    # 从图片识别起点
                    images = {}
                    for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
                        if cam_id in request.files:
                            file = request.files[cam_id]
                            img_data = file.read()
                            img = Image.open(BytesIO(img_data))
                            images[cam_id] = np.array(img)

                    # 支持多种参数名: goal_query, destination
                    goal_query = request.form.get('goal_query') or request.form.get('destination', '')
                    start_node_manual = request.form.get('start_node')
                    goal_node_manual = request.form.get('goal_node')
                else:
                    data = request.json or {}
                    images = {}
                    for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
                        if cam_id in data and data[cam_id]:
                            img_b64 = data[cam_id]
                            if ',' in img_b64:
                                img_b64 = img_b64.split(',')[1]
                            img_data = base64.b64decode(img_b64)
                            img = Image.open(BytesIO(img_data))
                            images[cam_id] = np.array(img)

                    # 支持多种参数名: goal_query, destination
                    goal_query = data.get('goal_query') or data.get('destination', '')
                    start_node_manual = data.get('start_node')
                    goal_node_manual = data.get('goal_node')

                # 确定起点
                if start_node_manual is not None:
                    start_node = int(start_node_manual)
                elif images:
                    # 方法1: 通过推理服务VPR识别起点
                    if self.inference_client.connected:
                        img_b64_dict = {}
                        for cam_id, img_arr in images.items():
                            img_pil = Image.fromarray(img_arr)
                            buffer = BytesIO()
                            img_pil.save(buffer, format='JPEG')
                            img_b64_dict[cam_id] = base64.b64encode(buffer.getvalue()).decode('utf-8')

                        response = self.inference_client.send_command('vpr_identify', images=img_b64_dict)
                        if response and response.get('status') == 'success':
                            result_data = response.get('data', {})
                            start_node = result_data.get('matched_node')

                    # 方法2: 本地VPR识别起点（fallback）
                    if start_node is None and self.feature_extractor is not None and self.topo_map is not None:
                        logger.info("使用本地VPR识别起点...")
                        try:
                            # 提取各相机的特征
                            query_features = {}
                            for cam_id, img_arr in images.items():
                                feat = self.feature_extractor.extract_feature(img_arr)
                                query_features[cam_id] = feat

                            # 使用topo_map的VPR进行多视角搜索
                            if query_features and self.topo_map.vpr.get_size() > 0:
                                results = self.topo_map.vpr.search_multi_view(query_features, k=1)
                                if results:
                                    start_node = results[0].node_id
                                    logger.info(f"本地VPR识别起点成功: node_id={start_node}, similarity={results[0].weighted_similarity:.3f}")
                        except Exception as local_e:
                            logger.warning(f"本地VPR识别起点失败: {local_e}")

                # 确定终点
                if goal_node_manual is not None:
                    goal_node = int(goal_node_manual)
                elif goal_query:
                    # 通过语义检索确定终点
                    # 先尝试推理服务
                    if self.inference_client.connected:
                        response = self.inference_client.send_command('semantic_search', query=goal_query)
                        if response and response.get('status') == 'success':
                            result_data = response.get('data', {})
                            best_match = result_data.get('best_match')
                            if best_match:
                                goal_node = best_match.get('id') or best_match.get('node_id')

                    # 如果推理服务未返回结果，回退到本地检索
                    if goal_node is None:
                        graph_data = self._load_graph_data()
                        if graph_data and graph_data.get('nodes'):
                            matched_nodes = self._local_semantic_search(graph_data['nodes'], goal_query)
                            if matched_nodes:
                                goal_node = matched_nodes[0].get('id')

                logger.info(f"智能路径规划: start_node={start_node}, goal_node={goal_node}, goal_query='{goal_query}'")

                if start_node is None:
                    return jsonify({'success': False, 'message': '无法确定起点（请上传图片或指定起点节点）'})
                if goal_node is None:
                    return jsonify({'success': False, 'message': '无法确定终点（请输入目标描述或指定终点节点）'})

                # 计算路径
                graph_data = self._load_graph_data()
                if graph_data:
                    path_result = self._dijkstra_shortest_path(graph_data, start_node, goal_node)
                    if path_result['success']:
                        path_result['start_node'] = start_node
                        path_result['goal_node'] = goal_node
                        path_result['goal_query'] = goal_query
                        return jsonify(path_result)

                return jsonify({'success': False, 'message': '路径规划失败'})

            except Exception as e:
                logger.error(f"智能路径规划错误: {e}", exc_info=True)
                return jsonify({'success': False, 'error': str(e)})

    def _load_graph_data(self) -> Optional[Dict]:
        """加载图数据（从推理服务或本地文件）"""
        # 优先使用缓存
        if self.loaded_graph_data:
            return self.loaded_graph_data

        # 从推理服务获取
        if self.inference_client.connected:
            response = self.inference_client.send_command('get_graph')
            if response and response.get('status') == 'success':
                data = response.get('data', {})
                if data.get('nodes'):
                    self.loaded_graph_data = data
                    return data

        # 从semantic_graph.json文件加载
        semantic_graph_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "deploy/logs/memory_data/semantic_graph.json"
        )
        if os.path.exists(semantic_graph_path):
            try:
                with open(semantic_graph_path, 'r') as f:
                    graph_data = json.load(f)

                # 转换为标准格式
                nodes = []
                for node in graph_data.get("nodes", []):
                    nodes.append({
                        'id': node.get('id'),
                        'is_keyframe': True if node.get('description') else False,
                        'semantic_labels': node.get('labels', []),
                        'scene_description': node.get('description', '')
                    })
                edges = []
                for link in graph_data.get("links", []):
                    edges.append({
                        'from': link.get('source'),
                        'to': link.get('target'),
                        'weight': link.get('weight', 1.0)
                    })

                result = {'nodes': nodes, 'edges': edges}
                self.loaded_graph_data = result
                return result
            except Exception as e:
                logger.warning(f"加载semantic_graph.json失败: {e}")

        return None

    def _dijkstra_shortest_path(self, graph_data: Dict, start: int, goal: int) -> Dict:
        """Dijkstra最短路径算法"""
        import heapq

        nodes = {n['id']: n for n in graph_data.get('nodes', [])}
        edges = graph_data.get('edges', [])

        if start not in nodes:
            return {'success': False, 'message': f'起点节点 {start} 不存在'}
        if goal not in nodes:
            return {'success': False, 'message': f'终点节点 {goal} 不存在'}

        # 构建邻接表（双向图）
        adj = {n['id']: [] for n in graph_data.get('nodes', [])}
        for edge in edges:
            from_node = edge['from']
            to_node = edge['to']
            weight = edge.get('weight', 1.0)
            if from_node in adj:
                adj[from_node].append((to_node, weight))
            if to_node in adj:
                adj[to_node].append((from_node, weight))

        # Dijkstra算法
        dist = {n: float('inf') for n in nodes}
        prev = {n: None for n in nodes}
        dist[start] = 0

        pq = [(0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == goal:
                break
            for v, w in adj.get(u, []):
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))

        # 重建路径
        if dist[goal] == float('inf'):
            return {'success': False, 'message': '未找到路径'}

        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()

        # 获取路径上每个节点的详细信息
        waypoints = []
        for node_id in path:
            node_info = nodes.get(node_id, {})
            waypoints.append({
                'node_id': node_id,
                'is_keyframe': node_info.get('is_keyframe', False),
                'semantic_labels': node_info.get('semantic_labels', []),
                'scene_description': node_info.get('scene_description', '')
            })

        return {
            'success': True,
            'path': path,
            'total_distance': dist[goal],
            'total_steps': len(path),
            'waypoints': waypoints
        }

    def _local_semantic_search(self, nodes: List[Dict], query: str) -> List[Dict]:
        """本地语义检索 - 简单的关键词匹配"""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = []
        for node in nodes:
            score = 0
            node_id = node.get('id')
            labels = node.get('semantic_labels', [])
            description = node.get('scene_description', '')

            # 检查标签匹配
            for label in labels:
                label_lower = label.lower()
                if query_lower in label_lower or label_lower in query_lower:
                    score += 2
                for word in query_words:
                    if word in label_lower:
                        score += 1

            # 检查描述匹配
            if description:
                desc_lower = description.lower()
                if query_lower in desc_lower:
                    score += 3
                for word in query_words:
                    if word in desc_lower:
                        score += 1

            if score > 0:
                results.append({
                    'id': node_id,
                    'score': score,
                    'semantic_labels': labels,
                    'scene_description': description
                })

        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:10]

    def _generate_mock_graph(self) -> Dict:
        """生成模拟图数据"""
        nodes = []
        for i in range(10):
            nodes.append({
                'id': i,
                'label': f'N{i}',
                'is_keyframe': i % 3 == 0,
                'visit_count': np.random.randint(1, 5),
                'semantic_labels': ['测试'],
                'scene_description': f'测试场景 {i}',
                'is_current': i == 5
            })

        edges = []
        for i in range(9):
            edges.append({'from': i, 'to': i + 1, 'weight': 1.0})
        edges.append({'from': 0, 'to': 5, 'weight': 2.5})
        edges.append({'from': 3, 'to': 8, 'weight': 2.0})

        return {'nodes': nodes, 'edges': edges, 'current_node': 5}

    def run(self, ws_host: str = 'localhost', ws_port: int = 9528):
        """启动服务器

        Args:
            ws_host: 推理服务主机地址
            ws_port: 推理服务端口（默认9528，ws_proxy_with_memory.py的端口）
        """
        if self.app is None:
            logger.error("Flask未安装")
            return

        # 启动前自动连接推理服务
        logger.info(f"尝试连接推理服务 ws://{ws_host}:{ws_port}...")
        success, msg = self.inference_client.connect(ws_host, ws_port)
        if success:
            logger.info(f"✓ 已连接推理服务: ws://{ws_host}:{ws_port}")
        else:
            logger.warning(f"✗ 推理服务连接失败: {msg}")
            logger.warning(f"  请确保 deploy/ws_proxy_with_memory.py 已启动")

        logger.info(f"启动记忆可视化服务器 v2.3，端口: {self.port}")
        logger.info(f"请访问: http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)


def main():
    parser = argparse.ArgumentParser(description='记忆导航可视化服务器 v2.3')
    parser.add_argument('--port', type=int, default=9530, help='Web服务器端口')
    parser.add_argument('--ws-host', type=str, default='localhost', help='推理服务主机地址')
    parser.add_argument('--ws-port', type=int, default=9528, help='推理服务端口（默认9528, ws_proxy_with_memory.py）')
    args = parser.parse_args()

    server = MemoryVisualizationServer(port=args.port)
    server.run(ws_host=args.ws_host, ws_port=args.ws_port)


if __name__ == '__main__':
    main()
