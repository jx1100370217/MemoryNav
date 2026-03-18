#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 可视化服务器 v4.0

功能:
1. 拓扑图可视化 (美化版)
2. VPR图片上传识别起点
3. 语义描述匹配检索终点
4. 智能路径规划
5. 数据库管理 (增删改查)

使用方式:
    conda activate internvla
    python scripts/memory_visualization_server.py --port 9530
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# 设置项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ===== 加载 VPR 统一配置 =====
from deploy.memory_nav.vpr_config_loader import load_vpr_config, get_threshold
_vpr_cfg = load_vpr_config()
VPR_METHOD = _vpr_cfg['vpr_method']
VPR_DEVICE = _vpr_cfg['device']
VPR_THRESHOLD = get_threshold(_vpr_cfg)
# ==============================

import numpy as np

try:
    from flask import Flask, render_template_string, jsonify, request, send_file
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask 未安装。请执行: pip install flask flask-cors")

# 导入 memory_nav 模块
try:
    from deploy.memory_nav import (
        MemoryNavigator, MemoryBuilder, MemoryGraph, MemoryVPR,
        MemoryNode, MemoryEdge, NavigationPlan, VPRResult,
    )
    MEMORY_NAV_AVAILABLE = True
except ImportError as e:
    MEMORY_NAV_AVAILABLE = False
    print(f"memory_nav 模块不可用: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# HTML模板 v4.0 - 美化版 + 数据库管理
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧭</text></svg>">
    <title>🧭 MemoryNav - 记忆导航可视化系统 v4.0</title>
    <script src="/static/vis-network.min.js"></script>
    <style>
        :root {
            --bg-dark: #0a0a1a;
            --bg-panel: #12122a;
            --bg-card: #1a1a3e;
            --bg-hover: #252550;
            --accent: #00d4ff;
            --accent-glow: rgba(0, 212, 255, 0.3);
            --success: #00ff88;
            --warning: #ffaa00;
            --danger: #ff4466;
            --text: #e8e8f0;
            --text-dim: #8888aa;
            --border: #2a2a5a;
            --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-2: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; 
            background: var(--bg-dark); 
            color: var(--text);
            min-height: 100vh;
        }
        
        /* 顶部导航栏 */
        .navbar {
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }
        
        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 20px;
            font-weight: 600;
        }
        
        .navbar-brand span {
            background: var(--gradient-2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .nav-tabs {
            display: flex;
            gap: 8px;
        }
        
        .nav-tab {
            padding: 8px 20px;
            border-radius: 20px;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--text-dim);
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
        }
        
        .nav-tab:hover {
            border-color: var(--accent);
            color: var(--accent);
        }
        
        .nav-tab.active {
            background: var(--accent);
            border-color: var(--accent);
            color: var(--bg-dark);
            font-weight: 600;
        }
        
        .nav-status {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 15px;
            background: var(--bg-card);
            font-size: 13px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* 主容器 */
        .main-container {
            display: flex;
            height: calc(100vh - 60px);
            margin-top: 60px;
        }
        
        /* 导航页面 */
        .page { display: none; width: 100%; }
        .page.active { display: flex; }
        
        /* 导航页布局 */
        #page-nav {
            display: none;
        }
        #page-nav.active {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 16px;
            padding: 16px;
        }
        
        .graph-container {
            background: var(--bg-panel);
            border-radius: 16px;
            padding: 16px;
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        
        .graph-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-2);
        }
        
        #graph { width: 100%; height: 100%; }
        
        .graph-controls {
            position: absolute;
            bottom: 20px;
            left: 20px;
            display: flex;
            gap: 8px;
        }
        
        .graph-btn {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            border: 1px solid var(--border);
            background: var(--bg-card);
            color: var(--text);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }
        
        .graph-btn:hover {
            background: var(--accent);
            color: var(--bg-dark);
            border-color: var(--accent);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px var(--accent-glow);
        }
        
        /* 侧边栏 */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 12px;
            overflow-y: auto;
            padding-right: 4px;
        }
        
        .sidebar::-webkit-scrollbar {
            width: 6px;
        }
        
        .sidebar::-webkit-scrollbar-track {
            background: var(--bg-dark);
            border-radius: 3px;
        }
        
        .sidebar::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        
        /* 卡片 */
        .card {
            background: var(--bg-panel);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid var(--border);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: var(--accent);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        
        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }
        
        .card-title {
            font-size: 15px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .card-title i {
            font-size: 18px;
        }
        
        /* 统计卡片 */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        
        .stat-item {
            background: var(--bg-card);
            padding: 12px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid transparent;
            transition: all 0.3s ease;
        }
        
        .stat-item:hover {
            border-color: var(--accent);
            transform: translateY(-2px);
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            background: var(--gradient-2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stat-label {
            font-size: 12px;
            color: var(--text-dim);
            margin-top: 4px;
        }
        
        /* 表单控件 */
        .form-group {
            margin-bottom: 12px;
        }
        
        .form-label {
            display: block;
            font-size: 12px;
            color: var(--text-dim);
            margin-bottom: 6px;
        }
        
        input, select, textarea {
            width: 100%;
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--bg-card);
            color: var(--text);
            font-size: 14px;
            transition: all 0.3s ease;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        
        input::placeholder {
            color: var(--text-dim);
        }
        
        /* 按钮 */
        .btn {
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: var(--gradient-2);
            color: var(--bg-dark);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px var(--accent-glow);
        }
        
        .btn-secondary {
            background: var(--bg-card);
            color: var(--text);
            border: 1px solid var(--border);
        }
        
        .btn-secondary:hover {
            border-color: var(--accent);
            color: var(--accent);
        }
        
        .btn-danger {
            background: var(--danger);
            color: white;
        }
        
        .btn-danger:hover {
            filter: brightness(1.1);
        }
        
        .btn-success {
            background: var(--success);
            color: var(--bg-dark);
        }
        
        .btn-block {
            width: 100%;
        }
        
        .btn-sm {
            padding: 6px 12px;
            font-size: 12px;
        }
        
        .btn-group {
            display: flex;
            gap: 8px;
        }
        
        /* 标签页 */
        .tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 12px;
            background: var(--bg-card);
            padding: 4px;
            border-radius: 8px;
        }
        
        .tab {
            flex: 1;
            padding: 8px;
            border-radius: 6px;
            border: none;
            background: transparent;
            color: var(--text-dim);
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s ease;
        }
        
        .tab:hover {
            color: var(--text);
        }
        
        .tab.active {
            background: var(--accent);
            color: var(--bg-dark);
            font-weight: 600;
        }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        /* 上传区域 */
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: var(--bg-card);
        }
        
        .upload-zone:hover {
            border-color: var(--accent);
            background: var(--bg-hover);
        }
        
        .upload-zone input { display: none; }
        
        .upload-icon {
            font-size: 32px;
            margin-bottom: 8px;
        }
        
        .upload-text {
            font-size: 14px;
            color: var(--text-dim);
        }
        
        .preview-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-top: 12px;
        }
        
        .preview-img {
            width: 100%;
            height: 70px;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        
        /* 4相机上传区 */
        .camera-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 8px;
        }
        
        .camera-slot {
            border: 2px dashed var(--border);
            border-radius: 8px;
            padding: 8px 4px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: var(--bg-card);
            min-height: 90px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        
        .camera-slot:hover {
            border-color: var(--accent);
            background: var(--bg-hover);
        }
        
        .camera-slot.has-image {
            border-color: var(--success);
            border-style: solid;
        }
        
        .camera-slot input { display: none; }
        
        .camera-slot .cam-label {
            font-size: 11px;
            font-weight: bold;
            color: var(--accent);
            margin-bottom: 4px;
        }
        
        .camera-slot .cam-angle {
            font-size: 10px;
            color: var(--text-dim);
        }
        
        .camera-slot .cam-preview {
            width: 100%;
            height: 55px;
            object-fit: cover;
            border-radius: 4px;
            margin-top: 4px;
        }
        
        .camera-slot .cam-icon {
            font-size: 20px;
            margin: 4px 0;
        }
        
        .camera-slot .cam-clear {
            position: absolute;
            top: 2px;
            right: 4px;
            font-size: 12px;
            cursor: pointer;
            color: var(--danger);
            opacity: 0.7;
            z-index: 2;
        }
        
        .camera-slot .cam-clear:hover {
            opacity: 1;
        }
        
        .robot-diagram {
            text-align: center;
            margin: 6px 0;
            padding: 6px;
            background: var(--bg-hover);
            border-radius: 8px;
            font-size: 11px;
            color: var(--text-dim);
        }
        
        .robot-diagram .robot-icon {
            font-size: 18px;
        }
        
        /* 确认弹窗 */
        .confirm-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.6);
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .confirm-box {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            min-width: 320px;
            max-width: 400px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        }
        
        .confirm-box .confirm-msg {
            font-size: 15px;
            margin-bottom: 20px;
            line-height: 1.5;
        }
        
        .confirm-box .confirm-actions {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        
        .confirm-box .btn-cancel {
            padding: 8px 20px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--bg-hover);
            color: var(--text);
            cursor: pointer;
        }
        
        .confirm-box .btn-confirm-danger {
            padding: 8px 20px;
            border-radius: 8px;
            border: none;
            background: var(--danger);
            color: white;
            cursor: pointer;
            font-weight: bold;
        }
        
                /* 结果框 */
        .result-box {
            background: var(--bg-card);
            border-radius: 10px;
            padding: 12px;
            margin-top: 12px;
        }
        
        .result-success {
            border-left: 3px solid var(--success);
        }
        
        .result-error {
            border-left: 3px solid var(--danger);
        }
        
        .result-title {
            font-weight: 600;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        /* 路径步骤 */
        .path-steps {
            margin-top: 12px;
        }
        
        .path-step {
            background: var(--bg-card);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 8px;
            border-left: 3px solid var(--accent);
            position: relative;
        }
        
        .path-step::before {
            content: '';
            position: absolute;
            left: -2px;
            top: 100%;
            width: 2px;
            height: 8px;
            background: var(--accent);
        }
        
        .path-step:last-child::before {
            display: none;
        }
        
        .step-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        
        .step-number {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: var(--accent);
            color: var(--bg-dark);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 700;
        }
        
        .step-info {
            flex: 1;
        }
        
        .step-from, .step-to {
            font-size: 13px;
        }
        
        .step-arrow {
            color: var(--accent);
            margin: 0 6px;
        }
        
        .step-angle {
            font-size: 12px;
            color: var(--text-dim);
            background: var(--bg-panel);
            padding: 2px 8px;
            border-radius: 10px;
        }
        
        .step-image {
            width: 100%;
            max-height: 120px;
            object-fit: contain;
            border-radius: 8px;
            margin-top: 8px;
        }
        
        /* 节点详情 */
        .node-detail-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }
        
        .node-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: var(--gradient-1);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }
        
        .node-name {
            font-size: 18px;
            font-weight: 600;
        }
        
        .node-id {
            font-size: 12px;
            color: var(--text-dim);
        }
        
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin: 12px 0;
        }
        
        .camera-img {
            width: 100%;
            height: 80px;
            object-fit: cover;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .camera-img:hover {
            transform: scale(1.05);
        }
        
        .neighbor-list {
            margin-top: 12px;
        }
        
        .neighbor-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 12px;
            background: var(--bg-card);
            border-radius: 8px;
            margin-bottom: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .neighbor-item:hover {
            background: var(--bg-hover);
            transform: translateX(4px);
        }
        
        .neighbor-name {
            font-size: 14px;
        }
        
        .neighbor-angle {
            font-size: 12px;
            color: var(--accent);
        }
        
        /* 数据库管理页面 */
        #page-db.active {
            display: block;
            padding: 16px;
        }
        
        .db-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            height: calc(100vh - 92px);
        }
        
        .db-panel {
            background: var(--bg-panel);
            border-radius: 16px;
            padding: 16px;
            border: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .db-panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }
        
        .db-panel-title {
            font-size: 16px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .db-toolbar {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }
        
        .search-box {
            flex: 1;
            position: relative;
        }
        
        .search-box input {
            padding-left: 36px;
        }
        
        .search-icon {
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-dim);
        }
        
        /* 数据表格 */
        .db-table-container {
            flex: 1;
            overflow-y: auto;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        
        .db-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .db-table th {
            position: sticky;
            top: 0;
            background: var(--bg-card);
            padding: 12px;
            text-align: left;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-dim);
            text-transform: uppercase;
            border-bottom: 1px solid var(--border);
        }
        
        .db-table td {
            padding: 10px 12px;
            font-size: 13px;
            border-bottom: 1px solid var(--border);
        }
        
        .db-table tr:hover td {
            background: var(--bg-hover);
        }
        
        .db-table .actions {
            display: flex;
            gap: 4px;
        }
        
        .icon-btn {
            width: 28px;
            height: 28px;
            border-radius: 6px;
            border: none;
            background: var(--bg-card);
            color: var(--text-dim);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }
        
        .icon-btn:hover {
            background: var(--accent);
            color: var(--bg-dark);
        }
        
        .icon-btn.danger:hover {
            background: var(--danger);
            color: white;
        }
        
        /* 模态框 */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(4px);
            z-index: 2000;
            align-items: center;
            justify-content: center;
        }
        
        .modal.active {
            display: flex;
        }
        
        .modal-content {
            background: var(--bg-panel);
            border-radius: 16px;
            padding: 24px;
            width: 400px;
            max-width: 90%;
            border: 1px solid var(--border);
            animation: modalIn 0.3s ease;
        }
        
        @keyframes modalIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        
        .modal-title {
            font-size: 18px;
            font-weight: 600;
        }
        
        .modal-close {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            border: none;
            background: var(--bg-card);
            color: var(--text-dim);
            cursor: pointer;
            font-size: 18px;
        }
        
        .modal-close:hover {
            background: var(--danger);
            color: white;
        }
        
        .modal-footer {
            display: flex;
            gap: 12px;
            margin-top: 20px;
            justify-content: flex-end;
        }
        
        /* 提示消息 */
        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            padding: 12px 20px;
            border-radius: 10px;
            background: var(--bg-panel);
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 10px;
            z-index: 3000;
            animation: toastIn 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        
        @keyframes toastIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .toast.success {
            border-color: var(--success);
        }
        
        .toast.error {
            border-color: var(--danger);
        }
        
        /* 加载动画 */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 20px;
            color: var(--text-dim);
        }
        
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* 空状态 */
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-dim);
        }
        
        .empty-icon {
            font-size: 48px;
            margin-bottom: 12px;
        }
        
        /* 子图匹配验证页面 */
        #page-sim.active {
            display: block;
            padding: 16px;
        }

        .sim-container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .sim-layout {
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 16px;
            height: calc(100vh - 92px);
        }

        .sim-panel {
            background: var(--bg-panel);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--border);
            overflow-y: auto;
        }

        .sim-panel-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }

        .sim-upload-area {
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: var(--bg-card);
            margin-bottom: 16px;
            position: relative;
        }

        .sim-upload-area:hover {
            border-color: var(--accent);
            background: var(--bg-hover);
        }

        .sim-upload-area.has-image {
            border-color: var(--success);
            border-style: solid;
            padding: 8px;
        }

        .sim-upload-area input { display: none; }

        .sim-preview {
            width: 100%;
            max-height: 200px;
            object-fit: contain;
            border-radius: 8px;
        }

        .sim-upload-icon { font-size: 32px; margin-bottom: 8px; }
        .sim-upload-text { font-size: 13px; color: var(--text-dim); }

        .sim-result-images {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .sim-result-img {
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        .sim-info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 12px;
        }

        .sim-info-item {
            background: var(--bg-card);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }

        .sim-info-value {
            font-size: 20px;
            font-weight: 700;
            background: var(--gradient-2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sim-info-label {
            font-size: 11px;
            color: var(--text-dim);
            margin-top: 2px;
        }

        .sim-settings {
            background: var(--bg-card);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 16px;
        }

        .sim-settings summary {
            cursor: pointer;
            font-size: 13px;
            color: var(--text-dim);
        }

        .sim-settings .form-group { margin-top: 10px; margin-bottom: 6px; }

                /* 模型打点页面 */
        #page-grounding.active {
            display: block;
            padding: 16px;
        }

        /* 隐藏类 */
        .hidden { display: none !important; }
        
        /* 响应式 */
        @media (max-width: 1200px) {
            #page-nav.active {
                grid-template-columns: 1fr;
            }
            
            .db-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- 顶部导航栏 -->
    <nav class="navbar">
        <div class="navbar-brand">
            <span>🧭</span>
            <span>MemoryNav</span>
        </div>
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="switchPage('nav')">🗺️ 导航</button>
            <button class="nav-tab" onclick="switchPage('db')">💾 数据管理</button>
            <button class="nav-tab" onclick="switchPage('sim')">🔍 子图匹配</button>
            <button class="nav-tab" onclick="switchPage('grounding')">🎯 模型打点</button>
        </div>
        <div class="nav-status">
            <div class="status-badge">
                <div class="status-dot"></div>
                <span id="status-text">已连接</span>
            </div>
            <div class="status-badge">
                <span>📊</span>
                <span id="nav-node-count">0</span> 节点
            </div>
        </div>
    </nav>
    
    <!-- 主容器 -->
    <div class="main-container">
        <!-- 导航页面 -->
        <div id="page-nav" class="page active">
            <!-- 图形区域 -->
            <div class="graph-container">
                <div id="graph"></div>
                <div class="graph-controls">
                    <button class="graph-btn" onclick="refreshGraph()" title="刷新">🔄</button>
                    <button class="graph-btn" onclick="fitGraph()" title="适应">📍</button>
                    <button class="graph-btn" onclick="togglePhysics()" title="物理">⚡</button>
                    <button class="graph-btn" onclick="resetGraphColors()" title="重置颜色">🎨</button>
                </div>
            </div>
            
            <!-- 侧边栏 -->
            <div class="sidebar">
                <!-- 统计 -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">📊 系统状态</div>
                    </div>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="stat-nodes">0</div>
                            <div class="stat-label">节点</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="stat-edges">0</div>
                            <div class="stat-label">连接</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="stat-current">-</div>
                            <div class="stat-label">当前</div>
                        </div>
                    </div>
                </div>
                
                <!-- 起点选择 -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">📍 定位起点</div>
                    </div>
                    <div class="tabs">
                        <button class="tab active" onclick="switchStartTab('vpr')">📷 图像定位</button>
                        <button class="tab" onclick="switchStartTab('manual')">✋ 手动选择</button>
                    </div>
                    <div id="start-vpr" class="tab-content active">
                        <div class="robot-diagram">
                            <div class="robot-icon">🤖</div>
                            <div>cam1: 左前37.5° | cam2: 右前37.5° | cam3: 右后37.5° | cam4: 左后37.5°</div>
                        </div>
                        <div class="camera-grid">
                            <div class="camera-slot" id="slot-cam1" onclick="document.getElementById('file-cam1').click()">
                                <input type="file" id="file-cam1" accept="image/*" onchange="onCamFileChange(1, this)">
                                <div class="cam-label">📷 Camera 1</div>
                                <div class="cam-icon">📷</div>
                                <div class="cam-angle">左前 37.5°</div>
                            </div>
                            <div class="camera-slot" id="slot-cam2" onclick="document.getElementById('file-cam2').click()">
                                <input type="file" id="file-cam2" accept="image/*" onchange="onCamFileChange(2, this)">
                                <div class="cam-label">📷 Camera 2</div>
                                <div class="cam-icon">📷</div>
                                <div class="cam-angle">右前 37.5°</div>
                            </div>
                            <div class="camera-slot" id="slot-cam4" onclick="document.getElementById('file-cam4').click()">
                                <input type="file" id="file-cam4" accept="image/*" onchange="onCamFileChange(4, this)">
                                <div class="cam-label">📷 Camera 4</div>
                                <div class="cam-icon">📷</div>
                                <div class="cam-angle">左后 37.5°</div>
                            </div>
                            <div class="camera-slot" id="slot-cam3" onclick="document.getElementById('file-cam3').click()">
                                <input type="file" id="file-cam3" accept="image/*" onchange="onCamFileChange(3, this)">
                                <div class="cam-label">📷 Camera 3</div>
                                <div class="cam-icon">📷</div>
                                <div class="cam-angle">右后 37.5°</div>
                            </div>
                        </div>
                        <div style="display: flex; gap: 6px; margin-top: 8px">
                            <button class="btn btn-primary" style="flex:1" onclick="locateByVPR()">🔍 VPR定位</button>
                            <button class="btn btn-secondary" style="flex:0 0 auto; padding: 8px 12px" onclick="clearAllCameras()">🗑️</button>
                        </div>
                        <div id="vpr-result"></div>
                    </div>
                    <div id="start-manual" class="tab-content">
                        <select id="start-node">
                            <option value="">选择起点节点...</option>
                        </select>
                    </div>
                </div>
                
                <!-- 终点选择 -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">🎯 选择终点</div>
                    </div>
                    <div class="tabs">
                        <button class="tab active" onclick="switchGoalTab('search')">🔍 搜索</button>
                        <button class="tab" onclick="switchGoalTab('list')">📋 列表</button>
                    </div>
                    <div id="goal-search" class="tab-content active">
                        <input type="text" id="semantic-query" placeholder="输入目的地，如：休息区、打印机...">
                        <button class="btn btn-primary btn-block" onclick="searchByText()" style="margin-top: 8px">🔍 搜索</button>
                        <div id="search-result"></div>
                    </div>
                    <div id="goal-list" class="tab-content">
                        <select id="goal-node">
                            <option value="">选择目标节点...</option>
                        </select>
                    </div>
                </div>
                
                <!-- 路径规划 -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">🧭 路径规划</div>
                    </div>
                    <div style="display: flex; gap: 12px; margin-bottom: 12px">
                        <div style="flex: 1">
                            <div class="form-label">起点</div>
                            <div id="route-start" style="font-weight: 600; color: var(--accent)">未选择</div>
                        </div>
                        <div style="flex: 1">
                            <div class="form-label">终点</div>
                            <div id="route-goal" style="font-weight: 600; color: var(--success)">未选择</div>
                        </div>
                    </div>
                    <button class="btn btn-success btn-block" onclick="planPath()">🚀 开始导航</button>
                    <div id="nav-result"></div>
                </div>
                
                <!-- 节点详情 -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">📍 节点详情</div>
                    </div>
                    <div id="node-detail">
                        <div class="empty-state">
                            <div class="empty-icon">👆</div>
                            <div>点击图中节点查看详情</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 数据库管理页面 -->
        <div id="page-db" class="page">
            <div class="db-container">
                <!-- 节点管理 -->
                <div class="db-panel">
                    <div class="db-panel-header">
                        <div class="db-panel-title">📍 节点管理</div>
                        <button class="btn btn-primary btn-sm" onclick="showAddNodeModal()">➕ 添加</button>
                    </div>
                    <div class="db-toolbar">
                        <div class="search-box">
                            <span class="search-icon">🔍</span>
                            <input type="text" id="node-search" placeholder="搜索节点..." oninput="filterNodes()">
                        </div>
                        <button class="btn btn-secondary btn-sm" onclick="refreshDbData()">🔄</button>
                    </div>
                    <div class="db-table-container">
                        <table class="db-table">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>名称</th>
                                    <th>邻居</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody id="nodes-tbody">
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- 边管理 -->
                <div class="db-panel">
                    <div class="db-panel-header">
                        <div class="db-panel-title">🔗 边管理</div>
                        <button class="btn btn-primary btn-sm" onclick="showAddEdgeModal()">➕ 添加</button>
                    </div>
                    <div class="db-toolbar">
                        <div class="search-box">
                            <span class="search-icon">🔍</span>
                            <input type="text" id="edge-search" placeholder="搜索边..." oninput="filterEdges()">
                        </div>
                        <div class="btn-group">
                            <button class="btn btn-secondary btn-sm" onclick="exportData()">📤 导出</button>
                            <button class="btn btn-secondary btn-sm" onclick="showImportModal()">📥 导入</button>
                        </div>
                    </div>
                    <div class="db-table-container">
                        <table class="db-table">
                            <thead>
                                <tr>
                                    <th>起点</th>
                                    <th>终点</th>
                                    <th>相机 / 地标</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody id="edges-tbody">
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 子图匹配验证页面 -->
        <div id="page-sim" class="page">
            <div class="sim-container">
                <div class="sim-layout">
                    <!-- 左侧: 上传和控制 -->
                    <div class="sim-panel">
                        <div class="sim-panel-title">🔍 子图匹配验证</div>
                        

                        <div class="form-label">原图 (相机图像)</div>
                        <div class="sim-upload-area" id="sim-orig-area" onclick="document.getElementById('sim-orig-file').click()">
                            <input type="file" id="sim-orig-file" accept="image/*" onchange="onSimFileChange('orig', this)">
                            <div class="sim-upload-icon">🖼️</div>
                            <div class="sim-upload-text">点击上传原图</div>
                        </div>
                        
                        <div class="form-label">子图 (Crop 图像)</div>
                        <div class="sim-upload-area" id="sim-crop-area" onclick="document.getElementById('sim-crop-file').click()">
                            <input type="file" id="sim-crop-file" accept="image/*" onchange="onSimFileChange('crop', this)">
                            <div class="sim-upload-icon">✂️</div>
                            <div class="sim-upload-text">点击上传子图</div>
                        </div>
                        
                        <details class="sim-settings">
                            <summary>⚙️ 高级设置</summary>
                            <div class="form-group">
                                <label class="form-label">置信度阈值</label>
                                <input type="range" id="sim-conf-threshold" min="0" max="1" step="0.05" value="0.3"
                                    oninput="document.getElementById('sim-conf-val').textContent = this.value">
                                <span id="sim-conf-val" style="font-size:13px; color:var(--accent)">0.3</span>
                            </div>
                            <div class="form-group">
                                <label class="form-label">最小特征匹配数</label>
                                <input type="number" id="sim-min-matches" value="8" min="4" max="50" style="width:80px">
                            </div>
                        </details>
                        
                        <div style="display:flex; gap:8px; margin-bottom:16px">
                            <button class="btn btn-primary" style="flex:1" onclick="runSubImageMatch()">🔍 开始匹配</button>
                            <button class="btn btn-secondary" style="flex:0 0 auto" onclick="clearSimInputs()">🗑️ 清除</button>
                        </div>
                        
                        <!-- 匹配结果信息 -->
                        <div id="sim-result-info"></div>
                    </div>
                    
                    <!-- 右侧: 结果展示 -->
                    <div class="sim-panel">
                        <div class="sim-panel-title">📊 匹配结果</div>
                        <div id="sim-result-display" style="text-align:center; color:var(--text-dim); padding:40px 0">
                            <div style="font-size:48px; margin-bottom:12px">🔍</div>
                            <div>上传原图和子图后点击"开始匹配"</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- 模型打点验证页面 -->
        <div id="page-grounding" class="page">
            <div class="sim-container">
                <div class="sim-layout">
                    <!-- 左侧: 控制面板 -->
                    <div class="sim-panel">
                        <div class="sim-panel-title">🎯 模型打点验证 (Qwen3.5)</div>
                        
                        <div class="form-label">上传图片</div>
                        <div class="sim-upload-area" id="gnd-img-area" onclick="document.getElementById('gnd-img-file').click()">
                            <input type="file" id="gnd-img-file" accept="image/*" onchange="onGndFileChange(this)">
                            <div class="sim-upload-icon">🖼️</div>
                            <div class="sim-upload-text">点击上传相机图像</div>
                        </div>

                        <div class="form-group" style="margin-top:12px">
                            <label class="form-label">地标名称 (landmark_name)</label>
                            <input type="text" id="gnd-landmark" placeholder="输入地标名称，如：电梯、打印机、玻璃门">
                        </div>


                        <div style="display:flex; gap:8px; margin-bottom:16px">
                            <button class="btn btn-primary" style="flex:1" onclick="runGrounding()">🎯 开始打点</button>
                            <button class="btn btn-secondary" style="flex:0 0 auto" onclick="clearGndInputs()">🗑️ 清除</button>
                        </div>

                        <!-- 打点结果信息 -->
                        <div id="gnd-result-info"></div>
                    </div>

                    <!-- 右侧: 结果展示 -->
                    <div class="sim-panel">
                        <div class="sim-panel-title">📊 打点结果</div>
                        <div id="gnd-result-display" style="text-align:center; color:var(--text-dim); padding:40px 0">
                            <div style="font-size:48px; margin-bottom:12px">🎯</div>
                            <div>上传图片并输入地标名称后点击"开始打点"</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    

        <!-- 添加节点模态框 -->
    <div id="add-node-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">➕ 添加节点</div>
                <button class="modal-close" onclick="closeModal('add-node-modal')">×</button>
            </div>
            <div class="form-group">
                <label class="form-label">节点 ID</label>
                <input type="text" id="new-node-id" placeholder="输入唯一ID">
            </div>
            <div class="form-group">
                <label class="form-label">节点名称</label>
                <input type="text" id="new-node-name" placeholder="如：A8休息区">
            </div>
            <div class="form-group">
                <label class="form-label">数据路径 (可选)</label>
                <input type="text" id="new-node-path" placeholder="数据文件夹路径">
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal('add-node-modal')">取消</button>
                <button class="btn btn-primary" onclick="addNode()">添加</button>
            </div>
        </div>
    </div>
    
    <!-- 编辑节点模态框 -->
    <div id="edit-node-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">✏️ 编辑节点</div>
                <button class="modal-close" onclick="closeModal('edit-node-modal')">×</button>
            </div>
            <input type="hidden" id="edit-node-id">
            <div class="form-group">
                <label class="form-label">节点名称</label>
                <input type="text" id="edit-node-name" placeholder="节点名称">
            </div>
            <div class="form-group">
                <label class="form-label">数据路径</label>
                <input type="text" id="edit-node-path" placeholder="数据文件夹路径">
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal('edit-node-modal')">取消</button>
                <button class="btn btn-primary" onclick="updateNode()">保存</button>
            </div>
        </div>
    </div>
    
    <!-- 添加边模态框 -->
    <div id="add-edge-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">➕ 添加边</div>
                <button class="modal-close" onclick="closeModal('add-edge-modal')">×</button>
            </div>
            <div class="form-group">
                <label class="form-label">起点节点</label>
                <select id="new-edge-from"></select>
            </div>
            <div class="form-group">
                <label class="form-label">终点节点</label>
                <select id="new-edge-to"></select>
            </div>
            <div class="form-group">
                <label class="form-label">相机 (camera_1~4)</label>
                <input type="text" id="new-edge-camera" value="camera_1" placeholder="camera_1~4">
                <label class="form-label">地标名称</label>
                <input type="text" id="new-edge-landmark" value="" placeholder="如: 电梯、玻璃门">
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal('add-edge-modal')">取消</button>
                <button class="btn btn-primary" onclick="addEdge()">添加</button>
            </div>
        </div>
    </div>
    
    <!-- 导入模态框 -->
    <div id="import-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">📥 导入数据</div>
                <button class="modal-close" onclick="closeModal('import-modal')">×</button>
            </div>
            <div class="form-group">
                <label class="form-label">数据目录路径</label>
                <input type="text" id="import-path" placeholder="/path/to/labeled_data">
            </div>
            <div class="form-group">
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer">
                    <input type="checkbox" id="import-clear" checked>
                    <span>清除现有数据</span>
                </label>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal('import-modal')">取消</button>
                <button class="btn btn-primary" onclick="importData()">导入</button>
            </div>
        </div>
    </div>

    <script>
        // 全局状态
        let network = null;
        let graphData = null;
        let selectedStart = null;
        let selectedGoal = null;
        let physicsEnabled = true;
        let dbNodes = [];
        let dbEdges = [];
        
        // 页面切换
        function switchPage(page) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.getElementById('page-' + page).classList.add('active');
            event.target.classList.add('active');
            
            if (page === 'db') {
                refreshDbData();
            }
        }
        
        // 刷新图数据
        async function refreshGraph() {
            try {
                const res = await fetch('/api/graph');
                const data = await res.json();
                if (data.success) {
                    graphData = data.data;
                    renderGraph(data.data);
                    updateStats(data.data);
                    updateSelects(data.data.nodes);
                }
            } catch (e) {
                showToast('加载失败: ' + e.message, 'error');
            }
        }
        
        // 渲染图形
        function renderGraph(data) {
            const nodes = new vis.DataSet(data.nodes.map(n => ({
                id: n.id,
                label: n.node_name,
                color: {
                    background: '#1a1a3e',
                    border: '#00d4ff',
                    highlight: { background: '#00d4ff', border: '#00ff88' },
                    hover: { background: '#252550', border: '#00d4ff' }
                },
                font: { color: '#e8e8f0', size: 12 },
                borderWidth: 2,
                shadow: { enabled: true, color: 'rgba(0,212,255,0.3)', size: 10 }
            })));
            
            const edges = new vis.DataSet(data.edges.map((e, i) => ({
                id: 'e' + i,
                from: e.from,
                to: e.to,
                arrows: e.bidirectional ? 
                    { to: { enabled: true, scaleFactor: 0.5 }, from: { enabled: true, scaleFactor: 0.5 } } :
                    { to: { enabled: true, scaleFactor: 0.5 } },
                color: { color: '#4cc9f0', opacity: 0.6, highlight: '#00ff88' },
                smooth: { type: 'continuous' }
            })));
            
            const options = {
                physics: {
                    enabled: physicsEnabled,
                    stabilization: { iterations: 200 },
                    barnesHut: { gravitationalConstant: -3000, springLength: 150 }
                },
                nodes: { shape: 'dot', size: 20 },
                edges: { width: 2 },
                interaction: { hover: true, tooltipDelay: 200 }
            };
            
            network = new vis.Network(document.getElementById('graph'), { nodes, edges }, options);
            network.on('click', p => { if (p.nodes.length > 0) showNodeDetail(p.nodes[0]); });
            network.on('doubleClick', p => { if (p.nodes.length > 0) setAsStart(p.nodes[0]); });
        }
        
        // 更新统计
        function updateStats(data) {
            document.getElementById('stat-nodes').textContent = data.nodes.length;
            document.getElementById('stat-edges').textContent = data.edges.length;
            document.getElementById('nav-node-count').textContent = data.nodes.length;
        }
        
        // 更新选择框
        function updateSelects(nodes) {
            const startSel = document.getElementById('start-node');
            const goalSel = document.getElementById('goal-node');
            
            const options = '<option value="">选择节点...</option>' + 
                nodes.map(n => `<option value="${n.id}">${n.node_name}${n.node_name_eng ? ' ('+n.node_name_eng+')' : ''}</option>`).join('');
            
            startSel.innerHTML = options;
            goalSel.innerHTML = options;
            
            startSel.onchange = e => { selectedStart = e.target.value; updateRoute(); };
            goalSel.onchange = e => { selectedGoal = e.target.value; updateRoute(); };
        }
        
        // 更新路径显示
        function updateRoute() {
            const startName = selectedStart ? (() => { const _n = graphData.nodes.find(n => n.id === selectedStart); return _n ? (_n.node_name + (_n.node_name_eng ? ' ('+_n.node_name_eng+')' : '')) : '未选择'; })() : '未选择';
            const goalName = selectedGoal ? (() => { const _n = graphData.nodes.find(n => n.id === selectedGoal); return _n ? (_n.node_name + (_n.node_name_eng ? ' ('+_n.node_name_eng+')' : '')) : '未选择'; })() : '未选择';
            document.getElementById('route-start').textContent = startName;
            document.getElementById('route-goal').textContent = goalName;
        }
        
        // 设为起点
        function setAsStart(nodeId) {
            selectedStart = nodeId;
            updateRoute();
            showToast('已设为起点: ' + (() => { const _n = graphData.nodes.find(n => n.id === nodeId); return _n ? _n.node_name + (_n.node_name_eng ? ' ('+_n.node_name_eng+')' : '') : nodeId; })(), 'success');
        }
        
        // 切换起点标签页
        function switchStartTab(tab) {
            document.querySelectorAll('#page-nav .card:nth-child(2) .tab').forEach((t, i) => {
                t.classList.toggle('active', (tab === 'vpr' ? i === 0 : i === 1));
            });
            document.getElementById('start-vpr').classList.toggle('active', tab === 'vpr');
            document.getElementById('start-manual').classList.toggle('active', tab === 'manual');
        }
        
        // 切换终点标签页
        function switchGoalTab(tab) {
            document.querySelectorAll('#page-nav .card:nth-child(3) .tab').forEach((t, i) => {
                t.classList.toggle('active', (tab === 'search' ? i === 0 : i === 1));
            });
            document.getElementById('goal-search').classList.toggle('active', tab === 'search');
            document.getElementById('goal-list').classList.toggle('active', tab === 'list');
        }
        
        // 预览图片
        // 4相机独立上传
        function onCamFileChange(camId, input) {
            const slot = document.getElementById('slot-cam' + camId);
            if (input.files && input.files[0]) {
                slot.classList.add('has-image');
                // 移除旧预览
                const oldPrev = slot.querySelector('.cam-preview');
                if (oldPrev) oldPrev.remove();
                const oldIcon = slot.querySelector('.cam-icon');
                if (oldIcon) oldIcon.style.display = 'none';
                // 添加预览图
                const img = document.createElement('img');
                img.className = 'cam-preview';
                img.src = URL.createObjectURL(input.files[0]);
                slot.appendChild(img);
                // 添加清除按钮
                if (!slot.querySelector('.cam-clear')) {
                    const clearBtn = document.createElement('span');
                    clearBtn.className = 'cam-clear';
                    clearBtn.textContent = '✕';
                    clearBtn.onclick = (e) => { e.stopPropagation(); clearCamera(camId); };
                    slot.appendChild(clearBtn);
                }
            }
        }
        
        function clearCamera(camId) {
            const slot = document.getElementById('slot-cam' + camId);
            const input = document.getElementById('file-cam' + camId);
            input.value = '';
            slot.classList.remove('has-image');
            const prev = slot.querySelector('.cam-preview');
            if (prev) prev.remove();
            const clearBtn = slot.querySelector('.cam-clear');
            if (clearBtn) clearBtn.remove();
            const icon = slot.querySelector('.cam-icon');
            if (icon) icon.style.display = '';
        }
        
        function clearAllCameras() {
            for (let i = 1; i <= 4; i++) clearCamera(i);
            document.getElementById('vpr-result').innerHTML = '';
        }
        
        // VPR定位 - 4相机独立上传
        async function locateByVPR() {
            const fd = new FormData();
            let count = 0;
            for (let i = 1; i <= 4; i++) {
                const input = document.getElementById('file-cam' + i);
                if (input.files && input.files[0]) {
                    fd.append('camera_' + i, input.files[0]);
                    count++;
                }
            }
            if (count === 0) {
                showToast('请至少上传1张相机图片', 'error');
                return;
            }
            
            document.getElementById('vpr-result').innerHTML = '<div class="loading"><div class="spinner"></div>定位中...</div>';
            
            try {
                const res = await fetch('/api/vpr', { method: 'POST', body: fd });
                const data = await res.json();
                
                if (data.success) {
                    selectedStart = data.matched_node_id;
                    updateRoute();
                    const offset = data.heading_offset || 0;
                    const offsetDir = offset > 0 ? '顺时针' : offset < 0 ? '逆时针' : '同向';
                    const offsetAbs = Math.abs(offset).toFixed(1);
                    document.getElementById('vpr-result').innerHTML = `
                        <div class="result-box result-success">
                            <div class="result-title">✅ 定位成功</div>
                            <div><strong>${data.matched_node_name}</strong>${data.matched_node_name_eng ? ' <span style="color:#888">('+data.matched_node_name_eng+')</span>' : ''}</div>
                            <div style="color: var(--text-dim); font-size: 13px">
                                相似度: ${(data.similarity * 100).toFixed(1)}% | 
                                置信度: ${(data.confidence * 100).toFixed(1)}%
                            </div>
                            <div style="color: var(--warning); font-size: 13px; margin-top: 4px">
                                🧭 朝向偏移: ${offsetDir} ${offsetAbs}° (shift=${data.best_shift})
                            </div>
                            <div style="color: var(--text-dim); font-size: 12px; margin-top: 4px">
                                各相机得分: ${Object.entries(data.camera_scores || {}).map(([k,v]) => k.replace('camera_','cam') + ':' + (v*100).toFixed(1) + '%').join(' | ')}
                            </div>
                        </div>
                    `;
                    if (network) network.selectNodes([selectedStart]);
                } else {
                    document.getElementById('vpr-result').innerHTML = `
                        <div class="result-box result-error">
                            <div class="result-title">❌ 定位失败</div>
                            <div>${data.error}</div>
                        </div>
                    `;
                }
            } catch (e) {
                document.getElementById('vpr-result').innerHTML = `
                    <div class="result-box result-error">
                        <div class="result-title">❌ 请求错误</div>
                        <div>${e.message}</div>
                    </div>
                `;
            }
        }
        
        // 文本搜索
        function searchByText() {
            const query = document.getElementById('semantic-query').value.trim().toLowerCase();
            if (!query) {
                showToast('请输入搜索内容', 'error');
                return;
            }
            
            const matches = graphData.nodes.filter(n => n.node_name.toLowerCase().includes(query) || (n.node_name_eng || '').toLowerCase().includes(query));
            
            if (matches.length) {
                let html = '<div class="neighbor-list">';
                matches.forEach(n => {
                    html += `<div class="neighbor-item" onclick="selectGoal('${n.id}')">
                        <span class="neighbor-name">${n.node_name}${n.node_name_eng ? ' <span style="color:#999;font-size:0.85em">('+n.node_name_eng+')</span>' : ''}</span>
                        <span>→</span>
                    </div>`;
                });
                html += '</div>';
                document.getElementById('search-result').innerHTML = html;
            } else {
                document.getElementById('search-result').innerHTML = `
                    <div class="result-box result-error">
                        <div>未找到匹配的节点</div>
                    </div>
                `;
            }
        }
        
        // 选择目标
        function selectGoal(id) {
            selectedGoal = id;
            updateRoute();
            if (network) network.selectNodes([id]);
            showToast('已选择目标: ' + (() => { const _n = graphData.nodes.find(n => n.id === id); return _n ? _n.node_name + (_n.node_name_eng ? ' ('+_n.node_name_eng+')' : '') : id; })(), 'success');
        }
        
        // 显示节点详情
        async function showNodeDetail(id) {
            try {
                const res = await fetch('/api/node/' + id);
                const data = await res.json();
                
                if (data.success) {
                    const n = data.node;
                    let html = `
                        <div class="node-detail-header">
                            <div class="node-icon">📍</div>
                            <div>
                                <div class="node-name">${n.node_name}${n.node_name_eng ? '<div style="color:#999;font-size:0.85em">'+n.node_name_eng+'</div>' : ''}</div>
                                <div class="node-id">ID: ${n.id}</div>
                            </div>
                        </div>
                    `;
                    
                    // 相机图片
                    if (n.camera_images && Object.keys(n.camera_images).length > 0) {
                        html += '<div class="camera-grid">';
                        for (let cam in n.camera_images) {
                            html += `<img class="camera-img" src="/api/image?path=${encodeURIComponent(n.camera_images[cam])}" alt="${cam}">`;
                        }
                        html += '</div>';
                    }
                    
                    // 邻居
                    if (n.neighbors && n.neighbors.length > 0) {
                        html += '<div class="form-label" style="margin-top: 12px">相邻节点</div><div class="neighbor-list">';
                        n.neighbors.forEach(nb => {
                            html += `<div class="neighbor-item" onclick="showNodeDetail('${nb.id}')">
                                <span class="neighbor-name">${nb.name}</span>
                                <span class="neighbor-angle">${nb.camera_name} → ${nb.landmark_name}</span>
                            </div>`;
                        });
                        html += '</div>';
                    }
                    
                    // 操作按钮
                    html += `
                        <div class="btn-group" style="margin-top: 12px">
                            <button class="btn btn-primary btn-sm" onclick="setAsStart('${n.id}')">设为起点</button>
                            <button class="btn btn-success btn-sm" onclick="selectGoal('${n.id}')">设为终点</button>
                        </div>
                    `;
                    
                    document.getElementById('node-detail').innerHTML = html;
                }
            } catch (e) {
                console.error(e);
            }
        }
        
        // 路径规划
        async function planPath() {
            const start = selectedStart || document.getElementById('start-node').value;
            const goal = selectedGoal || document.getElementById('goal-node').value;
            
            if (!start || !goal) {
                showToast('请选择起点和终点', 'error');
                return;
            }
            
            document.getElementById('nav-result').innerHTML = '<div class="loading"><div class="spinner"></div>规划中...</div>';
            
            try {
                const res = await fetch('/api/navigate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ start, goal })
                });
                const data = await res.json();
                
                if (data.success) {
                    let html = `
                        <div class="result-box result-success" style="margin-bottom: 12px">
                            <div class="result-title">✅ 路径规划成功</div>
                            <div style="font-size: 13px; color: var(--text-dim)">${data.path.join(' → ')}</div>
                        </div>
                        <div class="path-steps">
                    `;
                    
                    data.steps.forEach((step, i) => {
                        html += `
                            <div class="path-step">
                                <div class="step-header">
                                    <div class="step-number">${i + 1}</div>
                                    <div class="step-info">
                                        <span class="step-from">${step.from_node.name}</span>
                                        <span class="step-arrow">→</span>
                                        <span class="step-to">${step.to_node.name}</span>
                                    </div>
                                    <span class="step-angle">${step.camera_name} → ${step.landmark_name}</span>
                                </div>
                                <div style="display:flex; gap:8px; font-size:11px; color:var(--text-dim); margin:4px 0 4px 36px">
                                    
                                </div>
                                ${step.crop_image_paths && Object.keys(step.crop_image_paths).length > 0
                                    ? Object.entries(step.crop_image_paths).map(([scale, path]) =>
                                        `<div style="display:inline-block;margin:2px;text-align:center">
                                            <div style="font-size:10px;color:var(--text-dim)">${scale}</div>
                                            <img class="step-image" src="/api/image?path=${encodeURIComponent(path)}" style="max-width:120px">
                                        </div>`
                                    ).join('')
                                    : ''}
                            </div>
                        `;
                    });
                    
                    html += '</div>';
                    document.getElementById('nav-result').innerHTML = html;
                    
                    // 高亮路径 - 起点黄色，终点红色，中间节点绿色
                    if (network && data.path && data.path.length > 0) {
                        highlightPath(data.path);
                    }
                } else {
                    document.getElementById('nav-result').innerHTML = `
                        <div class="result-box result-error">
                            <div class="result-title">❌ 规划失败</div>
                            <div>${data.error || data.message}</div>
                        </div>
                    `;
                }
            } catch (e) {
                document.getElementById('nav-result').innerHTML = `
                    <div class="result-box result-error">
                        <div class="result-title">❌ 请求错误</div>
                        <div>${e.message}</div>
                    </div>
                `;
            }
        }
        
        // 高亮路径函数 - 起点黄色，终点红色，中间节点绿色
        function highlightPath(path) {
            if (!network || !path || path.length === 0) return;
            
            // 先重置所有节点颜色
            const allNodes = network.body.data.nodes.get();
            const resetUpdates = allNodes.map(n => ({
                id: n.id,
                color: {
                    background: '#1a1a3e',
                    border: '#00d4ff',
                    highlight: { background: '#00d4ff', border: '#00ff88' },
                    hover: { background: '#252550', border: '#00d4ff' }
                }
            }));
            network.body.data.nodes.update(resetUpdates);
            
            // 设置路径节点颜色
            const pathUpdates = path.map((nodeId, index) => {
                let color;
                if (index === 0) {
                    // 起点 - 黄色
                    color = {
                        background: '#FFD700',
                        border: '#FFA500',
                        highlight: { background: '#FFEC8B', border: '#FFD700' },
                        hover: { background: '#FFEC8B', border: '#FFA500' }
                    };
                } else if (index === path.length - 1) {
                    // 终点 - 红色
                    color = {
                        background: '#FF4466',
                        border: '#CC0033',
                        highlight: { background: '#FF6B6B', border: '#FF4466' },
                        hover: { background: '#FF6B6B', border: '#CC0033' }
                    };
                } else {
                    // 中间节点 - 绿色
                    color = {
                        background: '#00FF88',
                        border: '#00CC66',
                        highlight: { background: '#66FFAA', border: '#00FF88' },
                        hover: { background: '#66FFAA', border: '#00CC66' }
                    };
                }
                return { id: nodeId, color };
            });
            network.body.data.nodes.update(pathUpdates);
            
            // 高亮路径上的边
            const allEdges = network.body.data.edges.get();
            const edgeUpdates = allEdges.map(e => {
                // 检查边是否在路径上
                let isOnPath = false;
                for (let i = 0; i < path.length - 1; i++) {
                    if ((e.from === path[i] && e.to === path[i+1]) ||
                        (e.to === path[i] && e.from === path[i+1])) {
                        isOnPath = true;
                        break;
                    }
                }
                return {
                    id: e.id,
                    color: isOnPath ? 
                        { color: '#00FF88', opacity: 1 } : 
                        { color: '#4cc9f0', opacity: 0.4 },
                    width: isOnPath ? 4 : 2
                };
            });
            network.body.data.edges.update(edgeUpdates);
            
            // 聚焦到路径
            network.fit({ nodes: path, animation: true });
        }
        
        // 重置图形颜色
        function resetGraphColors() {
            if (!network) return;
            const allNodes = network.body.data.nodes.get();
            const resetNodes = allNodes.map(n => ({
                id: n.id,
                color: {
                    background: '#1a1a3e',
                    border: '#00d4ff',
                    highlight: { background: '#00d4ff', border: '#00ff88' },
                    hover: { background: '#252550', border: '#00d4ff' }
                }
            }));
            network.body.data.nodes.update(resetNodes);
            
            const allEdges = network.body.data.edges.get();
            const resetEdges = allEdges.map(e => ({
                id: e.id,
                color: { color: '#4cc9f0', opacity: 0.6 },
                width: 2
            }));
            network.body.data.edges.update(resetEdges);
        }
        
        // 图形控制
        function fitGraph() {
            if (network) network.fit();
        }
        
        function togglePhysics() {
            physicsEnabled = !physicsEnabled;
            if (network) {
                network.setOptions({ physics: { enabled: physicsEnabled } });
            }
            showToast(physicsEnabled ? '物理模拟已开启' : '物理模拟已关闭', 'success');
        }
        
        // ============ 数据库管理 ============
        
        // 刷新数据库数据
        async function refreshDbData() {
            try {
                const res = await fetch('/api/db/all');
                const data = await res.json();
                
                if (data.success) {
                    dbNodes = data.nodes;
                    dbEdges = data.edges;
                    renderNodesTable(data.nodes);
                    renderEdgesTable(data.edges);
                    updateEdgeSelects(data.nodes);
                }
            } catch (e) {
                showToast('加载数据失败', 'error');
            }
        }
        
        // 渲染节点表格
        function renderNodesTable(nodes) {
            const tbody = document.getElementById('nodes-tbody');
            tbody.innerHTML = nodes.map(n => `
                <tr>
                    <td>${n.id}</td>
                    <td>${n.node_name}${n.node_name_eng ? '<br><span style="color:#999;font-size:0.85em">'+n.node_name_eng+'</span>' : ''}</td>
                    <td>${n.neighbor_count}</td>
                    <td class="actions">
                        <button class="icon-btn" onclick="editNode('${n.id}', '${n.node_name}', '${n.data_path || ''}', '${n.node_name_eng || ''}')" title="编辑">✏️</button>
                        <button class="icon-btn danger" onclick="deleteNode('${n.id}')" title="删除">🗑️</button>
                    </td>
                </tr>
            `).join('');
        }
        
        // 渲染边表格
        function renderEdgesTable(edges) {
            const tbody = document.getElementById('edges-tbody');
            tbody.innerHTML = edges.map(e => `
                <tr>
                    <td>${e.from_name || e.from}</td>
                    <td>${e.to_name || e.to}</td>
                    <td>${e.camera_name} / ${e.landmark_name}</td>
                    <td class="actions">
                        <button class="icon-btn danger" onclick="deleteEdge('${e.from}', '${e.to}')" title="删除">🗑️</button>
                    </td>
                </tr>
            `).join('');
        }
        
        // 更新边选择框
        function updateEdgeSelects(nodes) {
            const options = nodes.map(n => `<option value="${n.id}">${n.node_name}${n.node_name_eng ? ' ('+n.node_name_eng+')' : ''}</option>`).join('');
            document.getElementById('new-edge-from').innerHTML = options;
            document.getElementById('new-edge-to').innerHTML = options;
        }
        
        // 过滤节点
        function filterNodes() {
            const q = document.getElementById('node-search').value.toLowerCase();
            const filtered = dbNodes.filter(n => 
                n.id.toLowerCase().includes(q) || n.node_name.toLowerCase().includes(q) || (n.node_name_eng || '').toLowerCase().includes(q)
            );
            renderNodesTable(filtered);
        }
        
        // 过滤边
        function filterEdges() {
            const q = document.getElementById('edge-search').value.toLowerCase();
            const filtered = dbEdges.filter(e => 
                e.from.toLowerCase().includes(q) || e.to.toLowerCase().includes(q) ||
                (e.from_name && e.from_name.toLowerCase().includes(q)) ||
                (e.to_name && e.to_name.toLowerCase().includes(q))
            );
            renderEdgesTable(filtered);
        }
        
        // 模态框控制
        function showAddNodeModal() {
            document.getElementById('new-node-id').value = '';
            document.getElementById('new-node-name').value = '';
            document.getElementById('new-node-path').value = '';
            document.getElementById('add-node-modal').classList.add('active');
        }
        
        function showAddEdgeModal() {
            document.getElementById('new-edge-camera').value = 'camera_1';
            document.getElementById('new-edge-landmark').value = '';
            document.getElementById('add-edge-modal').classList.add('active');
        }
        
        function showImportModal() {
            document.getElementById('import-path').value = '';
            document.getElementById('import-modal').classList.add('active');
        }
        
        function closeModal(id) {
            document.getElementById(id).classList.remove('active');
        }
        
        function editNode(id, name, path) {
            document.getElementById('edit-node-id').value = id;
            document.getElementById('edit-node-name').value = name;
            document.getElementById('edit-node-path').value = path;
            document.getElementById('edit-node-modal').classList.add('active');
        }
        
        // 添加节点
        async function addNode() {
            const id = document.getElementById('new-node-id').value.trim();
            const name = document.getElementById('new-node-name').value.trim();
            const path = document.getElementById('new-node-path').value.trim();
            
            if (!id || !name) {
                showToast('请填写ID和名称', 'error');
                return;
            }
            
            try {
                const res = await fetch('/api/db/node', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ id, name, data_path: path })
                });
                const data = await res.json();
                
                if (data.success) {
                    showToast('节点添加成功', 'success');
                    closeModal('add-node-modal');
                    refreshDbData();
                    refreshGraph();
                } else {
                    showToast(data.error, 'error');
                }
            } catch (e) {
                showToast('添加失败: ' + e.message, 'error');
            }
        }
        
        // 更新节点
        async function updateNode() {
            const id = document.getElementById('edit-node-id').value;
            const name = document.getElementById('edit-node-name').value.trim();
            const path = document.getElementById('edit-node-path').value.trim();
            
            try {
                const res = await fetch('/api/db/node/' + id, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, data_path: path })
                });
                const data = await res.json();
                
                if (data.success) {
                    showToast('节点更新成功', 'success');
                    closeModal('edit-node-modal');
                    refreshDbData();
                    refreshGraph();
                } else {
                    showToast(data.error, 'error');
                }
            } catch (e) {
                showToast('更新失败: ' + e.message, 'error');
            }
        }
        
        // 删除节点
        // 自定义确认弹窗
        function showConfirmDialog(message) {
            return new Promise((resolve) => {
                const overlay = document.createElement('div');
                overlay.className = 'confirm-overlay';
                overlay.innerHTML = `
                    <div class="confirm-box">
                        <div class="confirm-msg">${message}</div>
                        <div class="confirm-actions">
                            <button class="btn-cancel" id="confirm-cancel">取消</button>
                            <button class="btn-confirm-danger" id="confirm-ok">确定删除</button>
                        </div>
                    </div>
                `;
                document.body.appendChild(overlay);
                overlay.querySelector('#confirm-cancel').onclick = () => { overlay.remove(); resolve(false); };
                overlay.querySelector('#confirm-ok').onclick = () => { overlay.remove(); resolve(true); };
                overlay.onclick = (e) => { if (e.target === overlay) { overlay.remove(); resolve(false); } };
            });
        }
        
        async function deleteNode(id) {
            const ok = await showConfirmDialog(`确定删除节点 <strong>${id}</strong> 吗？<br>相关的边也会被删除。`);
            if (!ok) return;
            
            try {
                const res = await fetch('/api/db/node/' + id, { method: 'DELETE' });
                const data = await res.json();
                
                if (data.success) {
                    showToast('节点已删除', 'success');
                    refreshDbData();
                    refreshGraph();
                } else {
                    showToast(data.error, 'error');
                }
            } catch (e) {
                showToast('删除失败: ' + e.message, 'error');
            }
        }
        
        // 添加边
        async function addEdge() {
            const from_id = document.getElementById('new-edge-from').value;
            const to_id = document.getElementById('new-edge-to').value;
            const camera_name = document.getElementById('new-edge-camera').value || 'camera_1';
            const landmark_name = document.getElementById('new-edge-landmark').value || '';
            
            if (!from_id || !to_id) {
                showToast('请选择起点和终点', 'error');
                return;
            }
            
            if (from_id === to_id) {
                showToast('起点和终点不能相同', 'error');
                return;
            }
            
            try {
                const res = await fetch('/api/db/edge', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ from_id, to_id, camera_name, landmark_name })
                });
                const data = await res.json();
                
                if (data.success) {
                    showToast('边添加成功', 'success');
                    closeModal('add-edge-modal');
                    refreshDbData();
                    refreshGraph();
                } else {
                    showToast(data.error, 'error');
                }
            } catch (e) {
                showToast('添加失败: ' + e.message, 'error');
            }
        }
        
        // 删除边
        async function deleteEdge(from, to) {
            const ok = await showConfirmDialog(`确定删除边 <strong>${from} → ${to}</strong> 吗？`);
            if (!ok) return;
            
            try {
                const res = await fetch(`/api/db/edge?from=${from}&to=${to}`, { method: 'DELETE' });
                const data = await res.json();
                
                if (data.success) {
                    showToast('边已删除', 'success');
                    refreshDbData();
                    refreshGraph();
                } else {
                    showToast(data.error, 'error');
                }
            } catch (e) {
                showToast('删除失败: ' + e.message, 'error');
            }
        }
        
        // 导出数据
        async function exportData() {
            try {
                const res = await fetch('/api/db/export');
                const data = await res.json();
                
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'memory_graph_export.json';
                a.click();
                URL.revokeObjectURL(url);
                
                showToast('数据导出成功', 'success');
            } catch (e) {
                showToast('导出失败: ' + e.message, 'error');
            }
        }
        
        // 导入数据
        async function importData() {
            const path = document.getElementById('import-path').value.trim();
            const clear = document.getElementById('import-clear').checked;
            
            if (!path) {
                showToast('请输入数据路径', 'error');
                return;
            }
            
            try {
                const res = await fetch('/api/db/import', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path, clear })
                });
                const data = await res.json();
                
                if (data.success) {
                    showToast(`导入成功: ${data.nodes_count} 节点, ${data.edges_count} 边`, 'success');
                    closeModal('import-modal');
                    refreshDbData();
                    refreshGraph();
                } else {
                    showToast(data.error, 'error');
                }
            } catch (e) {
                showToast('导入失败: ' + e.message, 'error');
            }
        }
        
        // 提示消息
        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = 'toast ' + type;
            toast.innerHTML = `<span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span><span>${message}</span>`;
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'toastIn 0.3s ease reverse';
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        }
        
        // ============ 子图匹配验证 ============
        function onSimFileChange(type, input) {
            const area = document.getElementById('sim-' + type + '-area');
            if (input.files && input.files[0]) {
                const fileName = input.files[0].name;
                const reader = new FileReader();
                reader.onload = function(e) {
                    area.classList.add('has-image');
                    // 保留原始 input，只更新预览
                    let preview = area.querySelector('.sim-preview');
                    let nameDiv = area.querySelector('.sim-file-name');
                    if (!preview) {
                        // 隐藏上传提示
                        area.querySelectorAll('.sim-upload-icon, .sim-upload-text').forEach(el => el.style.display = 'none');
                        preview = document.createElement('img');
                        preview.className = 'sim-preview';
                        area.appendChild(preview);
                        nameDiv = document.createElement('div');
                        nameDiv.className = 'sim-file-name';
                        nameDiv.style.cssText = 'font-size:11px; color:var(--text-dim); margin-top:4px';
                        area.appendChild(nameDiv);
                    }
                    preview.src = e.target.result;
                    nameDiv.textContent = fileName;
                };
                reader.readAsDataURL(input.files[0]);
            }
        }

        function clearSimInputs() {
            ['orig', 'crop'].forEach(type => {
                const area = document.getElementById('sim-' + type + '-area');
                area.classList.remove('has-image');
                // 清除文件
                const input = document.getElementById('sim-' + type + '-file');
                input.value = '';
                // 移除预览
                const preview = area.querySelector('.sim-preview');
                if (preview) preview.remove();
                const nameDiv = area.querySelector('.sim-file-name');
                if (nameDiv) nameDiv.remove();
                // 恢复上传提示
                area.querySelectorAll('.sim-upload-icon, .sim-upload-text').forEach(el => el.style.display = '');
            });
            document.getElementById('sim-result-info').innerHTML = '';
            document.getElementById('sim-result-display').innerHTML = `
                <div style="text-align:center; color:var(--text-dim); padding:40px 0">
                    <div style="font-size:48px; margin-bottom:12px">🔍</div>
                    <div>上传原图和子图后点击"开始匹配"</div>
                </div>`;
        }

        async function runSubImageMatch() {
            const origInput = document.getElementById('sim-orig-file');
            const cropInput = document.getElementById('sim-crop-file');

            if (!origInput.files || !origInput.files[0]) {
                showToast('请上传原图', 'error'); return;
            }
            if (!cropInput.files || !cropInput.files[0]) {
                showToast('请上传子图', 'error'); return;
            }

            const fd = new FormData();
            fd.append('original', origInput.files[0]);
            fd.append('crop', cropInput.files[0]);
            fd.append('confidence_threshold', document.getElementById('sim-conf-threshold').value);
            fd.append('min_matches', document.getElementById('sim-min-matches').value);

            document.getElementById('sim-result-info').innerHTML = '<div class="loading"><div class="spinner"></div>匹配中...</div>';
            document.getElementById('sim-result-display').innerHTML = '<div style="text-align:center; padding:40px"><div class="spinner"></div></div>';

            try {
                const res = await fetch('/api/sub_image_match', { method: 'POST', body: fd });
                const data = await res.json();

                if (data.success) {
                    const r = data.result;
                    const statusColor = r.found ? 'var(--success)' : 'var(--danger)';
                    const statusText = r.found ? '✅ 匹配成功' : '❌ 匹配失败';

                    document.getElementById('sim-result-info').innerHTML = `
                        <div class="result-box" style="border-left: 3px solid ${statusColor}">
                            <div class="result-title" style="color:${statusColor}">${statusText}</div>
                            <div class="sim-info-grid">
                                <div class="sim-info-item">
                                    <div class="sim-info-value">${(r.confidence * 100).toFixed(1)}%</div>
                                    <div class="sim-info-label">置信度</div>
                                </div>
                                <div class="sim-info-item">
                                    <div class="sim-info-value">${r.elapsed_ms.toFixed(0)}ms</div>
                                    <div class="sim-info-label">耗时</div>
                                </div>
                            </div>
                            <div style="margin-top:10px; font-size:12px; color:var(--text-dim)">
                                <div>方法: ${r.method}</div>
                                ${r.found ? `<div>左上角: (${r.top_left_pct.x.toFixed(2)}, ${r.top_left_pct.y.toFixed(2)})</div>
                                <div>右下角: (${r.bottom_right_pct.x.toFixed(2)}, ${r.bottom_right_pct.y.toFixed(2)})</div>
                                <div>像素: (${r.bbox_pixel.x_min}, ${r.bbox_pixel.y_min}) → (${r.bbox_pixel.x_max}, ${r.bbox_pixel.y_max})</div>` : ''}
                            </div>
                        </div>`;

                    // Display annotated and matches images
                    let html = '<div class="sim-result-images">';
                    if (data.annotated_image) {
                        html += `<div>
                            <div style="font-size:13px; font-weight:600; margin-bottom:8px">📍 定位结果</div>
                            <img src="data:image/jpeg;base64,${data.annotated_image}" class="sim-result-img">
                        </div>`;
                    }
                    if (data.matches_image) {
                        html += `<div>
                            <div style="font-size:13px; font-weight:600; margin-bottom:8px">🔗 特征匹配</div>
                            <img src="data:image/jpeg;base64,${data.matches_image}" class="sim-result-img">
                        </div>`;
                    }
                    html += '</div>';
                    document.getElementById('sim-result-display').innerHTML = html;
                } else {
                    document.getElementById('sim-result-info').innerHTML = `
                        <div class="result-box result-error">
                            <div class="result-title">❌ 错误</div>
                            <div>${data.error}</div>
                        </div>`;
                    document.getElementById('sim-result-display').innerHTML = '';
                }
            } catch(e) {
                document.getElementById('sim-result-info').innerHTML = `
                    <div class="result-box result-error">
                        <div class="result-title">❌ 请求错误</div>
                        <div>${e.message}</div>
                    </div>`;
            }
        }

        
        // ============ 模型打点验证 ============
        function onGndFileChange(input) {
            const area = document.getElementById('gnd-img-area');
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    area.classList.add('has-image');
                    let preview = area.querySelector('.sim-preview');
                    if (!preview) {
                        area.querySelectorAll('.sim-upload-icon, .sim-upload-text').forEach(el => el.style.display = 'none');
                        preview = document.createElement('img');
                        preview.className = 'sim-preview';
                        area.appendChild(preview);
                    }
                    preview.src = e.target.result;
                };
                reader.readAsDataURL(input.files[0]);
            }
        }

        function clearGndInputs() {
            const area = document.getElementById('gnd-img-area');
            area.classList.remove('has-image');
            document.getElementById('gnd-img-file').value = '';
            const preview = area.querySelector('.sim-preview');
            if (preview) preview.remove();
            area.querySelectorAll('.sim-upload-icon, .sim-upload-text').forEach(el => el.style.display = '');
            document.getElementById('gnd-landmark').value = '';
            document.getElementById('gnd-result-info').innerHTML = '';
            document.getElementById('gnd-result-display').innerHTML = `
                <div style="text-align:center; color:var(--text-dim); padding:40px 0">
                    <div style="font-size:48px; margin-bottom:12px">🎯</div>
                    <div>上传图片并输入地标名称后点击"开始打点"</div>
                </div>`;
        }

        async function runGrounding() {
            const imgInput = document.getElementById('gnd-img-file');
            const landmark = document.getElementById('gnd-landmark').value.trim();

            if (!imgInput.files || !imgInput.files[0]) {
                showToast('请上传图片', 'error');
                return;
            }
            if (!landmark) {
                showToast('请输入地标名称', 'error');
                return;
            }

            document.getElementById('gnd-result-info').innerHTML = '<div class="loading"><div class="spinner"></div>Qwen3.5 打点推理中...</div>';
            document.getElementById('gnd-result-display').innerHTML = '<div style="text-align:center; padding:40px"><div class="spinner"></div></div>';

            try {
                const fd = new FormData();
                fd.append('image', imgInput.files[0]);
                fd.append('landmark_name', landmark);

                const res = await fetch('/api/point_grounding', { method: 'POST', body: fd });
                const data = await res.json();

                if (data.success) {
                    const r = data.result;
                    const statusColor = r.success ? 'var(--success)' : 'var(--danger)';
                    const statusText = r.success ? '✅ 打点成功' : '❌ 打点失败';

                    document.getElementById('gnd-result-info').innerHTML = `
                        <div class="result-box" style="border-left: 3px solid ${statusColor}">
                            <div class="result-title" style="color:${statusColor}">${statusText}</div>
                            <div class="sim-info-grid">
                                <div class="sim-info-item">
                                    <div class="sim-info-value">${r.success ? (r.confidence * 100).toFixed(0) + '%' : '-'}</div>
                                    <div class="sim-info-label">置信度</div>
                                </div>
                                <div class="sim-info-item">
                                    <div class="sim-info-value">${r.latency ? r.latency.toFixed(2) + 's' : '-'}</div>
                                    <div class="sim-info-label">耗时</div>
                                </div>
                            </div>
                            ${r.success ? `<div style="margin-top:10px; font-size:12px; color:var(--text-dim)">
                                <div>归一化坐标: [${r.point[0].toFixed(4)}, ${r.point[1].toFixed(4)}]</div>
                                <div>像素坐标: (${r.point_pixel[0]}, ${r.point_pixel[1]})</div>
                                <div>地标: ${landmark}</div>
                            </div>` : `<div style="margin-top:8px; font-size:12px; color:var(--danger)">${r.error || '未知错误'}</div>`}
                            ${r.raw_response ? `<div style="margin-top:8px; font-size:11px; color:var(--text-dim)">模型原始输出: ${r.raw_response.substring(0, 200)}</div>` : ''}
                        </div>`;

                    if (data.annotated_image) {
                        document.getElementById('gnd-result-display').innerHTML = `
                            <div class="sim-result-images">
                                <div>
                                    <div style="font-size:13px; font-weight:600; margin-bottom:8px">🎯 Qwen3.5 打点结果</div>
                                    <img src="data:image/jpeg;base64,${data.annotated_image}" class="sim-result-img">
                                </div>
                            </div>`;
                    }
                } else {
                    document.getElementById('gnd-result-info').innerHTML = `
                        <div class="result-box result-error">
                            <div class="result-title">❌ 错误</div>
                            <div>${data.error}</div>
                        </div>`;
                    document.getElementById('gnd-result-display').innerHTML = '';
                }
            } catch(e) {
                document.getElementById('gnd-result-info').innerHTML = `
                    <div class="result-box result-error">
                        <div class="result-title">❌ 请求错误</div>
                        <div>${e.message}</div>
                    </div>`;
            }
        }

        // 初始化
        window.onload = refreshGraph;
    </script>
</body>
</html>
'''


class MemoryNavServer:
    """记忆导航可视化服务器 v4.0"""
    
    def __init__(self, port: int = 9530, data_dir: str = None):
        self.port = port
        self.data_dir = data_dir or str(project_root / "merged_labeled_data")
        self.cache_path = str(project_root / "deploy/memory_nav/memory_cache")
        
        # memory_nav 组件
        self.memory_graph: Optional[MemoryGraph] = None
        self.memory_vpr: Optional[MemoryVPR] = None
        self.memory_navigator: Optional[MemoryNavigator] = None
        
        # Flask app
        if FLASK_AVAILABLE:
            self.app = Flask(__name__, static_folder='static', static_url_path='/static')
            CORS(self.app)
            self._setup_routes()
        
        # 初始化记忆系统
        self._init_memory()
    
    def _init_memory(self):
        """初始化记忆系统"""
        if not MEMORY_NAV_AVAILABLE:
            logger.error("memory_nav 模块不可用")
            return

        try:
            # 先创建 Navigator（内部创建 VPR extractor + SubImageMatcher）
            self.memory_navigator = MemoryNavigator(vpr_method=VPR_METHOD, device=VPR_DEVICE, preload_all_matchers=False)

            # 用 Navigator 的 load_memory 加载数据，内部复用同一个 extractor
            self.memory_navigator.load_memory(
                path=self.cache_path,
                data_dir=self.data_dir
            )

            self.memory_graph = self.memory_navigator.graph
            self.memory_vpr = self.memory_navigator.vpr

            logger.info(f"memory_nav 初始化成功: {len(self.memory_graph.nodes)} 节点")

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()

    def _save_graph(self):
        """保存图数据到缓存 (使用 MemoryGraph.save() 确保格式兼容)"""
        cache_path = str(project_root / "deploy/memory_nav/memory_cache_graph.pkl")
        self.memory_graph.save(cache_path)
        logger.info(f"图数据已保存: {len(self.memory_graph.nodes)} 节点")
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
        
        @self.app.route('/api/graph')
        def get_graph():
            """获取完整图数据"""
            if self.memory_graph is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            nodes = []
            edges = []
            edge_set = set()
            
            # 先收集所有有向边，用于检测双向
            directed_edges = set()
            for node_id, node in self.memory_graph.nodes.items():
                for edge in node.edges:
                    if edge.target_node_id in self.memory_graph.nodes:
                        directed_edges.add((node_id, edge.target_node_id))
            
            for node_id, node in self.memory_graph.nodes.items():
                nodes.append({
                    'id': node_id,
                    'label': f"{node_id}\n{node.node_name}" + (f"\n{getattr(node, 'node_name_eng', '')}" if getattr(node, 'node_name_eng', '') else ""),
                    'node_name': node.node_name,
                    'node_name_eng': getattr(node, 'node_name_eng', ''),
                    'neighbor_count': len(node.edges),
                    'has_features': node.fused_feature is not None,
                    'camera_images': node.camera_images,
                    'base_path': node.base_path,
                    'timestamp': node.timestamp
                })
                
                for edge in node.edges:
                    if edge.target_node_id not in self.memory_graph.nodes:
                        continue
                    edge_key = tuple(sorted([node_id, edge.target_node_id]))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        # 检测是否双向
                        reverse_exists = (edge.target_node_id, node_id) in directed_edges
                        edges.append({
                            'from': node_id,
                            'to': edge.target_node_id,
                            'camera_name': edge.camera_name,
                            'landmark_name': edge.landmark_name,
                            'landmark_name_eng': getattr(edge, 'landmark_name_eng', ''),
                            'crop_image_path': edge.crop_image_path,
                            'crop_image_paths': edge.crop_image_paths,
                            'target_name': edge.target_node_name,
                            'target_name_eng': getattr(edge, 'target_node_name_eng', ''),
                            'bidirectional': reverse_exists
                        })
            
            return jsonify({'success': True, 'data': {'nodes': nodes, 'edges': edges}})
        
        @self.app.route('/api/node/<node_id>')
        def get_node(node_id):
            """获取节点详情"""
            if self.memory_graph is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            if node_id not in self.memory_graph.nodes:
                return jsonify({'success': False, 'error': f'节点 {node_id} 不存在'})
            
            node = self.memory_graph.nodes[node_id]
            
            neighbors = []
            for edge in node.edges:
                neighbors.append({
                    'id': edge.target_node_id,
                    'name': edge.target_node_name,
                    'name_eng': getattr(edge, 'target_node_name_eng', ''),
                    'camera_name': edge.camera_name,
                    'landmark_name': edge.landmark_name,
                    'landmark_name_eng': getattr(edge, 'landmark_name_eng', ''),
                    'crop_image_path': edge.crop_image_path,
                    'crop_image_paths': edge.crop_image_paths,
                })
            
            return jsonify({
                'success': True,
                'node': {
                    'id': node_id,
                    'node_name': node.node_name,
                    'node_name_eng': getattr(node, 'node_name_eng', ''),
                    'camera_images': node.camera_images,
                    'base_path': node.base_path,
                    'timestamp': node.timestamp,
                    'neighbor_count': len(node.edges),
                    'has_features': node.fused_feature is not None,
                    'neighbors': neighbors
                }
            })
        
        @self.app.route('/api/navigate', methods=['POST'])
        def navigate():
            """路径规划"""
            if self.memory_navigator is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            data = request.json or {}
            start = data.get('start')
            goal = data.get('goal')
            
            if not start or not goal:
                return jsonify({'success': False, 'error': '请提供起点和终点'})
            
            self.memory_navigator.set_current_node(str(start))
            result = self.memory_navigator.navigate_to(str(goal), start_node_id=str(start))
            
            if result['success']:
                plan = result['plan']
                steps = [{
                    'step_index': s.step_index,
                    'from_node': {'id': s.from_node_id, 'name': s.from_node_name, 'name_eng': getattr(s, 'from_node_name_eng', '')},
                    'to_node': {'id': s.to_node_id, 'name': s.to_node_name, 'name_eng': getattr(s, 'to_node_name_eng', '')},
                    'camera_name': s.camera_name,
                    'landmark_name': s.landmark_name,
                    'landmark_name_eng': getattr(s, 'landmark_name_eng', ''),
                    'crop_image_path': s.crop_image_path,
                    'crop_image_paths': s.crop_image_paths,
                } for s in plan.steps]
                
                return jsonify({
                    'success': True,
                    'message': result['message'],
                    'path': plan.path,
                    'total_steps': plan.total_steps,
                    'steps': steps
                })
            
            return jsonify({'success': False, 'error': result['message']})
        
        @self.app.route('/api/destinations')
        def get_destinations():
            """获取所有目的地"""
            if self.memory_navigator is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            destinations = self.memory_navigator.get_all_destinations()
            return jsonify({
                'success': True,
                'destinations': [{'id': d[0], 'name': d[1]} for d in destinations]
            })
        
        @self.app.route('/api/vpr', methods=['POST'])
        def vpr_locate():
            """VPR定位"""
            if self.memory_navigator is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            import cv2
            camera_images = {}
            for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
                if cam_id in request.files:
                    file = request.files[cam_id]
                    nparr = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    camera_images[cam_id] = img
            
            if not camera_images:
                return jsonify({'success': False, 'error': '未上传图片'})
            
            result = self.memory_navigator.locate_by_images(camera_images)
            
            if result:
                return jsonify({
                    'success': True,
                    'matched_node_id': result.matched_node_id,
                    'matched_node_name': result.matched_node_name,
                    'matched_node_name_eng': getattr(result, 'matched_node_name_eng', ''),
                    'confidence': result.confidence,
                    'camera_scores': result.camera_scores,
                    'heading_offset': result.heading_offset,
                    'best_shift': result.best_shift,
                    'similarity': result.similarity
                })
            
            return jsonify({'success': False, 'error': 'VPR定位失败'})
        
        @self.app.route('/api/image')
        def get_image():
            """获取图片文件"""
            path = request.args.get('path', '')
            if not path:
                return jsonify({'error': '图片路径为空'}), 404
            # 相对路径转绝对路径（基于项目根目录）
            if not os.path.isabs(path):
                path = str(project_root / path)
            if not os.path.exists(path):
                return jsonify({'error': f'图片不存在: {path}'}), 404
            return send_file(path)
        
        @self.app.route('/api/status')
        def get_status():
            """获取系统状态"""
            return jsonify({
                'success': True,
                'memory_nav_available': self.memory_graph is not None,
                'total_nodes': len(self.memory_graph.nodes) if self.memory_graph else 0,
                'navigator_status': self.memory_navigator.get_status() if self.memory_navigator else None
            })
        
        # ========================================================================
        # 子图匹配验证 API
        # ========================================================================

        @self.app.route('/api/sub_image_match', methods=['POST'])
        def sub_image_match():
            """子图匹配验证"""
            if self.memory_navigator is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})

            import cv2
            import base64

            if 'original' not in request.files or 'crop' not in request.files:
                return jsonify({'success': False, 'error': '请上传原图和子图'})

            # 读取原图
            orig_file = request.files['original']
            orig_nparr = np.frombuffer(orig_file.read(), np.uint8)
            orig_img = cv2.imdecode(orig_nparr, cv2.IMREAD_COLOR)

            # 读取子图
            crop_file = request.files['crop']
            crop_nparr = np.frombuffer(crop_file.read(), np.uint8)
            crop_img = cv2.imdecode(crop_nparr, cv2.IMREAD_COLOR)

            if orig_img is None or crop_img is None:
                return jsonify({'success': False, 'error': '图片解码失败'})

            # 读取参数
            confidence_threshold = float(request.form.get('confidence_threshold', 0.3))
            min_matches = int(request.form.get('min_matches', 8))
            method = request.form.get('method', None)  # None = 使用默认方案

            # 临时调整参数
            old_conf = self.memory_navigator.sub_image_matcher.confidence_threshold
            old_min = self.memory_navigator.sub_image_matcher.min_matches
            self.memory_navigator.sub_image_matcher.confidence_threshold = confidence_threshold
            self.memory_navigator.sub_image_matcher.min_matches = min_matches

            try:
                result = self.memory_navigator.sub_image_matcher.match(orig_img, crop_img, method=method)
            finally:
                self.memory_navigator.sub_image_matcher.confidence_threshold = old_conf
                self.memory_navigator.sub_image_matcher.min_matches = old_min

            # 绘制标注图
            annotated = orig_img.copy()
            if result.found:
                cv2.rectangle(annotated, (result.x_min, result.y_min),
                              (result.x_max, result.y_max), (0, 255, 0), 3)
                label = f"Conf: {result.confidence:.3f}"
                cv2.putText(annotated, label,
                            (result.x_min, max(result.y_min - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(annotated, "No match found", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # 限制输出尺寸
            max_dim = 800
            h, w = annotated.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_b64 = base64.b64encode(buf).decode('utf-8')

            # 返回结果
            resp = {
                'success': True,
                'result': result.to_dict(),
                'annotated_image': annotated_b64,
                'matches_image': None
            }

            return jsonify(resp)

        @self.app.route('/api/sub_image_match_methods')
        def sub_image_match_methods():
            """获取可用的子图匹配方案列表"""
            from deploy.memory_nav.sub_image_matcher import list_strategies, STRATEGY_DISPLAY_NAMES
            methods = []
            for key in list_strategies():
                methods.append({
                    'key': key,
                    'name': STRATEGY_DISPLAY_NAMES.get(key, key),
                })
            default_method = 'dinov3'
            if self.memory_navigator:
                default_method = self.memory_navigator.sub_image_matcher.default_method
            return jsonify({
                'success': True,
                'methods': methods,
                'default': default_method,
            })

                # ========================================================================
        # 模型打点 API (Qwen3.5)
        # ========================================================================

        @self.app.route('/api/point_grounding', methods=['POST'])
        def point_grounding():
            """Qwen3.5 模型打点"""
            import cv2
            import base64

            landmark_name = request.form.get('landmark_name', '').strip()
            if not landmark_name:
                return jsonify({'success': False, 'error': '请输入地标名称 (landmark_name)'})

            # 获取图片: 上传文件 或 服务器路径
            image = None
            if 'image' in request.files:
                file = request.files['image']
                nparr = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif request.form.get('image_path'):
                img_path = request.form['image_path']
                if not os.path.isabs(img_path):
                    img_path = str(project_root / img_path)
                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                else:
                    return jsonify({'success': False, 'error': f'图片不存在: {img_path}'})

            if image is None:
                return jsonify({'success': False, 'error': '请上传图片或提供图片路径'})

            if self.memory_navigator is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})

            # 使用 MemoryNavigator 中的 Qwen3.5 打点器 (模型在 ws_proxy 启动时加载)
            grounder = self.memory_navigator.qwen35_grounder
            if not grounder.is_ready:
                try:
                    grounder.start()
                except Exception as e:
                    return jsonify({'success': False, 'error': f'Qwen3.5 启动失败: {e}'})

            # 执行打点
            result = grounder.predict(image, landmark_name)

            # 绘制标注图
            annotated = image.copy()
            h, w = annotated.shape[:2]
            if result.get('success') and result.get('point'):
                px, py = result['point_pixel']
                # 十字准星
                cv2.line(annotated, (px - 30, py), (px + 30, py), (0, 102, 255), 3)
                cv2.line(annotated, (px, py - 30), (px, py + 30), (0, 102, 255), 3)
                # 圆心
                cv2.circle(annotated, (px, py), 8, (0, 102, 255), -1)
                cv2.circle(annotated, (px, py), 10, (255, 255, 255), 2)
                # 同心圆
                cv2.circle(annotated, (px, py), 24, (0, 102, 255), 2)
                # 坐标标注
                label = f"[{result['point'][0]:.3f}, {result['point'][1]:.3f}]"
                cv2.putText(annotated, label, (px + 20, py - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 102, 255), 2)
                # 地标名称
                cv2.putText(annotated, f"Landmark: {landmark_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(annotated, f"FAILED: {landmark_name}", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # 限制输出尺寸
            max_dim = 800
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_b64 = base64.b64encode(buf).decode('utf-8')

            return jsonify({
                'success': True,
                'result': result,
                'annotated_image': annotated_b64,
            })

                # ========================================================================
        # 数据库管理 API
        # ========================================================================
        
        @self.app.route('/api/db/all')
        def get_all_data():
            """获取所有节点和边数据"""
            if self.memory_graph is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            nodes = []
            edges = []
            edge_set = set()
            
            for node_id, node in self.memory_graph.nodes.items():
                nodes.append({
                    'id': node_id,
                    'node_name': node.node_name,
                    'node_name_eng': getattr(node, 'node_name_eng', ''),
                    'data_path': getattr(node, 'base_path', ''),
                    'neighbor_count': len(node.edges)
                })
                
                for edge in node.edges:
                    if edge.target_node_id not in self.memory_graph.nodes:
                        continue
                    edge_key = tuple(sorted([node_id, edge.target_node_id]))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        tgt = self.memory_graph.nodes.get(edge.target_node_id)
                        edges.append({
                            'from': node_id,
                            'to': edge.target_node_id,
                            'from_name': node.node_name,
                            'from_name_eng': getattr(node, 'node_name_eng', ''),
                            'to_name': tgt.node_name if tgt else '',
                            'to_name_eng': getattr(tgt, 'node_name_eng', '') if tgt else '',
                            'camera_name': edge.camera_name,
                            'landmark_name': edge.landmark_name,
                            'landmark_name_eng': getattr(edge, 'landmark_name_eng', ''),
                            'crop_image_path': edge.crop_image_path,
                            'crop_image_paths': edge.crop_image_paths,
                        })
            
            return jsonify({'success': True, 'nodes': nodes, 'edges': edges})
        
        @self.app.route('/api/db/node', methods=['POST'])
        def add_node():
            """添加节点"""
            if self.memory_graph is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            data = request.json or {}
            node_id = str(data.get('id', ''))
            name = data.get('name', '')
            data_path = data.get('data_path', '')
            
            if not node_id or not name:
                return jsonify({'success': False, 'error': '缺少必要参数 (id, name)'})
            
            if node_id in self.memory_graph.nodes:
                return jsonify({'success': False, 'error': '节点ID已存在'})
            
            new_node = MemoryNode(
                node_id=node_id,
                node_name=name,
                node_name_eng=data.get('name_eng', ''),
                base_path=data_path
            )
            self.memory_graph.nodes[node_id] = new_node
            self._save_graph()
            return jsonify({'success': True})
        
        @self.app.route('/api/db/node/<node_id>', methods=['PUT'])
        def update_node(node_id):
            """更新节点"""
            if self.memory_graph is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            if node_id not in self.memory_graph.nodes:
                return jsonify({'success': False, 'error': '节点不存在'})
            
            data = request.json or {}
            node = self.memory_graph.nodes[node_id]
            
            if 'name' in data:
                node.node_name = data['name']
                if 'name_eng' in data:
                    node.node_name_eng = data['name_eng']
            if 'data_path' in data:
                node.base_path = data['data_path']
            
            self._save_graph()
            return jsonify({'success': True})
        
        @self.app.route('/api/db/node/<node_id>', methods=['DELETE'])
        def delete_node(node_id):
            """删除节点"""
            if self.memory_graph is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            if node_id not in self.memory_graph.nodes:
                return jsonify({'success': False, 'error': '节点不存在'})
            
            # 删除其他节点指向此节点的边
            for nid, node in self.memory_graph.nodes.items():
                node.edges = [e for e in node.edges if e.target_node_id != node_id]
            
            del self.memory_graph.nodes[node_id]
            self._save_graph()
            return jsonify({'success': True})
        
        @self.app.route('/api/db/edge', methods=['POST'])
        def add_edge():
            """添加边"""
            if self.memory_graph is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            data = request.json or {}
            from_id = str(data.get('from_id', ''))
            to_id = str(data.get('to_id', ''))
            camera_name = data.get('camera_name', 'camera_1')
            landmark_name = data.get('landmark_name', '')
            
            if not from_id or not to_id:
                return jsonify({'success': False, 'error': '缺少必要参数 (from_id, to_id)'})
            
            if from_id not in self.memory_graph.nodes:
                return jsonify({'success': False, 'error': f'起点节点 {from_id} 不存在'})
            if to_id not in self.memory_graph.nodes:
                return jsonify({'success': False, 'error': f'终点节点 {to_id} 不存在'})
            
            src_node = self.memory_graph.nodes[from_id]
            tgt_node = self.memory_graph.nodes[to_id]
            
            # 检查边是否已存在
            for e in src_node.edges:
                if e.target_node_id == to_id:
                    return jsonify({'success': False, 'error': '边已存在'})
            
            new_edge = MemoryEdge(
                target_node_id=to_id,
                target_node_name=tgt_node.node_name,
                target_node_name_eng=getattr(tgt_node, 'node_name_eng', ''),
                camera_name=camera_name,
                landmark_name=landmark_name,
                crop_image_path='',
            )
            src_node.edges.append(new_edge)
            self._save_graph()
            return jsonify({'success': True})
        
        @self.app.route('/api/db/edge', methods=['DELETE'])
        def delete_edge():
            """删除边"""
            if self.memory_graph is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            from_id = request.args.get('from', '')
            to_id = request.args.get('to', '')
            
            if not from_id or not to_id:
                return jsonify({'success': False, 'error': '缺少参数'})
            
            if from_id not in self.memory_graph.nodes:
                return jsonify({'success': False, 'error': '起点节点不存在'})
            
            src_node = self.memory_graph.nodes[from_id]
            original_len = len(src_node.edges)
            src_node.edges = [e for e in src_node.edges if e.target_node_id != to_id]
            
            if len(src_node.edges) == original_len:
                return jsonify({'success': False, 'error': '边不存在'})
            
            self._save_graph()
            return jsonify({'success': True})
        
        @self.app.route('/api/db/export')
        def export_data():
            """导出数据为JSON"""
            if self.memory_graph is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            nodes = []
            edges = []
            
            for node_id, node in self.memory_graph.nodes.items():
                nodes.append({
                    'id': node_id,
                    'name': node.node_name,
                    'name_eng': getattr(node, 'node_name_eng', ''),
                    'base_path': getattr(node, 'base_path', '')
                })
                
                for edge in node.edges:
                    edges.append({
                        'from': node_id,
                        'to': edge.target_node_id,
                        'camera_name': edge.camera_name,
                        'landmark_name': edge.landmark_name,
                        'landmark_name_eng': getattr(edge, 'landmark_name_eng', ''),
                            'crop_image_path': edge.crop_image_path,
                        'crop_image_paths': edge.crop_image_paths,
                    })
            
            return jsonify({
                'nodes': nodes,
                'edges': edges,
                'exported_at': str(np.datetime64('now'))
            })
        
        @self.app.route('/api/db/import', methods=['POST'])
        def import_data():
            """从目录导入数据"""
            if self.memory_graph is None:
                return jsonify({'success': False, 'error': 'memory_nav 未初始化'})
            
            data = request.json or {}
            path = data.get('path', '')
            clear = data.get('clear', True)
            
            if not path or not os.path.exists(path):
                return jsonify({'success': False, 'error': f'数据路径不存在: {path}'})
            
            try:
                if clear:
                    self.memory_graph.nodes.clear()
                
                # 重新构建
                builder = MemoryBuilder(
                    feature_extractor=self.memory_navigator.extractor,
                    feature_dim=self.memory_navigator.feature_dim,
                    vpr_method=VPR_METHOD, device=VPR_DEVICE
                )
                new_graph, new_vpr = builder.build_from_directory(
                    path, extract_features=True, save_path=self.cache_path
                )
                
                self.memory_graph = new_graph
                self.memory_vpr = new_vpr
                self.memory_navigator.set_memory(self.memory_graph, self.memory_vpr)
                
                return jsonify({
                    'success': True,
                    'nodes_count': len(self.memory_graph.nodes),
                    'edges_count': sum(len(n.edges) for n in self.memory_graph.nodes.values())
                })
            except Exception as e:
                logger.error(f"导入失败: {e}")
                return jsonify({'success': False, 'error': str(e)})
    
    def run(self):
        """启动服务器"""
        if not FLASK_AVAILABLE:
            logger.error("Flask 不可用")
            return
        
        logger.info(f"启动 MemoryNav 可视化服务器，端口: {self.port}")
        logger.info(f"访问: http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


def main():
    parser = argparse.ArgumentParser(description='MemoryNav 可视化服务器')
    parser.add_argument('--port', type=int, default=9530, help='服务器端口')
    parser.add_argument('--data', type=str, default=None, help='数据目录')
    args = parser.parse_args()
    
    server = MemoryNavServer(port=args.port, data_dir=args.data)
    server.run()


if __name__ == '__main__':
    main()
