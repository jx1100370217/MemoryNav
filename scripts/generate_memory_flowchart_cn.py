#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 详细流程图生成器 v2.0

生成记忆功能开启时的导航完整流程图，包括：
1. 系统初始化流程
2. 多视角独立特征提取流程 (不融合，保留各视角独立编码)
3. 多视角VPR回环检测流程 (独立FAISS索引 + 投票机制)
4. Dijkstra最短路径规划流程
5. 拓扑图管理流程 (双向图 + 节点合并)
6. 语义搜索流程 (同义词扩展 + 时序上下文 + 场景记忆)
7. 动作输出流程

v2.0 新特性:
- 多视角独立编码: camera_1~4 各自独立存储，不做加权融合
- 多视角投票VPR: 每个视角独立FAISS检索，投票决定匹配
- Dijkstra最短路径: 支持任意起点到目标的最短路径规划
- 双向拓扑图: NetworkX Graph (非DiGraph)，支持双向导航
- 节点合并: 相似度超过阈值自动合并节点

对标业界标准:
- DPV-SLAM with AnyLoc (2026): 自适应阈值机制
- TopoNav (2025): 拓扑图结构化记忆
- Meta-Memory (2025): 语义-空间联合推理
- VLM² (2025): 工作记忆 + 情景记忆双结构
- ReMEmbR (ICRA 2025): 时序空间信息整合

Author: Memory Navigation Team
Date: 2026-01-26
功能: 多视角独立VPR + Dijkstra最短路径 + 节点合并 + 双向导航
"""

from graphviz import Digraph
from pathlib import Path

# 颜色方案 - 专业且美观的配色
COLORS = {
    # 主题色
    'primary': '#2563EB',      # 蓝色 - 主要流程
    'secondary': '#7C3AED',    # 紫色 - 记忆相关
    'success': '#059669',      # 绿色 - 成功/输出
    'warning': '#D97706',      # 橙色 - 决策节点
    'danger': '#DC2626',       # 红色 - 关键节点
    'info': '#0891B2',         # 青色 - 信息节点

    # 背景色
    'bg_input': '#DBEAFE',     # 浅蓝 - 输入模块
    'bg_feature': '#E0E7FF',   # 浅紫 - 特征提取
    'bg_vpr': '#FEF3C7',       # 浅黄 - VPR模块
    'bg_memory': '#D1FAE5',    # 浅绿 - 记忆模块
    'bg_topo': '#FCE7F3',      # 浅粉 - 拓扑图
    'bg_output': '#F3E8FF',    # 浅紫 - 输出模块
    'bg_decision': '#FEE2E2',  # 浅红 - 决策模块
    'bg_semantic': '#E0F2FE',  # 浅天蓝 - 语义模块

    # 文字色
    'text_dark': '#1F2937',
    'text_light': '#F9FAFB',

    # 边框色
    'border': '#6B7280',
}


def create_memory_navigation_overview():
    """创建记忆导航系统总览图 v2.0 (多视角独立编码 + Dijkstra最短路径)"""
    dot = Digraph(comment='Memory Navigation System Overview v2.0')
    dot.attr(rankdir='TB', bgcolor='white', fontname='SimHei')
    dot.attr('node', fontname='SimHei', fontsize='11')
    dot.attr('edge', fontname='SimHei', fontsize='9')

    # 标题
    dot.node('title',
             '记忆导航系统架构总览 v2.0\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nMemory Navigation System Overview\n(多视角独立VPR + Dijkstra最短路径 + 节点合并)',
             shape='note', style='filled', fillcolor='#1E3A8A',
             fontcolor='white', fontsize='16', width='6')

    # 输入层
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入层 (Input Layer)', style='filled',
               fillcolor=COLORS['bg_input'], fontcolor=COLORS['text_dark'])
        c.node('ws_input',
               'WebSocket 输入\n━━━━━━━━━━━━━━━━━━━━\n• front_1 (推理用)\n• camera_1~4 (记忆用)\n• 任务指令 (instruction)',
               shape='folder', style='filled', fillcolor='#93C5FD')
        c.node('robot_state',
               '机器人状态\n━━━━━━━━━━━━━━━━━━━━\n• 位姿 (pose)\n• 里程计 (odom)\n• 时间戳 (timestamp)',
               shape='folder', style='filled', fillcolor='#93C5FD')

    # 特征提取层 (v2.0: 多视角独立编码)
    with dot.subgraph(name='cluster_feature') as c:
        c.attr(label='特征提取层 (Feature Extraction) - 多视角独立编码', style='filled',
               fillcolor=COLORS['bg_feature'], fontcolor=COLORS['text_dark'])
        c.node('longclip',
               'LongCLIP 特征提取器\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 512维视觉特征\n• 仅处理环视相机\n• GPU: cuda:1',
               shape='box3d', style='filled', fillcolor='#A5B4FC')
        c.node('multi_view_features',
               '[v2.0] 多视角独立编码\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• camera_1: +37.5° 前右\n• camera_2: -37.5° 前左\n• camera_3: -142.5° 后左\n• camera_4: +142.5° 后右\n• 各视角独立存储，不融合',
               shape='box3d', style='filled', fillcolor='#A5B4FC')

    # 记忆核心层 v2.0
    with dot.subgraph(name='cluster_memory') as c:
        c.attr(label='记忆核心层 (Memory Core) - v2.0 增强', style='filled',
               fillcolor=COLORS['bg_memory'], fontcolor=COLORS['text_dark'])
        c.node('vpr',
               '[v2.0] 多视角VPR\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 4个独立FAISS索引\n• 多视角投票机制\n• 加权相似度融合\n• 语义引导 + 时序验证\n• 最佳视角匹配',
               shape='component', style='filled', fillcolor='#6EE7B7')
        c.node('topo_map',
               '[v2.0] 拓扑地图管理\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 双向图 (nx.Graph)\n• Dijkstra最短路径\n• 自动节点合并\n• 双向边关系维护\n• GraphRAG 语义图',
               shape='component', style='filled', fillcolor='#6EE7B7')
        c.node('path_planner',
               '[v2.0] 路径规划器\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• Dijkstra算法\n• 任意起点导航\n• 路径点序列输出\n• 动作序列估计',
               shape='component', style='filled', fillcolor='#34D399')
        c.node('semantic_graph',
               '语义图管理\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 28组同义词字典\n• Levenshtein模糊匹配\n• 向量语义搜索',
               shape='component', style='filled', fillcolor='#34D399')

    # 决策层
    with dot.subgraph(name='cluster_decision') as c:
        c.attr(label='决策层 (Decision Layer)', style='filled',
               fillcolor=COLORS['bg_decision'], fontcolor=COLORS['text_dark'])
        c.node('memory_check',
               '记忆匹配检查\n━━━━━━━━━━━━━━━━━━━━━━\n1. 多视角VPR匹配\n2. 语义搜索匹配\n3. 最短路径规划',
               shape='diamond', style='filled', fillcolor='#FCA5A5')
        c.node('navigation_decision',
               '导航决策\n━━━━━━━━━━━━━━━━━━━━━━\n使用最短路径导航\n还是正常推理?',
               shape='diamond', style='filled', fillcolor='#FCA5A5')

    # 模型推理层
    with dot.subgraph(name='cluster_inference') as c:
        c.attr(label='模型推理层 (仅前置相机 front_1)', style='filled',
               fillcolor='#F3E8FF', fontcolor=COLORS['text_dark'])
        c.node('internvla',
               'InternVLA 模型推理\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 视觉-语言导航\n• 动作预测\n• 仅使用 front_1',
               shape='box3d', style='filled', fillcolor='#A78BFA')

    # 输出层
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出层 (Output Layer)', style='filled',
               fillcolor=COLORS['bg_output'], fontcolor=COLORS['text_dark'])
        c.node('action_output',
               '动作输出\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 线速度 (vx, vy)\n• 角速度 (omega)\n• 像素目标 (pixel_target)',
               shape='box', style='filled,rounded', fillcolor='#C4B5FD')
        c.node('task_status',
               '任务状态\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• executing (执行中)\n• completed (已完成)\n• failed (失败)',
               shape='box', style='filled,rounded', fillcolor='#C4B5FD')

    # 功能说明节点 v2.0
    dot.node('note_features',
             'v2.0 核心功能\n━━━━━━━━━━━━━━━━━━━━━━━━\n• 多视角独立FAISS索引\n• 投票机制VPR匹配\n• Dijkstra最短路径规划\n• 双向拓扑图导航\n• 自动节点合并 (sim>0.9)',
             shape='note', style='filled', fillcolor='#D1FAE5', fontcolor='#059669')

    # 连接关系
    dot.edge('title', 'ws_input', style='invis')

    # 前置相机路径
    dot.edge('ws_input', 'internvla', label='front_1\n(前置相机)', color='#7C3AED', penwidth='2')

    # 环视相机路径 - v2.0多视角独立编码
    dot.edge('ws_input', 'longclip', label='camera_1~4\n(环视相机)', color='#059669', penwidth='2')
    dot.edge('longclip', 'multi_view_features', label='4个独立512维特征')
    dot.edge('multi_view_features', 'vpr', label='多视角独立特征\n(不融合)')

    dot.edge('robot_state', 'topo_map', label='位姿信息', style='dashed')

    dot.edge('vpr', 'topo_map', label='多视角匹配结果')
    dot.edge('topo_map', 'path_planner', label='双向图结构')
    dot.edge('topo_map', 'semantic_graph', label='语义信息')

    dot.edge('path_planner', 'memory_check', label='最短路径')
    dot.edge('semantic_graph', 'memory_check', label='语义匹配', style='dashed')
    dot.edge('memory_check', 'navigation_decision', label='检查结果')

    dot.edge('internvla', 'action_output', label='推理动作', color='#7C3AED')
    dot.edge('navigation_decision', 'action_output', label='路径导航动作', color='#059669')
    dot.edge('action_output', 'task_status', label='执行反馈')

    dot.edge('note_features', 'vpr', style='dashed', constraint='false')

    return dot


def create_vpr_detection_flow():
    """创建VPR回环检测详细流程图 v2.0 (多视角独立搜索 + 投票机制)"""
    dot = Digraph(comment='VPR Loop Closure Detection Flow v2.0')
    dot.attr(rankdir='TB', bgcolor='white', fontname='SimHei')
    dot.attr('node', fontname='SimHei', fontsize='10')
    dot.attr('edge', fontname='SimHei', fontsize='9')

    # 标题
    dot.node('title',
             '[v2.0] 多视角VPR回环检测流程\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n(多视角独立FAISS + 投票机制 + 语义引导)',
             shape='note', style='filled', fillcolor='#7C3AED',
             fontcolor='white', fontsize='14', width='5')

    # 输入处理 - v2.0多视角
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入处理 - 多视角独立特征', style='filled', fillcolor=COLORS['bg_input'])
        c.node('query_features',
               '[v2.0] 多视角特征字典\n━━━━━━━━━━━━━━━━━━━━\nDict[camera_id, 512维特征]\n• camera_1: +37.5° 前右\n• camera_2: -37.5° 前左\n• camera_3: -142.5° 后左\n• camera_4: +142.5° 后右',
               shape='parallelogram', style='filled', fillcolor='#93C5FD')
        c.node('query_semantic',
               '语义标签\n━━━━━━━━━━━━━━━━\nSet[str]\n场景语义信息',
               shape='parallelogram', style='filled', fillcolor='#6EE7B7')
        c.node('current_time',
               '当前时间戳\n━━━━━━━━━━━━━━━━\ntime.time()',
               shape='parallelogram', style='filled', fillcolor='#93C5FD')

    # v2.0: 多视角独立FAISS搜索
    with dot.subgraph(name='cluster_faiss') as c:
        c.attr(label='[v2.0] 多视角独立FAISS搜索', style='filled', fillcolor=COLORS['bg_vpr'])
        c.node('faiss_cam1',
               'camera_1 FAISS\n━━━━━━━━━━━━━━\n前右视角索引\nTop-K搜索',
               shape='box3d', style='filled', fillcolor='#FDE68A')
        c.node('faiss_cam2',
               'camera_2 FAISS\n━━━━━━━━━━━━━━\n前左视角索引\nTop-K搜索',
               shape='box3d', style='filled', fillcolor='#FDE68A')
        c.node('faiss_cam3',
               'camera_3 FAISS\n━━━━━━━━━━━━━━\n后左视角索引\nTop-K搜索',
               shape='box3d', style='filled', fillcolor='#FDE68A')
        c.node('faiss_cam4',
               'camera_4 FAISS\n━━━━━━━━━━━━━━\n后右视角索引\nTop-K搜索',
               shape='box3d', style='filled', fillcolor='#FDE68A')

    # v2.0: 多视角投票机制
    with dot.subgraph(name='cluster_voting') as c:
        c.attr(label='[v2.0] 多视角投票机制', style='filled', fillcolor='#E0F2FE')
        c.node('vote_aggregator',
               '投票聚合器\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 统计各节点得票数\n• 计算加权相似度\n• 记录最佳匹配相机',
               shape='box', style='filled,rounded', fillcolor='#7DD3FC')
        c.node('vote_result',
               'MultiViewVPRResult\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• node_id: 最佳节点\n• best_camera: 最佳视角\n• voting_score: 投票分数\n• weighted_similarity: 加权相似度\n• matching_cameras: 匹配视角列表',
               shape='note', style='filled', fillcolor='#38BDF8')

    # 语义引导
    with dot.subgraph(name='cluster_semantic') as c:
        c.attr(label='语义引导匹配', style='filled', fillcolor=COLORS['bg_semantic'])
        c.node('semantic_adjust',
               '语义相似度调整\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• Jaccard相似度计算\n• 语义匹配加分 +0.1\n• 语义冲突降分 -0.05',
               shape='box', style='filled,rounded', fillcolor='#7DD3FC')
        c.node('semantic_db',
               '语义标签数据库\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nList[Set[str]]\n每个节点的语义标签',
               shape='cylinder', style='filled', fillcolor='#38BDF8')

    # v2.0: 多视角匹配判定
    with dot.subgraph(name='cluster_match_decision') as c:
        c.attr(label='[v2.0] 多视角匹配判定', style='filled',
               fillcolor=COLORS['bg_decision'])
        c.node('voting_check',
               '投票分数检查\n━━━━━━━━━━━━━━━━━━━━\nvoting_score ≥ 0.5?\n(≥2个视角匹配)',
               shape='diamond', style='filled', fillcolor='#FCA5A5')
        c.node('best_sim_check',
               '最佳视角检查\n━━━━━━━━━━━━━━━━━━━━\nbest_similarity ≥ 0.88?\n(单视角强匹配)',
               shape='diamond', style='filled', fillcolor='#FBBF24')
        c.node('weighted_sim_check',
               '加权相似度检查\n━━━━━━━━━━━━━━━━━━━━\nweighted_sim ≥ 0.78?\n(整体匹配)',
               shape='diamond', style='filled', fillcolor='#F59E0B')

    # 时序验证
    with dot.subgraph(name='cluster_temporal') as c:
        c.attr(label='时序一致性验证', style='filled', fillcolor=COLORS['bg_memory'])
        c.node('temporal_window',
               '时序窗口\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• deque(maxlen=3)\n• 存储最近匹配\n• node_id + similarity',
               shape='cylinder', style='filled', fillcolor='#6EE7B7')
        c.node('weighted_score',
               '加权评分\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 相同节点: +0.4×sim\n• 相邻节点(±2): +0.2×sim\n• 阈值: 0.5',
               shape='box', style='filled,rounded', fillcolor='#34D399')

    # 输出结果
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出结果', style='filled', fillcolor=COLORS['bg_output'])
        c.node('match_found',
               '[v2.0] 多视角匹配成功\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n返回 MultiViewVPRResult:\n• node_id + best_camera\n• 各视角相似度详情',
               shape='box', style='filled,rounded', fillcolor='#A7F3D0')
        c.node('no_match',
               '无匹配\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n返回 None\n记录候选到窗口',
               shape='box', style='filled,rounded', fillcolor='#FED7AA')

    # 连接 - v2.0多视角流程
    dot.edge('title', 'query_features', style='invis')

    # 多视角特征分发到各FAISS
    dot.edge('query_features', 'faiss_cam1', label='camera_1特征')
    dot.edge('query_features', 'faiss_cam2', label='camera_2特征')
    dot.edge('query_features', 'faiss_cam3', label='camera_3特征')
    dot.edge('query_features', 'faiss_cam4', label='camera_4特征')

    # FAISS搜索结果汇聚到投票
    dot.edge('faiss_cam1', 'vote_aggregator')
    dot.edge('faiss_cam2', 'vote_aggregator')
    dot.edge('faiss_cam3', 'vote_aggregator')
    dot.edge('faiss_cam4', 'vote_aggregator')

    dot.edge('vote_aggregator', 'vote_result')
    dot.edge('query_semantic', 'semantic_adjust', label='语义标签')
    dot.edge('current_time', 'voting_check', label='时间戳', style='dashed')

    dot.edge('vote_result', 'semantic_adjust', label='投票结果')
    dot.edge('semantic_db', 'semantic_adjust', style='dashed')

    dot.edge('semantic_adjust', 'voting_check', label='调整后分数')

    # 多视角匹配判定流程
    dot.edge('voting_check', 'match_found', label='≥50%视角匹配\n直接确认', color='#059669')
    dot.edge('voting_check', 'best_sim_check', label='<50%')

    dot.edge('best_sim_check', 'match_found', label='≥0.88\n单视角强匹配', color='#059669')
    dot.edge('best_sim_check', 'weighted_sim_check', label='<0.88')

    dot.edge('weighted_sim_check', 'weighted_score', label='≥0.78\n需时序验证')
    dot.edge('weighted_sim_check', 'no_match', label='<0.78', style='dashed')

    dot.edge('weighted_score', 'match_found', label='加权分≥0.5', color='#059669')
    dot.edge('weighted_score', 'temporal_window', label='加权分<0.5\n记录候选')

    dot.edge('temporal_window', 'weighted_score', label='验证查询', style='dashed')

    return dot


def create_memory_replay_flow():
    """创建记忆回放决策流程图 (- 增强版，支持模糊匹配)"""
    dot = Digraph(comment='Memory Replay Decision Flow')
    dot.attr(rankdir='TB', bgcolor='white', fontname='SimHei')
    dot.attr('node', fontname='SimHei', fontsize='10')
    dot.attr('edge', fontname='SimHei', fontsize='9')

    # 标题
    dot.node('title',
             '记忆回放决策流程\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n(支持模糊匹配 + 同义词扩展 + 跨语言)',
             shape='note', style='filled', fillcolor='#059669',
             fontcolor='white', fontsize='14', width='5')

    # 输入
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入信息', style='filled', fillcolor=COLORS['bg_input'])
        c.node('instruction',
               '任务指令\n━━━━━━━━━━━━━━━━━━━━━━\ninstruction: str\n例: "Go to the front desk"',
               shape='parallelogram', style='filled', fillcolor='#93C5FD')
        c.node('current_feature',
               '当前视觉特征\n━━━━━━━━━━━━━━━━━━━━━━\n512维 LongCLIP 特征',
               shape='parallelogram', style='filled', fillcolor='#93C5FD')

    # 指令解析
    with dot.subgraph(name='cluster_parse') as c:
        c.attr(label='指令解析', style='filled', fillcolor=COLORS['bg_semantic'])
        c.node('parse_instruction',
               '语义解析\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n提取:\n• 目标位置 (locations)\n• 动作类型 (actions)\n• 返回标记 (is_return)',
               shape='box', style='filled,rounded', fillcolor='#7DD3FC')
        c.node('synonym_expand',
               '同义词扩展\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• "front desk" → "前台"\n• "reception" → "前台"\n• 28组中英文同义词',
               shape='box', style='filled,rounded', fillcolor='#38BDF8')

    # 记忆检索
    with dot.subgraph(name='cluster_memory') as c:
        c.attr(label='记忆检索', style='filled', fillcolor=COLORS['bg_memory'])
        c.node('route_db',
               '路线数据库\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 已记录路线列表\n• 每条路线:\n  - route_id\n  - instruction\n  - keyframes\n  - trajectory',
               shape='cylinder', style='filled', fillcolor='#6EE7B7')
        c.node('exact_match',
               '精确匹配\n━━━━━━━━━━━━━━━━━━━━━━━━━━\ninstruction == saved_instruction',
               shape='hexagon', style='filled', fillcolor='#34D399')
        c.node('fuzzy_match',
               '模糊匹配\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 位置相似度 (权重3.0)\n• 动作相似度 (权重1.0)\n• 字符串相似度 (权重0.5)',
               shape='hexagon', style='filled', fillcolor='#10B981')

    # 决策流程
    with dot.subgraph(name='cluster_decision') as c:
        c.attr(label='决策流程', style='filled', fillcolor=COLORS['bg_decision'])
        c.node('exact_found',
               '精确匹配成功?\n━━━━━━━━━━━━━━━━━━━━',
               shape='diamond', style='filled', fillcolor='#FCA5A5')
        c.node('fuzzy_found',
               '模糊匹配成功?\n━━━━━━━━━━━━━━━━━━━━\n相似度 ≥ 0.65?',
               shape='diamond', style='filled', fillcolor='#F87171')
        c.node('position_match',
               '位置匹配检查\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n当前位置是否在\n已记录路线的起点附近?',
               shape='diamond', style='filled', fillcolor='#EF4444')

    # 相似度示例
    dot.node('similarity_example',
             '模糊匹配示例\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"Go to reception desk" vs "Go to front desk"\n→ 相似度: 0.879 ✓\n\n"Go to reception desk" vs "去前台"\n→ 相似度: 0.818 ✓ (跨语言)\n\n"Go to reception" vs "Go to sofa"\n→ 相似度: 0.314 ✗ (不匹配)',
             shape='note', style='filled', fillcolor='#D1FAE5', fontcolor='#059669')

    # 执行模式
    with dot.subgraph(name='cluster_modes') as c:
        c.attr(label='执行模式', style='filled', fillcolor=COLORS['bg_output'])
        c.node('replay_mode',
               '记忆回放模式\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 直接使用已记录动作\n• 沿着关键帧导航\n• 跳过模型推理\n• 效率提升 1.09x~2.26x',
               shape='box', style='filled,rounded', fillcolor='#A7F3D0')
        c.node('normal_mode',
               '正常推理模式\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 调用 InternVLAN 模型\n• 实时视觉推理\n• 记录新路线\n• 更新拓扑图',
               shape='box', style='filled,rounded', fillcolor='#C4B5FD')

    # 输出
    dot.node('action_output',
             '动作输出\n━━━━━━━━━━━━━━━━━━━━━━━━━━\naction: [[vx, vy, omega]]\npixel_target: [x, y] or None\ntask_status: str',
             shape='box', style='filled,rounded', fillcolor='#FBBF24')

    # 连接
    dot.edge('title', 'instruction', style='invis')
    dot.edge('instruction', 'parse_instruction')
    dot.edge('parse_instruction', 'synonym_expand')

    dot.edge('synonym_expand', 'exact_match')
    dot.edge('route_db', 'exact_match', style='dashed')
    dot.edge('route_db', 'fuzzy_match', style='dashed')

    dot.edge('exact_match', 'exact_found')

    dot.edge('exact_found', 'position_match', label='找到')
    dot.edge('exact_found', 'fuzzy_match', label='未找到')

    dot.edge('fuzzy_match', 'fuzzy_found')
    dot.edge('fuzzy_found', 'position_match', label='相似度≥0.65')
    dot.edge('fuzzy_found', 'normal_mode', label='相似度<0.65')

    dot.edge('current_feature', 'position_match', style='dashed')
    dot.edge('position_match', 'replay_mode', label='位置接近', color='#059669')
    dot.edge('position_match', 'normal_mode', label='位置不符')

    dot.edge('replay_mode', 'action_output')
    dot.edge('normal_mode', 'action_output')

    dot.edge('similarity_example', 'fuzzy_match', style='dashed', constraint='false')

    return dot


def create_semantic_search_flow():
    """创建语义搜索流程图"""
    dot = Digraph(comment='Semantic Search Flow')
    dot.attr(rankdir='TB', bgcolor='white', fontname='SimHei')
    dot.attr('node', fontname='SimHei', fontsize='10')
    dot.attr('edge', fontname='SimHei', fontsize='9')

    # 标题
    dot.node('title',
             '语义搜索流程\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nSemantic Search with Synonyms & Fuzzy Matching',
             shape='note', style='filled', fillcolor='#0891B2',
             fontcolor='white', fontsize='14', width='5')

    # 输入
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入查询', style='filled', fillcolor=COLORS['bg_input'])
        c.node('query',
               '查询文本\n━━━━━━━━━━━━━━━━━━━━━━\n例: "front desk"\n例: "corridor"\n例: "投影仪"',
               shape='parallelogram', style='filled', fillcolor='#93C5FD')

    # 同义词扩展
    with dot.subgraph(name='cluster_synonym') as c:
        c.attr(label='同义词扩展 (28组中英文)', style='filled', fillcolor=COLORS['bg_semantic'])
        c.node('synonym_dict',
               '同义词字典\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n前台: reception, front desk, 接待处...\n走廊: corridor, hallway, 过道...\n会议室: meeting room, conference...\n厨房: kitchen, 茶水间...\n沙发: sofa, couch...\n... (共28组)',
               shape='cylinder', style='filled', fillcolor='#7DD3FC')
        c.node('expand',
               '扩展查询词\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"front desk" →\n{front desk, 前台, reception,\n 接待处, reception desk}',
               shape='box', style='filled,rounded', fillcolor='#38BDF8')

    # 多维度匹配
    with dot.subgraph(name='cluster_match') as c:
        c.attr(label='多维度匹配', style='filled', fillcolor=COLORS['bg_memory'])
        c.node('label_match',
               '标签匹配\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 精确匹配: +1.5\n• 词级别匹配: +0.5\n• 标准化匹配: +1.0',
               shape='box', style='filled,rounded', fillcolor='#6EE7B7')
        c.node('fuzzy_match',
               'Levenshtein模糊匹配\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 编辑距离计算\n• 最大距离: 2\n• 模糊分数 > 0.6 加分',
               shape='box', style='filled,rounded', fillcolor='#34D399')
        c.node('desc_match',
               '描述匹配\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 场景描述文本\n• 词级别匹配',
               shape='box', style='filled,rounded', fillcolor='#10B981')

    # 分数计算
    with dot.subgraph(name='cluster_score') as c:
        c.attr(label='综合评分', style='filled', fillcolor=COLORS['bg_vpr'])
        c.node('combine_score',
               '加权综合\n━━━━━━━━━━━━━━━━━━━━━━━━━━\ntotal = label_score\n      + fuzzy_score\n      + description_score',
               shape='box', style='filled,rounded', fillcolor='#FDE68A')
        c.node('filter_sort',
               '过滤与排序\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 过滤 score > 0\n• 按分数降序排序\n• 返回 Top-K',
               shape='box', style='filled,rounded', fillcolor='#FBBF24')

    # 输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出结果', style='filled', fillcolor=COLORS['bg_output'])
        c.node('output',
               '搜索结果\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nList[(node_id, score, match_info)]\n\nmatch_info包含:\n• label_score\n• description_score\n• fuzzy_score\n• matched_labels',
               shape='box', style='filled,rounded', fillcolor='#C4B5FD')

    # 示例
    dot.node('example',
             '搜索示例\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n查询: "front desk"\n→ 节点1(前台): 得分=2.0\n   匹配标签=[接待处, 前台]\n\n查询: "corridor"\n→ 节点2(走廊): 得分=18.95\n   匹配标签=[过道, 走廊]\n\n查询: "kitchen"\n→ 节点4(茶水间): 得分=6.3\n   匹配标签=[厨房]',
             shape='note', style='filled', fillcolor='#D1FAE5', fontcolor='#059669')

    # 连接
    dot.edge('title', 'query', style='invis')
    dot.edge('query', 'expand')
    dot.edge('synonym_dict', 'expand', style='dashed')

    dot.edge('expand', 'label_match')
    dot.edge('expand', 'fuzzy_match')
    dot.edge('expand', 'desc_match')

    dot.edge('label_match', 'combine_score')
    dot.edge('fuzzy_match', 'combine_score')
    dot.edge('desc_match', 'combine_score')

    dot.edge('combine_score', 'filter_sort')
    dot.edge('filter_sort', 'output')

    dot.edge('example', 'output', style='dashed', constraint='false')

    return dot


def create_topological_map_flow():
    """创建拓扑地图管理流程图 v2.0 (双向图 + Dijkstra + 节点合并)"""
    dot = Digraph(comment='Topological Map Management Flow v2.0')
    dot.attr(rankdir='TB', bgcolor='white', fontname='SimHei')
    dot.attr('node', fontname='SimHei', fontsize='10')
    dot.attr('edge', fontname='SimHei', fontsize='9')

    # 标题
    dot.node('title',
             '[v2.0] 拓扑地图管理流程\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n(双向图 + Dijkstra最短路径 + 节点合并)',
             shape='note', style='filled', fillcolor='#DC2626',
             fontcolor='white', fontsize='14', width='5')

    # 输入观测 - v2.0多视角
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入观测 (add_observation) - v2.0', style='filled',
               fillcolor=COLORS['bg_input'])
        c.node('multi_view_features',
               '[v2.0] 多视角特征字典\n━━━━━━━━━━━━━━━━━━━━\nDict[camera_id, 512维]\n各视角独立存储',
               shape='parallelogram', style='filled', fillcolor='#93C5FD')
        c.node('rgb_image',
               'RGB图像 (front_1)\n━━━━━━━━━━━━━━━━\nH×W×3 np.ndarray\n仅用于存储',
               shape='parallelogram', style='filled', fillcolor='#93C5FD')
        c.node('action_sequence',
               '动作序列\n━━━━━━━━━━━━━━━━\nList[[vx, vy, omega]]\n边的动作存储',
               shape='parallelogram', style='filled', fillcolor='#93C5FD')
        c.node('semantic_info',
               '语义信息\n━━━━━━━━━━━━━━━━\n• scene_description\n• semantic_labels',
               shape='parallelogram', style='filled', fillcolor='#93C5FD')

    # v2.0: 多视角VPR检测
    with dot.subgraph(name='cluster_vpr') as c:
        c.attr(label='[v2.0] 多视角VPR检测', style='filled', fillcolor=COLORS['bg_vpr'])
        c.node('vpr_multi_view',
               '多视角VPR检测\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nvpr.is_revisited_multi_view()\n• 4个独立FAISS索引\n• 投票机制判定',
               shape='hexagon', style='filled', fillcolor='#FDE68A')
        c.node('revisit_found',
               '回环检测成功\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nMultiViewVPRResult:\n• node_id + best_camera\n• voting_score',
               shape='box', style='filled,rounded', fillcolor='#FEF08A')

    # v2.0: 节点合并机制
    with dot.subgraph(name='cluster_merge') as c:
        c.attr(label='[v2.0] 节点合并机制', style='filled', fillcolor=COLORS['bg_topo'])
        c.node('merge_check',
               '节点合并检查\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nweighted_similarity ≥ 0.9?\n(node_merge_threshold)',
               shape='diamond', style='filled', fillcolor='#F472B6')
        c.node('merge_nodes',
               '执行节点合并\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n_merge_node():\n• 特征EMA更新\n• 边重定向\n• 删除旧节点',
               shape='box', style='filled,rounded', fillcolor='#F9A8D4')
        c.node('create_new',
               '创建新节点\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nTopologicalNode(\n  node_id, multi_view_features,\n  rgb_image, ...)',
               shape='box', style='filled,rounded', fillcolor='#EC4899')

    # v2.0: 双向图管理
    with dot.subgraph(name='cluster_graph') as c:
        c.attr(label='[v2.0] 双向图管理 (nx.Graph)', style='filled', fillcolor=COLORS['bg_memory'])
        c.node('bidirectional_edge',
               '双向边创建\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n_add_bidirectional_edge():\n• forward_actions: 前进动作\n• backward_actions: 反转动作\n• weight: 边权重',
               shape='box', style='filled,rounded', fillcolor='#6EE7B7')
        c.node('reverse_actions',
               '动作反转计算\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n_compute_reverse_actions():\n• vx → -vx\n• vy → -vy\n• omega → -omega\n• 序列反转',
               shape='box', style='filled,rounded', fillcolor='#34D399')
        c.node('update_current',
               '更新当前节点\n━━━━━━━━━━━━━━━━━━━━━━━━━━\ncurrent_node_id = node_id\n(用于路径规划起点)',
               shape='box', style='filled,rounded', fillcolor='#6EE7B7')

    # v2.0: 多视角VPR索引更新
    with dot.subgraph(name='cluster_vpr_update') as c:
        c.attr(label='[v2.0] 多视角VPR索引更新', style='filled', fillcolor='#E0F2FE')
        c.node('add_to_vpr',
               'VPR 多视角索引更新\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nvpr.add_feature():\n• 4个独立FAISS索引\n• surround_features_dict\n• 语义标签',
               shape='cylinder', style='filled', fillcolor='#7DD3FC')

    # 语义图更新
    with dot.subgraph(name='cluster_semantic') as c:
        c.attr(label='语义图更新', style='filled', fillcolor=COLORS['bg_semantic'])
        c.node('add_to_semantic',
               '语义图更新\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nsemantic_graph.add_node()\n• 同义词索引\n• 仅关键帧',
               shape='cylinder', style='filled', fillcolor='#38BDF8')

    # 输出
    dot.node('output',
             '[v2.0] 返回结果\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n(node_id, is_new,\n revisit_info, merge_info)',
             shape='box', style='filled,rounded', fillcolor='#34D399')

    # 连接 v2.0流程
    dot.edge('title', 'multi_view_features', style='invis')
    dot.edge('multi_view_features', 'vpr_multi_view', label='多视角特征')
    dot.edge('action_sequence', 'bidirectional_edge', style='dashed')

    dot.edge('vpr_multi_view', 'revisit_found', label='检测到回环')
    dot.edge('vpr_multi_view', 'merge_check', label='无回环')

    dot.edge('revisit_found', 'merge_check', label='检查是否合并')

    dot.edge('merge_check', 'merge_nodes', label='sim ≥ 0.9\n合并节点')
    dot.edge('merge_check', 'create_new', label='sim < 0.9\n创建新节点')

    dot.edge('merge_nodes', 'bidirectional_edge')
    dot.edge('create_new', 'bidirectional_edge')
    dot.edge('create_new', 'add_to_vpr')

    dot.edge('bidirectional_edge', 'reverse_actions', label='计算反向动作')
    dot.edge('reverse_actions', 'update_current')

    dot.edge('update_current', 'add_to_semantic')
    dot.edge('semantic_info', 'add_to_semantic', style='dashed')

    dot.edge('add_to_semantic', 'output')

    return dot


def create_shortest_path_planning_flow():
    """创建Dijkstra最短路径规划流程图 v2.0"""
    dot = Digraph(comment='Dijkstra Shortest Path Planning Flow')
    dot.attr(rankdir='TB', bgcolor='white', fontname='SimHei')
    dot.attr('node', fontname='SimHei', fontsize='10')
    dot.attr('edge', fontname='SimHei', fontsize='9')

    # 标题
    dot.node('title',
             '[v2.0] Dijkstra最短路径规划流程\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n任意起点到目标的最短路径导航',
             shape='note', style='filled', fillcolor='#059669',
             fontcolor='white', fontsize='14', width='5')

    # 输入
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入参数', style='filled', fillcolor=COLORS['bg_input'])
        c.node('start_node',
               '起始节点\n━━━━━━━━━━━━━━━━\nstart_node: int\n或 current_node_id',
               shape='parallelogram', style='filled', fillcolor='#93C5FD')
        c.node('goal_node',
               '目标节点\n━━━━━━━━━━━━━━━━\ngoal_node: int\n(语义搜索得到)',
               shape='parallelogram', style='filled', fillcolor='#93C5FD')

    # 路径检查
    with dot.subgraph(name='cluster_check') as c:
        c.attr(label='路径检查', style='filled', fillcolor=COLORS['bg_decision'])
        c.node('node_exists',
               '节点存在检查\n━━━━━━━━━━━━━━━━━━━━\nstart_node in graph?\ngoal_node in graph?',
               shape='diamond', style='filled', fillcolor='#FCA5A5')
        c.node('same_node',
               '相同节点检查\n━━━━━━━━━━━━━━━━━━━━\nstart == goal?',
               shape='diamond', style='filled', fillcolor='#FBBF24')

    # Dijkstra算法
    with dot.subgraph(name='cluster_dijkstra') as c:
        c.attr(label='Dijkstra算法 (NetworkX)', style='filled', fillcolor=COLORS['bg_memory'])
        c.node('dijkstra_path',
               'nx.dijkstra_path()\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 计算最短路径序列\n• 基于边权重\n• 时间复杂度: O((V+E)logV)',
               shape='box3d', style='filled', fillcolor='#6EE7B7')
        c.node('dijkstra_length',
               'nx.dijkstra_path_length()\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 计算路径总距离\n• 边权重之和',
               shape='box3d', style='filled', fillcolor='#34D399')

    # 路径构建
    with dot.subgraph(name='cluster_build') as c:
        c.attr(label='路径构建', style='filled', fillcolor=COLORS['bg_vpr'])
        c.node('extract_waypoints',
               '提取路径点信息\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nfor each node in path:\n• node_id\n• semantic_labels\n• scene_description',
               shape='box', style='filled,rounded', fillcolor='#FDE68A')
        c.node('extract_actions',
               '提取动作序列\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nfor each edge in path:\n• forward_actions\n• 或 backward_actions\n(根据方向选择)',
               shape='box', style='filled,rounded', fillcolor='#FBBF24')

    # 输出结果
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出结果', style='filled', fillcolor=COLORS['bg_output'])
        c.node('path_result',
               'PathPlanResult\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n• success: bool\n• path: List[int] 节点序列\n• total_distance: float\n• total_steps: int 总步数\n• waypoints: List[Dict] 路径点\n• estimated_actions: List 动作序列',
               shape='note', style='filled', fillcolor='#C4B5FD')
        c.node('error_result',
               '错误结果\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nsuccess: False\nerror_message: str',
               shape='box', style='filled,rounded', fillcolor='#FCA5A5')

    # 使用示例
    dot.node('usage_example',
             '使用示例\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n# 从当前位置规划\nresult = plan_path_from_current(goal)\n\n# 从指定起点规划\nresult = plan_shortest_path(start, goal)\n\n# 执行导航\nfor waypoint in result.waypoints:\n    execute_actions(waypoint.actions)',
             shape='note', style='filled', fillcolor='#D1FAE5', fontcolor='#059669')

    # 连接
    dot.edge('title', 'start_node', style='invis')
    dot.edge('start_node', 'node_exists')
    dot.edge('goal_node', 'node_exists')

    dot.edge('node_exists', 'error_result', label='节点不存在', style='dashed', color='#DC2626')
    dot.edge('node_exists', 'same_node', label='存在')

    dot.edge('same_node', 'path_result', label='相同节点\n已到达', color='#059669')
    dot.edge('same_node', 'dijkstra_path', label='不同节点')

    dot.edge('dijkstra_path', 'dijkstra_length')
    dot.edge('dijkstra_length', 'extract_waypoints')

    dot.edge('extract_waypoints', 'extract_actions')
    dot.edge('extract_actions', 'path_result')

    dot.edge('usage_example', 'path_result', style='dashed', constraint='false')

    return dot


def create_full_navigation_pipeline():
    """创建完整导航流水线图 v2.0 (多视角独立编码 + Dijkstra)"""
    dot = Digraph(comment='Full Memory Navigation Pipeline v2.0')
    dot.attr(rankdir='LR', bgcolor='white', fontname='SimHei', nodesep='0.5')
    dot.attr('node', fontname='SimHei', fontsize='9')
    dot.attr('edge', fontname='SimHei', fontsize='8')

    # 标题
    dot.node('title',
             '[v2.0] 记忆导航系统完整流水线\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n(多视角独立VPR + Dijkstra最短路径 + 节点合并)',
             shape='note', style='filled', fillcolor='#1E40AF',
             fontcolor='white', fontsize='12', width='7')

    # 阶段1: WebSocket接收
    with dot.subgraph(name='cluster_s1') as c:
        c.attr(label='阶段1: WebSocket接收', style='filled', fillcolor='#DBEAFE')
        c.node('ws_recv',
               'WebSocket\n接收器\n━━━━━━━━━\n接收JSON\n消息',
               shape='box', style='filled,rounded', fillcolor='#60A5FA')
        c.node('parse_msg',
               '消息\n解析器\n━━━━━━━━━\n解析图像\n和指令',
               shape='box', style='filled,rounded', fillcolor='#60A5FA')

    # 阶段2: 预处理
    with dot.subgraph(name='cluster_s2') as c:
        c.attr(label='阶段2: 图像预处理', style='filled', fillcolor='#E0E7FF')
        c.node('decode_front',
               'front_1\n解码\n━━━━━━━━━\n前置相机\n仅用于推理',
               shape='box', style='filled,rounded', fillcolor='#F87171')
        c.node('decode_surround',
               'cam1~4\n解码\n━━━━━━━━━\n环视相机\n用于记忆',
               shape='box', style='filled,rounded', fillcolor='#818CF8')

    # 阶段3: 特征提取 - v2.0多视角独立
    with dot.subgraph(name='cluster_s3') as c:
        c.attr(label='阶段3: 特征提取 [v2.0多视角独立]', style='filled', fillcolor='#FCE7F3')
        c.node('longclip',
               'LongCLIP\n━━━━━━━━━━━\ncam1~4\n4×512维特征',
               shape='box3d', style='filled', fillcolor='#F472B6')
        c.node('multi_view',
               '[v2.0]\n独立编码\n━━━━━━━━━━━\n不融合\n各视角独立',
               shape='box3d', style='filled', fillcolor='#EC4899')

    # 阶段4: 记忆系统 - v2.0
    with dot.subgraph(name='cluster_s4') as c:
        c.attr(label='阶段4: 记忆系统 [v2.0]', style='filled', fillcolor='#D1FAE5')
        c.node('vpr',
               '[v2.0] VPR\n━━━━━━━━━━\n多视角FAISS\n投票机制',
               shape='hexagon', style='filled', fillcolor='#34D399')
        c.node('topo',
               '[v2.0] 拓扑图\n━━━━━━━━━━\n双向Graph\n节点合并',
               shape='hexagon', style='filled', fillcolor='#34D399')
        c.node('path_plan',
               '[v2.0] 路径\n━━━━━━━━━━\nDijkstra\n最短路径',
               shape='hexagon', style='filled', fillcolor='#10B981')
        c.node('semantic',
               '语义图\n━━━━━━━━━━\n28组同义词\n模糊搜索',
               shape='hexagon', style='filled', fillcolor='#0EA5E9')

    # 阶段5: 决策
    with dot.subgraph(name='cluster_s5') as c:
        c.attr(label='阶段5: 导航决策', style='filled', fillcolor='#FEE2E2')
        c.node('decision',
               '导航决策\n━━━━━━━━━━━\n最短路径?\n正常推理?',
               shape='diamond', style='filled', fillcolor='#F87171')

    # 阶段6A: 路径导航模式
    with dot.subgraph(name='cluster_s6a') as c:
        c.attr(label='阶段6A: 路径导航 [v2.0]', style='filled', fillcolor='#ECFDF5')
        c.node('path_nav',
               '[v2.0]\n路径导航\n━━━━━━━━━━━\n沿最短路径\n执行动作',
               shape='box', style='filled,rounded', fillcolor='#6EE7B7')

    # 阶段6B: 推理模式
    with dot.subgraph(name='cluster_s6b') as c:
        c.attr(label='阶段6B: 推理', style='filled', fillcolor='#F3E8FF')
        c.node('model',
               'InternVLA\n━━━━━━━━━━━━\nGPU: cuda:1\n动作推理',
               shape='box3d', style='filled', fillcolor='#A78BFA')

    # 阶段7: 输出
    with dot.subgraph(name='cluster_s7') as c:
        c.attr(label='阶段7: 输出', style='filled', fillcolor='#FEF3C7')
        c.node('output',
               '动作输出\n━━━━━━━━━━\n[[vx,vy,ω]]\npixel_target',
               shape='box', style='filled,rounded', fillcolor='#FBBF24')
        c.node('ws_send',
               'WebSocket\n发送器\n━━━━━━━━━━\nJSON响应',
               shape='box', style='filled,rounded', fillcolor='#FBBF24')

    # 连接流水线
    dot.edge('title', 'ws_recv', style='invis')

    dot.edge('ws_recv', 'parse_msg')
    dot.edge('parse_msg', 'decode_front', label='front_1')
    dot.edge('parse_msg', 'decode_surround', label='cam1~4')

    dot.edge('decode_front', 'model', label='仅推理', color='#7C3AED', penwidth='2')

    dot.edge('decode_surround', 'longclip')
    dot.edge('longclip', 'multi_view', label='4×512维')

    dot.edge('multi_view', 'vpr', label='多视角独立特征')

    dot.edge('vpr', 'topo', label='投票结果')
    dot.edge('topo', 'path_plan', label='双向图')
    dot.edge('topo', 'semantic')

    dot.edge('path_plan', 'decision', label='最短路径')
    dot.edge('semantic', 'decision', style='dashed', label='目标节点')

    dot.edge('decision', 'path_nav', label='路径导航')
    dot.edge('decision', 'model', label='正常推理')

    dot.edge('path_nav', 'output')
    dot.edge('model', 'output')

    dot.edge('output', 'ws_send')

    # 反馈循环
    dot.edge('topo', 'vpr', label='更新多视角索引', style='dashed', constraint='false')
    dot.edge('model', 'topo', label='添加节点/边', style='dashed', constraint='false')

    return dot


def create_feature_extraction_detail():
    """创建特征提取详细流程图 v2.0 (多视角独立编码)"""
    dot = Digraph(comment='Feature Extraction Detail v2.0')
    dot.attr(rankdir='TB', bgcolor='white', fontname='SimHei')
    dot.attr('node', fontname='SimHei', fontsize='10')
    dot.attr('edge', fontname='SimHei', fontsize='9')

    # 标题
    dot.node('title',
             '[v2.0] 特征提取详细流程\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n多视角独立编码 (不融合)',
             shape='note', style='filled', fillcolor='#0891B2',
             fontcolor='white', fontsize='14', width='5')

    # 输入图像 - v2.0带角度信息
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入图像 (5个相机) - v2.0带角度', style='filled', fillcolor=COLORS['bg_input'])
        c.node('cam0',
               'front_1 (前置)\n━━━━━━━━━━━━━━━━\n主相机\n仅用于模型推理\n不参与记忆模块',
               shape='folder', style='filled', fillcolor='#F87171')
        c.node('cam1',
               'camera_1 (+37.5°)\n━━━━━━━━━━━━━━━━\n前右视角\n独立512维特征',
               shape='folder', style='filled', fillcolor='#93C5FD')
        c.node('cam2',
               'camera_2 (-37.5°)\n━━━━━━━━━━━━━━━━\n前左视角\n独立512维特征',
               shape='folder', style='filled', fillcolor='#93C5FD')
        c.node('cam3',
               'camera_3 (-142.5°)\n━━━━━━━━━━━━━━━━\n后左视角\n独立512维特征',
               shape='folder', style='filled', fillcolor='#93C5FD')
        c.node('cam4',
               'camera_4 (+142.5°)\n━━━━━━━━━━━━━━━━\n后右视角\n独立512维特征',
               shape='folder', style='filled', fillcolor='#93C5FD')

    # LongCLIP处理
    with dot.subgraph(name='cluster_longclip') as c:
        c.attr(label='LongCLIP 特征提取器 (GPU: cuda:1)', style='filled',
               fillcolor=COLORS['bg_feature'])
        c.node('preprocess',
               '图像预处理\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• Resize to 224×224\n• Normalize\n• ToTensor',
               shape='box', style='filled,rounded', fillcolor='#A5B4FC')
        c.node('encoder',
               'ViT-B/16 编码器\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 12层Transformer\n• Patch size: 16×16\n• Hidden dim: 768',
               shape='box3d', style='filled', fillcolor='#818CF8')
        c.node('proj',
               '投影层\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n768 → 512\n归一化输出',
               shape='box', style='filled,rounded', fillcolor='#A5B4FC')

    # v2.0: 多视角独立特征输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='[v2.0] 多视角独立特征输出', style='filled', fillcolor=COLORS['bg_memory'])
        c.node('multi_view_features',
               '[v2.0] MultiViewFeatures\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• camera_features: Dict[str, 512维]\n  - camera_1: 前右特征\n  - camera_2: 前左特征\n  - camera_3: 后左特征\n  - camera_4: 后右特征\n• camera_angles: 各相机角度',
               shape='box', style='filled,rounded', fillcolor='#6EE7B7')

    # v2.0: 不再融合，直接送入多视角VPR
    with dot.subgraph(name='cluster_vpr') as c:
        c.attr(label='[v2.0] 多视角VPR (不融合)', style='filled',
               fillcolor=COLORS['bg_vpr'])
        c.node('vpr_cam1',
               'camera_1\nFAISS索引',
               shape='cylinder', style='filled', fillcolor='#FDE68A')
        c.node('vpr_cam2',
               'camera_2\nFAISS索引',
               shape='cylinder', style='filled', fillcolor='#FDE68A')
        c.node('vpr_cam3',
               'camera_3\nFAISS索引',
               shape='cylinder', style='filled', fillcolor='#FDE68A')
        c.node('vpr_cam4',
               'camera_4\nFAISS索引',
               shape='cylinder', style='filled', fillcolor='#FDE68A')

    # InternVLA
    dot.node('internvla',
             'InternVLA 模型推理\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 视觉-语言导航\n• 动作预测\n• 轨迹规划',
             shape='box3d', style='filled', fillcolor='#A78BFA')

    # 说明节点 v2.0
    dot.node('note_front',
             '注意: 前置相机 front_1\n仅用于 InternVLA 推理\n不参与 LongCLIP 特征提取',
             shape='note', style='filled', fillcolor='#FEE2E2', fontcolor='#DC2626')
    dot.node('note_multiview',
             '[v2.0] 多视角独立编码\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 各视角特征独立存储\n• 不做加权融合\n• 独立FAISS索引\n• 投票机制匹配',
             shape='note', style='filled', fillcolor='#D1FAE5', fontcolor='#059669')

    # 连接
    dot.edge('title', 'cam0', style='invis')

    dot.edge('cam0', 'internvla', label='仅用于推理', color='#7C3AED', penwidth='2')
    dot.edge('internvla', 'note_front', style='dashed')

    dot.edge('cam1', 'preprocess', color='#059669')
    dot.edge('cam2', 'preprocess', color='#059669')
    dot.edge('cam3', 'preprocess', color='#059669')
    dot.edge('cam4', 'preprocess', color='#059669', label='环视相机')

    dot.edge('preprocess', 'encoder')
    dot.edge('encoder', 'proj')
    dot.edge('proj', 'multi_view_features', label='4个独立512维特征')

    # v2.0: 独立特征直接送入各视角FAISS
    dot.edge('multi_view_features', 'vpr_cam1', label='camera_1特征')
    dot.edge('multi_view_features', 'vpr_cam2', label='camera_2特征')
    dot.edge('multi_view_features', 'vpr_cam3', label='camera_3特征')
    dot.edge('multi_view_features', 'vpr_cam4', label='camera_4特征')

    dot.edge('note_multiview', 'multi_view_features', style='dashed', constraint='false')

    return dot


def create_route_recording_flow():
    """创建路线记录流程图"""
    dot = Digraph(comment='Route Recording Flow')
    dot.attr(rankdir='TB', bgcolor='white', fontname='SimHei')
    dot.attr('node', fontname='SimHei', fontsize='10')
    dot.attr('edge', fontname='SimHei', fontsize='9')

    # 标题
    dot.node('title',
             '路线记录流程\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nRoute Recording Flow',
             shape='note', style='filled', fillcolor='#D97706',
             fontcolor='white', fontsize='14', width='5')

    # 记录控制
    with dot.subgraph(name='cluster_control') as c:
        c.attr(label='记录控制', style='filled', fillcolor=COLORS['bg_input'])
        c.node('start_cmd',
               '开始记录命令\n━━━━━━━━━━━━━━━━━━━━━━\nstart_route_recording()\nroute_id 自动生成',
               shape='box', style='filled,rounded', fillcolor='#60A5FA')
        c.node('stop_cmd',
               '停止记录命令\n━━━━━━━━━━━━━━━━━━━━━━\nstop_route_recording()\n保存到路线库',
               shape='box', style='filled,rounded', fillcolor='#F87171')

    # 帧处理
    with dot.subgraph(name='cluster_frame') as c:
        c.attr(label='帧处理循环', style='filled', fillcolor=COLORS['bg_feature'])
        c.node('frame_input',
               '接收帧数据\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 环视图像 (cam1~4)\n• 环视融合特征 (512维)\n• 动作输出 (action)',
               shape='parallelogram', style='filled', fillcolor='#A5B4FC')
        c.node('keyframe_check',
               '关键帧检测\n━━━━━━━━━━━━━━━━━━━━━━━━━━\npixel_target is not None?\nOR 帧间隔 ≥ 8?',
               shape='diamond', style='filled', fillcolor='#F472B6')

    # 关键帧处理
    with dot.subgraph(name='cluster_keyframe') as c:
        c.attr(label='关键帧处理', style='filled', fillcolor=COLORS['bg_memory'])
        c.node('vlm_describe',
               'VLM 场景描述\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nQwen2.5-VL-7B\nGPU: cuda:2\n生成语义描述',
               shape='box3d', style='filled', fillcolor='#6EE7B7')
        c.node('save_keyframe',
               '保存关键帧\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• RGB图像\n• 视觉特征\n• 语义描述\n• pixel_target',
               shape='cylinder', style='filled', fillcolor='#34D399')

    # 轨迹记录
    with dot.subgraph(name='cluster_trajectory') as c:
        c.attr(label='轨迹记录', style='filled', fillcolor=COLORS['bg_vpr'])
        c.node('record_action',
               '记录动作\n━━━━━━━━━━━━━━━━━━━━━━━━━━\ntrajectory.append({\n  action, timestamp,\n  frame_idx\n})',
               shape='box', style='filled,rounded', fillcolor='#FDE68A')
        c.node('update_topo',
               '更新拓扑图\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nadd_observation()\n创建/合并节点',
               shape='box', style='filled,rounded', fillcolor='#FBBF24')

    # 路线保存
    with dot.subgraph(name='cluster_save') as c:
        c.attr(label='路线保存', style='filled', fillcolor=COLORS['bg_output'])
        c.node('route_data',
               '路线数据结构\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{\n  route_id: str,\n  instruction: str,\n  keyframes: List[Keyframe],\n  trajectory: List[Action],\n  topo_nodes: List[int],\n  created_at: timestamp\n}',
               shape='note', style='filled', fillcolor='#C4B5FD')
        c.node('save_to_db',
               '保存到数据库\n━━━━━━━━━━━━━━━━━━━━━━━━━━\nmemory_save_path/\nroute_{id}.pkl',
               shape='cylinder', style='filled', fillcolor='#A78BFA')

    # 连接
    dot.edge('title', 'start_cmd', style='invis')

    dot.edge('start_cmd', 'frame_input', label='开始')
    dot.edge('frame_input', 'keyframe_check')

    dot.edge('keyframe_check', 'vlm_describe', label='是关键帧')
    dot.edge('keyframe_check', 'record_action', label='非关键帧')

    dot.edge('vlm_describe', 'save_keyframe')
    dot.edge('save_keyframe', 'record_action')

    dot.edge('record_action', 'update_topo')
    dot.edge('update_topo', 'frame_input', label='下一帧', style='dashed')

    dot.edge('stop_cmd', 'route_data', label='停止')
    dot.edge('route_data', 'save_to_db')

    return dot


def create_config_parameters():
    """创建配置参数说明图 """
    dot = Digraph(comment='Configuration Parameters')
    dot.attr(rankdir='TB', bgcolor='white', fontname='SimHei')
    dot.attr('node', fontname='SimHei', fontsize='9')
    dot.attr('edge', fontname='SimHei', fontsize='8')

    # 标题
    dot.node('title',
             '记忆导航系统配置参数\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nMemoryNavigationConfig Parameters',
             shape='note', style='filled', fillcolor='#4F46E5',
             fontcolor='white', fontsize='14', width='6')

    # 核心开关
    with dot.subgraph(name='cluster_core') as c:
        c.attr(label='核心开关', style='filled', fillcolor='#FEE2E2')
        c.node('memory_enabled',
               'memory_enabled\n━━━━━━━━━━━━━━━━━━━━\n类型: bool\n默认: True\n说明: 记忆功能总开关',
               shape='box', style='filled,rounded', fillcolor='#FCA5A5')

    # GPU配置
    with dot.subgraph(name='cluster_gpu') as c:
        c.attr(label='GPU配置', style='filled', fillcolor='#DBEAFE')
        c.node('gpu_config',
               'GPU设备配置\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• main_model_device: "1"\n• feature_extractor_device: "1"\n• vlm_device: "2"\n\n注: 排除GPU 0',
               shape='box', style='filled,rounded', fillcolor='#93C5FD')

    # VPR参数 
    with dot.subgraph(name='cluster_vpr') as c:
        c.attr(label='VPR参数 (增强)', style='filled', fillcolor='#D1FAE5')
        c.node('vpr_params',
               'VPR 参数\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n• similarity_threshold: 0.78\n• min_time_gap: 0.5s\n• high_confidence: 0.90\n• low_confidence: 0.72\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n语义引导参数:\n• semantic_weight: 0.1\n• semantic_penalty: 0.05\n空间一致性:\n• max_topological_jump: 5\n时序验证:\n• temporal_window_size: 3\n• weighted_threshold: 0.5',
               shape='box', style='filled,rounded', fillcolor='#6EE7B7')

    # 拓扑图参数
    with dot.subgraph(name='cluster_topo') as c:
        c.attr(label='拓扑图参数', style='filled', fillcolor='#FCE7F3')
        c.node('topo_params',
               '拓扑图参数\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• max_nodes: 1000\n• node_merge_threshold: 0.98',
               shape='box', style='filled,rounded', fillcolor='#F9A8D4')

    # 路线记忆参数 
    with dot.subgraph(name='cluster_route') as c:
        c.attr(label='路线记忆参数 (增强)', style='filled', fillcolor='#FEF3C7')
        c.node('route_params',
               '路线参数\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n• keyframe_interval: 8\n• max_memory_routes: 100\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n模糊匹配参数:\n• enable_fuzzy_match: True\n• fuzzy_match_threshold: 0.65\n相似度权重:\n• location_weight: 3.0\n• action_weight: 1.0\n• string_weight: 0.5',
               shape='box', style='filled,rounded', fillcolor='#FDE68A')

    # 语义图参数 
    with dot.subgraph(name='cluster_semantic') as c:
        c.attr(label='语义图参数', style='filled', fillcolor=COLORS['bg_semantic'])
        c.node('semantic_params',
               '语义图参数\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n同义词字典:\n• SYNONYM_DICT: 28组中英文\n• 支持: 前台/reception...\n• 支持: 走廊/corridor...\n模糊匹配:\n• max_levenshtein_distance: 2\n• fuzzy_score_threshold: 0.6\n向量搜索:\n• faiss_index_type: IndexFlatIP',
               shape='box', style='filled,rounded', fillcolor='#7DD3FC')

    # 环视融合参数
    with dot.subgraph(name='cluster_surround') as c:
        c.attr(label='环视融合参数', style='filled', fillcolor='#E0E7FF')
        c.node('surround_params',
               '环视参数\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• use_surround_cameras: True\n• surround_weight: 0.25\n  (每相机权重)',
               shape='box', style='filled,rounded', fillcolor='#A5B4FC')

    # VLM参数
    with dot.subgraph(name='cluster_vlm') as c:
        c.attr(label='VLM参数', style='filled', fillcolor='#F3E8FF')
        c.node('vlm_params',
               'VLM 参数\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n• vlm_enabled: True\n• vlm_model_path:\n  Qwen2.5-VL-7B-Instruct\n• vlm_max_new_tokens: 256\n• vlm_batch_size: 4',
               shape='box', style='filled,rounded', fillcolor='#C4B5FD')

    # 特征提取参数
    with dot.subgraph(name='cluster_feature') as c:
        c.attr(label='特征提取参数', style='filled', fillcolor='#ECFDF5')
        c.node('feature_params',
               '特征参数\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n• longclip_model_path:\n  checkpoints/longclip-B.pt\n• feature_dim: 512',
               shape='box', style='filled,rounded', fillcolor='#A7F3D0')

    # 连接
    dot.edge('title', 'memory_enabled', style='invis')
    dot.edge('memory_enabled', 'gpu_config', style='dotted')
    dot.edge('gpu_config', 'vpr_params', style='dotted')
    dot.edge('vpr_params', 'topo_params', style='dotted')
    dot.edge('topo_params', 'route_params', style='dotted')
    dot.edge('route_params', 'semantic_params', style='dotted')
    dot.edge('semantic_params', 'surround_params', style='dotted')
    dot.edge('surround_params', 'vlm_params', style='dotted')
    dot.edge('vlm_params', 'feature_params', style='dotted')

    return dot


def create_memory_disabled_flow():
    """创建关闭记忆功能时的导航流程图"""
    dot = Digraph(comment='Memory Disabled Navigation Flow')
    dot.attr(rankdir='TB', bgcolor='white', fontname='SimHei')
    dot.attr('node', fontname='SimHei', fontsize='10')
    dot.attr('edge', fontname='SimHei', fontsize='9')

    # 标题
    dot.node('title',
             '记忆关闭时的导航流程\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nNavigation Flow (memory_enabled=False)\n行为与 ws_proxy.py 完全一致',
             shape='note', style='filled', fillcolor='#6B7280',
             fontcolor='white', fontsize='14', width='6')

    # 输入层
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入层 (Input Layer)', style='filled',
               fillcolor=COLORS['bg_input'], fontcolor=COLORS['text_dark'])
        c.node('ws_input',
               'WebSocket 输入\n━━━━━━━━━━━━━━━━━━━━━━━━\n• front_1: 前置相机图像\n  (用于模型推理)\n• camera_1~4: 环视图像\n  (被忽略，不使用)\n• task: 导航指令',
               shape='folder', style='filled', fillcolor='#93C5FD')

    # 无记忆模块
    with dot.subgraph(name='cluster_no_memory') as c:
        c.attr(label='记忆模块状态', style='filled',
               fillcolor='#F3F4F6', fontcolor='#6B7280')
        c.node('no_memory',
               '记忆模块未初始化\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n✗ 无 MemoryNavigationAgent\n✗ 无 LongCLIP 特征提取\n✗ 无 VPR 回环检测\n✗ 无拓扑图更新\n✗ 无路线记忆\n✗ 无语义搜索',
               shape='box', style='filled,dashed',
               fillcolor='#E5E7EB', fontcolor='#9CA3AF')

    # 图像预处理
    with dot.subgraph(name='cluster_preprocess') as c:
        c.attr(label='图像预处理', style='filled', fillcolor=COLORS['bg_feature'])
        c.node('decode',
               '图像解码\n━━━━━━━━━━━━━━━━━━━━━━\nBase64 → RGB\n仅处理 front_1',
               shape='box', style='filled,rounded', fillcolor='#A5B4FC')
        c.node('resize',
               '图像缩放\n━━━━━━━━━━━━━━━━━━━━━━\n→ 640×480\n(模型输入尺寸)',
               shape='box', style='filled,rounded', fillcolor='#A5B4FC')

    # 模型推理
    with dot.subgraph(name='cluster_model') as c:
        c.attr(label='模型推理 (InternVLA)', style='filled',
               fillcolor=COLORS['bg_output'], fontcolor=COLORS['text_dark'])
        c.node('model',
               'InternVLA-N1 模型\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n• 输入: RGB图像 + 指令\n• 输出: 动作序列 / 轨迹\n• GPU: 根据配置\n• 预加载，非懒加载',
               shape='box3d', style='filled', fillcolor='#C4B5FD')

    # 动作转换
    with dot.subgraph(name='cluster_action') as c:
        c.attr(label='动作转换', style='filled', fillcolor=COLORS['bg_vpr'])
        c.node('convert',
               '动作转换\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n离散动作 → 机器人控制\n[vx, vy, omega]',
               shape='box', style='filled,rounded', fillcolor='#FDE68A')
        c.node('small_action_check',
               '小动作检测\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n所有值 < 0.5?\n→ 自动停止',
               shape='diamond', style='filled', fillcolor='#FBBF24')

    # 输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出', style='filled', fillcolor=COLORS['bg_memory'])
        c.node('output',
               'WebSocket 响应\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{\n  status: "success",\n  action: [[vx, vy, omega], ...],\n  task_status: "executing"/"end",\n  pixel_target: [x, y] or null\n}\n\n✗ 无 memory_info 字段',
               shape='box', style='filled,rounded', fillcolor='#6EE7B7')

    # 连接
    dot.edge('title', 'ws_input', style='invis')
    dot.edge('ws_input', 'decode', label='仅 front_1')
    dot.edge('ws_input', 'no_memory', style='dashed', label='camera_1~4\n被忽略')
    dot.edge('decode', 'resize')
    dot.edge('resize', 'model')
    dot.edge('model', 'convert')
    dot.edge('convert', 'small_action_check')
    dot.edge('small_action_check', 'output')

    return dot


def create_camera_usage_diagram():
    """创建相机图像使用说明图"""
    dot = Digraph(comment='Camera Usage Diagram')
    dot.attr(rankdir='LR', bgcolor='white', fontname='SimHei')
    dot.attr('node', fontname='SimHei', fontsize='10')
    dot.attr('edge', fontname='SimHei', fontsize='9')

    # 标题
    dot.node('title',
             '相机图像使用分工\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nCamera Image Usage',
             shape='note', style='filled', fillcolor='#0891B2',
             fontcolor='white', fontsize='14', width='4')

    # 输入相机
    with dot.subgraph(name='cluster_cameras') as c:
        c.attr(label='输入相机 (5个)', style='filled', fillcolor=COLORS['bg_input'])
        c.node('front_1',
               'front_1\n(前置相机)\n━━━━━━━━━━━━\n主相机',
               shape='folder', style='filled', fillcolor='#60A5FA')
        c.node('camera_1',
               'camera_1\n(左前)\n━━━━━━━━━━━━\n环视相机',
               shape='folder', style='filled', fillcolor='#93C5FD')
        c.node('camera_2',
               'camera_2\n(右前)\n━━━━━━━━━━━━\n环视相机',
               shape='folder', style='filled', fillcolor='#93C5FD')
        c.node('camera_3',
               'camera_3\n(左后)\n━━━━━━━━━━━━\n环视相机',
               shape='folder', style='filled', fillcolor='#93C5FD')
        c.node('camera_4',
               'camera_4\n(右后)\n━━━━━━━━━━━━\n环视相机',
               shape='folder', style='filled', fillcolor='#93C5FD')

    # 用途分工
    with dot.subgraph(name='cluster_usage') as c:
        c.attr(label='用途分工', style='filled', fillcolor=COLORS['bg_feature'])
        c.node('model_inference',
               'InternVLA 模型推理\n━━━━━━━━━━━━━━━━━━━━━━━━\n• 视觉-语言导航\n• 动作预测\n• 轨迹规划',
               shape='box3d', style='filled', fillcolor='#A78BFA')
        c.node('memory_module',
               '记忆模块\n━━━━━━━━━━━━━━━━━━━━━━━━\n• LongCLIP 特征提取\n• VPR 回环检测 (语义引导)\n• 拓扑图管理\n• 语义图 (同义词+模糊匹配)',
               shape='box3d', style='filled', fillcolor='#6EE7B7')

    # 说明
    dot.node('note_front',
             '前置相机 front_1\n仅用于模型推理\n不参与记忆模块',
             shape='note', style='filled', fillcolor='#FEE2E2', fontcolor='#DC2626')
    dot.node('note_surround',
             '环视相机 camera_1~4\n仅用于记忆模块\n不参与模型推理',
             shape='note', style='filled', fillcolor='#D1FAE5', fontcolor='#059669')

    # 连接
    dot.edge('title', 'front_1', style='invis')

    dot.edge('front_1', 'model_inference', label='仅此相机', color='#7C3AED', penwidth='2')
    dot.edge('model_inference', 'note_front', style='dashed')

    dot.edge('camera_1', 'memory_module', color='#059669')
    dot.edge('camera_2', 'memory_module', color='#059669')
    dot.edge('camera_3', 'memory_module', color='#059669')
    dot.edge('camera_4', 'memory_module', color='#059669', label='4个环视相机')
    dot.edge('memory_module', 'note_surround', style='dashed')

    return dot


def main():
    """生成所有流程图 v2.0"""
    output_dir = Path(__file__).parent.parent / "docs" / "diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("记忆导航系统流程图生成器 v2.0")
    print("(多视角独立VPR + Dijkstra最短路径 + 节点合并)")
    print("=" * 60)

    diagrams = [
        ("memory_navigation_overview", create_memory_navigation_overview(), "[v2.0] 系统总览图"),
        ("memory_disabled_flow", create_memory_disabled_flow(), "记忆关闭时的导航流程"),
        ("camera_usage_diagram", create_camera_usage_diagram(), "相机图像使用分工"),
        ("vpr_detection_flow", create_vpr_detection_flow(), "[v2.0] 多视角VPR回环检测流程"),
        ("topological_map_flow", create_topological_map_flow(), "[v2.0] 拓扑地图管理流程"),
        ("shortest_path_planning", create_shortest_path_planning_flow(), "[v2.0] Dijkstra最短路径规划"),
        ("memory_replay_flow", create_memory_replay_flow(), "记忆回放决策流程 (模糊匹配)"),
        ("semantic_search_flow", create_semantic_search_flow(), "语义搜索流程 (同义词+模糊)"),
        ("full_navigation_pipeline", create_full_navigation_pipeline(), "[v2.0] 完整导航流水线"),
        ("feature_extraction_detail", create_feature_extraction_detail(), "[v2.0] 特征提取详细流程"),
        ("route_recording_flow", create_route_recording_flow(), "路线记录流程"),
        ("config_parameters", create_config_parameters(), "配置参数说明"),
    ]

    for name, diagram, description in diagrams:
        output_path = output_dir / name
        diagram.render(str(output_path), format='png', cleanup=True)
        diagram.render(str(output_path), format='pdf', cleanup=True)
        print(f"✓ 生成 {description}: {output_path}.png / .pdf")

    print("\n" + "=" * 60)
    print(f"所有流程图已保存到: {output_dir}")
    print("=" * 60)

    # 生成索引文件 v2.0
    index_content = """# 记忆导航系统流程图索引 v2.0

## v2.0 新特性

本版本包含以下重要更新:

1. **多视角独立编码** - camera_1~4 各视角特征独立存储，不做加权融合
2. **多视角投票VPR** - 每个视角独立FAISS索引，投票机制决定匹配
3. **Dijkstra最短路径** - 支持从任意起点到目标的最短路径规划
4. **双向拓扑图** - NetworkX Graph (非DiGraph)，支持双向导航
5. **自动节点合并** - 相似度超过阈值(0.9)自动合并节点

## 图表列表

| 图表名称 | 文件名 | 说明 |
|---------|--------|------|
"""
    for name, _, description in diagrams:
        index_content += f"| {description} | `{name}.png` | [查看PDF]({name}.pdf) |\n"

    index_content += """
## 查看建议

### 记忆功能开启时 (memory_enabled=True)
1. **[v2.0] 系统总览图** - 首先查看，了解整体架构和v2.0核心功能
2. **相机图像使用分工** - 理解前置相机和环视相机的分工
3. **[v2.0] 完整导航流水线** - 了解多视角独立编码的数据流向
4. **[v2.0] 多视角VPR回环检测** - 理解独立FAISS和投票机制
5. **[v2.0] Dijkstra最短路径规划** - 理解任意起点的路径规划
6. **[v2.0] 拓扑地图管理** - 理解双向图和节点合并
7. **[v2.0] 特征提取详细流程** - 理解多视角独立编码
8. **语义搜索流程** - 理解同义词扩展和模糊匹配
9. **配置参数说明** - 参数调优参考

### 记忆功能关闭时 (memory_enabled=False)
1. **记忆关闭时的导航流程** - 了解关闭记忆时的简化流程

## 相机配置 v2.0

| 相机 | 角度 | 位置 | 用途 |
|------|------|------|------|
| front_1 | 0° | 正前 | 仅模型推理 |
| camera_1 | +37.5° | 前右 | 多视角VPR |
| camera_2 | -37.5° | 前左 | 多视角VPR |
| camera_3 | -142.5° | 后左 | 多视角VPR |
| camera_4 | +142.5° | 后右 | 多视角VPR |

## v2.0 核心功能详情

### 多视角独立VPR
- **独立FAISS索引**: 每个相机视角有独立的IndexFlatIP索引
- **投票机制**: 统计各节点在多少视角获得高相似度
- **匹配条件**:
  - 投票分数 ≥ 50% (至少2个视角匹配)
  - 或最佳视角相似度 ≥ 0.88 (单视角强匹配)
  - 或加权平均相似度 ≥ 0.78 (整体匹配)

### Dijkstra最短路径规划
- **算法**: NetworkX的dijkstra_path()
- **输入**: 起始节点 + 目标节点
- **输出**: PathPlanResult包含路径序列、总距离、路径点信息、动作序列
- **特点**: 支持从当前位置或任意节点出发

### 双向拓扑图
- **图类型**: nx.Graph (无向图，支持双向导航)
- **边信息**:
  - forward_actions: 正向动作序列
  - backward_actions: 反向动作序列 (自动计算)
  - weight: 边权重
- **节点合并**: 相似度 > 0.9 时自动合并，重定向边关系

### 语义引导 (保留)
- 语义标签数据库: 每个节点存储语义标签集合
- Jaccard相似度: 计算语义标签重叠度
- 语义加分/降分: 匹配加分+0.1, 冲突降分-0.05

## 生成时间

生成于: 2026-01-26 (v2.0)
"""

    index_path = output_dir / "README.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    print(f"✓ 生成索引文件: {index_path}")


if __name__ == "__main__":
    main()
