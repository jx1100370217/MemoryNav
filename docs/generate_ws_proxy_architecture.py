#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉记忆导航系统详细架构图生成器 (ws_proxy_with_memory.py)

基于实际代码生成详细的模型架构、记忆记录流程和推理流程图
包含每一层的shape描述和详细的数据流
"""

from graphviz import Digraph
import os

# 配色方案
COLORS = {
    'input': '#E3F2FD',
    'input_line': '#1976D2',
    'feature': '#FCE4EC',
    'feature_line': '#C2185B',
    'vlm': '#FFF3E0',
    'vlm_line': '#F57C00',
    'vpr': '#E8F5E9',
    'vpr_line': '#388E3C',
    'topo': '#F3E5F5',
    'topo_line': '#7B1FA2',
    'nav': '#E0F7FA',
    'nav_line': '#00ACC1',
    'output': '#FFF9C4',
    'output_line': '#FBC02D',
    'memory': '#FFE0B2',
    'memory_line': '#F57C00',
    'ws': '#E1F5FE',
    'ws_line': '#0277BD',
}

FONT = 'SimHei'  # 使用中文字体（黑体）


def create_model_architecture():
    """生成详细的模型架构图 - 包含每一层的shape和所有组件"""
    dot = Digraph('MemoryNav_Model_Architecture', comment='视觉记忆导航系统模型架构')
    
    dot.attr(rankdir='TB', size='28,40', dpi='300',
             nodesep='0.7', ranksep='0.9', bgcolor='white',
             fontname=FONT, fontsize='12')
    
    dot.attr('node', shape='box', style='rounded,filled', penwidth='2',
             fontname=FONT, fontsize='10')
    dot.attr('edge', fontname=FONT, fontsize='9', penwidth='1.5')
    
    # ============================================================
    # 输入层
    # ============================================================
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='📥 输入层 (WebSocket数据)', style='filled',
               fillcolor=COLORS['input'], color=COLORS['input_line'],
               penwidth='3', fontsize='16', fontname=FONT)
        
        c.node('Front_Camera', 
               '📷 前置相机 (front_1)\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '原始图像: PIL.Image\\n'
               'Resize: (640, 480)\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'RGB图像: [H=480, W=640, C=3]\\n'
               'dtype: uint8',
               fillcolor='white', shape='folder')
        
        c.node('Surround_Cameras',
               '📷 环视相机 (camera_1~4)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'camera_1: +37.5° (前右)\\n'
               'camera_2: -37.5° (前左)\\n'
               'camera_3: -142.5° (后左)\\n'
               'camera_4: +142.5° (后右)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '每张图像: [480, 640, 3]\\n'
               'dtype: uint8',
               fillcolor='white', shape='folder')
        
        c.node('Instruction',
               '📝 导航指令\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '类型: str\\n'
               '示例: "穿过马路后左转"',
               fillcolor='white', shape='note')
        
        c.node('Depth_Pose',
               '📊 深度图 & 位姿\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'Depth: [480, 640] float32\\n'
               'Pose: [4, 4] float32\\n'
               'Intrinsic: [4, 4] float32',
               fillcolor='white', shape='cylinder')

    # ============================================================
    # LongCLIP 视觉特征提取器
    # ============================================================
    with dot.subgraph(name='cluster_longclip') as c:
        c.attr(label='🔍 LongCLIP 视觉特征提取器', style='filled',
               fillcolor=COLORS['feature'], color=COLORS['feature_line'],
               penwidth='3', fontsize='16')
        
        c.node('LongCLIP_Preprocess',
               'LongCLIP Preprocessor\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输入: RGB [H, W, 3]\\n'
               'Resize & Normalize\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: Tensor [3, 224, 224]',
               fillcolor='white')
        
        c.node('LongCLIP_Vision',
               'LongCLIP Vision Encoder\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Conv1: Patch Embedding\\n'
               '  [B, 3, 224, 224] → [B, 768, 7, 7]\\n'
               'Positional Embedding\\n'
               'Transformer (12 Layers):\\n'
               '  Hidden Size: 768\\n'
               '  Heads: 12\\n'
               '  Self-Attention + FFN\\n'
               'LayerNorm\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: CLS Token [B, 768]',
               fillcolor='#FFE0E6', shape='box3d', penwidth='3')
        
        c.node('Feature_Projection',
               '特征投影\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Linear Projection\\n'
               '[B, 768] → [B, 512]\\n'
               'L2 Normalization\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: Feature Vector [512]',
               fillcolor=COLORS['output'], color=COLORS['output_line'],
               shape='parallelogram')

    # ============================================================
    # 环视相机特征融合
    # ============================================================
    with dot.subgraph(name='cluster_surround_fusion') as c:
        c.attr(label='🔄 环视相机特征融合', style='filled',
               fillcolor=COLORS['feature'], color=COLORS['feature_line'],
               penwidth='3', fontsize='16')
        
        c.node('Extract_Surround_Features',
               '提取环视特征\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'For each camera_i in [1,2,3,4]:\\n'
               '  feature_i = LongCLIP(image_i)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: Dict{cam_id: [512]}',
               fillcolor='white')
        
        c.node('Weighted_Fusion',
               '加权融合\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Weight per camera: 0.25\\n'
               'fused = Σ (weight_i × feature_i)\\n'
               'Normalize: L2 norm\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: Fused Feature [512]',
               fillcolor=COLORS['output'], color=COLORS['output_line'],
               shape='parallelogram')

    # ============================================================
    # VLM 场景描述生成器 (Qwen3-VL)
    # ============================================================
    with dot.subgraph(name='cluster_vlm') as c:
        c.attr(label='🧠 Qwen3-VL 场景描述生成器 (关键帧)', 
               style='filled', fillcolor=COLORS['vlm'],
               color=COLORS['vlm_line'], penwidth='3', fontsize='16')
        
        c.node('VLM_Condition',
               '⚡ 触发条件\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'pixel_target ≠ None\\n'
               '(关键帧检测)',
               fillcolor='#FFEBEE', shape='diamond')
        
        c.node('VLM_Processor',
               'Qwen3-VL Processor\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输入: 4张环视图像\\n'
               'Image Preprocessing\\n'
               'Prompt Construction\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: input_ids, pixel_values',
               fillcolor='white')
        
        c.node('VLM_Model',
               'Qwen2.5-VL-8B Model\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Vision Encoder: 提取图像特征\\n'
               'Language Decoder: 生成描述\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Max New Tokens: 256\\n'
               'Device: cuda:1',
               fillcolor='#FFE0E6', shape='box3d', penwidth='3')
        
        c.node('Scene_Description',
               '📝 场景描述输出\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'scene_description: str\\n'
               '示例: "当前位于室内走廊，\\n'
               '左侧有窗户，右侧是墙壁，\\n'
               '前方可见电梯门"\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'semantic_labels: List[str]\\n'
               '示例: ["走廊", "窗户", "电梯"]',
               fillcolor=COLORS['output'], color=COLORS['output_line'],
               shape='note')

    # ============================================================
    # 视觉位置识别 (VPR)
    # ============================================================
    with dot.subgraph(name='cluster_vpr') as c:
        c.attr(label='🎯 视觉位置识别 (VPR)', style='filled',
               fillcolor=COLORS['vpr'], color=COLORS['vpr_line'],
               penwidth='3', fontsize='16')
        
        c.node('FAISS_Index',
               'FAISS 索引\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'IndexFlatIP (内积搜索)\\n'
               'Feature Dimension: 512\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Database:\\n'
               '  features: [N, 512]\\n'
               '  node_ids: [N]\\n'
               '  timestamps: [N]',
               fillcolor='white', shape='cylinder')
        
        c.node('Similarity_Search',
               '相似度搜索\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '查询特征: [512]\\n'
               'Top-K Search (k=10)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: [(node_id, similarity)]',
               fillcolor='white')
        
        c.node('Loop_Closure',
               '回环检测\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '条件:\\n'
               '  1. similarity > 0.85\\n'
               '  2. time_gap > 30s\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: (node_id, similarity)\\n'
               '      或 None',
               fillcolor=COLORS['output'], color=COLORS['output_line'],
               shape='diamond')

    # ============================================================
    # 拓扑地图管理器
    # ============================================================
    with dot.subgraph(name='cluster_topo') as c:
        c.attr(label='🗺️ 拓扑地图管理器', style='filled',
               fillcolor=COLORS['topo'], color=COLORS['topo_line'],
               penwidth='3', fontsize='16')
        
        c.node('Create_Node',
               '创建/更新节点\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'TopologicalNode:\\n'
               '  node_id: int\\n'
               '  visual_feature: [512]\\n'
               '  rgb_image: [480, 640, 3]\\n'
               '  surround_images: Dict\\n'
               '  timestamp: float\\n'
               '  scene_description: str\\n'
               '  semantic_labels: List[str]\\n'
               '  pixel_target: [2] (关键帧)\\n'
               '  is_keyframe: bool',
               fillcolor='white', shape='box3d')
        
        c.node('NetworkX_Graph',
               'NetworkX 有向图\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '节点: TopologicalNode\\n'
               '边: (from, to, actions)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '最短路径搜索\\n'
               'nx.shortest_path()',
               fillcolor='white', shape='cylinder')
        
        c.node('GraphRAG_Semantic',
               'GraphRAG 语义地图\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'node_metadata: Dict\\n'
               '  scene_description\\n'
               '  semantic_labels\\n'
               '  visit_count\\n'
               'label_index: Dict[str, List[int]]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '语义搜索\\n'
               'semantic_search(query, k=5)',
               fillcolor='white', shape='cylinder')

    # ============================================================
    # InternVLA-N1 导航模型
    # ============================================================
    with dot.subgraph(name='cluster_nav_model') as c:
        c.attr(label='🤖 InternVLA-N1 导航模型', style='filled',
               fillcolor=COLORS['nav'], color=COLORS['nav_line'],
               penwidth='3', fontsize='16')
        
        c.node('History_Buffer',
               '历史帧缓冲\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '最大历史帧数: 8\\n'
               'rgb_list: List[np.ndarray]\\n'
               'depth_list: List[np.ndarray]\\n'
               'pose_list: List[np.ndarray]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '采样策略: np.linspace',
               fillcolor='white')
        
        c.node('InternVLA_Encoder',
               'InternVLA-N1 视觉编码器\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输入: RGB+Depth+Pose\\n'
               'Resize: (384, 384)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'CNN Backbone\\n'
               'Multi-scale Feature Extraction\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: Feature Maps',
               fillcolor='#E1F5FE', shape='box3d', penwidth='3')
        
        c.node('Language_Encoder',
               '语言编码器\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输入: instruction (str)\\n'
               'Tokenization\\n'
               'Embedding\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: Text Embedding',
               fillcolor='#E1F5FE', shape='box3d')
        
        c.node('InternVLA_Fusion',
               '多模态融合\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Vision-Language Fusion\\n'
               'Cross-attention\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: Fused Features',
               fillcolor='#E1F5FE', shape='box3d')
        
        c.node('InternVLA_Decoder',
               'InternVLA-N1 解码器\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '双系统输出:\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '1. 离散动作序列\\n'
               '   output_action: List[int]\\n'
               '   0=STOP, 1=前进, 2=左转\\n'
               '   3=右转, 5=向下看\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '2. 连续轨迹\\n'
               '   output_trajectory: [33, 2]\\n'
               '   增量坐标 [dx, dy]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '3. 像素目标\\n'
               '   output_pixel: [y, x]\\n'
               '   关键帧标记',
               fillcolor='#E1F5FE', shape='box3d', penwidth='3')

    # ============================================================
    # 输出转换
    # ============================================================
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='📤 输出转换', style='filled',
               fillcolor=COLORS['output'], color=COLORS['output_line'],
               penwidth='3', fontsize='16')
        
        c.node('Action_Converter',
               '动作转换器\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '离散动作 → 机器人控制:\\n'
               '  统计: forward, left, right\\n'
               '  x = forward × 0.25m\\n'
               '  yaw = (left - right) × π/24\\n'
               '  输出: [[x, y, yaw]]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '轨迹点 → 累积坐标:\\n'
               '  cumsum(delta_xy)\\n'
               '  输出: [[x1,y1,0], ..., [xn,yn,0]]',
               fillcolor='white')
        
        c.node('Pixel_Normalizer',
               '像素目标归一化\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输入: [y, x] (绝对坐标)\\n'
               '图像尺寸: 480×640\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '归一化:\\n'
               '  x_norm = x / 640\\n'
               '  y_norm = y / 480\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: [x_norm, y_norm]',
               fillcolor='white')
        
        c.node('Response_JSON',
               '📦 WebSocket 响应\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '{\\n'
               '  "status": "success",\\n'
               '  "id": robot_id,\\n'
               '  "pts": timestamp,\\n'
               '  "task_status": "executing/end",\\n'
               '  "action": [[x, y, yaw], ...],\\n'
               '  "pixel_target": [x, y],\\n'
               '  "memory_info": {...}\\n'
               '}',
               fillcolor=COLORS['output'], color=COLORS['output_line'],
               shape='note', penwidth='3')

    # ============================================================
    # Edges
    # ============================================================
    
    # 输入 → LongCLIP
    dot.edge('Front_Camera', 'LongCLIP_Preprocess', color=COLORS['input_line'])
    dot.edge('Surround_Cameras', 'Extract_Surround_Features', color=COLORS['input_line'])
    
    # LongCLIP 流程
    dot.edge('LongCLIP_Preprocess', 'LongCLIP_Vision')
    dot.edge('LongCLIP_Vision', 'Feature_Projection')
    
    # 环视融合
    dot.edge('Extract_Surround_Features', 'Weighted_Fusion', label='camera_1~4 features')
    
    # VPR 流程
    dot.edge('Weighted_Fusion', 'Similarity_Search', label='query feature')
    dot.edge('FAISS_Index', 'Similarity_Search', style='dashed')
    dot.edge('Similarity_Search', 'Loop_Closure')
    
    # 拓扑地图
    dot.edge('Loop_Closure', 'Create_Node', label='new/revisited')
    dot.edge('Create_Node', 'NetworkX_Graph')
    dot.edge('Create_Node', 'FAISS_Index', label='add feature', style='dashed')
    
    # VLM 流程 (关键帧)
    dot.edge('VLM_Condition', 'VLM_Processor', label='关键帧', color=COLORS['vlm_line'])
    dot.edge('Surround_Cameras', 'VLM_Processor', style='dotted', color=COLORS['vlm_line'])
    dot.edge('VLM_Processor', 'VLM_Model')
    dot.edge('VLM_Model', 'Scene_Description')
    dot.edge('Scene_Description', 'GraphRAG_Semantic', label='semantic info')
    dot.edge('GraphRAG_Semantic', 'Create_Node', style='dashed')
    
    # InternVLA 导航流程
    dot.edge('Front_Camera', 'History_Buffer', color=COLORS['nav_line'])
    dot.edge('Depth_Pose', 'History_Buffer', color=COLORS['nav_line'])
    dot.edge('History_Buffer', 'InternVLA_Encoder')
    dot.edge('Instruction', 'Language_Encoder', color=COLORS['nav_line'])
    dot.edge('InternVLA_Encoder', 'InternVLA_Fusion')
    dot.edge('Language_Encoder', 'InternVLA_Fusion')
    dot.edge('InternVLA_Fusion', 'InternVLA_Decoder')
    
    # 输出转换
    dot.edge('InternVLA_Decoder', 'Action_Converter', label='action/trajectory')
    dot.edge('InternVLA_Decoder', 'Pixel_Normalizer', label='pixel_target')
    dot.edge('InternVLA_Decoder', 'VLM_Condition', label='pixel≠None?', style='dashed')
    dot.edge('Action_Converter', 'Response_JSON')
    dot.edge('Pixel_Normalizer', 'Response_JSON')
    dot.edge('Create_Node', 'Response_JSON', label='memory_info', style='dashed')
    
    return dot


def create_memory_recording_flow():
    """生成记忆记录流程图"""
    dot = Digraph('Memory_Recording_Flow', comment='视觉记忆记录流程')
    
    dot.attr(rankdir='TB', size='22,30', dpi='300',
             nodesep='0.8', ranksep='1.0', bgcolor='white',
             fontname=FONT, fontsize='12')
    
    dot.attr('node', shape='box', style='rounded,filled', penwidth='2',
             fontname=FONT, fontsize='10')
    dot.attr('edge', fontname=FONT, fontsize='9', penwidth='1.5')
    
    # ============================================================
    # 启动记录
    # ============================================================
    with dot.subgraph(name='cluster_start') as c:
        c.attr(label='🎬 启动记忆记录', style='filled',
               fillcolor='#E8F5E9', color='#388E3C', penwidth='3', fontsize='16')
        
        c.node('Start_Recording',
               'START_MEMORY 指令\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'instruction = 任务描述\\n'
               'route_id = f"route_{timestamp}"\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '初始化 RouteMemory:\\n'
               '  node_sequence = []\\n'
               '  action_history = []\\n'
               '  keyframe_indices = []\\n'
               '  visual_features = []',
               fillcolor='white')

    # ============================================================
    # 导航循环
    # ============================================================
    with dot.subgraph(name='cluster_loop') as c:
        c.attr(label='🔁 导航循环 (每帧)', style='filled',
               fillcolor='#E3F2FD', color='#1976D2', penwidth='3', fontsize='16')
        
        c.node('Receive_Frame',
               '接收帧数据\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'RGB图像 [480, 640, 3]\\n'
               '环视图像 camera_1~4\\n'
               '导航指令',
               fillcolor='#BBDEFB')
        
        c.node('Extract_Feature',
               '提取视觉特征\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'LongCLIP: camera_1~4\\n'
               '环视融合 [512]\\n'
               'VPR 回环检测',
               fillcolor='#BBDEFB')
        
        c.node('VLA_Inference',
               'InternVLA-N1 推理\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '输出动作序列\\n'
               '输出像素目标 (关键帧)',
               fillcolor='#BBDEFB')
        
        c.node('Check_Keyframe',
               '是否关键帧？\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'pixel_target ≠ None',
               fillcolor='#FFF9C4', shape='diamond')
        
        c.node('VLM_Generate',
               'VLM 场景描述生成\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'Qwen3-VL: camera_1~4\\n'
               '生成 scene_description\\n'
               '提取 semantic_labels',
               fillcolor='#FFE0B2')
        
        c.node('Add_Topo_Node',
               '添加拓扑节点\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'TopologicalNode:\\n'
               '  visual_feature\\n'
               '  rgb_image\\n'
               '  surround_images\\n'
               '  scene_description\\n'
               '  semantic_labels\\n'
               '  is_keyframe',
               fillcolor='#BBDEFB')
        
        c.node('Record_Step',
               '记录导航步骤\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'node_sequence.append(node_id)\\n'
               'action_history.append(action)\\n'
               'visual_features.append(feature)\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '如果是关键帧:\\n'
               '  keyframe_indices.append(idx)\\n'
               '  keyframe_images.append(rgb)',
               fillcolor='#BBDEFB')
        
        c.node('Save_Realtime',
               '实时保存到磁盘\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '每个关键帧触发保存:\\n'
               '  {route_id}.pkl\\n'
               '  {route_id}_features.npy\\n'
               '  {route_id}_keyframes/',
               fillcolor='#C8E6C9')

    # ============================================================
    # 停止记录
    # ============================================================
    with dot.subgraph(name='cluster_stop') as c:
        c.attr(label='🛑 停止记录', style='filled',
               fillcolor='#FCE4EC', color='#C2185B', penwidth='3', fontsize='16')
        
        c.node('Stop_Recording',
               'STOP_MEMORY 指令\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'route.is_complete = True\\n'
               'route.end_timestamp = now',
               fillcolor='white')
        
        c.node('Save_Route',
               '保存完整路线\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '持久化到磁盘:\\n'
               '  路线元数据 .pkl\\n'
               '  视觉特征 .npy\\n'
               '  关键帧图像 .jpg\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '保存语义图:\\n'
               '  semantic_metadata.json\\n'
               '  semantic_graph.json',
               fillcolor='white')
        
        c.node('Route_Stats',
               '路线统计\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'total_nodes: int\\n'
               'total_keyframes: int\\n'
               'duration: float\\n'
               'semantic_nodes: int',
               fillcolor='#FFF9C4', shape='note')

    # ============================================================
    # Edges
    # ============================================================
    dot.edge('Start_Recording', 'Receive_Frame')
    dot.edge('Receive_Frame', 'Extract_Feature')
    dot.edge('Extract_Feature', 'VLA_Inference')
    dot.edge('VLA_Inference', 'Check_Keyframe')
    dot.edge('Check_Keyframe', 'VLM_Generate', label='是')
    dot.edge('Check_Keyframe', 'Add_Topo_Node', label='否')
    dot.edge('VLM_Generate', 'Add_Topo_Node')
    dot.edge('Add_Topo_Node', 'Record_Step')
    dot.edge('Record_Step', 'Save_Realtime', label='关键帧')
    dot.edge('Record_Step', 'Receive_Frame', label='下一帧')
    dot.edge('Save_Realtime', 'Receive_Frame', style='dashed')
    dot.edge('Receive_Frame', 'Stop_Recording', label='STOP', style='dashed', color='red')
    dot.edge('Stop_Recording', 'Save_Route')
    dot.edge('Save_Route', 'Route_Stats')
    
    return dot


def create_inference_flow():
    """生成推理流程图（包含记忆复用）"""
    dot = Digraph('Inference_Flow', comment='推理流程（含记忆复用）')
    
    dot.attr(rankdir='TB', size='24,32', dpi='300',
             nodesep='0.8', ranksep='1.0', bgcolor='white',
             fontname=FONT, fontsize='12')
    
    dot.attr('node', shape='box', style='rounded,filled', penwidth='2',
             fontname=FONT, fontsize='10')
    dot.attr('edge', fontname=FONT, fontsize='9', penwidth='1.5')
    
    # ============================================================
    # WebSocket 输入
    # ============================================================
    with dot.subgraph(name='cluster_ws_input') as c:
        c.attr(label='📡 WebSocket 输入', style='filled',
               fillcolor=COLORS['ws'], color=COLORS['ws_line'],
               penwidth='3', fontsize='16')
        
        c.node('WS_Receive',
               'WebSocket 接收消息\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'JSON 格式:\\n'
               '  id: robot_id\\n'
               '  pts: timestamp\\n'
               '  task: instruction\\n'
               '  images: {front_1, camera_1~4}\\n'
               '  depth: base64 (可选)\\n'
               '  pose: [4,4] (可选)',
               fillcolor='white', shape='parallelogram')
        
        c.node('Decode_Images',
               '解码图像\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'Base64 → PIL.Image\\n'
               'Resize to (640, 480)\\n'
               'Convert to numpy array',
               fillcolor='white')

    # ============================================================
    # 特殊指令处理
    # ============================================================
    with dot.subgraph(name='cluster_special') as c:
        c.attr(label='⚡ 特殊指令处理', style='filled',
               fillcolor='#FFF3E0', color='#F57C00',
               penwidth='3', fontsize='16')
        
        c.node('Check_Special',
               '检查特殊指令\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'STOP / RETURN /\\n'
               'START_MEMORY / STOP_MEMORY /\\n'
               'turn left / turn right / go straight',
               fillcolor='white', shape='diamond')
        
        c.node('Direct_Control',
               '直接控制\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'turn left → [0, 0, π/12]\\n'
               'turn right → [0, 0, -π/12]\\n'
               'go straight → [1, 0, 0]\\n'
               '跳过模型推理',
               fillcolor='#FFE0B2')
        
        c.node('Return_Nav',
               '返回导航\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '启动返回导航器\\n'
               '使用拓扑图路径\\n'
               '或轨迹回溯',
               fillcolor='#FFE0B2')

    # ============================================================
    # 记忆复用检查
    # ============================================================
    with dot.subgraph(name='cluster_replay') as c:
        c.attr(label='🔄 记忆复用检查', style='filled',
               fillcolor=COLORS['memory'], color=COLORS['memory_line'],
               penwidth='3', fontsize='16')
        
        c.node('Check_Replay',
               '检查记忆复用\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '条件:\\n'
               '  1. 任务刚开始 (request_count=0)\\n'
               '  2. 或任务刚变化\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '搜索匹配路线:\\n'
               '  find_matching_route(instruction)',
               fillcolor='white', shape='diamond')
        
        c.node('Load_Route',
               '加载匹配路线\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '从内存或磁盘加载:\\n'
               '  route_id\\n'
               '  action_history\\n'
               '  keyframe_indices\\n'
               '  visual_features',
               fillcolor='white')
        
        c.node('Replay_Action',
               '复用动作\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'action = route.action_history[step]\\n'
               'step += 1\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '跳过模型推理\\n'
               'skipped_inference = True',
               fillcolor='#C8E6C9', penwidth='3')
        
        c.node('Check_Replay_Complete',
               '复用完成？\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'step >= total_steps',
               fillcolor='#FFF9C4', shape='diamond')

    # ============================================================
    # 正常推理流程
    # ============================================================
    with dot.subgraph(name='cluster_inference') as c:
        c.attr(label='🧠 正常推理流程', style='filled',
               fillcolor='#E1F5FE', color='#0277BD',
               penwidth='3', fontsize='16')
        
        c.node('Check_Task_Change',
               '检查任务变化\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'instruction != last_task?\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '如果变化: Agent.reset()',
               fillcolor='white', shape='diamond')
        
        c.node('VLA_Step',
               'InternVLA-N1 推理\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'agent.step(\\n'
               '  rgb, depth, pose,\\n'
               '  instruction, intrinsic,\\n'
               '  look_down\\n'
               ')\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '历史帧采样 (最多8帧)\\n'
               '多模态编码与融合',
               fillcolor='#BBDEFB', shape='box3d', penwidth='3')
        
        c.node('Check_Action5',
               '检测动作5？\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'output_action[0] == 5\\n'
               '(向下看)',
               fillcolor='#FFF9C4', shape='diamond')
        
        c.node('Lookdown_Inference',
               'Look-Down 推理\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'agent.step(\\n'
               '  同样输入,\\n'
               '  look_down=True\\n'
               ')\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '重新推理获取精细动作',
               fillcolor='#BBDEFB')
        
        c.node('Dual_Output',
               '双系统输出\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '1. output_action: List[int]\\n'
               '   离散动作序列\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '2. output_trajectory: [33, 2]\\n'
               '   连续轨迹点 (增量)\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '3. output_pixel: [y, x]\\n'
               '   像素目标 (关键帧)',
               fillcolor='#BBDEFB')

    # ============================================================
    # 记忆处理
    # ============================================================
    with dot.subgraph(name='cluster_memory_process') as c:
        c.attr(label='💾 记忆处理', style='filled',
               fillcolor=COLORS['memory'], color=COLORS['memory_line'],
               penwidth='3', fontsize='16')
        
        c.node('Extract_Surround',
               '提取环视特征\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'LongCLIP(camera_1~4)\\n'
               '环视融合 [512]',
               fillcolor='white')
        
        c.node('VPR_Check',
               'VPR 回环检测\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'FAISS 搜索\\n'
               'is_revisited?',
               fillcolor='white', shape='diamond')
        
        c.node('VLM_Keyframe',
               'VLM 处理 (关键帧)\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'pixel_target ≠ None?\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'Qwen3-VL 生成:\\n'
               '  scene_description\\n'
               '  semantic_labels',
               fillcolor='white')
        
        c.node('Update_Topo',
               '更新拓扑图\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '添加/更新节点\\n'
               '添加边 (动作)\\n'
               '更新 FAISS 索引\\n'
               '更新 GraphRAG',
               fillcolor='white')
        
        c.node('Record_If_Active',
               '记录步骤 (如果记录中)\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'if route_memory.is_recording():\\n'
               '  record_step(node_id, action)\\n'
               '  if is_keyframe:\\n'
               '    save_keyframe_image()',
               fillcolor='white')

    # ============================================================
    # 输出处理
    # ============================================================
    with dot.subgraph(name='cluster_output_process') as c:
        c.attr(label='📤 输出处理', style='filled',
               fillcolor=COLORS['output'], color=COLORS['output_line'],
               penwidth='3', fontsize='16')
        
        c.node('Convert_Action',
               '动作转换\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '离散动作 → [x, y, yaw]\\n'
               '或\\n'
               '轨迹点 → 累积坐标',
               fillcolor='white')
        
        c.node('Check_Small_Action',
               '小动作检测\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '33个点 && 所有值 < 0.5?\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '自动转换为 STOP',
               fillcolor='white', shape='diamond')
        
        c.node('Normalize_Pixel',
               '像素目标归一化\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '[y, x] → [x/640, y/480]',
               fillcolor='white')
        
        c.node('Build_Response',
               '构建响应\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'JSON:\\n'
               '  status, id, pts\\n'
               '  task_status\\n'
               '  action\\n'
               '  pixel_target\\n'
               '  memory_info',
               fillcolor='white')
        
        c.node('Visualize_Save',
               '可视化保存 (关键帧)\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '标注图像\\n'
               '环视拼接图\\n'
               '元数据 JSON',
               fillcolor='white')
        
        c.node('WS_Send',
               'WebSocket 发送响应\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'JSON 响应',
               fillcolor='white', shape='parallelogram')

    # ============================================================
    # Edges
    # ============================================================
    
    # 输入流程
    dot.edge('WS_Receive', 'Decode_Images')
    dot.edge('Decode_Images', 'Check_Special')
    
    # 特殊指令
    dot.edge('Check_Special', 'Direct_Control', label='直接控制')
    dot.edge('Check_Special', 'Return_Nav', label='返回')
    dot.edge('Direct_Control', 'Build_Response')
    dot.edge('Return_Nav', 'Build_Response')
    
    # 记忆复用
    dot.edge('Check_Special', 'Check_Replay', label='正常任务')
    dot.edge('Check_Replay', 'Load_Route', label='找到匹配')
    dot.edge('Load_Route', 'Replay_Action')
    dot.edge('Replay_Action', 'Check_Replay_Complete')
    dot.edge('Check_Replay_Complete', 'Build_Response', label='完成')
    dot.edge('Check_Replay_Complete', 'WS_Receive', label='继续', style='dashed')
    
    # 正常推理
    dot.edge('Check_Replay', 'Check_Task_Change', label='无匹配')
    dot.edge('Check_Task_Change', 'VLA_Step')
    dot.edge('VLA_Step', 'Check_Action5')
    dot.edge('Check_Action5', 'Lookdown_Inference', label='是')
    dot.edge('Check_Action5', 'Dual_Output', label='否')
    dot.edge('Lookdown_Inference', 'Dual_Output')
    
    # 记忆处理
    dot.edge('Dual_Output', 'Extract_Surround')
    dot.edge('Extract_Surround', 'VPR_Check')
    dot.edge('VPR_Check', 'VLM_Keyframe', label='新位置')
    dot.edge('VPR_Check', 'Update_Topo', label='已访问')
    dot.edge('VLM_Keyframe', 'Update_Topo')
    dot.edge('Update_Topo', 'Record_If_Active')
    
    # 输出处理
    dot.edge('Record_If_Active', 'Convert_Action')
    dot.edge('Convert_Action', 'Check_Small_Action')
    dot.edge('Check_Small_Action', 'Normalize_Pixel', label='否')
    dot.edge('Check_Small_Action', 'Build_Response', label='是 (STOP)', color='red')
    dot.edge('Normalize_Pixel', 'Build_Response')
    dot.edge('Build_Response', 'Visualize_Save', style='dashed', label='关键帧')
    dot.edge('Visualize_Save', 'WS_Send')
    dot.edge('Build_Response', 'WS_Send')
    dot.edge('WS_Send', 'WS_Receive', label='下一帧', style='dotted')
    
    return dot


def main():
    output_dir = './docs'
    os.makedirs(output_dir, exist_ok=True)
    print("正在生成架构图...")
    
    try:
        # 模型架构图
        print("1. 生成模型架构图...")
        model_dot = create_model_architecture()
        model_dot.render(f'{output_dir}/ws_proxy_model_architecture', format='png', cleanup=True)
        print(f"✅ 模型架构图: {output_dir}/ws_proxy_model_architecture.png")
        
        # 记忆记录流程图
        print("2. 生成记忆记录流程图...")
        recording_dot = create_memory_recording_flow()
        recording_dot.render(f'{output_dir}/ws_proxy_memory_recording_flow', format='png', cleanup=True)
        print(f"✅ 记忆记录流程图: {output_dir}/ws_proxy_memory_recording_flow.png")
        
        # 推理流程图
        print("3. 生成推理流程图...")
        inference_dot = create_inference_flow()
        inference_dot.render(f'{output_dir}/ws_proxy_inference_flow', format='png', cleanup=True)
        print(f"✅ 推理流程图: {output_dir}/ws_proxy_inference_flow.png")
        
        print("\n" + "="*60)
        print("✅ 所有流程图生成完成！")
        print("="*60)
        print(f"\n输出目录: {output_dir}/")
        print(f"  1. ws_proxy_model_architecture.png - 详细模型结构图")
        print(f"  2. ws_proxy_memory_recording_flow.png - 记忆记录流程图")
        print(f"  3. ws_proxy_inference_flow.png - 推理流程图（含记忆复用）")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()










