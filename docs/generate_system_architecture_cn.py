#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉记忆导航系统架构图生成器 - 简洁中文版

生成清晰、美观的系统架构图，适合文档和演示使用
"""

from graphviz import Digraph
import os

# 配色方案 - 使用更柔和的颜色
COLORS = {
    # 主要模块颜色
    'input': '#E3F2FD',          # 浅蓝 - 输入
    'memory': '#E8F5E9',         # 浅绿 - 记忆模块
    'feature': '#FFF3E0',        # 浅橙 - 特征提取
    'navigation': '#F3E5F5',     # 浅紫 - 导航模块
    'output': '#FFECB3',         # 浅黄 - 输出

    # 边框颜色
    'input_border': '#1976D2',
    'memory_border': '#388E3C',
    'feature_border': '#F57C00',
    'navigation_border': '#7B1FA2',
    'output_border': '#FFA000',

    # 特殊颜色
    'white': '#FFFFFF',
    'highlight': '#FFCDD2',
}

# 中文字体
FONT = 'SimHei'


def create_system_overview():
    """生成系统整体架构概览图 - 简洁版"""
    dot = Digraph('MemoryNav_System_Architecture',
                  comment='视觉记忆导航系统架构图',
                  format='png')

    # 全局属性 - 优化布局
    dot.attr(rankdir='TB',  # 从上到下
             size='16,20',
             dpi='300',
             nodesep='0.6',
             ranksep='0.8',
             bgcolor='white',
             fontname=FONT,
             fontsize='14',
             splines='ortho')  # 正交边线

    # 节点默认属性
    dot.attr('node',
             shape='box',
             style='rounded,filled',
             penwidth='2',
             fontname=FONT,
             fontsize='11',
             margin='0.3,0.2')

    # 边默认属性
    dot.attr('edge',
             fontname=FONT,
             fontsize='10',
             penwidth='1.5',
             color='#666666')

    # ================================================================
    # 第1层: 输入层
    # ================================================================
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='📥 数据输入层',
               style='filled,rounded',
               fillcolor=COLORS['input'],
               color=COLORS['input_border'],
               penwidth='2',
               fontsize='14',
               fontname=FONT,
               labeljust='c')

        c.node('ws_input',
               'WebSocket 接口\n(端口 9528)',
               fillcolor=COLORS['white'])

        c.node('front_camera',
               '前置相机\n(front_1)',
               fillcolor=COLORS['white'])

        c.node('surround_cameras',
               '环视相机\n(camera_1~4)',
               fillcolor=COLORS['white'])

        c.node('task_input',
               '导航任务\n(指令文本)',
               fillcolor=COLORS['white'])

    # ================================================================
    # 第2层: 特征提取层
    # ================================================================
    with dot.subgraph(name='cluster_feature') as c:
        c.attr(label='🔍 特征提取层',
               style='filled,rounded',
               fillcolor=COLORS['feature'],
               color=COLORS['feature_border'],
               penwidth='2',
               fontsize='14',
               fontname=FONT,
               labeljust='c')

        c.node('longclip',
               'LongCLIP 特征提取器\n(视觉编码)',
               fillcolor=COLORS['white'],
               shape='box3d')

        c.node('feature_fusion',
               '环视特征融合\n(加权平均)',
               fillcolor=COLORS['white'])

    # ================================================================
    # 第3层: 记忆管理层
    # ================================================================
    with dot.subgraph(name='cluster_memory') as c:
        c.attr(label='💾 记忆管理层',
               style='filled,rounded',
               fillcolor=COLORS['memory'],
               color=COLORS['memory_border'],
               penwidth='2',
               fontsize='14',
               fontname=FONT,
               labeljust='c')

        c.node('vpr',
               'VPR 视觉位置识别\n(FAISS 索引)',
               fillcolor=COLORS['white'],
               shape='cylinder')

        c.node('topo_map',
               '拓扑地图\n(NetworkX 图)',
               fillcolor=COLORS['white'],
               shape='cylinder')

        c.node('route_memory',
               '路线记忆\n(轨迹存储)',
               fillcolor=COLORS['white'],
               shape='cylinder')

        c.node('vlm_scene',
               'VLM 场景描述\n(Qwen2.5-VL)',
               fillcolor=COLORS['highlight'],
               shape='box3d')

        c.node('graphrag',
               'GraphRAG 语义图\n(语义检索)',
               fillcolor=COLORS['white'],
               shape='cylinder')

    # ================================================================
    # 第4层: 导航决策层
    # ================================================================
    with dot.subgraph(name='cluster_navigation') as c:
        c.attr(label='🤖 导航决策层',
               style='filled,rounded',
               fillcolor=COLORS['navigation'],
               color=COLORS['navigation_border'],
               penwidth='2',
               fontsize='14',
               fontname=FONT,
               labeljust='c')

        c.node('internvla',
               'InternVLA-N1 导航模型\n(双系统架构)',
               fillcolor=COLORS['highlight'],
               shape='box3d',
               penwidth='3')

        c.node('return_nav',
               '返回导航器\n(路径规划)',
               fillcolor=COLORS['white'])

        c.node('memory_replay',
               '记忆复用\n(动作回放)',
               fillcolor=COLORS['white'])

    # ================================================================
    # 第5层: 输出层
    # ================================================================
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='📤 动作输出层',
               style='filled,rounded',
               fillcolor=COLORS['output'],
               color=COLORS['output_border'],
               penwidth='2',
               fontsize='14',
               fontname=FONT,
               labeljust='c')

        c.node('action_convert',
               '动作转换器\n(离散→连续)',
               fillcolor=COLORS['white'])

        c.node('ws_output',
               'WebSocket 响应\n(JSON格式)',
               fillcolor=COLORS['white'])

    # ================================================================
    # 数据流连接
    # ================================================================

    # 输入层 → 特征提取层
    dot.edge('ws_input', 'front_camera', style='dashed')
    dot.edge('ws_input', 'surround_cameras', style='dashed')
    dot.edge('ws_input', 'task_input', style='dashed')

    dot.edge('front_camera', 'longclip', label='RGB图像')
    dot.edge('surround_cameras', 'longclip', label='环视图像')
    dot.edge('longclip', 'feature_fusion')

    # 特征提取层 → 记忆管理层
    dot.edge('feature_fusion', 'vpr', label='融合特征')
    dot.edge('vpr', 'topo_map', label='节点管理')
    dot.edge('topo_map', 'route_memory', label='路线记录')

    # VLM场景描述
    dot.edge('surround_cameras', 'vlm_scene',
             label='关键帧', style='dashed', color=COLORS['feature_border'])
    dot.edge('vlm_scene', 'graphrag', label='语义信息')
    dot.edge('graphrag', 'topo_map', label='语义标注', style='dashed')

    # 记忆管理层 → 导航决策层
    dot.edge('task_input', 'internvla', label='任务指令', color=COLORS['navigation_border'])
    dot.edge('front_camera', 'internvla', label='当前观测', color=COLORS['navigation_border'])
    dot.edge('topo_map', 'return_nav', label='拓扑路径')
    dot.edge('route_memory', 'memory_replay', label='历史动作')

    # 导航决策层 → 输出层
    dot.edge('internvla', 'action_convert', label='动作/轨迹', penwidth='2')
    dot.edge('return_nav', 'action_convert', label='返回动作', style='dashed')
    dot.edge('memory_replay', 'action_convert', label='复用动作', style='dashed')
    dot.edge('action_convert', 'ws_output', label='控制指令', penwidth='2')

    return dot


def create_module_detail():
    """生成各模块详细架构图"""
    dot = Digraph('MemoryNav_Module_Detail',
                  comment='视觉记忆导航系统模块详情',
                  format='png')

    dot.attr(rankdir='LR',  # 从左到右
             size='20,12',
             dpi='300',
             nodesep='0.5',
             ranksep='1.0',
             bgcolor='white',
             fontname=FONT,
             fontsize='12')

    dot.attr('node',
             shape='box',
             style='rounded,filled',
             penwidth='2',
             fontname=FONT,
             fontsize='10')

    dot.attr('edge', fontname=FONT, fontsize='9', penwidth='1.5')

    # ================================================================
    # 特征提取模块
    # ================================================================
    with dot.subgraph(name='cluster_feature_detail') as c:
        c.attr(label='特征提取模块',
               style='filled',
               fillcolor='#FFF3E0',
               color='#F57C00',
               penwidth='2')

        c.node('img_input', '图像输入\n[480×640×3]', fillcolor='white')
        c.node('preprocess', '预处理\nResize 224×224', fillcolor='white')
        c.node('vit_encoder', 'ViT 编码器\n12层 Transformer', fillcolor='#FFCCBC', shape='box3d')
        c.node('cls_token', 'CLS Token\n[768]', fillcolor='white')
        c.node('projection', '线性投影\n[768]→[512]', fillcolor='white')
        c.node('l2_norm', 'L2归一化\n[512]', fillcolor='#FFCC80')

    # ================================================================
    # VPR 模块
    # ================================================================
    with dot.subgraph(name='cluster_vpr_detail') as c:
        c.attr(label='VPR 视觉位置识别',
               style='filled',
               fillcolor='#E8F5E9',
               color='#388E3C',
               penwidth='2')

        c.node('query_feat', '查询特征\n[512]', fillcolor='white')
        c.node('faiss_index', 'FAISS 索引\nIndexFlatIP', fillcolor='white', shape='cylinder')
        c.node('topk_search', 'Top-K 搜索\nK=10', fillcolor='white')
        c.node('loop_detect', '回环检测\nsim>0.85', fillcolor='#C8E6C9', shape='diamond')

    # ================================================================
    # 导航模块
    # ================================================================
    with dot.subgraph(name='cluster_nav_detail') as c:
        c.attr(label='InternVLA-N1 导航',
               style='filled',
               fillcolor='#F3E5F5',
               color='#7B1FA2',
               penwidth='2')

        c.node('history_buf', '历史缓冲\n最大8帧', fillcolor='white')
        c.node('rgbd_encode', 'RGBD 编码', fillcolor='white')
        c.node('lang_encode', '语言编码', fillcolor='white')
        c.node('fusion', '多模态融合\nCross-Attention', fillcolor='#E1BEE7', shape='box3d')
        c.node('decoder', '动作解码器', fillcolor='#E1BEE7', shape='box3d')
        c.node('action_out', '离散动作\n[0,1,2,3,5]', fillcolor='#CE93D8')
        c.node('traj_out', '连续轨迹\n[33×2]', fillcolor='#CE93D8')
        c.node('pixel_out', '像素目标\n[y,x]', fillcolor='#CE93D8')

    # 连接
    dot.edge('img_input', 'preprocess')
    dot.edge('preprocess', 'vit_encoder')
    dot.edge('vit_encoder', 'cls_token')
    dot.edge('cls_token', 'projection')
    dot.edge('projection', 'l2_norm')

    dot.edge('l2_norm', 'query_feat')
    dot.edge('query_feat', 'topk_search')
    dot.edge('faiss_index', 'topk_search')
    dot.edge('topk_search', 'loop_detect')

    dot.edge('history_buf', 'rgbd_encode')
    dot.edge('rgbd_encode', 'fusion')
    dot.edge('lang_encode', 'fusion')
    dot.edge('fusion', 'decoder')
    dot.edge('decoder', 'action_out')
    dot.edge('decoder', 'traj_out')
    dot.edge('decoder', 'pixel_out')

    return dot


def create_data_flow():
    """生成数据流程图"""
    dot = Digraph('MemoryNav_Data_Flow',
                  comment='视觉记忆导航系统数据流程',
                  format='png')

    dot.attr(rankdir='TB',
             size='14,18',
             dpi='300',
             nodesep='0.5',
             ranksep='0.6',
             bgcolor='white',
             fontname=FONT,
             fontsize='12')

    dot.attr('node',
             shape='box',
             style='rounded,filled',
             penwidth='2',
             fontname=FONT,
             fontsize='10')

    dot.attr('edge', fontname=FONT, fontsize='9', penwidth='1.5')

    # 开始
    dot.node('start', '🚀 开始', shape='ellipse', fillcolor='#C8E6C9')

    # 接收请求
    dot.node('recv_ws', '接收 WebSocket 请求\n(JSON 数据)', fillcolor='#E3F2FD')

    # 解析数据
    dot.node('parse_data', '解析图像和任务\n(Base64 解码)', fillcolor='#E3F2FD')

    # 判断特殊指令
    dot.node('check_special', '特殊指令?', shape='diamond', fillcolor='#FFF9C4')

    # 特殊指令处理
    dot.node('handle_special', '处理特殊指令\n(STOP/RETURN等)', fillcolor='#FFECB3')

    # 记忆复用检查
    dot.node('check_replay', '记忆复用?', shape='diamond', fillcolor='#FFF9C4')

    # 记忆复用
    dot.node('replay_action', '复用历史动作\n(跳过模型推理)', fillcolor='#E8F5E9')

    # 模型推理
    dot.node('model_infer', 'InternVLA-N1 推理\n(动作生成)', fillcolor='#F3E5F5', shape='box3d')

    # 关键帧检测
    dot.node('check_keyframe', '关键帧?', shape='diamond', fillcolor='#FFF9C4')

    # VLM处理
    dot.node('vlm_process', 'VLM 场景描述\n(语义提取)', fillcolor='#FFF3E0')

    # 记忆更新
    dot.node('update_memory', '更新拓扑图\n(VPR/记忆)', fillcolor='#E8F5E9')

    # 动作转换
    dot.node('convert_action', '动作转换\n(离散→控制)', fillcolor='#FFECB3')

    # 发送响应
    dot.node('send_response', '发送 WebSocket 响应\n(JSON)', fillcolor='#E3F2FD')

    # 结束
    dot.node('end', '✅ 完成', shape='ellipse', fillcolor='#C8E6C9')

    # 连接
    dot.edge('start', 'recv_ws')
    dot.edge('recv_ws', 'parse_data')
    dot.edge('parse_data', 'check_special')

    dot.edge('check_special', 'handle_special', label='是')
    dot.edge('check_special', 'check_replay', label='否')

    dot.edge('handle_special', 'send_response')

    dot.edge('check_replay', 'replay_action', label='是')
    dot.edge('check_replay', 'model_infer', label='否')

    dot.edge('replay_action', 'convert_action')

    dot.edge('model_infer', 'check_keyframe')

    dot.edge('check_keyframe', 'vlm_process', label='是')
    dot.edge('check_keyframe', 'update_memory', label='否')

    dot.edge('vlm_process', 'update_memory')
    dot.edge('update_memory', 'convert_action')

    dot.edge('convert_action', 'send_response')
    dot.edge('send_response', 'end')

    return dot


def main():
    """主函数 - 生成所有架构图"""
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("🎨 开始生成视觉记忆导航系统架构图 (简洁中文版)")
    print("=" * 60)

    try:
        # 1. 系统概览图
        print("\n📐 生成系统架构概览图...")
        overview_dot = create_system_overview()
        overview_path = os.path.join(output_dir, 'system_architecture_cn')
        overview_dot.render(overview_path, format='png', cleanup=True)
        print(f"   ✅ 已保存: {overview_path}.png")

        # 2. 模块详情图
        print("\n📦 生成模块详情图...")
        module_dot = create_module_detail()
        module_path = os.path.join(output_dir, 'module_detail_cn')
        module_dot.render(module_path, format='png', cleanup=True)
        print(f"   ✅ 已保存: {module_path}.png")

        # 3. 数据流程图
        print("\n🔄 生成数据流程图...")
        flow_dot = create_data_flow()
        flow_path = os.path.join(output_dir, 'data_flow_cn')
        flow_dot.render(flow_path, format='png', cleanup=True)
        print(f"   ✅ 已保存: {flow_path}.png")

        print("\n" + "=" * 60)
        print("🎉 所有架构图生成完成!")
        print("=" * 60)
        print(f"\n📁 输出目录: {output_dir}/")
        print("\n生成的文件:")
        print("  1. system_architecture_cn.png - 系统架构概览")
        print("  2. module_detail_cn.png       - 模块详情")
        print("  3. data_flow_cn.png           - 数据流程")
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
