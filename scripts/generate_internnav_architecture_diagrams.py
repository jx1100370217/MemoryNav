"""
InternNav项目详细架构图生成器
基于实际代码生成详细的模型架构、训练流程和推理流程图
包含每一层的shape描述和详细的数据流
"""

from graphviz import Digraph
import os

# 配色方案
COLORS = {
    'input': '#E3F2FD',
    'input_line': '#1976D2',
    's2_vlm': '#FCE4EC',
    's2_vlm_line': '#C2185B',
    's1_navdp': '#FFF3E0',
    's1_navdp_line': '#F57C00',
    'encoder': '#E8F5E9',
    'encoder_line': '#388E3C',
    'decoder': '#F3E5F5',
    'decoder_line': '#7B1FA2',
    'loss': '#FFEBEE',
    'loss_line': '#D32F2F',
    'output': '#E0F7FA',
    'output_line': '#00ACC1',
    'tensor': '#FFF9C4',
    'tensor_line': '#FBC02D',
    'data': '#F3E5F5',
    'data_line': '#9C27B0',
    'process': '#E8EAF6',
    'process_line': '#3F51B5',
}

FONT = 'SimHei'  # 使用中文字体（黑体）

def create_internvlan1_model_architecture():
    """生成InternVLA-N1详细模型架构图"""
    dot = Digraph('InternVLA_N1_Model_Architecture', comment='InternVLA-N1双系统导航模型详细架构')

    dot.attr(rankdir='TB', size='32,48', dpi='300',
             nodesep='0.8', ranksep='1.0', bgcolor='white',
             fontname=FONT, fontsize='12')

    dot.attr('node', shape='box', style='rounded,filled', penwidth='2',
             fontname=FONT, fontsize='10')
    dot.attr('edge', fontname=FONT, fontsize='9', penwidth='1.5')

    # ============================================================
    # 输入层
    # ============================================================
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='📥 输入层', style='filled',
               fillcolor=COLORS['input'], color=COLORS['input_line'],
               penwidth='3', fontsize='18', fontname=FONT)

        c.node('RGB_History',
               '📷 RGB图像历史\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               '当前帧 + 历史帧\\n'
               'List[PIL.Image]\\n'
               'Resize: (width, height)\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'num_history = 3\\n'
               'total_frames = num_history + 1',
               fillcolor='white', shape='folder')

        c.node('Depth',
               '🌊 深度图\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'Depth Map\\n'
               'Shape: [H, W, 1]\\n'
               'Range: 0.1-5.0m',
               fillcolor='white', shape='folder')

        c.node('Instruction',
               '📝 导航指令\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '"Go to the kitchen and..."\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '类型: str',
               fillcolor='white', shape='note')

        c.node('Pose',
               '🧭 位姿信息\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'Position + Rotation\\n'
               'Shape: [7] (xyz + quat)',
               fillcolor='white')

    # ============================================================
    # System 2: Qwen2.5-VL 高级规划器
    # ============================================================
    with dot.subgraph(name='cluster_s2') as c:
        c.attr(label='🧠 System 2: Qwen2.5-VL 高级规划器 (S2)',
               style='filled', fillcolor=COLORS['s2_vlm'],
               color=COLORS['s2_vlm_line'], penwidth='3', fontsize='18')

        # Processor
        c.node('Processor',
               '⚙️ AutoProcessor\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '图像预处理:\\n'
               '  • pixel_values: [B, N_img, C, H, W]\\n'
               '  • image_grid_thw: [B, N_img, 3]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '文本分词:\\n'
               '  • input_ids: [B, seq_len]\\n'
               '  • attention_mask: [B, seq_len]',
               fillcolor='white')

        # Vision Tower
        c.node('Vision_Tower',
               '🖼️ Qwen2.5-VL Vision Tower\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输入: pixel_values\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Patch Embed + ViT Blocks\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: image_embeds\\n'
               '  [B, N_img_tokens, hidden_dim]',
               fillcolor='#FFEBEE', shape='component')

        # Text Embedding
        c.node('Text_Embed',
               '📝 Text Embeddings\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输入: input_ids [B, N_text]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Embedding(vocab_size, 2560)',
               fillcolor='#FFEBEE', shape='component')

        # Transformer
        c.node('QwenVL_Transformer',
               '🔄 Qwen2.5-VL Transformer\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Config: Qwen/Qwen2.5-VL-7B-Instruct\\n'
               'Hidden Size: 2560\\n'
               'Num Layers: 28\\n'
               'Attention Heads: 20\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Flash Attention 2\\n'
               'RoPE Position Encoding\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输入: Concat(Image + Text) Embeds\\n'
               '输出: last_hidden_state [B, seq_len, 2560]',
               fillcolor='#FFE0E6', shape='box3d', penwidth='3')

        # LM Head (for text generation)
        c.node('LM_Head',
               '💬 Language Model Head\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Linear(2560 -> vocab_size)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '生成文本输出:\\n'
               '  • 像素坐标: "(x, y)"\\n'
               '  • 或离散动作: "↑←→↓STOP"',
               fillcolor=COLORS['tensor'], color=COLORS['tensor_line'])

        # Latent Queries
        c.node('Latent_Queries',
               '🎯 Latent Query Generation\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Step 1: 在input_ids末尾添加\\n'
               '  TRAJ_START_TOKEN (151665)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Step 2: 插入N个TRAJ_TOKEN\\n'
               '  learnable latent_queries\\n'
               '  [1, n_query, 2560]\\n'
               '  默认 n_query = 100\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Step 3: 通过Transformer\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: traj_latents\\n'
               '  [B, 100, 2560]',
               fillcolor=COLORS['tensor'], color=COLORS['tensor_line'],
               shape='parallelogram')

    # ============================================================
    # System 1: NavDP 低级运动控制器
    # ============================================================
    with dot.subgraph(name='cluster_s1') as c:
        c.attr(label='🤖 System 1: NavDP 扩散策略导航 (S1)',
               style='filled', fillcolor=COLORS['s1_navdp'],
               color=COLORS['s1_navdp_line'], penwidth='3', fontsize='18')

        # NavDP输入准备
        c.node('NavDP_Input_Prep',
               '📦 NavDP输入准备\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'RGB-D Images (当前观测)\\n'
               'Latent from S2: [B, 100, 2560]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '异步模式额外输入:\\n'
               '  • 像素目标RGB图像\\n'
               '  • 像素目标深度图',
               fillcolor='white')

        # RGBD Backbone
        c.node('RGBD_Backbone',
               '🔍 RGBD Encoder Backbone\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '基础模型: Depth-Anything-V2-Small\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'RGB分支:\\n'
               '  • Input: [B, T, H, W, 3]\\n'
               '  • ViT Encoder\\n'
               '  • Output: [B, T*256, 384]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Depth分支:\\n'
               '  • Input: [B, T, H, W, 1]\\n'
               '  • Replicate to 3 channels\\n'
               '  • ViT Encoder\\n'
               '  • Output: [B, T*256, 384]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Fusion:\\n'
               '  • Concat: [B, 2*T*256, 384]\\n'
               '  • + Learnable Pos Encoding\\n'
               '  • TransformerDecoder(2 layers)\\n'
               '  • Query: [B, memory_size*16, 384]\\n'
               '  • Project to token_dim=512\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Output: rgbd_embed [B, M*16, 512]',
               fillcolor='#FFE0B2', shape='component', penwidth='3')

        # Goal Encoders
        c.node('Goal_Encoders',
               '🎯 多模态目标编码器\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '1. Point Goal Encoder:\\n'
               '   Linear(3 -> 512)\\n'
               '   Input: [B, 3] (相对坐标)\\n'
               '   Output: [B, 1, 512]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '2. Image Goal Encoder:\\n'
               '   DepthAnything (6 channels)\\n'
               '   Input: [B, H, W, 6]\\n'
               '   Output: [B, 1, 512]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '3. Pixel Goal Encoder:\\n'
               '   DepthAnything (7 channels)\\n'
               '   Input: [B, H, W, 7]\\n'
               '   Output: [B, 1, 512]',
               fillcolor='#FFE0B2', shape='component')

        # Diffusion Process
        c.node('Diffusion_Process',
               '🌊 扩散去噪过程 (DDPM)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Scheduler: DDPMScheduler\\n'
               '  • num_train_timesteps: 10\\n'
               '  • beta_schedule: squaredcos_cap_v2\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '推理过程 (采样K=32条轨迹):\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '1. 初始化噪声:\\n'
               '   noisy_action ~ N(0,I)\\n'
               '   Shape: [K*B, predict_size, 3]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '2. 迭代去噪 (T=10 steps):\\n'
               '   for t in [9,8,...,1,0]:\\n'
               '     • Embed: action_embed = Linear(3->512)\\n'
               '     • Time Embed: time_emb = SinPosEmb(t)\\n'
               '     • Condition: [time, goal*3, rgbd]\\n'
               '     • Transformer Decode:\\n'
               '       - Input: action_embed + pos_embed\\n'
               '       - Memory: cond_embed + cond_pos_embed\\n'
               '       - Causal Mask (predict_size)\\n'
               '     • Predict noise_pred\\n'
               '     • Update: action = scheduler.step()\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出: denoised_actions [K*B, P, 3]',
               fillcolor='#E1BEE7', shape='box3d', penwidth='3')

        # Transformer Decoder
        c.node('Action_Decoder',
               '🎬 Action Transformer Decoder\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Architecture:\\n'
               '  • Layers: temporal_depth (6)\\n'
               '  • Hidden dim: 512\\n'
               '  • Heads: 8\\n'
               '  • FFN dim: 2048\\n'
               '  • Activation: GELU\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Input Embedding:\\n'
               '  action_embed + output_pos_embed\\n'
               'Memory (Condition):\\n'
               '  [time, goal, goal, goal, rgbd]\\n'
               '  + cond_pos_embed\\n'
               'Causal Mask: 上三角\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Output: [B, predict_size, 512]\\n'
               'Action Head: Linear(512 -> 3)',
               fillcolor='#FFE0B2')

        # Critic Network
        c.node('Critic_Network',
               '⚖️ Critic Network (轨迹评估)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输入: 预测的K条轨迹\\n'
               'Condition: [0, 0, 0, 0, rgbd]\\n'
               'Memory Mask: mask前4个goal位置\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Transformer Decoder\\n'
               '+ LayerNorm\\n'
               '+ Mean Pooling\\n'
               '+ Linear(512 -> 1)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Output: critic_values [K*B]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '轨迹选择:\\n'
               '  • Top-8 positive (最高分)\\n'
               '  • Top-8 negative (最低分)',
               fillcolor='#C5CAE9', shape='component')

        # Trajectory Generation
        c.node('Traj_Generation',
               '📈 轨迹生成与选择\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '1. 累积和生成轨迹:\\n'
               '   trajectory = cumsum(actions/4.0)\\n'
               '   Shape: [K*B, predict_size, 3]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '2. 根据Critic选择最佳轨迹\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '3. 转换为离散动作:\\n'
               '   • 连续模式: traj_to_actions()\\n'
               '   • 离散模式: chunk_token()\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Output: action_list (4-8步)',
               fillcolor=COLORS['tensor'], color=COLORS['tensor_line'])

    # ============================================================
    # 输出
    # ============================================================
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='📤 最终输出', style='filled',
               fillcolor=COLORS['output'], color=COLORS['output_line'],
               penwidth='3', fontsize='18')

        c.node('S2_Output',
               '🎯 S2输出\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '情况1: 像素目标\\n'
               '  • pixel_coord: [2]\\n'
               '  • traj_latents: [B, 100, 2560]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '情况2: 离散动作\\n'
               '  • actions: [↑,←,→,STOP]',
               fillcolor='white', shape='note')

        c.node('S1_Output',
               '🤖 S1输出\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '低级运动控制指令\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '同步模式: 4步动作\\n'
               '异步模式: 8步动作\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'action_indices: [0,1,2,3,5]\\n'
               '0=STOP, 1=↑, 2=←, 3=→, 5=↓',
               fillcolor='white', shape='note')

        c.node('Final_Action',
               '✅ 执行动作\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '发送给机器人执行\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '可能包含:\\n'
               '  • 线速度\\n'
               '  • 角速度\\n'
               '  • 停止信号',
               fillcolor=COLORS['output'], color=COLORS['output_line'],
               shape='doubleoctagon', penwidth='3')

    # ============================================================
    # Edges (数据流)
    # ============================================================
    # 输入到S2
    dot.edge('RGB_History', 'Processor', color=COLORS['input_line'])
    dot.edge('Instruction', 'Processor', color=COLORS['input_line'])
    dot.edge('Pose', 'Processor', style='dashed')

    dot.edge('Processor', 'Vision_Tower', label='pixel_values')
    dot.edge('Processor', 'Text_Embed', label='input_ids')

    dot.edge('Vision_Tower', 'QwenVL_Transformer', label='image_embeds')
    dot.edge('Text_Embed', 'QwenVL_Transformer', label='text_embeds')

    dot.edge('QwenVL_Transformer', 'LM_Head', label='hidden_states', penwidth='2')
    dot.edge('LM_Head', 'S2_Output', label='text output')

    dot.edge('QwenVL_Transformer', 'Latent_Queries',
             label='generate_latents()', style='dashed', color=COLORS['s2_vlm_line'])

    # S2 to S1
    dot.edge('Latent_Queries', 'NavDP_Input_Prep',
             label='traj_latents\n[B,100,2560]',
             penwidth='3', color=COLORS['s1_navdp_line'])
    dot.edge('RGB_History', 'NavDP_Input_Prep', style='dashed')
    dot.edge('Depth', 'NavDP_Input_Prep', style='dashed')

    # S1 processing
    dot.edge('NavDP_Input_Prep', 'RGBD_Backbone', label='RGB-D')
    dot.edge('NavDP_Input_Prep', 'Goal_Encoders', label='latent/goal', style='dashed')

    dot.edge('RGBD_Backbone', 'Diffusion_Process', label='rgbd_embed\n[B,M*16,512]')
    dot.edge('Goal_Encoders', 'Diffusion_Process', label='goal_embed\n[B,1,512]')

    dot.edge('Diffusion_Process', 'Action_Decoder',
             label='iterative\ndenoising', style='dashed')
    dot.edge('Action_Decoder', 'Diffusion_Process',
             label='noise_pred', style='dashed')

    dot.edge('Diffusion_Process', 'Critic_Network', label='K trajectories')
    dot.edge('Critic_Network', 'Traj_Generation', label='critic scores')
    dot.edge('Diffusion_Process', 'Traj_Generation', label='actions')

    # Output
    dot.edge('Traj_Generation', 'S1_Output', label='action_list')
    dot.edge('S1_Output', 'Final_Action', penwidth='3')
    dot.edge('S2_Output', 'Final_Action', label='if no latent', style='dashed')

    return dot


def create_training_flow():
    """生成NavDP训练流程图"""
    dot = Digraph('NavDP_Training_Flow', comment='NavDP训练流程详细版')

    dot.attr(rankdir='TB', size='24,32', dpi='300',
             nodesep='0.8', ranksep='1.0', bgcolor='white',
             fontname=FONT, fontsize='12')

    dot.attr('node', shape='box', style='rounded,filled', penwidth='2',
             fontname=FONT, fontsize='10')
    dot.attr('edge', fontname=FONT, fontsize='9', penwidth='1.5')

    # ============================================================
    # 数据加载
    # ============================================================
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='📦 数据加载流程', style='filled',
               fillcolor='#E8EAF6', color='#3F51B5', penwidth='3', fontsize='16')

        c.node('Dataset_Root',
               '📁 Dataset Root\\n'
               '━━━━━━━━━━━━━━━━━━\\n'
               'LeRobot格式数据集\\n'
               '目录结构:\\n'
               '  scene_dir/\\n'
               '    trajectory_dir/\\n'
               '      rgb/ (图像序列)\\n'
               '      depth/ (深度序列)\\n'
               '      data.json (轨迹数据)\\n'
               '      path.ply (路径点云)',
               fillcolor='white', shape='folder')

        c.node('NavDP_Dataset',
               '🔄 NavDP_Base_Dataset\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '参数:\\n'
               '  • memory_size: 8 (历史帧数)\\n'
               '  • predict_size: 24 (预测步数)\\n'
               '  • image_size: 224\\n'
               '  • batch_size: 64\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '加载并处理:\\n'
               '  • RGB图像归一化\\n'
               '  • 深度图裁剪 [0.1, 5.0]m\\n'
               '  • 路径点云解析\\n'
               '  • 动作轨迹插值',
               fillcolor='white')

        c.node('Data_Sample',
               '📊 单个样本\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '包含:\\n'
               '  • batch_rgb: [M, H, W, 3]\\n'
               '  • batch_depth: [M, H, W, 1]\\n'
               '  • batch_pg: [3] (point goal)\\n'
               '  • batch_ig: [H, W, 6] (img goal)\\n'
               '  • batch_tg: [H, W, 7] (pixel goal)\\n'
               '  • batch_labels: [P, 3] (动作)\\n'
               '  • batch_augments: [P, 3]\\n'
               '  • batch_label_critic: [1]\\n'
               '  • batch_augment_critic: [1]',
               fillcolor='white')

        c.node('DataLoader',
               '📤 DistributedDataLoader\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'DDP训练配置:\\n'
               '  • DistributedSampler\\n'
               '  • num_workers: 8\\n'
               '  • pin_memory: True\\n'
               '  • drop_last: True',
               fillcolor='white')

    # ============================================================
    # 模型初始化
    # ============================================================
    with dot.subgraph(name='cluster_init') as c:
        c.attr(label='🔧 模型初始化', style='filled',
               fillcolor='#FFF9E6', color='#FF6F00', penwidth='3', fontsize='16')

        c.node('Init_Model',
               '🏗️ NavDPNet初始化\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '1. RGBD Encoder (DepthAnything)\\n'
               '2. Goal Encoders (3种)\\n'
               '3. Transformer Decoder (6层)\\n'
               '4. Action Head & Critic Head\\n'
               '5. Noise Scheduler (DDPM)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '参数量: ~100M',
               fillcolor='white')

        c.node('DDP_Wrap',
               '🔗 DDP包装\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'DistributedDataParallel\\n'
               '  find_unused_parameters=True\\n'
               '  gradient_as_bucket_view=True',
               fillcolor='white')

        c.node('Optimizer',
               '⚡ Optimizer & Scheduler\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Adam Optimizer:\\n'
               '  • lr: 1e-4\\n'
               '  • weight_decay: 0\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'LinearLR Scheduler:\\n'
               '  • start_factor: 1.0\\n'
               '  • end_factor: 0.5\\n'
               '  • total_iters: 10000',
               fillcolor='white')

    # ============================================================
    # 训练循环
    # ============================================================
    with dot.subgraph(name='cluster_train') as c:
        c.attr(label='🔁 训练循环 (每个Epoch)', style='filled',
               fillcolor='#FCE4EC', color='#C2185B', penwidth='3', fontsize='16')

        c.node('Forward_Pass',
               '➡️ Forward Pass\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '1. RGBD Encoding:\\n'
               '   rgbd_embed = RGBD_Backbone(rgb, depth)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '2. Goal Encoding:\\n'
               '   pg_embed = PointEncoder(point_goal)\\n'
               '   ig_embed = ImageEncoder(image_goal)\\n'
               '   tg_embed = PixelEncoder(pixel_goal)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '3. 添加噪声到GT动作:\\n'
               '   t ~ Uniform(0, T)\\n'
               '   ε ~ N(0,I)\\n'
               '   noisy_action = √(ᾱₜ)·action + √(1-ᾱₜ)·ε\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '4. 噪声预测 (2个分支):\\n'
               '   • No-Goal: pred_ng\\n'
               '   • Multi-Goal: pred_mg (27种组合)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '5. Critic预测:\\n'
               '   • Label trajectory: critic_label\\n'
               '   • Augment trajectory: critic_augment',
               fillcolor='#BBDEFB', shape='component', penwidth='2')

        c.node('Loss_Computation',
               '📉 损失计算\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '1. Action Loss (扩散损失):\\n'
               '   ng_loss = MSE(pred_ng, noise_ng)\\n'
               '   mg_loss = MSE(pred_mg, noise_mg)\\n'
               '   action_loss = 0.5·ng + 0.5·mg\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '2. Critic Loss (质量评估):\\n'
               '   cr_loss = MSE(critic_pred, critic_gt)\\n'
               '           + MSE(aug_pred, aug_gt)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '3. Auxiliary Loss (辅助损失):\\n'
               '   aux_loss = 0.5·MSE(pg, ig_pred)\\n'
               '            + 0.5·MSE(pg, tg_pred)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '总损失:\\n'
               '   Loss = 0.8·action + 0.2·critic + 0.5·aux',
               fillcolor='#F8BBD0', penwidth='2')

        c.node('Backward',
               '⬅️ Backward & Update\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '1. loss.backward()\\n'
               '2. DDP梯度同步\\n'
               '3. Gradient Clipping (可选)\\n'
               '4. optimizer.step()\\n'
               '5. scheduler.step()\\n'
               '6. optimizer.zero_grad()',
               fillcolor='#C5CAE9')

        c.node('Logging',
               '📊 日志记录\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '记录指标:\\n'
               '  • Total Loss\\n'
               '  • Action Loss (ng & mg)\\n'
               '  • Critic Loss\\n'
               '  • Auxiliary Loss\\n'
               '  • Learning Rate\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '输出到TensorBoard',
               fillcolor='white')

    # ============================================================
    # 保存与评估
    # ============================================================
    with dot.subgraph(name='cluster_save') as c:
        c.attr(label='💾 保存与评估', style='filled',
               fillcolor='#C8E6C9', color='#388E3C', penwidth='3', fontsize='16')

        c.node('Checkpoint',
               '💾 Checkpoint保存\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '保存频率: 每N步\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '保存内容:\\n'
               '  • model.state_dict()\\n'
               '  • optimizer.state_dict()\\n'
               '  • scheduler.state_dict()\\n'
               '  • epoch, step\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '保存路径:\\n'
               '  output_dir/navdp.ckpt',
               fillcolor='white', shape='folder')

        c.node('Evaluation',
               '📈 模型评估\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '在Habitat仿真环境中:\\n'
               '  • 点目标导航成功率\\n'
               '  • SPL (Success weighted by Path Length)\\n'
               '  • Collision Rate\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '在VLN任务中:\\n'
               '  • Navigation Error (NE)\\n'
               '  • Oracle Success (OS)\\n'
               '  • Success Rate (SR)',
               fillcolor='white')

    # Edges
    dot.edge('Dataset_Root', 'NavDP_Dataset')
    dot.edge('NavDP_Dataset', 'Data_Sample', label='__getitem__')
    dot.edge('Data_Sample', 'DataLoader', label='collate')
    dot.edge('DataLoader', 'Forward_Pass', label='batch')

    dot.edge('Init_Model', 'DDP_Wrap')
    dot.edge('DDP_Wrap', 'Optimizer')
    dot.edge('Optimizer', 'Forward_Pass', style='dashed')

    dot.edge('Forward_Pass', 'Loss_Computation')
    dot.edge('Loss_Computation', 'Backward')
    dot.edge('Backward', 'Logging')
    dot.edge('Logging', 'Forward_Pass', label='Next Batch', style='dashed')

    dot.edge('Backward', 'Checkpoint', label='每N步', style='dotted')
    dot.edge('Checkpoint', 'Evaluation', label='定期评估', style='dotted')
    dot.edge('Evaluation', 'Forward_Pass', label='继续训练', style='dashed')

    return dot


def create_inference_flow():
    """生成InternVLA-N1推理流程图"""
    dot = Digraph('InternVLA_N1_Inference_Flow', comment='InternVLA-N1推理流程（双系统异步协作）')

    dot.attr(rankdir='TB', size='28,40', dpi='300',
             nodesep='0.8', ranksep='1.2', bgcolor='white',
             fontname=FONT, fontsize='12')

    dot.attr('node', shape='box', style='rounded,filled', penwidth='2',
             fontname=FONT, fontsize='10')
    dot.attr('edge', fontname=FONT, fontsize='9', penwidth='1.5')

    # ============================================================
    # 初始化
    # ============================================================
    with dot.subgraph(name='cluster_init') as c:
        c.attr(label='🚀 系统初始化', style='filled',
               fillcolor='#E8EAF6', color='#3F51B5', penwidth='3', fontsize='16')

        c.node('Agent_Init',
               '🤖 InternVLAN1Agent初始化\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '1. 加载InternVLA-N1模型\\n'
               '   • S2: Qwen2.5-VL-7B\\n'
               '   • S1: NavDP\\n'
               '2. 设置推理模式:\\n'
               '   • sync (同步)\\n'
               '   • async (异步)\\n'
               '3. 相机参数初始化\\n'
               '4. 创建线程锁',
               fillcolor='white')

        c.node('Thread_Start',
               '🧵 启动S2推理线程\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'def s2_thread_func():\\n'
               '  while True:\\n'
               '    if s2_input.should_infer:\\n'
               '      执行S2推理\\n'
               '      更新s2_output\\n'
               '    else:\\n'
               '      sleep(0.5)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '后台持续运行',
               fillcolor='#E1BEE7')

    # ============================================================
    # 主循环 - Episode开始
    # ============================================================
    with dot.subgraph(name='cluster_episode') as c:
        c.attr(label='🔄 Episode主循环', style='filled',
               fillcolor='#FCE4EC', color='#C2185B', penwidth='3', fontsize='16')

        c.node('Episode_Start',
               '🎬 Episode开始\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '接收任务:\\n'
               '  • instruction: 导航指令\\n'
               '  • start_pose: 初始位置\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'agent.reset()\\n'
               '  • 清空历史\\n'
               '  • 重置状态\\n'
               '  • episode_step = 0',
               fillcolor='white', shape='hexagon', penwidth='3')

        c.node('Obs_Capture',
               '📸 获取观测\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'obs = env.get_observation()\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '包含:\\n'
               '  • rgb: [H, W, 3]\\n'
               '  • depth: [H, W, 1]\\n'
               '  • pose: [7] (xyz+quat)\\n'
               '  • instruction: str',
               fillcolor='white')

    # ============================================================
    # S2推理分支（异步线程）
    # ============================================================
    with dot.subgraph(name='cluster_s2') as c:
        c.attr(label='🧠 S2推理分支（异步线程）', style='filled',
               fillcolor='#FFF3E0', color='#F57C00', penwidth='3', fontsize='16')

        c.node('S2_Trigger',
               '🔔 触发S2推理\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '条件检查:\\n'
               '  1. dual_forward_step==0 OR\\n'
               '  2. dual_forward_step>=sys2_max_step\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'if 触发:\\n'
               '  with s2_input_lock:\\n'
               '    s2_input.rgb = rgb\\n'
               '    s2_input.depth = depth\\n'
               '    s2_input.instruction = inst\\n'
               '    s2_input.should_infer = True',
               fillcolor='white', shape='diamond')

        c.node('S2_Infer',
               '🔮 S2推理 (Qwen2.5-VL)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Step 1: 图像历史准备\\n'
               '  • 当前帧 + num_history帧\\n'
               '  • 均匀采样历史\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Step 2: 构建Prompt\\n'
               '  "你是导航助手...\\n'
               '   任务: <instruction>\\n'
               '   历史观测: <images>\\n'
               '   当前观测: <image>"\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Step 3: VLM推理\\n'
               '  output_ids = model.generate()\\n'
               '  llm_output = decode(output_ids)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Step 4: 解析输出\\n'
               '  if 包含数字:\\n'
               '    → 像素坐标 "(x,y)"\\n'
               '    → 生成latent特征\\n'
               '  else:\\n'
               '    → 离散动作 "↑←→STOP"',
               fillcolor='#FFCCBC', shape='component', penwidth='3')

        c.node('Latent_Gen',
               '✨ 生成Latent特征\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'generate_latents():\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '1. 添加TRAJ_START_TOKEN\\n'
               '2. 插入100个TRAJ_TOKEN\\n'
               '3. 通过Transformer\\n'
               '4. 提取最后100个token\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Output: [1, 100, 2560]',
               fillcolor='#FFCCBC')

        c.node('S2_Output_Update',
               '📝 更新S2输出\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'with s2_output_lock:\\n'
               '  s2_output.output_pixel = pixel\\n'
               '  s2_output.output_action = actions\\n'
               '  s2_output.output_latent = latent\\n'
               '  s2_output.idx = current_step',
               fillcolor='white')

    # ============================================================
    # S1推理分支（主线程）
    # ============================================================
    with dot.subgraph(name='cluster_s1') as c:
        c.attr(label='🤖 S1推理分支（主线程）', style='filled',
               fillcolor='#E8F5E9', color='#388E3C', penwidth='3', fontsize='16')

        c.node('Check_S2_Output',
               '🔍 检查S2输出\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'with s2_output_lock:\\n'
               '  if s2_output.output_latent:\\n'
               '    使用S1\\n'
               '  elif s2_output.output_action:\\n'
               '    直接返回S2动作\\n'
               '  else:\\n'
               '    等待S2完成',
               fillcolor='white', shape='diamond')

        c.node('S1_Prepare',
               '📦 S1输入准备\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '获取当前观测:\\n'
               '  • rgb (用于RGBD编码)\\n'
               '  • depth\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '获取S2 latent:\\n'
               '  traj_latents = s2_output.latent\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '异步模式额外输入:\\n'
               '  • pixel_goal_rgb\\n'
               '  • pixel_goal_depth',
               fillcolor='white')

        c.node('S1_Infer',
               '🎯 S1推理 (NavDP)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'generate_traj(traj_latents, rgb, depth)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Step 1: RGBD编码\\n'
               '  rgbd_embed = RGBD_Backbone(rgb, depth)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Step 2: 扩散去噪 (采样32条轨迹)\\n'
               '  noisy ~ N(0,I) [32, 24, 3]\\n'
               '  for t in [9..0]:\\n'
               '    noise_pred = Decoder(\\n'
               '      noisy, t, latent, rgbd)\\n'
               '    noisy = scheduler.step()\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Step 3: Critic评分\\n'
               '  critic_scores = Critic(trajs)\\n'
               '  best_traj = trajs[argmax(scores)]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'Step 4: 转换为动作序列\\n'
               '  actions = traj_to_actions(best_traj)\\n'
               '  返回前4-8步',
               fillcolor='#C8E6C9', shape='component', penwidth='3')

        c.node('Action_Select',
               '🎮 动作选择\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'action_list = s1_output.idx\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '取第一个动作执行:\\n'
               'action = action_list[0]\\n'
               'action_list = action_list[1:]\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '动作映射:\\n'
               '  0: STOP\\n'
               '  1: MOVE_FORWARD\\n'
               '  2: TURN_LEFT\\n'
               '  3: TURN_RIGHT\\n'
               '  5: LOOK_DOWN',
               fillcolor='white')

    # ============================================================
    # 执行与循环
    # ============================================================
    with dot.subgraph(name='cluster_exec') as c:
        c.attr(label='⚡ 动作执行与状态更新', style='filled',
               fillcolor='#E0F7FA', color='#00ACC1', penwidth='3', fontsize='16')

        c.node('Execute_Action',
               '🏃 执行动作\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'obs = env.step(action)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '机器人执行物理动作\\n'
               '获取新的观测',
               fillcolor='white', shape='doubleoctagon', penwidth='3')

        c.node('Update_State',
               '🔄 状态更新\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'episode_step += 1\\n'
               'dual_forward_step += 1\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '更新历史:\\n'
               '  rgb_list.append(rgb)\\n'
               '  depth_list.append(depth)\\n'
               '  pose_list.append(pose)',
               fillcolor='white')

        c.node('Check_Done',
               '✅ 检查终止条件\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               '满足以下任一条件:\\n'
               '  1. action == STOP\\n'
               '  2. 到达目标位置\\n'
               '  3. 超过最大步数\\n'
               '  4. 碰撞次数过多\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'if done:\\n'
               '  计算指标\\n'
               '  保存日志\\n'
               'else:\\n'
               '  继续循环',
               fillcolor='white', shape='diamond')

    # ============================================================
    # Episode结束
    # ============================================================
    with dot.subgraph(name='cluster_end') as c:
        c.attr(label='🏁 Episode结束', style='filled',
               fillcolor='#C8E6C9', color='#388E3C', penwidth='3', fontsize='16')

        c.node('Metrics',
               '📊 计算评估指标\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'VLN-CE指标:\\n'
               '  • Navigation Error (NE)\\n'
               '  • Success Rate (SR)\\n'
               '  • Oracle Success (OS)\\n'
               '  • SPL (Success weighted by PL)\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'VLN-PE指标:\\n'
               '  • Path Length\\n'
               '  • Steps\\n'
               '  • Collision Count',
               fillcolor='white')

        c.node('Next_Episode',
               '🔁 下一个Episode\\n'
               '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n'
               'if 还有任务:\\n'
               '  agent.reset()\\n'
               '  开始新Episode\\n'
               'else:\\n'
               '  汇总统计\\n'
               '  保存结果\\n'
               '  结束评估',
               fillcolor='white', shape='hexagon')

    # ============================================================
    # Edges（数据流）
    # ============================================================
    # 初始化流程
    dot.edge('Agent_Init', 'Thread_Start', penwidth='2')
    dot.edge('Thread_Start', 'Episode_Start', style='dashed', label='后台运行')

    # Episode主流程
    dot.edge('Episode_Start', 'Obs_Capture', penwidth='2')
    dot.edge('Obs_Capture', 'S2_Trigger', penwidth='2')

    # S2分支（异步）
    dot.edge('S2_Trigger', 'S2_Infer', label='if 触发', color=COLORS['s2_vlm_line'])
    dot.edge('S2_Infer', 'Latent_Gen', label='if 像素目标', color=COLORS['s2_vlm_line'])
    dot.edge('Latent_Gen', 'S2_Output_Update', color=COLORS['s2_vlm_line'])
    dot.edge('S2_Infer', 'S2_Output_Update', label='if 离散动作',
             style='dashed', color=COLORS['s2_vlm_line'])

    # S1分支（主线程）
    dot.edge('S2_Trigger', 'Check_S2_Output', label='主线程继续')
    dot.edge('S2_Output_Update', 'Check_S2_Output',
             label='异步更新', style='dotted', color=COLORS['s2_vlm_line'])

    dot.edge('Check_S2_Output', 'S1_Prepare', label='有latent',
             color=COLORS['s1_navdp_line'])
    dot.edge('S1_Prepare', 'S1_Infer', color=COLORS['s1_navdp_line'], penwidth='2')
    dot.edge('S1_Infer', 'Action_Select', color=COLORS['s1_navdp_line'])

    dot.edge('Check_S2_Output', 'Action_Select',
             label='有S2动作', style='dashed')

    # 执行流程
    dot.edge('Action_Select', 'Execute_Action', penwidth='3')
    dot.edge('Execute_Action', 'Update_State', penwidth='2')
    dot.edge('Update_State', 'Check_Done', penwidth='2')

    # 循环or结束
    dot.edge('Check_Done', 'Obs_Capture',
             label='继续', style='dashed', color='#666')
    dot.edge('Check_Done', 'Metrics', label='done', penwidth='2')
    dot.edge('Metrics', 'Next_Episode')
    dot.edge('Next_Episode', 'Episode_Start',
             label='下一个Episode', style='dotted')

    return dot


def main():
    output_dir = './docs'
    os.makedirs(output_dir, exist_ok=True)
    print("="*60)
    print("开始生成InternNav项目架构图...")
    print("="*60)

    try:
        # 1. 生成模型架构图
        print("\n📐 生成模型架构图...")
        model_dot = create_internvlan1_model_architecture()
        model_path = f'{output_dir}/internvla_n1_model_architecture'
        model_dot.render(model_path, format='png', cleanup=True)
        print(f"✅ 模型架构图: {model_path}.png")

        # 2. 生成训练流程图
        print("\n📚 生成训练流程图...")
        train_dot = create_training_flow()
        train_path = f'{output_dir}/navdp_training_flow'
        train_dot.render(train_path, format='png', cleanup=True)
        print(f"✅ 训练流程图: {train_path}.png")

        # 3. 生成推理流程图
        print("\n🚀 生成推理流程图...")
        infer_dot = create_inference_flow()
        infer_path = f'{output_dir}/internvla_n1_inference_flow'
        infer_dot.render(infer_path, format='png', cleanup=True)
        print(f"✅ 推理流程图: {infer_path}.png")

        print("\n" + "="*60)
        print("🎉 所有架构图生成完成!")
        print("="*60)
        print(f"\n📁 输出目录: {os.path.abspath(output_dir)}/")
        print("\n生成的文件:")
        print(f"  1. internvla_n1_model_architecture.png - InternVLA-N1详细模型架构")
        print(f"  2. navdp_training_flow.png - NavDP训练流程")
        print(f"  3. internvla_n1_inference_flow.png - 双系统异步推理流程")
        print("\n" + "="*60)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
