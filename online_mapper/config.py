"""online_mapper 全局配置"""
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class OnlineMapperConfig:
    # IO
    input_dir: str = "memory_test_data"
    output_dir: str = "online_mapper/output/merged_labeled_data"
    vpr_config_path: str = "deploy/vpr_config.yaml"

    # 关键帧多触发条件
    vpr_dissim_threshold: float = 0.50
    accumulated_translation: float = 1.5
    accumulated_rotation: float = 0.6
    info_gain_threshold: float = 0.05

    # 闭环
    loop_closure_min_gap: int = 8
    loop_closure_vpr_threshold: float = 0.78  # 上限 (auto-tune 取 min)
    loop_closure_top_k: int = 5
    loop_closure_geom_verify: bool = True
    loop_closure_min_inliers: int = 15

    # frontier / NBV
    nbv_info_weight: float = 1.0
    nbv_cost_weight: float = 0.3
    nbv_semantic_weight: float = 0.5

    # depth
    depth_model_id: str = "pretrained/depth-anything-v2-small-hf"
    depth_device: str = "cuda:0"
    enable_depth: bool = True
    # depth backend: "da_v2" (Depth-Anything-V2-Small) | "vggt" (VGGT-1B, metric)
    depth_backend: str = "vggt"
    vggt_model_path: str = "pretrained/vggt-1b/model.pt"
    vggt_window_size: int = 4
    vggt_dtype: str = "bf16"

    # semantics
    enable_grounding_dino: bool = True
    grounding_dino_model_id: str = "pretrained/grounding-dino-base"
    enable_qwen_naming: bool = True   # 默认开 (vLLM 应已启动)
    qwen_base_url: str = "http://localhost:8199/v1"
    qwen_gpu: str = "1"

    # door plate retrospective
    enable_door_plate_detection: bool = True
    door_plate_min_score: float = 0.30
    # 门牌扫描节流: 非关键帧每 N 帧扫一次 (1=每帧默认, 越大越省延迟但会漏门牌)
    # 漏门牌的后果: 投票票数不够 (MultiFrameVoter min_frames=2), 对应 landmark
    # 节点建不出, 如 '关爱室' 可能因此缺失.
    plate_scan_every_n_frames: int = 3

    # connection builder (real next_positions)
    enable_real_connections: bool = True
    connection_sim_threshold: float = 0.40
    # target 方向打点器: "qwen" (Qwen35PointGrounder) 或 "gdino"
    # (Grounding-DINO, 对开放场景整体失锚更稳定).
    pointer_backend: str = "qwen"

    # visual odometry
    enable_real_vo: bool = True
    # vo backend: "orb" (MonoVO) | "vggt" (复用 VGGTDepthEstimator 位姿, 零额外推理)
    vo_backend: str = "vggt"

    # occupancy
    grid_resolution: float = 0.2
    # occ backend: "depth_row" (1D 射线) | "vggt" (dense 点云直填)
    occ_backend: str = "vggt"
    grid_size: int = 200

    # 起始 ID
    start_id: int = 1

    # 最小帧间隔
    min_keyframe_frame_interval: int = 3
