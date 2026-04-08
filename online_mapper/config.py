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
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_device: str = "cuda:0"
    enable_depth: bool = True

    # semantics
    enable_grounding_dino: bool = True
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-base"
    enable_qwen_naming: bool = True   # 默认开 (vLLM 应已启动)
    qwen_base_url: str = "http://localhost:8199/v1"
    qwen_gpu: str = "1"

    # door plate retrospective
    enable_door_plate_detection: bool = True
    door_plate_min_score: float = 0.30

    # connection builder (real next_positions)
    enable_real_connections: bool = True
    connection_sim_threshold: float = 0.40

    # visual odometry
    enable_real_vo: bool = True

    # occupancy
    grid_resolution: float = 0.2
    grid_size: int = 200

    # 起始 ID
    start_id: int = 1

    # 最小帧间隔
    min_keyframe_frame_interval: int = 3
