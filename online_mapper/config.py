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
    plate_scan_every_n_frames: int = 1

    # connection builder (real next_positions)
    enable_real_connections: bool = True
    connection_sim_threshold: float = 0.40

    # visual odometry
    enable_real_vo: bool = True
    # vo backend: 仅支持 "vggt" (ORB MonoVO 已删除); 其他值 → 常速代理 (0.5,0.02)
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

    # ==== finalize canonical merge ====
    # Pass D/E brand-attach / SHOP-attach 帧距阈值
    plate_attach_max_frame_gap: int = 12
    # _rebuild_topology_neighbors_spatial 跨 segment 断档阈值 (ms)
    topology_cross_gap_ms: int = 60_000

    # ==== ConnectionBuilder geometric prior (connection_builder.py) ====
    # 跨 segment 真断档时 geo_bonus 清零 (ms)
    cb_gap_fuse_ms: int = 300_000
    # 同 segment 内几何先验权重 (sim += alpha * cos)
    cb_alpha_normal: float = 0.5
    # person 占据中心 >15% 时 cam 行 sim 惩罚
    cb_person_penalty: float = 0.30
    # |cx/w - 0.5| <= 此值 → Qwen 居中 fallback, 触发几何投影替换
    cb_fallback_center_tol: float = 0.03

    # ==== AutoSubImageExtractor corridor matching ====
    cb_max_corridor_frames: int = 30
    cb_corridor_sample_count: int = 3

    # ==== JunctionDetector ====
    junction_open_depth_thresh: float = 1.8
