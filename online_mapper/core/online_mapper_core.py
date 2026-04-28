"""OnlineMapperCore — 主流式建图循环 (v2)

主要改进:
- 真实 VO (ORB + EssentialMatrix + 深度 scale) 替换常速代理
- Qwen 语义命名 (use_qwen=True), 接 vLLM 8199
- DoorPlateTracker 跨帧累积门牌检测, 选 bbox 最大帧作代表
- ConnectionBuilder (AutoSubImageExtractor 子类) 生成真实
  next_positions, sim>=0.40 阈值过滤
- LoopCloser auto-tune threshold + 几何验证
"""
import os, sys, time, json, logging, cv2, base64, numpy as np, math
from pathlib import Path
from typing import Dict, List

# 动态项目根 (online_mapper/core/online_mapper_core.py → 上 2 级 = MemoryNav/).
# 替换原硬编码 '/home/ubuntu/Disk/codes/jianxiong/MemoryNav' 以支持任意部署路径.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from online_mapper.config import OnlineMapperConfig
from online_mapper.core.stream_loader import StreamLoader
from online_mapper.geometry.depth_estimator import DepthEstimator, build_depth_estimator
from online_mapper.geometry.pose_graph import PoseGraph
from online_mapper.geometry.occupancy import OccupancyGrid
from online_mapper.geometry.visual_odometry import build_visual_odometry
from online_mapper.topology.graph import TopoGraph, TopoNode
from online_mapper.topology.keyframe_selector import KeyframeSelector
from online_mapper.topology.loop_closure import LoopCloser
from online_mapper.topology.frontier_nbv import FrontierNBV
from online_mapper.semantics.open_set_detector import OpenSetDetector
from online_mapper.semantics.scene_graph import SceneGraph, SceneObject
from online_mapper.semantics.door_plate_tracker import DoorPlateTracker, PlateObservation
from online_mapper.semantics.node_naming import (
    NodeName, merge_names, is_brand_like, select_organization,
    resolve_global_uniqueness,
)
from online_mapper.semantics.hallucination_filter import (
    QwenVerifier, MultiFrameVoter, NameVote, NameDeduplicator,
    STRICT_DETECT_TEXT_PROMPT,
)
from online_mapper.semantics.node_category import (
    NodeCategoryClassifier, NodeCategory, JunctionKind, CategoryDecision,
    FUNCTION_AREA_WHITELIST, LANDMARK_FACILITY_WHITELIST,
    BUILDING_LANDMARK_PATTERNS,
)
from online_mapper.semantics.colocation_merger import ColocationMerger
from online_mapper.geometry.junction_detector import JunctionDetector
from online_mapper.semantics import semantic_dedup
from online_mapper.io.merged_data_writer import MergedDataWriter

logger = logging.getLogger(__name__)

CAM_IDS = ['camera_1', 'camera_2', 'camera_3', 'camera_4']

# ---------------------------------------------------------------------------
# Canonical-merge regex / constants (used by _merge_by_canonical_name passes)
# ---------------------------------------------------------------------------
import re as _re_canon
# Pass A 楼栋 base 提取: 'A座入口' / 'A座大堂' / 'A座' → 'A座'
_BUILDING_BASE_RE = _re_canon.compile(r"^([A-Za-z0-9]+(?:号)?(?:座|楼|栋))(?:入口|大堂|电梯)?$")
# Pass B 复合 plate (X座电梯/X号楼楼梯/X层电梯)
_COMPOSITE_PREFIX_RE = _re_canon.compile(
    r"^([A-Za-z]|[0-9]{1,3}号|[0-9]{1,3}层|[0-9]{1,3})(座|楼|栋)?(电梯|楼梯)$")
# Pass C 数字柜 (1号柜)
_CABINET_RE = _re_canon.compile(r"^([0-9]{1,3})号柜$")
# Pass F final recenter: 提取楼栋 base
_BLD_BASE_RE = _re_canon.compile(r"^([A-Za-z0-9]+(?:号)?(?:座|楼|栋))")

# Pass A: 同 canonical 但物理位置远的 node 不应合并 (例: 起点电梯厅 vs H 座电梯厅)
_SPATIAL_MERGE_DIST_M = 3.0
# Pass D: 保安亭/岗亭 scene_describe 文本集合
_BOOTH_TOKENS = {"岗亭", "保安亭", "值班亭", "门岗"}
# Pass C: 柜区分类常量
_CABINET_AREA_TO_KIND = {
    "外卖柜区": "外卖柜", "储物柜区": "储物柜", "快递柜区": "快递柜",
}
_CABINET_KIND_PLATES = ("外卖柜", "储物柜", "快递柜")
_CABINET_KIND_TO_SHORT = {"外卖柜": "外卖柜", "储物柜": "储物柜", "快递柜": "快递柜"}
# Pass C: 明显近视证据阈值 (plate bbox area)
_CABINET_NEAR_VIEW_AREA = 10000

# 各 pass 帧距阈值 (合并/匹配时允许的 frame_idx gap)
_PASS_B_FRAME_GAP = 12
_PASS_C_FRAME_GAP = 16
_PASS_D_BUILDING_GAP = 12   # case B: 已有保安亭节点 找最近 building 的 gap
_PASS_D_SCENE_GAP = 10      # case A: scene 提示岗亭 找最近 building 的 gap
_PASS_E_FRAME_GAP = 12      # SHOP brand attach 帧距
_PASS_C_NEAR_WINDOW = 12    # 柜 plate 周围 area 加权窗口


class OnlineMapperCore:
    def __init__(self, cfg: OnlineMapperConfig, shared_vpr_extractor=None):
        """
        Args:
            shared_vpr_extractor: 可选的复用 VPR extractor (如 MemoryNavigator.extractor).
                                   传入后 NodeDistanceEstimator 不再新建模型, 节省显存.
        """
        self.cfg = cfg
        self.loader = StreamLoader(cfg.input_dir)

        # VPR
        from online_mapper.vpr.node_distance_estimator import NodeDistanceEstimator
        self.vpr = NodeDistanceEstimator(
            cfg.vpr_config_path,
            similarity_threshold=cfg.vpr_dissim_threshold,
            min_frame_interval=cfg.min_keyframe_frame_interval,
            extractor=shared_vpr_extractor,
        )

        # geometry
        self.depth = build_depth_estimator(cfg) if cfg.enable_depth else None
        self.vo = build_visual_odometry(cfg, self.depth) if cfg.enable_real_vo else None
        self.pose_graph = PoseGraph()
        self.occ = OccupancyGrid(cfg.grid_size, cfg.grid_resolution)

        # topology
        self.topo = TopoGraph()
        self.kf_selector = KeyframeSelector(cfg)
        self.loop_closer = LoopCloser(cfg, self.vpr)
        self.nbv = FrontierNBV(cfg)

        # semantics
        self.detector = OpenSetDetector(cfg)
        self.scene_graph = SceneGraph()
        self.door_tracker = DoorPlateTracker() if cfg.enable_door_plate_detection else None
        self.plate_voter = MultiFrameVoter(min_frames=2, min_cameras=2,
                                           allow_single_frame_whitelist=True)
        self.verifier: QwenVerifier = None  # 延迟到 namer 就绪后创建
        self.category_clf = NodeCategoryClassifier()
        self.junction_detector = (JunctionDetector(self.depth,
                                                    open_depth_thresh=cfg.junction_open_depth_thresh)
                                   if self.depth else None)
        # 跟踪 frame_idx -> 该帧产生的 confirmed plate 名称(集合)
        self._frame_plate_hits: Dict[int, set] = {}

        # naming
        self.namer = None
        if cfg.enable_qwen_naming:
            try:
                from online_mapper.semantics.auto_landmark_namer import AutoLandmarkNamer
                self.namer = AutoLandmarkNamer(use_qwen=True, gpu=cfg.qwen_gpu)
                if not (self.namer._qwen_server and self.namer._qwen_server.is_ready):
                    logger.warning("Qwen namer not ready; falling back to placeholder names")
                    self.namer = None
                else:
                    logger.info("Qwen namer ready (vLLM 8199)")
                    self.verifier = QwenVerifier(self.namer._qwen_server)
            except Exception as e:
                logger.warning(f"Qwen namer init failed: {e}")
                self.namer = None

        # writer
        self.writer = MergedDataWriter(cfg.output_dir)

        # state
        self.next_node_id = cfg.start_id
        self.last_kf_features = None
        self.last_kf_node_id = None
        self.node_features: Dict[str, Dict[str, np.ndarray]] = {}
        self.node_frame_idx: Dict[str, int] = {}
        self.node_payloads: Dict[str, Dict] = {}
        self.robot_x = self.robot_y = self.robot_theta = 0.0

        # cached frames for ConnectionBuilder
        self._all_frames_cache: List[Dict] = []
        from collections import deque as _deque
        self._recent_scene_winners: _deque = _deque(maxlen=3)

        # logs
        self.log_lines: List[Dict] = []
        self.metrics = {
            "n_nodes": 0, "n_edges": 0, "n_loop_closures": 0,
            "n_semantic_merges": 0, "n_frames": 0,
            "n_keyframes_triggered": 0, "n_door_plates": 0,
            "n_connections": 0, "n_named_landmarks": 0,
            "vo_mode": "real" if cfg.enable_real_vo else "constant",
            "runtime_s": {"depth": 0.0, "vpr": 0.0, "detect": 0.0, "vo": 0.0, "name": 0.0, "total": 0.0},
        }

    # ------------------------------------------------------------------
    def _extract_features(self, cameras):
        return self.vpr.extract_camera_features(cameras)

    def _vpr_sim(self, fa, fb):
        return semantic_dedup.cyclic_cosine(fa, fb)

    def _vo_motion(self, front_img, depth_map):
        if self.vo is not None:
            return self.vo.estimate(front_img, depth_map)
        return 0.5, 0.02

    def _b64(self, img):
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('utf-8')

    # ------------------------------------------------------------------
    def run(self):
        """离线一次性跑完整个 StreamLoader, 跑完自动 finalize."""
        self._run_start_ts = time.time()
        for frame in self.loader:
            self.process_frame(frame)
        self.finalize()

    # ------------------------------------------------------------------
    def process_frame(self, frame: Dict) -> Dict:
        """处理单帧 (供 ws_proxy 在线建图模式调用).

        编排器: 各 _stage_* 顺序固定, 副作用顺序与原 v2 行为完全等价.

        Args:
            frame: dict with keys {timestamp, frame_idx, cameras: {camera_1..4: path}}

        Returns:
            log_entry dict (与写入 online_mapping_log.jsonl 的一致)
        """
        if not hasattr(self, "_run_start_ts"):
            self._run_start_ts = time.time()
        self.metrics["n_frames"] += 1
        fidx = frame["frame_idx"]
        self._all_frames_cache.append(frame)

        # Layer 1 几何: depth + VO + 累积位姿 + occupancy + KF 累计量
        dtrans, drot, info_gain = self._stage_motion_and_occupancy(frame)

        # Layer 2 拓扑: VPR 特征 + sim_to_last + 主动闭环 (每帧, 不依赖 KF)
        feats, sim_to_last = self._stage_vpr_features(frame)
        self._stage_loop_closure(feats, fidx, frame)

        # KF 触发判定 (首帧强制 trigger)
        triggered, reason = self.kf_selector.should_trigger(fidx, sim_to_last)
        if self.last_kf_features is None:
            triggered, reason = True, "first_frame"

        log_entry = {
            "frame_idx": fidx, "ts": frame["timestamp"],
            "vpr_sim_to_last": sim_to_last,
            "info_gain": info_gain,
            "vo_dtrans": dtrans, "vo_drot": drot,
            "robot_pose": [self.robot_x, self.robot_y, self.robot_theta],
            "occupancy": self.occ.stats(),
            "keyframe": triggered, "reason": reason,
        }

        # 门牌扫描 (节流, KF 必扫). 注意: 必须在 KF 主体之前, 后者读 _frame_plate_hits.
        self._stage_plate_scan_throttled(frame, fidx, triggered)

        if triggered:
            # KF 主体: 4-cam GD detect → semantic merge or (junction → scene_naming
            # → category → 创建 node). REJECT 路径会自负责 append+return.
            early_return = self._stage_create_keyframe(frame, fidx, feats, log_entry)
            if early_return:
                return log_entry
            self.kf_selector.reset(fidx)

        self.log_lines.append(log_entry)
        return log_entry

    # ------------------------------------------------------------------
    def _stage_motion_and_occupancy(self, frame: Dict):
        """Layer 1 几何: depth → VO → 累积 robot pose → occupancy → KF 累计量.

        Returns: (dtrans, drot, info_gain)
        副作用: self.robot_{x,y,theta} 累加; self.kf_selector.update_motion;
                self.metrics["runtime_s"]["depth"|"vo"] 累加.
        """
        front_img = cv2.imread(frame["cameras"]["camera_1"])

        t1 = time.time()
        depth_map = None
        if self.depth and self.depth.available and front_img is not None:
            depth_map = self.depth.estimate(front_img)
        self.metrics["runtime_s"]["depth"] += time.time() - t1

        tvo = time.time()
        dtrans, drot = self._vo_motion(front_img, depth_map)
        self.metrics["runtime_s"]["vo"] += time.time() - tvo

        self.robot_theta += drot
        self.robot_x += dtrans * math.cos(self.robot_theta)
        self.robot_y += dtrans * math.sin(self.robot_theta)

        info_gain = 0.0
        pts_cam = getattr(self.depth, "last_points_camera", None) if self.depth else None
        depth_conf = getattr(self.depth, "last_depth_conf", None) if self.depth else None
        if pts_cam is not None and getattr(self.cfg, "occ_backend", "vggt") == "vggt":
            info_gain = self.occ.integrate_pointcloud(
                pts_cam, self.robot_x, self.robot_y, self.robot_theta,
                conf=depth_conf,
            )
        elif depth_map is not None:
            row = depth_map[depth_map.shape[0] // 2]
            row = row[::max(1, len(row) // 64)]
            info_gain = self.occ.integrate(self.robot_x, self.robot_y,
                                            self.robot_theta, row)

        self.kf_selector.update_motion(dtrans, drot, info_gain)
        return dtrans, drot, info_gain

    # ------------------------------------------------------------------
    def _stage_vpr_features(self, frame: Dict):
        """提取 4-cam VPR 特征 + 计算到 last KF 的相似度.

        Returns: (feats, sim_to_last)  — sim_to_last is None if no last KF.
        """
        tv = time.time()
        feats = self._extract_features(frame["cameras"])
        self.metrics["runtime_s"]["vpr"] += time.time() - tv
        sim_to_last = None
        if self.last_kf_features is not None:
            sim_to_last = self._vpr_sim(feats, self.last_kf_features)
        return feats, sim_to_last

    # ------------------------------------------------------------------
    def _stage_loop_closure(self, feats, fidx: int, frame: Dict):
        """主动全局闭环检测 (每帧). 副作用: pose_graph/topo edges,
        metrics["n_loop_closures"], self._last_lc_frame, self._lc_log.

        过滤规则:
        - candidates 中 == last_kf_node_id 的直接丢弃 (这是 'VPR 还在最近 KF
          视野' 而非真闭环 — 机器人没远离, 不应当成 loop closure).
        - 剩余候选才视为"真闭环检测", 计入 n_loop_closures + 加 pose_graph edge.
        """
        if len(self.node_features) < 2:
            return
        lc = self.loop_closer.detect(
            "_current_frame", feats, self.node_features,
            self.node_frame_idx, fidx)
        if not lc:
            return
        # 5.5 fix: 提前剔除 self loop, 让 n_loop_closures 反映真实闭环数
        if self.last_kf_node_id:
            n_self = sum(1 for nid, _ in lc if nid == self.last_kf_node_id)
            if n_self:
                self.metrics.setdefault("n_loop_self_skipped", 0)
                self.metrics["n_loop_self_skipped"] += n_self
            lc = [(nid, sim) for nid, sim in lc if nid != self.last_kf_node_id]
        if not lc:
            return  # 全是 self loop, 不算"真闭环检测"

        verified_lc = []
        for old_id, sim in lc:
            if self.cfg.loop_closure_geom_verify:
                ok = self.loop_closer.geometric_verify(
                    frame["cameras"]["camera_1"],
                    self.topo.nodes[old_id].cameras.get("camera_1"),
                    min_inliers=self.cfg.loop_closure_min_inliers)
                if not ok:
                    continue
            verified_lc.append((old_id, sim))
        if not verified_lc:
            return
        # 节流: 距上次闭环 ≥3 帧才记 (条件等价原 v2)
        if not getattr(self, "_last_lc_frame", -100) or \
                fidx - getattr(self, "_last_lc_frame", -100) >= 3:
            self.metrics["n_loop_closures"] += 1
            self._last_lc_frame = fidx
            log_lc = {"frame_idx": fidx, "matches": verified_lc}
            if not hasattr(self, "_lc_log"):
                self._lc_log = []
            self._lc_log.append(log_lc)
            logger.info(f"[LoopClose] frame {fidx} -> {verified_lc}")
            # 加 pose_graph loop edge: 取首个候选 (LoopCloser 已按 sim 降序排过)
            if self.last_kf_node_id:
                for old_id, sim in verified_lc:
                    self.topo.add_edge(self.last_kf_node_id, old_id)
                    self.pose_graph.add_edge(
                        self.last_kf_node_id, old_id,
                        0.0, 0.0, 0.0, info=sim, kind="loop")
                    self.metrics.setdefault("n_loop_edges_added", 0)
                    self.metrics["n_loop_edges_added"] += 1
                    break  # 一次 LC 只加一条边 (与原 v2 verified_lc[:1] 等价)

    # ------------------------------------------------------------------
    def _stage_plate_scan_throttled(self, frame: Dict, fidx: int, triggered: bool):
        """门牌扫描节流: KF 必扫; 否则每 N 帧扫一次."""
        plate_scan_every = int(getattr(self.cfg, "plate_scan_every_n_frames", 2))
        should_scan_plate = (
            self.door_tracker is not None
            and self.detector.gd_available
            and (triggered or plate_scan_every <= 1 or (fidx % plate_scan_every == 0))
        )
        if should_scan_plate:
            tp = time.time()
            self._scan_door_plates(frame, fidx)
            self.metrics["runtime_s"].setdefault("plate_scan", 0.0)
            self.metrics["runtime_s"]["plate_scan"] += time.time() - tp
        else:
            self.metrics.setdefault("plate_scan_skipped", 0)
            self.metrics["plate_scan_skipped"] += 1

    # ------------------------------------------------------------------
    def _stage_create_keyframe(self, frame: Dict, fidx: int, feats, log_entry: Dict) -> bool:
        """KF 主体. 返回 True 表示 REJECT 路径已 append log_entry, 外层 early-return.

        流程: 4-cam GD detect → semantic merge 检查 → (若不合并) junction →
        plate_text → multi-cam scene → category 决策 → REJECT or 创建节点.
        """
        self.metrics["n_keyframes_triggered"] += 1

        # 4-cam Grounding-DINO open-set detect
        td = time.time()
        cam_objects = {}
        for c in CAM_IDS:
            img = cv2.imread(frame["cameras"][c])
            if img is None:
                continue
            cam_objects[c] = self.detector.detect(img)
        self.metrics["runtime_s"]["detect"] += time.time() - td

        # GD 信号: 取最高分 label 作 landmark; 若含 door/plate 则置 room
        room, landmark, best_obj_score = "", "", 0
        for c, dets in cam_objects.items():
            for d in dets:
                if d["score"] > best_obj_score:
                    best_obj_score = d["score"]
                    landmark = d["label"]
                if "door" in d["label"].lower() or "plate" in d["label"].lower():
                    room = d["label"]

        # 同 room/landmark + VPR 高 → 直接合并到旧节点
        merge_target = semantic_dedup.find_merge_target(
            room, landmark, feats, self.topo.nodes, self.node_features)
        if merge_target:
            log_entry["semantic_merge_into"] = merge_target
            self.metrics["n_semantic_merges"] += 1
            if self.last_kf_node_id and self.last_kf_node_id != merge_target:
                self.topo.add_edge(self.last_kf_node_id, merge_target)
            return False  # 走外层 append

        # 不合并: 走 junction → scene → category 链
        junction_kind = self._detect_junction(frame, log_entry)
        plate_text, plate_verified = self._select_confirmed_plate_for_frame(fidx)
        scene_describe, scene_verified = self._stage_multi_cam_scene_naming(
            frame, fidx, junction_kind)

        decision = self.category_clf.classify(
            plate_text=plate_text,
            plate_text_verified=plate_verified,
            scene_describe=scene_describe,
            scene_verified=scene_verified,
            junction_kind=junction_kind,
            gd_landmark=landmark,
        )
        log_entry["category_decision"] = {
            "category": decision.category.value,
            "name": decision.final_name_cn,
            "reason": decision.reason,
            "scene_describe": scene_describe,
            "scene_verified": scene_verified,
            "plate_text": plate_text,
            "plate_verified": plate_verified,
            "gd_landmark": landmark,
        }

        if decision.category == NodeCategory.REJECT:
            self.metrics.setdefault("kf_rejected_by_category", 0)
            self.metrics["kf_rejected_by_category"] += 1
            logger.info(f"[KF-REJECT] frame {fidx}: {decision.reason}")
            self.kf_selector.reset(fidx)
            self.last_kf_features = feats
            self.log_lines.append(log_entry)
            return True  # early return

        self._register_keyframe_node(frame, fidx, feats, decision, cam_objects, log_entry, landmark)
        return False

    # ------------------------------------------------------------------
    def _detect_junction(self, frame: Dict, log_entry: Dict):
        """4-cam 深度 junction 分类. 副作用: log_entry["junction"]."""
        junction_kind = JunctionKind.UNKNOWN
        if self.junction_detector is not None:
            try:
                jinfo = self.junction_detector.classify(frame["cameras"])
                junction_kind = jinfo.kind
                log_entry["junction"] = {
                    "kind": jinfo.kind.value,
                    "open_cams": jinfo.open_cams,
                    "n_open": jinfo.n_open,
                }
            except Exception as e:
                logger.debug(f"junction detect fail: {e}")
        return junction_kind

    # ------------------------------------------------------------------
    def _select_confirmed_plate_for_frame(self, fidx: int):
        """从该帧 _frame_plate_hits 里挑一个 voter-confirmed plate.
        优先级: confirmed brand-like > confirmed 其他 > 无.
        Returns: (plate_text, plate_verified)."""
        hits = list(self._frame_plate_hits.get(fidx, set()))

        def _hit_priority(name):
            if not self.plate_voter.is_confirmed(name):
                return -1
            return 0 if is_brand_like(name) else 1

        hits.sort(key=_hit_priority, reverse=True)
        for name in hits:
            if self.plate_voter.is_confirmed(name):
                return name, True
        return None, False

    # ------------------------------------------------------------------
    def _stage_multi_cam_scene_naming(self, frame: Dict, fidx: int, junction_kind):
        """4-cam Qwen describe_scene + canonical 投票 + 4 种豁免链.

        Returns: (scene_describe, scene_verified).
        副作用: self._recent_scene_winners.append(winner | None).

        豁免链 (consensus≥2 时):
          - top_count ≥ 3 → strong consensus, verified
          - winner ∈ FUNCTION_AREA canonical → func_area bypass, verified
          - junction ∈ {CROSS, T_JUNCTION} → junction bypass, verified
          - winner 出现在最近 3 帧 _recent_scene_winners → temporal confirmed
          - 否则 → tentative, 不建 node
        """
        if self.namer is None or not self.namer._qwen_server:
            return None, False
        try:
            scene_cands = []
            for cam_id in ('camera_1', 'camera_2', 'camera_3', 'camera_4'):
                img_path = frame["cameras"].get(cam_id)
                if not img_path:
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                try:
                    r = self.namer._qwen_server.describe_scene(self._b64(img))
                except Exception:
                    continue
                if r.get("status") == "ok":
                    cand = (r.get("name_cn") or "").strip()
                    if cand and cand not in ("未知", "未知位置"):
                        scene_cands.append((cam_id, cand))

            def _canonicalize(name: str) -> str:
                """Normalize raw scene name to canonical form so synonyms collapse
                into one vote (e.g. '打印机房' / '打印机室' both → '打印区')."""
                for kw, cn in FUNCTION_AREA_WHITELIST.items():
                    if kw in name:
                        return cn
                for kw, cn in LANDMARK_FACILITY_WHITELIST.items():
                    if kw in name:
                        return cn
                for pat in BUILDING_LANDMARK_PATTERNS:
                    if pat.match(name):
                        return name
                return name

            def _specificity_rank(name: str) -> int:
                for kw in FUNCTION_AREA_WHITELIST:
                    if kw in name:
                        return 0
                for kw in LANDMARK_FACILITY_WHITELIST:
                    if kw in name:
                        return 1
                for pat in BUILDING_LANDMARK_PATTERNS:
                    if pat.match(name):
                        return 2
                return 3

            if not scene_cands:
                return None, False

            canonical_cands = [(c, _canonicalize(n)) for c, n in scene_cands]
            from collections import Counter
            vote_cnt = Counter(n for _, n in canonical_cands)
            top = vote_cnt.most_common()
            top_count = top[0][1]
            if top_count < 2:
                self._recent_scene_winners.append(None)
                logger.info(f"[MultiCamScene] fidx={fidx} cands={scene_cands} "
                            f"votes={dict(vote_cnt)} → NO CONSENSUS (all cams differ), skip node")
                return None, False

            top_names = [n for n, c in top if c == top_count]
            top_names.sort(key=_specificity_rank)
            winner = top_names[0]
            # FUNCTION_AREA canonical 值 (前台/打印区/关爱室/外卖柜区/保安亭 等) 是 Qwen
            # 低幻觉的具体 landmark, 独立于场景也能 2/4 bypass temporal. '大堂' 等泛词
            # 不在此 set (被 category_clf reject). LANDMARK_FACILITY (含'电梯厅') 不
            # bypass, 防 N11 电梯厅_2 类 multi-cam joint hallucination.
            is_junction = junction_kind in (JunctionKind.CROSS, JunctionKind.T_JUNCTION)
            is_func_area_strong = winner in set(FUNCTION_AREA_WHITELIST.values())

            scene_describe, scene_verified = None, False
            if top_count >= 3:
                scene_describe, scene_verified = winner, True
                logger.info(f"[MultiCamScene] fidx={fidx} cands={scene_cands} "
                            f"votes={dict(vote_cnt)} → '{winner}' (strong consensus="
                            f"{top_count}/{len(canonical_cands)})")
            elif is_func_area_strong:
                scene_describe, scene_verified = winner, True
                logger.info(f"[MultiCamScene] fidx={fidx} cands={scene_cands} "
                            f"votes={dict(vote_cnt)} → '{winner}' (func_area bypass, "
                            f"consensus={top_count}/{len(canonical_cands)})")
            elif is_junction:
                scene_describe, scene_verified = winner, True
                logger.info(f"[MultiCamScene] fidx={fidx} cands={scene_cands} "
                            f"votes={dict(vote_cnt)} → '{winner}' (junction-"
                            f"{junction_kind.value} bypass, consensus="
                            f"{top_count}/{len(canonical_cands)})")
            elif winner in self._recent_scene_winners:
                scene_describe, scene_verified = winner, True
                logger.info(f"[MultiCamScene] fidx={fidx} cands={scene_cands} "
                            f"votes={dict(vote_cnt)} → '{winner}' (temporal confirmed, "
                            f"consensus={top_count}/{len(canonical_cands)})")
            else:
                logger.info(f"[MultiCamScene] fidx={fidx} cands={scene_cands} "
                            f"votes={dict(vote_cnt)} → tentative '{winner}' (consensus="
                            f"{top_count}/{len(canonical_cands)}, awaits temporal confirm)")

            self._recent_scene_winners.append(winner)
            return scene_describe, scene_verified
        except Exception as e:
            logger.debug(f"multi-cam scene desc fail: {e}")
            return None, False

    # ------------------------------------------------------------------
    def _register_keyframe_node(self, frame: Dict, fidx: int, feats,
                                 decision, cam_objects: Dict, log_entry: Dict,
                                 gd_landmark: str):
        """创建 TopoNode + scene_graph + pose_graph + 邻接边 + NBV.
        副作用: next_node_id++, last_kf_node_id, last_kf_features, node_payloads."""
        self.metrics.setdefault("kf_accepted_by_category", {})
        cat_key = decision.category.value
        self.metrics["kf_accepted_by_category"][cat_key] = \
            self.metrics["kf_accepted_by_category"].get(cat_key, 0) + 1

        nid = str(self.next_node_id)
        self.next_node_id += 1
        node = TopoNode(node_id=nid, timestamp=frame["timestamp"],
                        frame_idx=fidx, cameras=dict(frame["cameras"]),
                        landmark_name=decision.final_name_cn or gd_landmark,
                        room=decision.final_name_cn or "unknown")
        node.position_name = decision.final_name_cn
        node.position_name_eng = decision.final_name_en
        node.category = decision.category.value
        if decision.category == NodeCategory.SHOP:
            node.name_struct = NodeName(organization=decision.final_name_cn)
        else:
            node.name_struct = NodeName(
                category=decision.final_name_cn,
                category_en=decision.final_name_en,
            )
        for c, dets in cam_objects.items():
            for d in dets:
                lab = d.get("label")
                if lab and lab not in node.name_struct.nearby_landmarks:
                    node.name_struct.nearby_landmarks.append(lab)
        self.topo.add_node(node)
        self.node_features[nid] = feats
        self.node_frame_idx[nid] = fidx

        sobjs = []
        for c, dets in cam_objects.items():
            for d in dets:
                sobjs.append(SceneObject(label=d["label"], bbox=d["bbox"],
                                         score=d["score"], camera=c))
        self.scene_graph.add_node(nid, room=node.room, objects=sobjs)

        # pose_graph: 加节点 + 与 last KF 的相对 odom 边 (机器人坐标系下)
        self.pose_graph.add_node(nid, self.robot_x, self.robot_y, self.robot_theta)
        if self.last_kf_node_id:
            self.topo.add_edge(self.last_kf_node_id, nid)
            last_pn = self.pose_graph.nodes[self.last_kf_node_id]
            c, s = math.cos(last_pn.theta), math.sin(last_pn.theta)
            dx_w, dy_w = self.robot_x - last_pn.x, self.robot_y - last_pn.y
            edx = c * dx_w + s * dy_w
            edy = -s * dx_w + c * dy_w
            self.pose_graph.add_edge(self.last_kf_node_id, nid,
                                      edx, edy, self.robot_theta - last_pn.theta,
                                      info=1.0, kind="odom")

        best, _top5 = self.nbv.score_and_pick(
            self.occ, self.robot_x, self.robot_y,
            [n.landmark_name for n in self.topo.nodes.values()][-3:])
        if best:
            log_entry["nbv_pick"] = best

        self.node_payloads[nid] = {
            "node": node,
            "objects": cam_objects,
            "front_img": frame["cameras"]["camera_1"],
        }

        self.last_kf_node_id = nid
        self.last_kf_features = feats

    # ------------------------------------------------------------------
    def finalize(self):
        """收尾: 语义节点/合并/拓扑重建/写出文件. 幂等."""
        if getattr(self, "_finalized", False):
            logger.warning("finalize() called twice; skipping")
            return
        self._finalized = True
        t0 = getattr(self, "_run_start_ts", time.time())
        self.metrics["runtime_s"]["total"] = time.time() - t0
        self.metrics["n_nodes"] = len(self.topo.nodes)
        self.metrics["n_edges"] = len(self.topo.edges)
        self._finalize()

    # ------------------------------------------------------------------
    def _scan_door_plates(self, frame, fidx):
        """每帧扫描门牌候选 (严格 prompt + 二次验证 + 多帧投票).

        性能优化: 该函数原本 4 个 camera 的 Qwen VQA 串行执行, 占总耗时 ~74%.
        改为两阶段 + bbox 级并发:
          阶段 1 (串行): cv2.imread + GD detect 收集所有 bbox.
          阶段 2 (并发): Qwen strict_prompt + verify 并发发到 vLLM,
                        让 vLLM continuous batching 发挥作用.
          阶段 3 (串行): 汇总结果, 线程安全地 voter.add / tracker.add.
        语义等价原始串行版: bbox 的接受/拒绝判定逻辑一字不差, 只是顺序异步.
        voter/tracker 的插入顺序最终按 (cam_id, bbox_idx) 重新确定, 保证可复现.
        """
        if self.namer is None or not self.namer._qwen_server:
            return
        import json as _json, re as _re
        from concurrent.futures import ThreadPoolExecutor

        # ---- 阶段 1: 串行 GD detect (本地 GPU, Qwen 不参与) ----
        det_items = []  # [(cam_id, img, d, crop_for_qwen)]
        for cam_id in CAM_IDS:
            cam_path = frame["cameras"].get(cam_id)
            if not cam_path: continue
            img = cv2.imread(cam_path)
            if img is None: continue
            dets = self.detector.detect(img, queries=["door plate", "room number sign"])
            for d in dets:
                if d["score"] < self.cfg.door_plate_min_score:
                    continue
                try:
                    x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                    h, w = img.shape[:2]
                    bw = x2 - x1; bh = y2 - y1
                    mx = max(20, int(bw * 0.6)); my = max(20, int(bh * 0.6))
                    x1c = max(0, x1 - mx); y1c = max(0, y1 - my)
                    x2c = min(w, x2 + mx); y2c = min(h, y2 + my)
                    crop = img[y1c:y2c, x1c:x2c]
                    if crop.size == 0: continue
                    short_side = min(crop.shape[0], crop.shape[1])
                    crop_for_qwen = img if short_side < 300 else crop
                except Exception:
                    continue
                det_items.append((cam_id, img, d, crop_for_qwen, (x1, y1, x2, y2)))

        if not det_items:
            return

        # ---- 阶段 2: 并发跑 Qwen strict + (可能) verify ----
        qwen = self.namer._qwen_server

        def _strict_one(item):
            cam_id, img, d, crop_for_qwen, bbox_int = item
            try:
                raw = qwen._chat(
                    STRICT_DETECT_TEXT_PROMPT, self._b64(crop_for_qwen), max_tokens=120)
                raw_clean = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
                m = _re.search(r'\{.*\}', raw_clean, flags=_re.DOTALL)
                if not m: return (item, None)
                try:
                    obj = _json.loads(m.group())
                except Exception:
                    return (item, None)
                return (item, obj)
            except Exception as e:
                logger.debug(f"plate strict fail: {e}")
                return (item, None)

        # max_workers=4 和 CAM_IDS 长度匹配. vLLM 内部 continuous batching 消化.
        with ThreadPoolExecutor(max_workers=4) as ex:
            strict_results = list(ex.map(_strict_one, det_items))

        # 解析出需要 verify 的, 并发 verify_text
        parsed = []  # [(item, obj, need_verify)]
        for item, obj in strict_results:
            if obj is None or not obj.get("found"):
                continue
            text = (obj.get("text") or "").strip()
            name_cn = (obj.get("name_cn") or "").strip()
            confidence = (obj.get("confidence") or "").lower()
            if confidence == "low":
                self.metrics.setdefault("plate_drops_low_conf", 0)
                self.metrics["plate_drops_low_conf"] += 1
                continue
            if not (text or name_cn):
                continue
            verify_claim = text or name_cn
            skip_verify = (confidence == "high") or not (
                self.verifier and self.verifier.available)
            parsed.append((item, obj, None if skip_verify else verify_claim))

        # 并发 verify_text (仅对 need_verify 的)
        to_verify = [(i, p) for i, p in enumerate(parsed) if p[2] is not None]
        verify_results = {}
        if to_verify:
            def _verify_one(arg):
                i, (item, obj, claim) = arg
                cam_id, img, d, _cfq, _bbint = item
                try:
                    return (i, self.verifier.verify_text(img, claim))
                except Exception:
                    return (i, False)
            with ThreadPoolExecutor(max_workers=4) as ex:
                for i, ok in ex.map(_verify_one, to_verify):
                    verify_results[i] = ok

        # ---- 阶段 3: 串行 merge 到 voter + tracker (保证顺序可复现) ----
        import re as _re2
        for idx, (item, obj, claim) in enumerate(parsed):
            cam_id, img, d, _cfq, (x1, y1, x2, y2) = item
            if claim is not None:
                ok = verify_results.get(idx, False)
                if not ok:
                    self.metrics.setdefault("plate_drops_verify", 0)
                    self.metrics["plate_drops_verify"] += 1
                    logger.debug(f"[DoorPlate-VERIFY-DROP] "
                                 f"fidx={fidx} cam={cam_id} claim='{claim}'")
                    continue
            text = (obj.get("text") or "").strip()
            name_cn = (obj.get("name_cn") or "").strip()
            name_en = (obj.get("name_en") or "").strip()
            confidence = (obj.get("confidence") or "medium").lower()
            if text and _re2.fullmatch(r"[A-Za-z][A-Za-z0-9\.\- &']{3,30}", text):
                vote_name = text
            else:
                vote_name = name_cn or text
            self.plate_voter.add(NameVote(
                name=vote_name, frame_idx=fidx,
                camera=cam_id, area=float((x2 - x1) * (y2 - y1)),
                confidence=confidence,
            ))
            self._frame_plate_hits.setdefault(fidx, set()).add(vote_name)
            self.door_tracker.add(PlateObservation(
                frame_idx=fidx, timestamp=frame["timestamp"],
                cameras=dict(frame["cameras"]),
                camera=cam_id, bbox=d["bbox"], score=d["score"],
                text=text, name_cn=name_cn, name_en=name_en,
                pose=(self.robot_x, self.robot_y, self.robot_theta),
            ))
            self.metrics["n_door_plates"] += 1

    # ------------------------------------------------------------------
    def _create_door_plate_nodes(self):
        """从 DoorPlateTracker 取每个 plate 的 best 帧, 创建 semantic node
        仅接受 voter 确认 + 分类器接受的 plate. 已被 keyframe 节点吸收的
        plate 跳过 (避免重复)。"""
        if not self.door_tracker:
            return
        voter_stats = self.plate_voter.stats()
        self.metrics["plate_voter"] = voter_stats
        logger.info(f"[DoorPlate] voter stats: {voter_stats}")
        confirmed = set(self.plate_voter.confirmed_names())
        # 已经被 keyframe 节点拿走的 plate 名 (避免重复建)
        already_used = set()
        for n in self.topo.nodes.values():
            pn = getattr(n, "position_name", None)
            if pn:
                already_used.add(pn)
        import re as _re3
        for key, obs in self.door_tracker.all_best().items():
            if obs is None:
                continue
            # 与 _scan_door_plates 里 vote_name 的逻辑保持一致
            if obs.text and _re3.fullmatch(r"[A-Za-z][A-Za-z0-9\.\- &']{3,30}", obs.text):
                vote_key = obs.text
            else:
                vote_key = obs.name_cn or obs.text
            if vote_key not in confirmed:
                self.metrics.setdefault("plate_drops_unconfirmed", 0)
                self.metrics["plate_drops_unconfirmed"] += 1
                logger.info(f"[DoorPlate-DROP-UNCONFIRMED] '{vote_key}' "
                            f"(only {len(self.plate_voter.votes_for(vote_key))} vote(s))")
                continue
            # 通过分类器
            decision = self.category_clf.classify(
                plate_text=vote_key, plate_text_verified=True,
                junction_kind=JunctionKind.UNKNOWN,
            )
            if decision.category == NodeCategory.REJECT:
                self.metrics.setdefault("plate_drops_category", 0)
                self.metrics["plate_drops_category"] += 1
                logger.info(f"[DoorPlate-DROP-CATEGORY] '{vote_key}' {decision.reason}")
                continue
            # 第一遍: 跳过 brand-like SHOP, 留到第二遍 attach 处理
            if decision.category == NodeCategory.SHOP and is_brand_like(vote_key):
                continue
            if decision.final_name_cn in already_used:
                self.metrics.setdefault("plate_drops_already_attached", 0)
                self.metrics["plate_drops_already_attached"] += 1
                logger.info(f"[DoorPlate-SKIP-DUP] '{decision.final_name_cn}' "
                            f"already on a keyframe node")
                continue
            already_used.add(decision.final_name_cn)
            nid = str(self.next_node_id)
            self.next_node_id += 1
            node = TopoNode(
                node_id=nid, timestamp=obs.timestamp, frame_idx=obs.frame_idx,
                cameras=dict(obs.cameras),
                landmark_name=decision.final_name_cn,
                room=decision.final_name_cn)
            node.position_name = decision.final_name_cn
            node.position_name_eng = decision.final_name_en
            node.category = decision.category.value
            # 标记: 此节点由 plate best-view 创建, display frame 就是 at-the-plate 视角
            # brand attach 阶段不应被 brand 的 best-view 覆盖
            node._from_plate_best = True
            if decision.category == NodeCategory.SHOP:
                node.name_struct = NodeName(organization=decision.final_name_cn)
            else:
                node.name_struct = NodeName(
                    category=decision.final_name_cn,
                    category_en=decision.final_name_en,
                )
            self.topo.add_node(node)
            self.node_features[nid] = self._extract_features(obs.cameras)
            self.node_frame_idx[nid] = obs.frame_idx
            self.pose_graph.add_node(nid, obs.pose[0], obs.pose[1], obs.pose[2])
            # 邻接关系不在此处建立: finalize 阶段统一用空间最近邻 + 时间相邻
            # (`_rebuild_topology_neighbors_spatial`) 重建, 防止 door-plate
            # 节点的临时 prev/next 连接污染最终拓扑.
            self.scene_graph.add_node(nid, room=obs.name_cn or "unknown", objects=[])
            logger.info(f"[DoorPlate] node {nid} '{obs.name_cn or obs.text}' "
                        f"frame={obs.frame_idx} area={obs.area:.0f}")

        # ============ 第二遍: brand-like plate attach to nearby functional node ============
        ATTACH_GAP = self.cfg.plate_attach_max_frame_gap
        for key, obs in self.door_tracker.all_best().items():
            if obs is None:
                continue
            if obs.text and _re3.fullmatch(r"[A-Za-z][A-Za-z0-9\.\- &']{3,30}", obs.text):
                vote_key = obs.text
            else:
                vote_key = obs.name_cn or obs.text
            if vote_key not in confirmed:
                continue
            if not is_brand_like(vote_key):
                continue
            # 找帧距最近的"非 SHOP / 非空 category"的 node
            nearest_nid = None
            nearest_gap = ATTACH_GAP + 1
            for nid_existing, n_existing in self.topo.nodes.items():
                ts_e = getattr(n_existing, "name_struct", None)
                # target 必须有 functional/room/landmark category 才能接受 brand attach
                if ts_e is None or not ts_e.category:
                    continue
                gap = abs(n_existing.frame_idx - obs.frame_idx)
                if gap < nearest_gap:
                    nearest_gap = gap
                    nearest_nid = nid_existing
            if nearest_nid is not None and nearest_gap <= ATTACH_GAP:
                target = self.topo.nodes[nearest_nid]
                ts = target.name_struct
                replaced_org = False
                if not ts.organization:
                    ts.organization = vote_key
                    replaced_org = True
                elif vote_key != ts.organization and vote_key not in ts.nearby_plates:
                    new_votes = len(self.plate_voter.votes_for(vote_key))
                    old_votes = len(self.plate_voter.votes_for(ts.organization))
                    if new_votes > old_votes * 1.5:
                        ts.nearby_plates.append(ts.organization)
                        ts.organization = vote_key
                        replaced_org = True
                    else:
                        ts.nearby_plates.append(vote_key)
                # 重定位 (仅显示层):
                #   - target 由 plate best-view 创建 (_from_plate_best=True) → 保留
                #     原帧. 因为 plate best-view 已经是"到这个房间/设施门口"的
                #     最佳视角 (bbox 最大 = 离牌子最近), brand 是别处的副标, relocate
                #     到 brand 帧会离开功能中心. 例: 关爱室 (由 '关爱室' plate 建
                #     frame 41=门口近视) 被 NEUMANN (在别处 frame 38) attach 时保留.
                #   - target 由 keyframe trigger 创建 (无 _from_plate_best) →
                #     keyframe 的 trigger 帧 (acc_rot/acc_trans/vpr<0.5) 位置是
                #     累计运动决定的, 相对"任意". brand 的 best-view 帧反而是"离
                #     brand 牌子最近"的语义中心, 是更好的视角. 例: 前台 (keyframe
                #     trigger 在 1770097836 中段) 被 DEEPROUTE.AI (best frame 1770097843
                #     = 前台柜台近视) attach 时, relocate 到 DEEPROUTE.AI 帧更合理.
                target_from_plate = getattr(target, "_from_plate_best", False)
                # When a brand plate is attached to a keyframe-sourced node,
                # relocate display to the brand best-view frame — its plate
                # bbox is usually much larger and the crop includes the
                # brand sign as a navigation landmark.
                if replaced_org and not target_from_plate:
                    target.timestamp = obs.timestamp
                    target.cameras = dict(obs.cameras)
                    logger.info(f"[DoorPlate-RELOCATE-DISPLAY] node {nearest_nid} "
                                f"display ts -> {obs.timestamp} "
                                f"(brand '{vote_key}' best view, keyframe-sourced target); "
                                f"topology frame_idx stays at {target.frame_idx}")
                elif replaced_org:
                    logger.info(f"[DoorPlate-ATTACH-KEEP-DISPLAY] node {nearest_nid} "
                                f"({target.position_name or target.category}) "
                                f"plate-sourced, keeping original plate best-view "
                                f"at ts={target.timestamp}")
                target.position_name = ts.display_cn()
                target.position_name_eng = ts.display_en()
                self.metrics.setdefault("plate_attached_to_keyframe", 0)
                self.metrics["plate_attached_to_keyframe"] += 1
                logger.info(f"[DoorPlate-ATTACH] brand '{vote_key}' -> "
                            f"node {nearest_nid} (gap={nearest_gap}, "
                            f"display='{target.position_name}')")
                continue
            # 无 nearby functional node: 创建独立 SHOP node
            decision = self.category_clf.classify(
                plate_text=vote_key, plate_text_verified=True,
                junction_kind=JunctionKind.UNKNOWN,
            )
            if decision.category == NodeCategory.REJECT:
                continue
            nid = str(self.next_node_id)
            self.next_node_id += 1
            node = TopoNode(
                node_id=nid, timestamp=obs.timestamp, frame_idx=obs.frame_idx,
                cameras=dict(obs.cameras),
                landmark_name=decision.final_name_cn,
                room=decision.final_name_cn)
            node.position_name = decision.final_name_cn
            node.position_name_eng = decision.final_name_en
            node.category = decision.category.value
            node.name_struct = NodeName(organization=vote_key)
            node.position_name = node.name_struct.display_cn()
            node.position_name_eng = node.name_struct.display_en()
            self.topo.add_node(node)
            self.node_features[nid] = self._extract_features(obs.cameras)
            self.node_frame_idx[nid] = obs.frame_idx
            self.pose_graph.add_node(nid, obs.pose[0], obs.pose[1], obs.pose[2])
            self.scene_graph.add_node(nid, room=obs.name_cn or "unknown", objects=[])
            logger.info(f"[DoorPlate-SHOP-STANDALONE] {nid} '{vote_key}' "
                        f"frame={obs.frame_idx}")

    # ------------------------------------------------------------------
    def _finalize(self):
        out_root = Path(self.cfg.output_dir)
        # finalize 各阶段耗时, 暴露在 metrics["runtime_s"]["finalize"] 里
        ft = self.metrics["runtime_s"].setdefault("finalize", {})

        def _time_stage(name):
            class _T:
                def __enter__(_s):
                    _s.t0 = time.time(); return _s
                def __exit__(_s, *a):
                    ft[name] = round(ft.get(name, 0.0) + time.time() - _s.t0, 3)
            return _T()

        # 1. 创建门牌 semantic nodes (基于 best frame)
        with _time_stage("door_plate_nodes"):
            self._create_door_plate_nodes()
        self.metrics["n_nodes"] = len(self.topo.nodes)

        # 1.2 Pose graph 全局优化: 闭环边 (LoopCloser 加的 dx=dy=dtheta=0 约束)
        # 把累积漂移的 VO 位姿拉回一致. 必须在 ColocationMerger / 拓扑重建之前
        # 运行 (后两者都依赖 spatial dist), 也必须在所有节点都加进 pose_graph
        # 之后 (door_plate 节点也已添加).
        po_t = time.time()
        n_pg_edges_before = len(self.pose_graph.edges)
        n_loop_edges = sum(1 for e in self.pose_graph.edges if e.kind == "loop")
        if n_pg_edges_before >= 1 and len(self.pose_graph.nodes) >= 2:
            self.pose_graph.optimize(iters=30)
            self.metrics["pose_graph_optimized"] = {
                "n_nodes": len(self.pose_graph.nodes),
                "n_edges": n_pg_edges_before,
                "n_loop_edges": n_loop_edges,
                "runtime_s": round(time.time() - po_t, 3),
            }
            ft["pose_optimize"] = round(time.time() - po_t, 3)
            logger.info(f"[PoseGraphOpt] {len(self.pose_graph.nodes)} nodes, "
                        f"{n_pg_edges_before} edges ({n_loop_edges} loop) "
                        f"optimized in {time.time() - po_t:.2f}s")

        # 1.5 Co-location merge: 同物理位置的不同 category node 合并
        with _time_stage("coloc_merge"):
            coloc = ColocationMerger(frame_gap=5, vpr_sim=0.70, spatial_dist=1.0)
            coloc_alias = coloc.merge(self.topo.nodes, self.node_features, self.pose_graph)
        self.metrics["coloc_merge"] = {
            "pairs_examined": coloc.stats.pairs_examined,
            "merges": coloc.stats.merges,
            "by_reason": dict(coloc.stats.by_reason),
            "aliases": dict(coloc.stats.aliases),
        }
        if coloc_alias:
            # 用空 names dict 应用 alias (names 还没生成)
            self._apply_node_alias(coloc_alias, {})

        # 1.6 同名 BUILDING_LANDMARK / LANDMARK_FACILITY 合并:
        # ColocationMerger 受限于 frame_gap/spatial_dist, 在多视角观察 (例如
        # 走过 C 座 3 次, 每次离 5m+) 同栋楼时不会合并. 这里强制按 canonical
        # name 合并, 并把 anchor 的 display 帧搬到该名字 plate bbox 最大的帧
        # (door_tracker.best(name)), 给后续 ConnectionBuilder/导航更近视角.
        with _time_stage("canonical_merge"):
            same_name_alias = self._merge_by_canonical_name()
            if same_name_alias:
                self._apply_node_alias(same_name_alias, {})
                self.metrics.setdefault("same_name_merge", {})
                self.metrics["same_name_merge"]["aliases"] = dict(same_name_alias)
                self.metrics["same_name_merge"]["count"] = len(same_name_alias)

        # 1.7 拓扑邻接重建: 用空间最近邻 + 时间相邻并集
        # 之前的 keyframe-only 链 + door-plate prev/next 边在 coloc merge
        # 后会留下 stale edge (e.g. 5↔7), 这里全部清空重建.
        # k_spatial=1: 每个 node 只连最近的 1 个空间邻居 (双向 union)
        # k_temporal=1: 再加每个 node 的 frame_idx prev + next, 保证连通
        with _time_stage("topology_rebuild"):
            self._rebuild_topology_neighbors_spatial(k_spatial=1, k_temporal=1)

        # 2. 命名 (从 node attribute 直接读)
        names = self._generate_names()

        # 2.5 名称去重: 合并同名且 VPR 相似的节点, 其余加后缀
        with _time_stage("name_dedup"):
            dedup = NameDeduplicator(merge_vpr_threshold=0.78)
            names, alias_map = dedup.resolve(names, self.topo.nodes, self.node_features)
            self.metrics["name_dedup"] = dict(dedup.stats)
            if alias_map:
                self._apply_node_alias(alias_map, names)
        self.metrics["n_nodes"] = len(self.topo.nodes)
        self.metrics["n_edges"] = len(self.topo.edges)

        # 3. 写出 base node 目录
        with _time_stage("write_node_dirs"):
            for nid, node in self.topo.nodes.items():
                ns = getattr(node, "name_struct", None)
                self.writer.write_node(
                    node_id=nid, timestamp=node.timestamp,
                    cameras=node.cameras,
                    position_name=names[nid]["cn"],
                    position_name_eng=names[nid]["en"],
                    next_positions=[],
                    name_struct=ns.to_dict() if ns is not None else None,
                )

        # Motion-based body heading per node (atan2 of displacement to an
        # adjacent keyframe). More robust than pose.theta when VO yaw drifts.
        try:
            import math as _math
            ordered = sorted(self.topo.nodes.keys(),
                              key=lambda i: self.topo.nodes[i].frame_idx)
            for i, nid in enumerate(ordered):
                if nid not in self.pose_graph.nodes:
                    continue
                cur = self.pose_graph.nodes[nid]
                dx = dy = None
                # prefer forward-to-next delta; fallback to prev-to-self
                if i + 1 < len(ordered) and ordered[i + 1] in self.pose_graph.nodes:
                    nxt = self.pose_graph.nodes[ordered[i + 1]]
                    dx, dy = nxt.x - cur.x, nxt.y - cur.y
                if dx is None or abs(dx) + abs(dy) < 1e-3:
                    if i > 0 and ordered[i - 1] in self.pose_graph.nodes:
                        prv = self.pose_graph.nodes[ordered[i - 1]]
                        dx, dy = cur.x - prv.x, cur.y - prv.y
                if dx is not None and abs(dx) + abs(dy) > 1e-3:
                    cur.motion_theta = _math.atan2(dy, dx)
                    logger.info(f"[MotionHeading] node {nid}: "
                                f"θ_motion={_math.degrees(cur.motion_theta):+.1f}° "
                                f"(θ_pose={_math.degrees(cur.theta):+.1f}°)")
        except Exception as e:
            logger.warning(f"motion_heading compute failed: {e}")

        # 4. ConnectionBuilder: 真实 next_positions
        if self.cfg.enable_real_connections:
            with _time_stage("connection_builder"):
                try:
                    from online_mapper.topology.connection_builder import ConnectionBuilder
                    cb = ConnectionBuilder(
                        sim_threshold=self.cfg.connection_sim_threshold,
                        qwen_gpu=self.cfg.qwen_gpu, namer=self.namer,
                        depth_estimator=self.depth,
                        detector=self.detector,
                        gap_fuse_ms=self.cfg.cb_gap_fuse_ms,
                        alpha_normal=self.cfg.cb_alpha_normal,
                        person_penalty=self.cfg.cb_person_penalty,
                        fallback_center_tol=self.cfg.cb_fallback_center_tol,
                        max_corridor_frames=self.cfg.cb_max_corridor_frames,
                        corridor_sample_count=self.cfg.cb_corridor_sample_count,
                    )
                    node_dirs = {nid: out_root / nid for nid in self.topo.nodes}
                    for nid, node in self.topo.nodes.items():
                        nbs = [self.topo.nodes[nbid] for nbid in node.neighbors
                               if nbid in self.topo.nodes]
                        if not nbs:
                            continue
                        nps = cb.build_for_node(node, nbs, node_dirs[nid], node_dirs,
                                                 self._all_frames_cache,
                                                 pose_graph=self.pose_graph)
                        if nps:
                            # 重写 node json
                            self._patch_node_with_nexts(out_root / nid, nps,
                                                         names[nid], names)
                            self.metrics["n_connections"] += len(nps)
                            for np_ in nps:
                                if np_.get("landmark_name", " ").strip() not in ("", " "):
                                    self.metrics["n_named_landmarks"] += 1
                    cb.cleanup()
                except Exception as e:
                    logger.error(f"ConnectionBuilder failed: {e}")
                    import traceback; traceback.print_exc()

        # 5. scene_graph / pose_graph / log
        # metrics.json 留到最后写, 让 visualize 阶段产出的 metrics["visualizations"]
        # 也能落盘.
        self.scene_graph.save(str(out_root.parent / "scene_graph.json"))
        with open(out_root.parent / "pose_graph.json", "w", encoding="utf-8") as f:
            json.dump(self.pose_graph.to_dict(), f, ensure_ascii=False, indent=2)
        with open(out_root.parent / "online_mapping_log.jsonl", "w", encoding="utf-8") as f:
            for line in self.log_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.metrics["loop_threshold_used"] = self.loop_closer.current_threshold()

        # DEBUG dump: 每个 plate 的所有 voter 观测 (frame_idx/camera/area/conf), 用于 round-4 分析
        try:
            voter_dump = {
                name: [
                    {"frame_idx": v.frame_idx, "camera": v.camera,
                     "area": v.area, "confidence": v.confidence}
                    for v in votes
                ] for name, votes in self.plate_voter._votes.items()
            }
            with open(out_root.parent / "plate_voter_dump.json", "w", encoding="utf-8") as f:
                json.dump(voter_dump, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"voter dump failed: {e}")

        # 6. 自动生成可视化 (pose_graph.png / occupancy.png / keyframe_timeline.png /
        # scene_overview.txt). matplotlib 不在 import 时加载, 这里 lazy import 避免
        # 把 GUI 依赖塞进 ws_proxy 在线模式. 失败不阻塞主流程.
        # visualize 必须在 metrics.json 写入之前, 让产出的 metrics["visualizations"] 落盘.
        with _time_stage("visualize"):
            try:
                from online_mapper.viz.visualize import render_mapping_visuals
                viz_out = render_mapping_visuals(str(out_root.parent), mapper=self)
                if viz_out:
                    self.metrics["visualizations"] = list(viz_out.keys())
                    logger.info(f"[Visualize] generated: {list(viz_out.keys())}")
            except Exception as e:
                logger.warning(f"visualize failed: {e}")

        # 7. metrics.json (最后写, 含 visualizations + finalize timing 完整信息)
        with open(out_root.parent / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)

        logger.info(f"Online mapping complete: {self.metrics}")

    # ------------------------------------------------------------------
    def _merge_by_canonical_name(self) -> Dict[str, str]:
        """同名 / 复合 plate 节点合并 + 帧 recenter 编排器.

        6 个 Pass 顺序固定 (互相依赖 alias 累积):
          A. 同 base building / 同 canonical name 的节点强制合并
          B. 复合 plate (X座电梯) 重命名最近节点
          C. 数字+柜 (2号柜) → 外卖/储物/快递柜 命名
          D. 保安亭 (scene 或独立 node) attach 到最近建筑
          E. SHOP 品牌节点 attach 到附近 FUNCTION/BUILDING/ROOM
          F. 所有楼栋节点 final recenter 到 plate best-view 帧

        return: {sub_id -> anchor_id} alias map, 调用方传给 _apply_node_alias.
        """
        alias: Dict[str, str] = {}
        self._pass_a_building_landmark_merge(alias)
        self._pass_b_composite_plate_rename(alias)
        self._pass_c_cabinet_plate(alias)
        self._pass_d_booth_attach(alias)
        self._pass_e_shop_attach(alias)
        self._pass_f_final_recenter(alias)
        return alias

    # ------------------------------------------------------------------
    # Canonical-merge helpers (used by Pass A-F)
    # ------------------------------------------------------------------
    @staticmethod
    def _canon_node_name(node) -> str:
        """节点 canonical 名: name_struct.category|organization 优先, 否则 position_name."""
        ns = getattr(node, "name_struct", None)
        if ns and (ns.category or ns.organization):
            base = ns.category or ns.organization
        else:
            base = (getattr(node, "position_name", "") or "").split("·")[0]
        return base.strip()

    @staticmethod
    def _canon_building_base(name: str) -> str:
        """楼栋 base 提取: 'A座入口' / 'A座大堂' / 'A座' → 'A座'."""
        if not name:
            return ""
        m = _BUILDING_BASE_RE.match(name.strip())
        return m.group(1) if m else name.strip()

    def _canon_best_frame_for_voter(self, plate_text: str):
        """直接遍历 voter 找 plate_text 的代表帧 (max bbox area).

        注意: door_tracker 与 plate_voter 用不同 key (前者 text 优先, 后者 name_cn
        优先). 复合 plate (H座电梯) 大概率只在 voter 里, tracker 里是裸 'H座'.
        Pass B/C 必须以 voter._votes 为来源.
        """
        votes = self.plate_voter.votes_for(plate_text)
        if not votes:
            return None, None
        best = max(votes, key=lambda v: v.area or 0)
        return best.frame_idx, best.area or 0.0

    def _canon_best_obs_for_group(self, name: str, alias_names: List[str]):
        """选某 building/landmark group 的代表帧.

        优先级:
          1. '入口' variant — 机器人站在入口前, 位置语义最具体
          2. '大堂' variant
          3. 裸 base name
        再在选定 variant 里取 plate bbox 面积最大那帧 (不能跨 variant 取 max area,
        否则远看整栋楼的 '座' 字 bbox 也大, 会选到机器人已经走过入口的帧).
        door_tracker + voter 双来源: tracker 有 timestamp/cameras 信息, voter 更
        可靠 index (vote_name 可能与 tracker key 不一致).
        """
        all_names = [name] + [n for n in alias_names if n != name]
        entries = [n for n in all_names if "入口" in n]
        lobbies = [n for n in all_names if "大堂" in n]
        bases = [n for n in all_names if n not in entries and n not in lobbies]

        def _resolve(group_names):
            best_obs, best_area = None, 0.0
            for n in group_names:
                if self.door_tracker:
                    obs = self.door_tracker.best(n)
                    if obs is not None and obs.area > best_area:
                        best_obs, best_area = obs, obs.area
                votes = self.plate_voter.votes_for(n)
                if votes:
                    vbest = max(votes, key=lambda v: v.area or 0.0)
                    if (vbest.area or 0.0) > best_area:
                        # 从 tracker 找该 frame 的 obs 拿 timestamp
                        best_obs_fallback = None
                        if self.door_tracker:
                            for nn in group_names:
                                for obs in self.door_tracker._observations.get(nn, []):
                                    if obs.frame_idx == vbest.frame_idx:
                                        best_obs_fallback = obs
                                        break
                                if best_obs_fallback:
                                    break
                        if best_obs_fallback is not None:
                            best_obs, best_area = best_obs_fallback, vbest.area or 0.0
                        elif best_obs is None:
                            # 没 tracker obs 就用 voter 帧手工构造 obs
                            from online_mapper.semantics.door_plate_tracker import PlateObservation
                            ts, cams = None, {}
                            for ln in self.log_lines:
                                if ln.get("frame_idx") == vbest.frame_idx:
                                    ts = ln.get("ts")
                                    break
                            if hasattr(self, "_all_frames_cache") and self._all_frames_cache:
                                for fr in self._all_frames_cache:
                                    if fr.get("frame_idx") == vbest.frame_idx:
                                        cams = dict(fr.get("cameras") or {})
                                        ts = ts or fr.get("timestamp")
                                        break
                            if ts and cams:
                                best_obs = PlateObservation(
                                    frame_idx=vbest.frame_idx, timestamp=ts,
                                    cameras=cams, camera=vbest.camera,
                                    bbox=[0, 0, 0, 0], score=0.0, text=n,
                                    name_cn=n, name_en="",
                                    pose=(0.0, 0.0, 0.0), area=vbest.area or 0.0)
                                best_area = vbest.area or 0.0
            return best_obs

        for group in (entries, lobbies, bases):
            obs = _resolve(group)
            if obs is not None:
                return obs
        return None

    def _canon_spatial_cluster(self, ids_group):
        """同 canonical name 但物理位置远的 node 不应合并 (例: 起点电梯厅 vs H 座电梯厅)."""
        if not self.pose_graph:
            return [ids_group]
        clusters = []
        for i in ids_group:
            if i not in self.pose_graph.nodes:
                clusters.append([i])
                continue
            pi = self.pose_graph.nodes[i]
            placed = False
            for c in clusters:
                for j in c:
                    if j not in self.pose_graph.nodes:
                        continue
                    pj = self.pose_graph.nodes[j]
                    d = ((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2) ** 0.5
                    if d <= _SPATIAL_MERGE_DIST_M:
                        c.append(i)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                clusters.append([i])
        return clusters

    @staticmethod
    def _canon_composite_specificity(plate_text: str) -> int:
        """Pass B 复合 plate 优先级评分: 越大越具体 ('X座电梯' > 'X层电梯' > '11电梯')."""
        m = _COMPOSITE_PREFIX_RE.fullmatch(plate_text)
        if not m:
            return 0
        prefix, suffix_type, _kind = m.group(1), m.group(2), m.group(3)
        s = 0
        if suffix_type in ("座", "栋"):
            s += 30
        elif suffix_type == "楼":
            s += 25
        elif "层" in prefix:
            s += 10
        else:
            s += 5
        if len(prefix) == 1 and prefix.isalpha():
            s += 3
        return s

    # ------------------------------------------------------------------
    def _pass_a_building_landmark_merge(self, alias: Dict[str, str]):
        """Pass A: 同 base building / 同 canonical name 的 BUILDING_LANDMARK +
        LANDMARK_FACILITY 节点强制合并, 并搬 anchor 帧到 plate best-area 帧."""
        from online_mapper.semantics.node_category import NodeCategory

        groups: Dict[tuple, List[str]] = {}
        for nid, node in self.topo.nodes.items():
            cat = getattr(node, "category", "") or ""
            if cat not in (NodeCategory.BUILDING_LANDMARK.value,
                           NodeCategory.LANDMARK_FACILITY.value):
                continue
            name = self._canon_node_name(node)
            if not name:
                continue
            key_name = (self._canon_building_base(name)
                        if cat == NodeCategory.BUILDING_LANDMARK.value else name)
            groups.setdefault((cat, key_name), []).append(nid)

        for (cat, base_name), ids in groups.items():
            if len(ids) < 2:
                continue
            variants = list({self._canon_node_name(self.topo.nodes[i]) for i in ids})
            best_obs = self._canon_best_obs_for_group(base_name, variants)
            for sub_ids in self._canon_spatial_cluster(ids):
                if len(sub_ids) < 2:
                    continue
                # Anchor = 最晚 frame_idx (机器人沿路径走过 landmark 时, 更晚的
                # keyframe 通常离 landmark 更近 → plate bbox 更大, cam_crop 更准)
                anchor_id = max(sub_ids, key=lambda i: self.topo.nodes[i].frame_idx)
                if cat == NodeCategory.BUILDING_LANDMARK.value:
                    target = self.topo.nodes[anchor_id]
                    ns = getattr(target, "name_struct", None)
                    if ns is not None:
                        ns.category = base_name
                        ns.category_en = base_name
                    target.position_name = base_name
                    target.position_name_eng = base_name
                    target.room = base_name
                for sid in sub_ids:
                    if sid != anchor_id and sid not in alias:
                        alias[sid] = anchor_id
                if best_obs is not None:
                    target = self.topo.nodes[anchor_id]
                    if target.frame_idx != best_obs.frame_idx:
                        logger.info(f"[SameName-RELOCATE] node {anchor_id} '{base_name}' "
                                    f"frame {target.frame_idx} -> {best_obs.frame_idx} "
                                    f"(plate area={best_obs.area:.0f})")
                        target.frame_idx = best_obs.frame_idx
                        target.timestamp = best_obs.timestamp
                        target.cameras = dict(best_obs.cameras)
                        if anchor_id in self.node_features:
                            self.node_features[anchor_id] = self._extract_features(best_obs.cameras)
                        self.node_frame_idx[anchor_id] = best_obs.frame_idx
                logger.info(f"[SameName] '{base_name}' [{cat}] cluster={sub_ids}: "
                            f"keep {anchor_id}, absorb {[i for i in sub_ids if i != anchor_id]}")

    # ------------------------------------------------------------------
    def _pass_b_composite_plate_rename(self, alias: Dict[str, str]):
        """Pass B: 复合 plate ('H座电梯') 优先, 重命名最近 BUILDING/LANDMARK 节点
        并搬到 plate best-view 帧. 按 specificity → area 降序分配, 每 anchor 一次."""
        from online_mapper.semantics.node_category import NodeCategory

        candidates = []
        for plate_text in list(self.plate_voter._votes.keys()):
            if not _COMPOSITE_PREFIX_RE.fullmatch(plate_text or ""):
                continue
            if not self.plate_voter.is_confirmed(plate_text):
                continue
            target_frame, area = self._canon_best_frame_for_voter(plate_text)
            if target_frame is None:
                continue
            candidates.append((plate_text, target_frame, area,
                               self._canon_composite_specificity(plate_text)))
        candidates.sort(key=lambda x: (-x[3], -x[2]))

        anchored = set()
        for plate_text, target_frame, _area, _spec in candidates:
            cands = [
                nid for nid, n in self.topo.nodes.items()
                if nid not in alias and nid not in anchored
                and (getattr(n, "category", "") in (
                    NodeCategory.BUILDING_LANDMARK.value,
                    NodeCategory.LANDMARK_FACILITY.value))
                and abs(n.frame_idx - target_frame) <= _PASS_B_FRAME_GAP
            ]
            if not cands:
                continue
            anchor = min(cands, key=lambda i: abs(
                self.topo.nodes[i].frame_idx - target_frame))
            target = self.topo.nodes[anchor]
            ns = getattr(target, "name_struct", None)
            if ns is not None:
                ns.category = plate_text
                ns.category_en = plate_text
            target.position_name = plate_text
            target.position_name_eng = plate_text
            target.room = plate_text
            # 搬 timestamp/frame/cameras 到 plate 最大 bbox 那帧 (用户最近视角).
            # door_tracker 优先 (有 ts/cameras), 否则从 _all_frames_cache 反查
            # voter 的 best vote frame_idx (tracker 的 key 可能是单字母 'H').
            votes = self.plate_voter.votes_for(plate_text)
            best_v = max(votes, key=lambda v: v.area or 0) if votes else None
            obs = self.door_tracker.best(plate_text) if self.door_tracker else None
            new_frame = new_ts = new_cams = None
            if obs is not None:
                new_frame, new_ts, new_cams = obs.frame_idx, obs.timestamp, dict(obs.cameras)
            elif best_v is not None and self._all_frames_cache:
                for fr in self._all_frames_cache:
                    if fr.get("frame_idx") == best_v.frame_idx:
                        new_frame = best_v.frame_idx
                        new_ts = fr.get("timestamp")
                        new_cams = dict(fr.get("cameras") or {})
                        break
            if new_frame is not None and target.frame_idx != new_frame and new_ts and new_cams:
                target.frame_idx = new_frame
                target.timestamp = new_ts
                target.cameras = new_cams
                if anchor in self.node_features:
                    self.node_features[anchor] = self._extract_features(new_cams)
                self.node_frame_idx[anchor] = new_frame
            anchored.add(anchor)
            logger.info(f"[CompositePlate] '{plate_text}' spec={_spec} → node {anchor} ts→{new_ts}")

    # ------------------------------------------------------------------
    def _pass_c_cabinet_plate(self, alias: Dict[str, str]):
        """Pass C: 数字+柜 (2号柜/1号柜) 命名外卖/储物/快递柜区.

        kind 由该 N号柜 plate 周围 confirmed 的 '外卖柜'/'储物柜'/'快递柜' plate
        实证决定 (Qwen scene_describe 把外卖柜区误读成储物柜区曾发生).
        近视证据 area > _CABINET_NEAR_VIEW_AREA 时优先取 '外卖柜'/'快递柜'.
        """
        def _nearest_plate_area(plate_name: str, f: int, window: int) -> float:
            votes = self.plate_voter.votes_for(plate_name)
            if not votes:
                return 0.0
            near = [v for v in votes if abs(v.frame_idx - f) <= window]
            if not near:
                return 0.0
            return max((v.area or 0.0) for v in near)

        for plate_text in list(self.plate_voter._votes.keys()):
            m = _CABINET_RE.fullmatch(plate_text or "")
            if not m:
                continue
            if not self.plate_voter.is_confirmed(plate_text):
                continue
            target_frame, _ = self._canon_best_frame_for_voter(plate_text)
            if target_frame is None:
                continue
            num = m.group(1)
            cands = [
                nid for nid, n in self.topo.nodes.items()
                if nid not in alias
                and (getattr(n, "position_name", "") or "").split("·")[0] in _CABINET_AREA_TO_KIND
                and abs(n.frame_idx - target_frame) <= _PASS_C_FRAME_GAP
            ]
            if not cands:
                continue
            anchor = min(cands, key=lambda i: abs(
                self.topo.nodes[i].frame_idx - target_frame))
            target = self.topo.nodes[anchor]
            base_canon = (target.position_name or "").split("·")[0]

            kind_scores = {}
            for kname in _CABINET_KIND_PLATES:
                if self.plate_voter.is_confirmed(kname):
                    a = _nearest_plate_area(kname, target_frame, _PASS_C_NEAR_WINDOW)
                    if a > 0:
                        kind_scores[kname] = a
            chosen_kind = None
            if kind_scores:
                outbox = kind_scores.get("外卖柜", 0)
                express = kind_scores.get("快递柜", 0)
                if outbox >= _CABINET_NEAR_VIEW_AREA or express >= _CABINET_NEAR_VIEW_AREA:
                    chosen_kind = "外卖柜" if outbox >= express else "快递柜"
                else:
                    chosen_kind = max(kind_scores.items(), key=lambda kv: kv[1])[0]
            else:
                chosen_kind = base_canon.replace("区", "") if base_canon.endswith("区") else None
                if chosen_kind not in _CABINET_KIND_TO_SHORT:
                    chosen_kind = None
            if chosen_kind is None:
                continue
            short = _CABINET_KIND_TO_SHORT[chosen_kind]
            composite = f"{num}号{short}"
            ns = getattr(target, "name_struct", None)
            if ns is not None:
                ns.category = composite
                ns.category_en = composite
                # 中文 org 保留, 英文 org (EXHIOH/HIOF) drop
                if not (ns.organization and not any(c.isascii() and c.isalnum()
                                                     for c in ns.organization)):
                    ns.organization = ""
            target.position_name = composite
            target.position_name_eng = composite
            target.room = composite
            logger.info(f"[CabinetPlate] '{plate_text}' near frame {target_frame} "
                        f"kind={chosen_kind} → '{composite}' on node {anchor} "
                        f"(was: {base_canon})")

    # ------------------------------------------------------------------
    def _pass_d_booth_attach(self, alias: Dict[str, str]):
        """Pass D: 保安亭/岗亭 attach 到最近 BUILDING_LANDMARK.

        case B (优先): 已建 FUNCTION_AREA=保安亭 节点 → 合并到最近 building, 并把
                       building 改名为 'X座保安亭' (保留 building anchor).
        case A: scene 提示岗亭但没建独立节点 → 直接 rename 最近 building.
        """
        from online_mapper.semantics.node_category import NodeCategory

        booth_frames = []
        for ln in self.log_lines:
            cd = ln.get("category_decision") or {}
            sd = cd.get("scene_describe") or ""
            if sd in _BOOTH_TOKENS:
                booth_frames.append(ln.get("frame_idx"))

        building_nodes = [
            (nid, n) for nid, n in self.topo.nodes.items()
            if nid not in alias
            and getattr(n, "category", "") == NodeCategory.BUILDING_LANDMARK.value
        ]
        booth_nodes = [
            (nid, n) for nid, n in self.topo.nodes.items()
            if nid not in alias
            and (getattr(n, "name_struct", None) is not None
                 and getattr(n.name_struct, "category", "") == "保安亭")
        ]

        used_buildings = set()
        # case B
        for (bn_id, bn) in list(booth_nodes):
            if bn_id in alias:
                continue
            if not building_nodes:
                break
            nearest = min(building_nodes, key=lambda p: abs(p[1].frame_idx - bn.frame_idx))
            if abs(nearest[1].frame_idx - bn.frame_idx) > _PASS_D_BUILDING_GAP:
                continue
            bld_name = (self._canon_node_name(nearest[1]) or "").strip()
            if not bld_name:
                continue
            composite = f"{bld_name}保安亭"
            ns = getattr(bn, "name_struct", None)
            if ns is not None:
                ns.category = composite
                ns.category_en = composite
                ns.organization = ""
            bn.position_name = composite
            bn.position_name_eng = composite
            bn.room = composite
            alias[nearest[0]] = bn_id
            used_buildings.add(nearest[0])
            logger.info(f"[Booth-B] merge {nearest[0]}({bld_name}) → {bn_id} ({composite})")

        # case A
        for bf in booth_frames:
            if bf is None:
                continue
            cands = [
                (nid, n) for nid, n in self.topo.nodes.items()
                if nid not in alias and nid not in used_buildings
                and getattr(n, "category", "") == NodeCategory.BUILDING_LANDMARK.value
                and abs(n.frame_idx - bf) <= _PASS_D_SCENE_GAP
            ]
            if not cands:
                continue
            nid_b, n_b = min(cands, key=lambda p: abs(p[1].frame_idx - bf))
            bld_name = (self._canon_node_name(n_b) or "").strip()
            composite = f"{bld_name}保安亭"
            ns = getattr(n_b, "name_struct", None)
            if ns is not None:
                # 保留楼栋前缀, 把 base category 改成复合名; organization (HIOF) 可保留
                ns.category = composite
                ns.category_en = composite
            n_b.position_name = ns.display_cn() if ns else composite
            n_b.position_name_eng = ns.display_en() if ns else composite
            n_b.room = composite
            used_buildings.add(nid_b)
            logger.info(f"[Booth-A] scene=岗亭 frame {bf} → rename node {nid_b} '{bld_name}' → '{composite}'")

    # ------------------------------------------------------------------
    def _pass_e_shop_attach(self, alias: Dict[str, str]):
        """Pass E: SHOP 节点 (HIOF/EXHIOH) brand-attach 到相近非 SHOP 节点
        (FUNCTION/BUILDING/LANDMARK/ROOM), 以 FUNCTION 为主名, SHOP 作 organization."""
        from online_mapper.semantics.node_category import NodeCategory

        for sid, snode in list(self.topo.nodes.items()):
            if sid in alias:
                continue
            if getattr(snode, "category", "") != NodeCategory.SHOP.value:
                continue
            sns = getattr(snode, "name_struct", None)
            if sns is None or not sns.organization:
                continue
            nearby = [
                (fid, fn) for fid, fn in self.topo.nodes.items()
                if fid not in alias and fid != sid
                and getattr(fn, "category", "") in (
                    NodeCategory.FUNCTION_AREA.value,
                    NodeCategory.BUILDING_LANDMARK.value,
                    NodeCategory.LANDMARK_FACILITY.value,
                    NodeCategory.ROOM_NAMED.value,
                    NodeCategory.ROOM_NUMBERED.value)
                and abs(fn.frame_idx - snode.frame_idx) <= _PASS_E_FRAME_GAP
            ]
            if not nearby:
                continue
            anchor_id, anchor_node = min(
                nearby, key=lambda p: abs(p[1].frame_idx - snode.frame_idx))
            ans = getattr(anchor_node, "name_struct", None)
            if ans is None:
                continue
            if not ans.organization:
                ans.organization = sns.organization
            elif ans.organization != sns.organization:
                if sns.organization not in ans.nearby_plates:
                    ans.nearby_plates.append(sns.organization)
            anchor_node.position_name = ans.display_cn()
            anchor_node.position_name_eng = ans.display_en()
            alias[sid] = anchor_id
            logger.info(f"[ShopAttach] merge SHOP {sid}({sns.organization}) → "
                        f"{anchor_id}({anchor_node.position_name})")

    # ------------------------------------------------------------------
    def _pass_f_final_recenter(self, alias: Dict[str, str]):
        """Pass F: 所有 BUILDING_LANDMARK / 含保安亭/电梯/楼梯 的节点强制把 display
        帧搬到其 canonical plate 的 best-area 帧 (优先 '入口' variant + multi-cam).

        Pass A 内部 fallback 对单字母 tracker key 不稳, 这里用 voter 做最终 re-center.
        """
        for nid, node in self.topo.nodes.items():
            if nid in alias:
                continue
            name = (getattr(node, "position_name", "") or "").split("·")[0]
            m = _BLD_BASE_RE.match(name)
            if m:
                self._canon_final_recenter(node, m.group(1))

    def _canon_final_recenter(self, node, canonical_base: str):
        """搬 display 帧策略: 优先 multi-camera 可见帧 (机器人正面对地标),
        再退化到单 camera max area. 入口 variant 优先于裸 base."""
        relevant = set()
        for k in self.plate_voter._votes.keys():
            if not k:
                continue
            if k == canonical_base:
                relevant.add(k)
            elif k.startswith(canonical_base) and any(s in k for s in ("入口", "大堂", "电梯", "楼梯")):
                relevant.add(k)
        if not relevant:
            return

        from collections import defaultdict as _dd

        def _pick_multi(keys):
            fc = _dd(set); fa = _dd(float)
            for k in keys:
                for v in self.plate_voter.votes_for(k):
                    fc[v.frame_idx].add(v.camera)
                    fa[v.frame_idx] = max(fa[v.frame_idx], v.area or 0)
            multi = [fi for fi, cs in fc.items() if len(cs) >= 2]
            if multi:
                return max(multi, key=lambda fi: fa[fi])
            return None

        def _pick_single(keys):
            best_f, best_a = None, 0.0
            for k in keys:
                for v in self.plate_voter.votes_for(k):
                    if (v.area or 0) > best_a:
                        best_a = v.area or 0
                        best_f = v.frame_idx
            return best_f

        entry_keys = [k for k in relevant if "入口" in k]
        base_keys = [k for k in relevant if k == canonical_base]
        all_keys = list(relevant)

        # 优先级链: entry multi → base multi → all multi → entry single → base single
        target_frame = (_pick_multi(entry_keys) or _pick_multi(base_keys)
                        or _pick_multi(all_keys)
                        or _pick_single(entry_keys) or _pick_single(base_keys))
        if target_frame is None or target_frame == node.frame_idx:
            return
        for fr in self._all_frames_cache:
            if fr.get("frame_idx") == target_frame:
                node.frame_idx = target_frame
                node.timestamp = fr.get("timestamp") or node.timestamp
                cams = dict(fr.get("cameras") or {})
                if cams:
                    node.cameras = cams
                # 重建 VPR features 以保持一致
                nid = None
                for k, v in self.topo.nodes.items():
                    if v is node:
                        nid = k
                        break
                if nid and nid in self.node_features and cams:
                    try:
                        self.node_features[nid] = self._extract_features(cams)
                        self.node_frame_idx[nid] = target_frame
                    except Exception:
                        pass
                logger.info(f"[Recenter] node -> frame {target_frame} ts={node.timestamp} "
                            f"(canonical={canonical_base}, via keys {list(relevant)[:3]})")
                break

    # ------------------------------------------------------------------
    def _rebuild_topology_neighbors_spatial(self, k_spatial: int = 2, k_temporal: int = 1):
        """Clear all topology edges, then rebuild as the union of:
          - Each node's k_spatial nearest neighbors by pose distance
          - Each node's k_temporal nearest neighbors by frame_idx (immediate
            prev + next, capped at k_temporal each side)

        Pose-based KNN gives the spatially-correct chain, even when the
        robot's frame_idx ordering doesn't match the spatial layout (e.g.
        the robot revisits a region). Temporal KNN ensures connectivity.
        Both directions of each edge are added (bidirectional).
        """
        if len(self.topo.nodes) < 2:
            return
        # Reset adjacency
        self.topo.edges = set()
        for node in self.topo.nodes.values():
            node.neighbors = set()

        ids = list(self.topo.nodes.keys())
        # ---- spatial KNN ----
        poses: Dict[str, tuple] = {}
        for nid in ids:
            if nid in self.pose_graph.nodes:
                pn = self.pose_graph.nodes[nid]
                poses[nid] = (pn.x, pn.y)
        # Cross-gap filter: reject edges between nodes whose timestamps differ
        # by more than CROSS_GAP_MS and have no bridging keyframe between them.
        ts_map: Dict[str, int] = {}
        for nid in ids:
            try:
                ts_map[nid] = int(self.topo.nodes[nid].timestamp)
            except Exception:
                pass
        CROSS_GAP_MS = self.cfg.topology_cross_gap_ms
        def _cross_gap(a: str, b: str) -> bool:
            ta, tb = ts_map.get(a), ts_map.get(b)
            if ta is None or tb is None:
                return False
            dt = abs(ta - tb)
            if dt <= CROSS_GAP_MS:
                return False
            # timestamp 差大, 检查中间是否有其他 node 作为 bridging keyframe
            lo, hi = min(ta, tb), max(ta, tb)
            bridged = any(lo < ts < hi for kid, ts in ts_map.items() if kid not in (a, b))
            return not bridged
        added_spatial = 0
        dropped_cross_gap = 0
        for nid in ids:
            if nid not in poses:
                continue
            x, y = poses[nid]
            dists = []
            for other in ids:
                if other == nid or other not in poses:
                    continue
                ox, oy = poses[other]
                d = ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5
                dists.append((d, other))
            dists.sort()
            for _, other in dists[:k_spatial]:
                if _cross_gap(nid, other):
                    dropped_cross_gap += 1
                    continue
                edge = (min(nid, other), max(nid, other))
                if edge not in self.topo.edges:
                    self.topo.add_edge(nid, other)
                    added_spatial += 1

        # ---- temporal KNN ----
        ordered = sorted(ids, key=lambda i: self.topo.nodes[i].frame_idx)
        added_temporal = 0
        for i, nid in enumerate(ordered):
            for j in range(1, k_temporal + 1):
                for nb_idx in (i - j, i + j):
                    if 0 <= nb_idx < len(ordered):
                        other = ordered[nb_idx]
                        edge = (min(nid, other), max(nid, other))
                        if edge not in self.topo.edges:
                            self.topo.add_edge(nid, other)
                            added_temporal += 1

        self.metrics["topology_rebuild"] = {
            "k_spatial": k_spatial, "k_temporal": k_temporal,
            "spatial_edges": added_spatial,
            "temporal_edges": added_temporal,
            "total_edges": len(self.topo.edges),
        }
        logger.info(f"[TopoRebuild] spatial={added_spatial} temporal={added_temporal} "
                    f"total={len(self.topo.edges)}")

    def _fill_temporal_neighbors(self):
        """Ensure each surviving node connects to its frame_idx-immediate
        predecessor and successor in the topology, for bidirectional walks.
        """
        if len(self.topo.nodes) < 2:
            return
        ordered = sorted(self.topo.nodes.keys(),
                         key=lambda i: self.topo.nodes[i].frame_idx)
        added = 0
        for i in range(len(ordered) - 1):
            a = ordered[i]; b = ordered[i + 1]
            edge = (min(a, b), max(a, b))
            if edge not in self.topo.edges:
                self.topo.add_edge(a, b)
                added += 1
        self.metrics.setdefault("temporal_edges_added", 0)
        self.metrics["temporal_edges_added"] += added
        logger.info(f"[TempNeighbors] added {added} edges to ensure prev/next connectivity")

    def _apply_node_alias(self, alias_map: Dict[str, str], names: Dict[str, Dict[str, str]]):
        """删除被合并的节点, 把它们的邻接关系转到 anchor 上。"""
        if not alias_map:
            return
        # 1. 重写 topo edges
        new_edges = set()
        for a, b in self.topo.edges:
            a2 = alias_map.get(a, a)
            b2 = alias_map.get(b, b)
            if a2 == b2:
                continue
            new_edges.add((min(a2, b2), max(a2, b2)))
        self.topo.edges = new_edges
        # 2. 邻接表重建
        for nid, node in self.topo.nodes.items():
            node.neighbors = set()
        for a, b in self.topo.edges:
            if a in self.topo.nodes and b in self.topo.nodes:
                self.topo.nodes[a].neighbors.add(b)
                self.topo.nodes[b].neighbors.add(a)
        # 3. 删除被合并的节点
        for old_id, anchor_id in alias_map.items():
            self.topo.nodes.pop(old_id, None)
            self.node_features.pop(old_id, None)
            self.node_frame_idx.pop(old_id, None)
            self.node_payloads.pop(old_id, None)
            names.pop(old_id, None)
            # 同步 anchor 的 frame_idx 到合并后的 node.frame_idx
            if anchor_id in self.topo.nodes:
                self.node_frame_idx[anchor_id] = self.topo.nodes[anchor_id].frame_idx
            # scene_graph
            self.scene_graph.scene_nodes.pop(old_id, None)
            for floor, rooms in self.scene_graph.floors.items():
                for room, lst in rooms.items():
                    if old_id in lst:
                        lst.remove(old_id)
            # pose_graph
            self.pose_graph.nodes.pop(old_id, None)
        # 4. pose_graph edges
        self.pose_graph.edges = [
            e for e in self.pose_graph.edges
            if e.a not in alias_map and e.b not in alias_map
        ]
        # 5. 清理空房间, 并按节点最终 room 重建 floors 索引: Pass B/C 改名
        # 后 (H座 → 11层电梯, 储物柜区 → 1号储物柜) 旧 room 还挂着空 ID list,
        # 新 room 又没在 floors 里. 这里清空再按当前节点状态重建.
        for floor in list(self.scene_graph.floors.keys()):
            self.scene_graph.floors[floor] = {}
        for nid, sn in list(self.scene_graph.scene_nodes.items()):
            node = self.topo.nodes.get(nid)
            current_room = (node.room or sn.room) if node else sn.room
            if not current_room:
                current_room = "unknown"
            sn.room = current_room
            self.scene_graph.floors.setdefault("F1", {}).setdefault(current_room, []).append(nid)
        logger.info(f"[NameDedup] alias applied: removed {len(alias_map)} nodes "
                    f"({alias_map}); scene_graph floors rebuilt")

    # ------------------------------------------------------------------
    def _patch_node_with_nexts(self, node_dir: Path, nps, name_self, all_names):
        """将 ConnectionBuilder 输出的 next_positions 写回 node_position_info.json"""
        info_path = node_dir / "node_position_info.json"
        if not info_path.exists():
            return
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)

        out_nps = []
        for np_ in nps:
            np_ = dict(np_)
            np_.pop("_match_sim", None)
            nid = np_.get("position_id")
            if nid in all_names:
                np_["position_name"] = all_names[nid]["cn"]
                np_["position_name_eng"] = all_names[nid]["en"]
            for k in ("big_box", "mid_box", "small_box",
                      "landmark_name", "landmark_name_eng",
                      "position_name", "position_name_eng",
                      "camera_name", "position_id", "pixel_box"):
                np_.setdefault(k, "")
            out_nps.append(np_)
        info["next_positions"] = out_nps
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    def _generate_names(self):
        # 优先从结构化 name_struct 渲染 display name; 兼容旧 position_name
        out = {}
        for nid, node in self.topo.nodes.items():
            ns = getattr(node, "name_struct", None)
            if ns is not None and (ns.category or ns.organization):
                out[nid] = {"cn": ns.display_cn(), "en": ns.display_en()}
                continue
            cn = getattr(node, "position_name", None) or node.landmark_name or f"节点_{nid}"
            en = getattr(node, "position_name_eng", None) or ""
            out[nid] = {"cn": cn, "en": en}
        return out

