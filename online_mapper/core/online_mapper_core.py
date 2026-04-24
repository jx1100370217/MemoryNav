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

sys.path.insert(0, '/home/ubuntu/Disk/codes/jianxiong/MemoryNav')

from online_mapper.config import OnlineMapperConfig
from online_mapper.core.stream_loader import StreamLoader
from online_mapper.geometry.depth_estimator import DepthEstimator, build_depth_estimator
from online_mapper.geometry.pose_graph import PoseGraph
from online_mapper.geometry.occupancy import OccupancyGrid
from online_mapper.geometry.visual_odometry import MonoVO, build_visual_odometry
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
    cn_to_en, FUNCTION_AREA_WHITELIST, LANDMARK_FACILITY_WHITELIST,
    BUILDING_LANDMARK_PATTERNS,
)
from online_mapper.semantics.colocation_merger import ColocationMerger
from online_mapper.geometry.junction_detector import JunctionDetector
from online_mapper.semantics import semantic_dedup
from online_mapper.io.merged_data_writer import MergedDataWriter

logger = logging.getLogger(__name__)

CAM_IDS = ['camera_1', 'camera_2', 'camera_3', 'camera_4']


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
        self.junction_detector = JunctionDetector(self.depth) if self.depth else None
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

        tv = time.time()
        feats = self._extract_features(frame["cameras"])
        self.metrics["runtime_s"]["vpr"] += time.time() - tv

        sim_to_last = None
        if self.last_kf_features is not None:
            sim_to_last = self._vpr_sim(feats, self.last_kf_features)

        # 主动全局闭环检测 (每帧, 不依赖 keyframe 创建)
        if len(self.node_features) >= 2:
            lc = self.loop_closer.detect(
                "_current_frame", feats, self.node_features,
                self.node_frame_idx, fidx)
            if lc:
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
                if verified_lc:
                    if not getattr(self, "_last_lc_frame", -100) or \
                            fidx - getattr(self, "_last_lc_frame", -100) >= 3:
                        self.metrics["n_loop_closures"] += 1
                        self._last_lc_frame = fidx
                        log_lc = {"frame_idx": fidx, "matches": verified_lc}
                        if not hasattr(self, "_lc_log"):
                            self._lc_log = []
                        self._lc_log.append(log_lc)
                        logger.info(f"[LoopClose] frame {fidx} -> {verified_lc}")
                        if self.last_kf_node_id:
                            for old_id, sim in verified_lc[:1]:
                                if old_id != self.last_kf_node_id:
                                    self.topo.add_edge(self.last_kf_node_id, old_id)
                                    self.pose_graph.add_edge(
                                        self.last_kf_node_id, old_id,
                                        0.0, 0.0, 0.0, info=sim, kind="loop")

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

        # 门牌扫描节流: 关键帧必扫; 否则每 N 帧扫一次 (大幅缓解高 VPR 相似段的延迟)
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

        if triggered:
            self.metrics["n_keyframes_triggered"] += 1

            td = time.time()
            cam_objects = {}
            for c in CAM_IDS:
                img = cv2.imread(frame["cameras"][c])
                if img is None:
                    continue
                dets = self.detector.detect(img)
                cam_objects[c] = dets
            self.metrics["runtime_s"]["detect"] += time.time() - td

            room = ""
            landmark = ""
            best_obj_score = 0
            for c, dets in cam_objects.items():
                for d in dets:
                    if d["score"] > best_obj_score:
                        best_obj_score = d["score"]
                        landmark = d["label"]
                    if "door" in d["label"].lower() or "plate" in d["label"].lower():
                        room = d["label"]

            merge_target = semantic_dedup.find_merge_target(
                room, landmark, feats, self.topo.nodes, self.node_features)

            if merge_target:
                log_entry["semantic_merge_into"] = merge_target
                self.metrics["n_semantic_merges"] += 1
                if self.last_kf_node_id and self.last_kf_node_id != merge_target:
                    self.topo.add_edge(self.last_kf_node_id, merge_target)
            else:
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

                plate_text = None
                plate_verified = False
                hits = list(self._frame_plate_hits.get(fidx, set()))
                def _hit_priority(name):
                    if not self.plate_voter.is_confirmed(name):
                        return -1
                    return 0 if is_brand_like(name) else 1
                hits.sort(key=_hit_priority, reverse=True)
                for name in hits:
                    if self.plate_voter.is_confirmed(name):
                        plate_text = name
                        plate_verified = True
                        break

                scene_describe = None
                scene_verified = False
                if self.namer is not None and self.namer._qwen_server:
                    try:
                        # Multi-cam scene recognition: query Qwen on each of
                        # the 4 cameras, then vote on canonical name.
                        scene_cands = []
                        verify_img_map = {}
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
                                    verify_img_map[cand] = img

                        def _canonicalize(name: str) -> str:
                            """Normalize raw scene name to canonical form so
                            synonyms collapse into one vote (e.g. '打印机房'
                            and '打印机室' both map to '打印区')."""
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

                        if scene_cands:
                            # Each cam votes once on its canonical name.
                            # Require ≥2 cam consensus; tie-break by
                            # specificity rank.
                            canonical_cands = [(c, _canonicalize(n)) for c, n in scene_cands]
                            from collections import Counter
                            vote_cnt = Counter(n for _, n in canonical_cands)
                            top = vote_cnt.most_common()
                            top_count = top[0][1]
                            if top_count >= 2:
                                top_names = [n for n, c in top if c == top_count]
                                top_names.sort(key=_specificity_rank)
                                winner = top_names[0]
                                # Whitelisted categories (FUNCTION_AREA,
                                # LANDMARK_FACILITY) bypass temporal check —
                                # their matched keywords are high-trust signals.
                                # Other categories (BUILDING_LANDMARK, generic
                                # names) require a preceding keyframe to have
                                # voted the same canonical to guard against
                                # multi-cam joint hallucination.
                                in_strong_wl = any(kw in winner for kw in FUNCTION_AREA_WHITELIST) \
                                    or any(kw in winner for kw in LANDMARK_FACILITY_WHITELIST)
                                if in_strong_wl or winner in self._recent_scene_winners:
                                    scene_describe = winner
                                    scene_verified = True
                                    logger.info(f"[MultiCamScene] fidx={fidx} "
                                                f"cands={scene_cands} votes={dict(vote_cnt)} "
                                                f"→ '{scene_describe}' (consensus={top_count}/"
                                                f"{len(canonical_cands)})")
                                else:
                                    scene_describe = None
                                    scene_verified = False
                                    logger.info(f"[MultiCamScene] fidx={fidx} "
                                                f"cands={scene_cands} votes={dict(vote_cnt)} "
                                                f"→ tentative '{winner}' "
                                                f"(awaits temporal confirm)")
                                self._recent_scene_winners.append(winner)
                            else:
                                scene_describe = None
                                scene_verified = False
                                self._recent_scene_winners.append(None)
                                logger.info(f"[MultiCamScene] fidx={fidx} "
                                            f"cands={scene_cands} votes={dict(vote_cnt)} "
                                            f"→ NO CONSENSUS (all cams differ), skip node")
                    except Exception as e:
                        logger.debug(f"multi-cam scene desc fail: {e}")

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
                    return log_entry

                self.metrics.setdefault("kf_accepted_by_category", {})
                cat_key = decision.category.value
                self.metrics["kf_accepted_by_category"][cat_key] = \
                    self.metrics["kf_accepted_by_category"].get(cat_key, 0) + 1

                nid = str(self.next_node_id)
                self.next_node_id += 1
                node = TopoNode(node_id=nid, timestamp=frame["timestamp"],
                                frame_idx=fidx, cameras=dict(frame["cameras"]),
                                landmark_name=decision.final_name_cn or landmark,
                                room=decision.final_name_cn or "unknown")
                node.position_name = decision.final_name_cn
                node.position_name_eng = decision.final_name_en
                node.category = decision.category.value
                if decision.category == NodeCategory.SHOP:
                    node.name_struct = NodeName(
                        organization=decision.final_name_cn,
                    )
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

                best, top5 = self.nbv.score_and_pick(
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

            self.kf_selector.reset(fidx)

        self.log_lines.append(log_entry)
        return log_entry

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
        ATTACH_GAP = 12
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

        # 1. 创建门牌 semantic nodes (基于 best frame)
        self._create_door_plate_nodes()
        self.metrics["n_nodes"] = len(self.topo.nodes)

        # 1.5 Co-location merge: 同物理位置的不同 category node 合并
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
        self._rebuild_topology_neighbors_spatial(k_spatial=1, k_temporal=1)

        # 2. 命名 (从 node attribute 直接读)
        names = self._generate_names()

        # 2.5 名称去重: 合并同名且 VPR 相似的节点, 其余加后缀
        dedup = NameDeduplicator(merge_vpr_threshold=0.78)
        names, alias_map = dedup.resolve(names, self.topo.nodes, self.node_features)
        self.metrics["name_dedup"] = dict(dedup.stats)
        if alias_map:
            self._apply_node_alias(alias_map, names)
        self.metrics["n_nodes"] = len(self.topo.nodes)
        self.metrics["n_edges"] = len(self.topo.edges)

        # 3. 写出 base node 目录
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
            try:
                from online_mapper.topology.connection_builder import ConnectionBuilder
                cb = ConnectionBuilder(
                    sim_threshold=self.cfg.connection_sim_threshold,
                    qwen_gpu=self.cfg.qwen_gpu, namer=self.namer,
                    depth_estimator=self.depth,
                    detector=self.detector)
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

        # 5. scene_graph / pose_graph / log / metrics
        self.scene_graph.save(str(out_root.parent / "scene_graph.json"))
        with open(out_root.parent / "pose_graph.json", "w", encoding="utf-8") as f:
            json.dump(self.pose_graph.to_dict(), f, ensure_ascii=False, indent=2)
        with open(out_root.parent / "online_mapping_log.jsonl", "w", encoding="utf-8") as f:
            for line in self.log_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.metrics["loop_threshold_used"] = self.loop_closer.current_threshold()
        with open(out_root.parent / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)

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

        logger.info(f"Online mapping complete: {self.metrics}")

    # ------------------------------------------------------------------
    def _merge_by_canonical_name(self) -> Dict[str, str]:
        """同 base building 的 BUILDING_LANDMARK 节点 + 同 canonical name 的
        LANDMARK_FACILITY 节点强制合并; 并把"复合 plate"(如 'H座电梯', '2号外卖柜')
        作为更具体的 display name 优先采用.

        BUILDING_LANDMARK 的 base 提取规则:
          'C座' / 'C座入口' / 'C座大堂' → base 'C座' (全归同栋楼)
          '13号楼' / '13号楼入口' → base '13号楼'
          'D栋' / 'D栋入口' → base 'D栋'

        anchor 选: 离 plate-best frame 最近的 node, 并把 timestamp/cameras 搬到
        bbox 最大那帧, 给用户看到最近最大的招牌视角.

        return: {sub_id -> anchor_id} alias map, 调用方传给 _apply_node_alias.
        """
        from online_mapper.semantics.node_category import NodeCategory
        import re as _re
        alias: Dict[str, str] = {}

        # base building 提取: '<X>座/楼/栋' 取裸名, 去掉 入口/大堂 后缀
        base_re = _re.compile(r"^([A-Za-z0-9]+(?:号)?(?:座|楼|栋))(?:入口|大堂|电梯)?$")
        def _building_base(name: str) -> str:
            if not name: return ""
            m = base_re.match(name.strip())
            return m.group(1) if m else name.strip()

        def _node_canon(node) -> str:
            ns = getattr(node, "name_struct", None)
            if ns and (ns.category or ns.organization):
                base = ns.category or ns.organization
            else:
                base = (getattr(node, "position_name", "") or "").split("·")[0]
            return base.strip()

        # ---- Pass A: 同 base building / 同 canonical 强制合并 ----
        groups: Dict[tuple, List[str]] = {}
        for nid, node in self.topo.nodes.items():
            cat = getattr(node, "category", "") or ""
            if cat not in (NodeCategory.BUILDING_LANDMARK.value,
                           NodeCategory.LANDMARK_FACILITY.value):
                continue
            name = _node_canon(node)
            if not name:
                continue
            # building 用 base 收纳; landmark facility 用原 canonical
            key_name = (_building_base(name)
                        if cat == NodeCategory.BUILDING_LANDMARK.value else name)
            groups.setdefault((cat, key_name), []).append(nid)

        def _best_obs_for_group(name: str, alias_names: List[str]):
            """选该 group 的代表帧.
            优先级:
              1. 带 '入口' 的 variant (X座入口) — 机器人站在入口前, 位置语义最具体
              2. 带 '大堂' 的 variant
              3. 裸 base name (X座)
            再在选定的 variant 里取 plate bbox 面积最大那帧.
            (不能无脑跨 variant 取 max area, 因为远看整栋楼 '座' 字 bbox 也大,
             会选到机器人已经走过入口后的帧.)
            用 door_tracker + voter 双来源: door_tracker 有 timestamp/cameras,
            voter 更可靠地 index 了 plate name (voter 用的 vote_name 可能与
            tracker key 不一致).
            """
            all_names = [name] + [n for n in alias_names if n != name]
            entries = [n for n in all_names if "入口" in n]
            lobbies = [n for n in all_names if "大堂" in n]
            bases   = [n for n in all_names if n not in entries and n not in lobbies]

            def _resolve(group_names):
                best_obs, best_area = None, 0.0
                for n in group_names:
                    # 先试 tracker (有 timestamp/cameras 信息)
                    if self.door_tracker:
                        obs = self.door_tracker.best(n)
                        if obs is not None and obs.area > best_area:
                            best_obs, best_area = obs, obs.area
                    # 再查 voter, 如果 tracker 没这 key 或 area 较小
                    votes = self.plate_voter.votes_for(n)
                    if votes:
                        vbest = max(votes, key=lambda v: v.area or 0.0)
                        if (vbest.area or 0.0) > best_area:
                            # 从 tracker 找该 (frame, cam) 的 obs 拿 timestamp
                            best_obs_fallback = None
                            if self.door_tracker:
                                for nn in group_names:
                                    for obs in self.door_tracker._observations.get(nn, []):
                                        if obs.frame_idx == vbest.frame_idx:
                                            best_obs_fallback = obs
                                            break
                                    if best_obs_fallback: break
                            if best_obs_fallback is not None:
                                best_obs, best_area = best_obs_fallback, vbest.area or 0.0
                            elif best_obs is None:
                                # 没 tracker obs 就用 voter 帧手工构造 obs
                                from online_mapper.semantics.door_plate_tracker import PlateObservation
                                # 从 log_lines 找该 frame 的 timestamp/cameras
                                ts, cams = None, {}
                                for ln in self.log_lines:
                                    if ln.get("frame_idx") == vbest.frame_idx:
                                        ts = ln.get("ts")
                                        break
                                # 再从 _all_frames_cache (finalize 阶段常存在) 取 cameras
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
                                        bbox=[0,0,0,0], score=0.0, text=n,
                                        name_cn=n, name_en="",
                                        pose=(0.0,0.0,0.0), area=vbest.area or 0.0)
                                    best_area = vbest.area or 0.0
                return best_obs

            for group in (entries, lobbies, bases):
                obs = _resolve(group)
                if obs is not None:
                    return obs
            return None

        for (cat, base_name), ids in groups.items():
            if len(ids) < 2:
                continue
            # 同组内的实际 plate 名变体集合 (X座 + X座入口 + X座大堂 ...)
            variants = list({_node_canon(self.topo.nodes[i]) for i in ids})
            best_obs = _best_obs_for_group(base_name, variants)
            if best_obs is not None:
                anchor_id = min(ids, key=lambda i: abs(
                    self.topo.nodes[i].frame_idx - best_obs.frame_idx))
            else:
                anchor_id = min(ids, key=lambda i: self.topo.nodes[i].frame_idx)
            # 把 anchor 改名成 base name (统一 X座入口/X座大堂 → X座)
            if cat == NodeCategory.BUILDING_LANDMARK.value:
                target = self.topo.nodes[anchor_id]
                ns = getattr(target, "name_struct", None)
                if ns is not None:
                    ns.category = base_name
                    ns.category_en = base_name
                target.position_name = base_name
                target.position_name_eng = base_name
                target.room = base_name
            for sid in ids:
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
            logger.info(f"[SameName] '{base_name}' [{cat}] variants={variants}: "
                        f"keep {anchor_id}, absorb {[i for i in ids if i != anchor_id]}")

        # ---- 工具: 直接遍历 voter 找 plate_text 的代表帧 ----
        # 注意: door_tracker 与 plate_voter 用不同 key (前者 text 优先, 后者
        # name_cn 优先). 复合 plate (H座电梯) 大概率只在 voter 里, tracker 里
        # 是裸 'H座'. 因此 Pass B/C 必须以 voter._votes 为来源.
        def _best_frame_for_voter(plate_text: str):
            votes = self.plate_voter.votes_for(plate_text)
            if not votes:
                return None, None
            best = max(votes, key=lambda v: v.area or 0)
            return best.frame_idx, best.area or 0.0

        # ---- Pass B: 复合 plate 优先, 重命名节点 (H座 -> H座电梯) ----
        # 按 specificity 排序, 每个 anchor 只被 rename 一次, 挑最具体的复合 plate:
        #   X座电梯 / X座楼梯  (楼栋级)         -- 最具体
        #   X号楼电梯 / X栋电梯
        #   X层电梯   (仅楼层, 不知道是哪栋)   -- 最宽泛
        prefix_pat = _re.compile(
            r"^([A-Za-z]|[0-9]{1,3}号|[0-9]{1,3}层|[0-9]{1,3})(座|楼|栋)?(电梯|楼梯)$")

        def _composite_specificity(plate_text: str) -> int:
            """越大越具体; 优先分配."""
            m = prefix_pat.fullmatch(plate_text)
            if not m: return 0
            prefix, suffix_type, kind = m.group(1), m.group(2), m.group(3)
            s = 0
            if suffix_type in ("座", "栋"):    # 明确楼栋
                s += 30
            elif suffix_type == "楼":            # X号楼 明确楼栋
                s += 25
            elif "层" in prefix:                # 仅楼层
                s += 10
            else:                                # 裸数字 (e.g. 11电梯) 最差
                s += 5
            # 字母楼栋 (A-Z) 比数字更稳 (楼栋编号常是字母)
            if len(prefix) == 1 and prefix.isalpha():
                s += 3
            return s

        candidates_B = []
        for plate_text in list(self.plate_voter._votes.keys()):
            if not prefix_pat.fullmatch(plate_text or ""):
                continue
            if not self.plate_voter.is_confirmed(plate_text):
                continue
            target_frame, area = _best_frame_for_voter(plate_text)
            if target_frame is None:
                continue
            candidates_B.append((plate_text, target_frame, area,
                                  _composite_specificity(plate_text)))
        # 高 specificity 优先; 同级按 area 降序
        candidates_B.sort(key=lambda x: (-x[3], -x[2]))
        anchored = set()
        for plate_text, target_frame, _area, _spec in candidates_B:
            cands = [
                nid for nid, n in self.topo.nodes.items()
                if nid not in alias and nid not in anchored
                and (getattr(n, "category", "") in (
                    NodeCategory.BUILDING_LANDMARK.value,
                    NodeCategory.LANDMARK_FACILITY.value))
                and abs(n.frame_idx - target_frame) <= 12
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
            # 搬 timestamp/frame/cameras 到 plate 最大 bbox 那帧, 给用户最近视角.
            # 优先用 door_tracker (有 timestamp/cameras), 否则从 _all_frames_cache
            # 按 voter 的 best vote frame_idx 反查 (door_tracker 可能以 text 单字母
            # 为 key 找不到复合 plate, e.g. 'H' vs 'H座电梯').
            votes = self.plate_voter.votes_for(plate_text)
            best_v = max(votes, key=lambda v: v.area or 0) if votes else None
            obs = None
            if self.door_tracker:
                obs = self.door_tracker.best(plate_text)
            new_frame = None; new_ts = None; new_cams = None
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

        # ---- Pass C: 数字+柜 复合 plate (2号柜/1号柜) 命名外卖柜区/储物柜区 ----
        cabinet_pat = _re.compile(r"^([0-9]{1,3})号柜$")
        cabinet_categories = {
            "外卖柜区": "外卖柜",
            "储物柜区": "储物柜",
            "快递柜区": "快递柜",
        }
        # 辅助: 返回帧 f 附近 (±window) 内某 plate 的最大 area
        def _nearest_plate_area(plate_name: str, f: int, window: int = 12) -> float:
            votes = self.plate_voter.votes_for(plate_name)
            if not votes:
                return 0.0
            near = [v for v in votes if abs(v.frame_idx - f) <= window]
            if not near: return 0.0
            return max((v.area or 0.0) for v in near)

        # 外卖柜 vs 储物柜 判断: 哪个 plate 在该 N 号柜附近 frame 里出现的 area 更大,
        # 就是哪种柜. Qwen 的 scene_describe '储物柜区' / '外卖柜区' 不总准(Qwen 把
        # 外卖柜帧读成"储物柜区"曾发生), 用 plate 实证更可靠.
        KIND_PLATES = {"外卖柜": "外卖柜", "储物柜": "储物柜", "快递柜": "快递柜"}
        KIND_TO_SHORT = {"外卖柜": "外卖柜", "储物柜": "储物柜", "快递柜": "快递柜"}
        KIND_TO_ROOM = {"外卖柜": "外卖柜区", "储物柜": "储物柜区", "快递柜": "快递柜区"}

        for plate_text in list(self.plate_voter._votes.keys()):
            m = cabinet_pat.fullmatch(plate_text or "")
            if not m: continue
            if not self.plate_voter.is_confirmed(plate_text): continue
            target_frame, _ = _best_frame_for_voter(plate_text)
            if target_frame is None: continue
            num = m.group(1)
            cands = [
                nid for nid, n in self.topo.nodes.items()
                if nid not in alias
                and (getattr(n, "position_name", "") or "").split("·")[0] in cabinet_categories
                and abs(n.frame_idx - target_frame) <= 16
            ]
            if not cands: continue
            anchor = min(cands, key=lambda i: abs(
                self.topo.nodes[i].frame_idx - target_frame))
            target = self.topo.nodes[anchor]
            base_canon = (target.position_name or "").split("·")[0]
            # 根据 N 号柜 plate 周围 confirmed 的 '外卖柜'/'储物柜'/'快递柜' plate
            # 选 kind. 偏好规则:
            #   - 专用命名 (2 字招牌 "外卖柜"/"快递柜") 读到的文字比 scene describe
            #     "储物柜"/"储物柜区" 更可信 (Qwen 有把外卖柜区当储物柜区的倾向).
            #   - 加 prior boost 给 '外卖柜'/'快递柜', 只有在它们在附近明显缺席
            #     (area 低于 10% of 储物柜) 时才选 '储物柜'.
            window = 12
            kind_scores = {}
            for kname in KIND_PLATES:
                if self.plate_voter.is_confirmed(kname):
                    a = _nearest_plate_area(kname, target_frame, window=window)
                    if a > 0:
                        kind_scores[kname] = a
            chosen_kind = None
            if kind_scores:
                outbox = kind_scores.get("外卖柜", 0)
                express = kind_scores.get("快递柜", 0)
                storage = kind_scores.get("储物柜", 0)
                # 若 '外卖柜' 或 '快递柜' 有近视证据 (area > 10000), 直接选它;
                # 两者都有则选 area 更大的. 否则退化到 '储物柜'.
                if outbox >= 10000 or express >= 10000:
                    chosen_kind = "外卖柜" if outbox >= express else "快递柜"
                else:
                    chosen_kind = max(kind_scores.items(), key=lambda kv: kv[1])[0]
            else:
                chosen_kind = base_canon.replace("区", "") if base_canon.endswith("区") else None
                if chosen_kind not in KIND_TO_SHORT:
                    chosen_kind = None
            if chosen_kind is None: continue
            short = KIND_TO_SHORT[chosen_kind]
            composite = f"{num}号{short}"
            ns = getattr(target, "name_struct", None)
            if ns is not None:
                ns.category = composite
                ns.category_en = composite
                # 清掉 EXHIOH 这种伪 brand (它本身就是 hallucinated plate)
                if ns.organization and not any(c.isascii() and c.isalnum()
                                                for c in ns.organization):
                    pass  # 中文 org 保留
                else:
                    # 英文 org 若与柜区无关 (EXHIOH/HIOF 等), drop
                    ns.organization = ""
            target.position_name = composite
            target.position_name_eng = composite
            target.room = composite
            logger.info(f"[CabinetPlate] '{plate_text}' near frame {target_frame} "
                        f"kind={chosen_kind} → '{composite}' on node {anchor} "
                        f"(was: {base_canon})")

        # ---- Pass D: 保安亭 ----
        # 策略 A: 若 scene 识别为 "岗亭"/"保安亭" 的帧(即使 reject 未建 node)
        # 存在, 且附近有 BUILDING_LANDMARK, 重命名该建筑节点为 "X座保安亭".
        # 策略 B: 若已建 FUNCTION_AREA=保安亭 节点, 合并到最近建筑.
        BOOTH_TOKENS = {"岗亭", "保安亭", "值班亭", "门岗"}
        booth_frames = []
        for ln in self.log_lines:
            cd = ln.get("category_decision") or {}
            sd = cd.get("scene_describe") or ""
            if sd in BOOTH_TOKENS:
                booth_frames.append(ln.get("frame_idx"))

        building_nodes = [
            (nid, n) for nid, n in self.topo.nodes.items()
            if nid not in alias
            and getattr(n, "category", "") == NodeCategory.BUILDING_LANDMARK.value
        ]
        booth_node_nodes = [
            (nid, n) for nid, n in self.topo.nodes.items()
            if nid not in alias
            and (getattr(n, "name_struct", None) is not None
                 and getattr(n.name_struct, "category", "") == "保安亭")
        ]

        used_buildings = set()
        # 先 case B (已有保安亭 node)
        for (bn_id, bn) in list(booth_node_nodes):
            if bn_id in alias: continue
            if not building_nodes: break
            nearest = min(building_nodes,
                          key=lambda p: abs(p[1].frame_idx - bn.frame_idx))
            if abs(nearest[1].frame_idx - bn.frame_idx) > 12: continue
            bld_name = (_node_canon(nearest[1]) or "").strip()
            if not bld_name: continue
            composite = f"{bld_name}保安亭"
            ns = getattr(bn, "name_struct", None)
            if ns is not None:
                ns.category = composite; ns.category_en = composite
                ns.organization = ""
            bn.position_name = composite
            bn.position_name_eng = composite
            bn.room = composite
            alias[nearest[0]] = bn_id
            used_buildings.add(nearest[0])
            logger.info(f"[Booth-B] merge {nearest[0]}({bld_name}) → {bn_id} ({composite})")

        # case A: scene 提示岗亭但没单独建节点; 直接 rename 最近 building
        for bf in booth_frames:
            if bf is None: continue
            cands = [
                (nid, n) for nid, n in self.topo.nodes.items()
                if nid not in alias and nid not in used_buildings
                and getattr(n, "category", "") == NodeCategory.BUILDING_LANDMARK.value
                and abs(n.frame_idx - bf) <= 10
            ]
            if not cands: continue
            nid_b, n_b = min(cands, key=lambda p: abs(p[1].frame_idx - bf))
            bld_name = (_node_canon(n_b) or "").strip()
            composite = f"{bld_name}保安亭"
            ns = getattr(n_b, "name_struct", None)
            if ns is not None:
                # 保留楼栋前缀, 把 base category 改成复合名; organization (如 HIOF) 可保留
                ns.category = composite
                ns.category_en = composite
            n_b.position_name = ns.display_cn() if ns else composite
            n_b.position_name_eng = ns.display_en() if ns else composite
            n_b.room = composite
            used_buildings.add(nid_b)
            logger.info(f"[Booth-A] scene=岗亭 frame {bf} → rename node {nid_b} '{bld_name}' → '{composite}'")

        # ---- Pass E: HIOF/EXHIOH 这类 'SHOP' 节点 brand-attach 到相近 FUNCTION_AREA ----
        # 如果 SHOP 节点名(organization) 与相邻非 SHOP FUNCTION 节点在 frame 差 ≤ 12,
        # 应当合并 SHOP 进 FUNCTION, 以 FUNCTION 为主名, SHOP 作 organization.
        for sid, snode in list(self.topo.nodes.items()):
            if sid in alias: continue
            if getattr(snode, "category", "") != NodeCategory.SHOP.value: continue
            sns = getattr(snode, "name_struct", None)
            if sns is None or not sns.organization: continue
            # 找相邻非 SHOP 节点
            nearby = [
                (fid, fn) for fid, fn in self.topo.nodes.items()
                if fid not in alias and fid != sid
                and getattr(fn, "category", "") in (
                    NodeCategory.FUNCTION_AREA.value,
                    NodeCategory.BUILDING_LANDMARK.value,
                    NodeCategory.LANDMARK_FACILITY.value,
                    NodeCategory.ROOM_NAMED.value,
                    NodeCategory.ROOM_NUMBERED.value)
                and abs(fn.frame_idx - snode.frame_idx) <= 12
            ]
            if not nearby: continue
            anchor_id, anchor_node = min(
                nearby, key=lambda p: abs(p[1].frame_idx - snode.frame_idx))
            ans = getattr(anchor_node, "name_struct", None)
            if ans is None: continue
            # 融合 organization
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

        # ---- Pass F: 所有 BUILDING_LANDMARK / 含保安亭/电梯/楼梯 的节点
        # 强制把 display 帧搬到其 canonical plate 的 best-area 帧 (优先 "入口"
        # 变体). Pass A 内部 fallback 对单字母 tracker key 不稳, 这里用 voter
        # 直接作最终 re-center, 确保用户看到离招牌最近的帧.
        def _final_recenter(node, canonical_base: str):
            """帧选择策略: 优先 multi-camera 可见帧 (说明机器人正面对地标),
            再退化到单 camera max area. 入口变体优先于裸 base.
            """
            relevant = set()
            for k in self.plate_voter._votes.keys():
                if not k: continue
                if k == canonical_base: relevant.add(k)
                elif k.startswith(canonical_base) and any(s in k for s in ("入口","大堂","电梯","楼梯")):
                    relevant.add(k)
            if not relevant: return

            from collections import defaultdict as _dd
            def _pick_best(keys):
                """在 keys 中找: 优先 ≥2 cam 帧, 再退化到单 cam max area."""
                frame_cams = _dd(set)
                frame_area = _dd(float)
                for k in keys:
                    for v in self.plate_voter.votes_for(k):
                        frame_cams[v.frame_idx].add(v.camera)
                        frame_area[v.frame_idx] = max(frame_area[v.frame_idx], v.area or 0)
                if not frame_area: return None
                multi = [fi for fi, cs in frame_cams.items() if len(cs) >= 2]
                if multi:
                    return max(multi, key=lambda fi: frame_area[fi])
                return max(frame_area.items(), key=lambda x: x[1])[0]

            entry_keys = [k for k in relevant if "入口" in k]
            base_keys = [k for k in relevant if k == canonical_base]
            all_keys = list(relevant)

            # 优先级链: entry multi-cam → base multi-cam → all multi-cam
            #         → entry single → base single
            def _pick_multi(keys):
                from collections import defaultdict as _dd2
                fc = _dd2(set); fa = _dd2(float)
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
                            best_a = v.area or 0; best_f = v.frame_idx
                return best_f

            target_frame = (_pick_multi(entry_keys) or _pick_multi(base_keys)
                            or _pick_multi(all_keys)
                            or _pick_single(entry_keys) or _pick_single(base_keys))
            if target_frame is None or target_frame == node.frame_idx:
                return
            # 从 _all_frames_cache 取 ts 与 cameras
            for fr in self._all_frames_cache:
                if fr.get("frame_idx") == target_frame:
                    node.frame_idx = target_frame
                    node.timestamp = fr.get("timestamp") or node.timestamp
                    cams = dict(fr.get("cameras") or {})
                    if cams: node.cameras = cams
                    # 重建 VPR features 以保持一致
                    nid = None
                    for k, v in self.topo.nodes.items():
                        if v is node: nid = k; break
                    if nid and nid in self.node_features and cams:
                        try:
                            self.node_features[nid] = self._extract_features(cams)
                            self.node_frame_idx[nid] = target_frame
                        except Exception: pass
                    logger.info(f"[Recenter] node -> frame {target_frame} ts={node.timestamp} "
                                f"(canonical={canonical_base}, via keys {list(relevant)[:3]})")
                    break

        import re as _re_f
        bld_base_re = _re_f.compile(r"^([A-Za-z0-9]+(?:号)?(?:座|楼|栋))")
        for nid, node in self.topo.nodes.items():
            if nid in alias: continue
            name = (getattr(node, "position_name", "") or "").split("·")[0]
            m = bld_base_re.match(name)
            if m:
                _final_recenter(node, m.group(1))

        return alias

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
        CROSS_GAP_MS = 60 * 1000  # 60s
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

    def _generate_names_legacy(self):
        """直接采用分类器在 keyframe 时已经决定的最终名.
        EN 字段确保是真正的英文 (走 cn_to_en map)."""
        names = {}
        for nid, node in self.topo.nodes.items():
            cn = getattr(node, "position_name", None) or node.landmark_name or f"节点_{nid}"
            en = getattr(node, "position_name_eng", None) or ""
            # 若 en 为空 / 与 cn 相同 / 含中文, 通过映射表翻译
            if (not en) or en == cn or any('\u4e00' <= c <= '\u9fff' for c in en):
                en = cn_to_en(cn) or f"node_{nid}"
            names[nid] = {"cn": cn, "en": en}
        return names
