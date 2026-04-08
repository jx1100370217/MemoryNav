"""OnlineMapperCore — 主流式建图循环 (v2)

主要改进:
- 真实 VO (ORB + EssentialMatrix + 深度 scale) 替换常速代理
- Qwen 语义命名 (use_qwen=True), 接 vLLM 8199
- DoorPlateTracker 跨帧累积门牌检测, 选 bbox 最大帧作代表
- ConnectionBuilder (offline_mapper.AutoSubImageExtractor 子类) 生成真实
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
from online_mapper.geometry.visual_odometry import MonoVO
from online_mapper.topology.graph import TopoGraph, TopoNode
from online_mapper.topology.keyframe_selector import KeyframeSelector
from online_mapper.topology.loop_closure import LoopCloser
from online_mapper.topology.frontier_nbv import FrontierNBV
from online_mapper.semantics.open_set_detector import OpenSetDetector
from online_mapper.semantics.scene_graph import SceneGraph, SceneObject
from online_mapper.semantics.door_plate_tracker import DoorPlateTracker, PlateObservation
from online_mapper.semantics.hallucination_filter import (
    QwenVerifier, MultiFrameVoter, NameVote, NameDeduplicator,
    STRICT_DETECT_TEXT_PROMPT,
)
from online_mapper.semantics.node_category import (
    NodeCategoryClassifier, NodeCategory, JunctionKind, CategoryDecision,
    cn_to_en,
)
from online_mapper.semantics.colocation_merger import ColocationMerger
from online_mapper.geometry.junction_detector import JunctionDetector
from online_mapper.semantics import semantic_dedup
from online_mapper.output.merged_data_writer import MergedDataWriter

logger = logging.getLogger(__name__)

CAM_IDS = ['camera_1', 'camera_2', 'camera_3', 'camera_4']


class OnlineMapperCore:
    def __init__(self, cfg: OnlineMapperConfig):
        self.cfg = cfg
        self.loader = StreamLoader(cfg.input_dir)

        # VPR
        from offline_mapper.node_distance_estimator import NodeDistanceEstimator
        self.vpr = NodeDistanceEstimator(cfg.vpr_config_path,
                                         similarity_threshold=cfg.vpr_dissim_threshold,
                                         min_frame_interval=cfg.min_keyframe_frame_interval)

        # geometry
        self.depth = build_depth_estimator(cfg) if cfg.enable_depth else None
        self.vo = MonoVO() if cfg.enable_real_vo else None
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
                from offline_mapper.auto_landmark_namer import AutoLandmarkNamer
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
        t0 = time.time()
        for frame in self.loader:
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
            if depth_map is not None:
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
            # 这样即使语义合并不新建 node, 也能记录 loop closure
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
                        # 只记录最近的(避免每帧重复)
                        if not getattr(self, "_last_lc_frame", -100) or \
                                fidx - getattr(self, "_last_lc_frame", -100) >= 3:
                            self.metrics["n_loop_closures"] += 1
                            self._last_lc_frame = fidx
                            log_lc = {"frame_idx": fidx, "matches": verified_lc}
                            if not hasattr(self, "_lc_log"):
                                self._lc_log = []
                            self._lc_log.append(log_lc)
                            logger.info(f"[LoopClose] frame {fidx} -> {verified_lc}")
                            # 加边 (从最近 keyframe node 到匹配旧 node)
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

            # ---- door-plate retrospective: 在每帧 (不仅 keyframe) 上都跑 GD 找门牌 ----
            if self.door_tracker is not None and self.detector.gd_available:
                # 仅做轻量门牌候选 (限定 query) 以控成本
                self._scan_door_plates(frame, fidx)

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
                    # ---- 路口检测 ----
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

                    # ---- 已确认的 plate 名 (本帧 + 投票通过) ----
                    confirmed_plate = None
                    plate_text = None
                    plate_verified = False
                    for name in self._frame_plate_hits.get(fidx, set()):
                        if self.plate_voter.is_confirmed(name):
                            confirmed_plate = name
                            plate_text = name
                            plate_verified = True
                            break

                    # ---- describe_scene 候选 + verify ----
                    scene_describe = None
                    scene_verified = False
                    if self.namer is not None and self.namer._qwen_server:
                        try:
                            front = cv2.imread(frame["cameras"]["camera_1"])
                            if front is not None:
                                r = self.namer._qwen_server.describe_scene(self._b64(front))
                                if r.get("status") == "ok":
                                    cand = (r.get("name_cn") or "").strip()
                                    if cand and cand not in ("未知", "未知位置"):
                                        scene_describe = cand
                                        if (self.verifier
                                                and self.verifier.available
                                                and self.verifier.verify_scene(front, cand)):
                                            scene_verified = True
                        except Exception as e:
                            logger.debug(f"scene desc fail: {e}")

                    # ---- 分类器决策 ----
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
                    }
                    if decision.category == NodeCategory.REJECT:
                        self.metrics.setdefault("kf_rejected_by_category", 0)
                        self.metrics["kf_rejected_by_category"] += 1
                        logger.info(f"[KF-REJECT] frame {fidx}: {decision.reason}")
                        self.kf_selector.reset(fidx)
                        self.last_kf_features = feats  # 仍更新 vpr last
                        self.log_lines.append(log_entry)
                        continue

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
                    # 把分类决策的最终名挂在 node 上 (使用 attribute monkeypatch)
                    node.position_name = decision.final_name_cn
                    node.position_name_eng = decision.final_name_en
                    node.category = decision.category.value
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

                    # 闭环已在帧级别检测 (见上方主动全局闭环段)

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

        self.metrics["runtime_s"]["total"] = time.time() - t0
        self.metrics["n_nodes"] = len(self.topo.nodes)
        self.metrics["n_edges"] = len(self.topo.edges)
        self._finalize()

    # ------------------------------------------------------------------
    def _scan_door_plates(self, frame, fidx):
        """每帧扫描门牌候选 (严格 prompt + 二次验证 + 多帧投票)"""
        if self.namer is None or not self.namer._qwen_server:
            return
        import json as _json, re as _re
        for cam_id in CAM_IDS:
            cam_path = frame["cameras"].get(cam_id)
            if not cam_path:
                continue
            img = cv2.imread(cam_path)
            if img is None:
                continue
            dets = self.detector.detect(img, queries=["door plate", "room number sign"])
            for d in dets:
                if d["score"] < self.cfg.door_plate_min_score:
                    continue
                try:
                    x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                    h, w = img.shape[:2]
                    bw = x2 - x1; bh = y2 - y1
                    # 自适应 margin: 小 bbox 给大相对扩边, 大 bbox 给小绝对扩边
                    mx = max(20, int(bw * 0.6))
                    my = max(20, int(bh * 0.6))
                    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
                    x2 = min(w, x2 + mx); y2 = min(h, y2 + my)
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    # 如果扩边后仍然太小 (<300px short side), 直接用全图
                    short_side = min(crop.shape[0], crop.shape[1])
                    if short_side < 300:
                        crop_for_qwen = img
                    else:
                        crop_for_qwen = crop
                    # --- 严格 prompt: 要求 confidence, 不确定返回 false ---
                    raw = self.namer._qwen_server._chat(
                        STRICT_DETECT_TEXT_PROMPT, self._b64(crop_for_qwen), max_tokens=120)
                    raw_clean = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
                    m = _re.search(r'\{.*\}', raw_clean, flags=_re.DOTALL)
                    if not m:
                        continue
                    try:
                        obj = _json.loads(m.group())
                    except Exception:
                        continue
                    if not obj.get("found"):
                        continue
                    text = (obj.get("text") or "").strip()
                    name_cn = (obj.get("name_cn") or "").strip()
                    name_en = (obj.get("name_en") or "").strip()
                    confidence = (obj.get("confidence") or "").lower()
                    if confidence == "low":
                        self.metrics.setdefault("plate_drops_low_conf", 0)
                        self.metrics["plate_drops_low_conf"] += 1
                        continue
                    if not (text or name_cn):
                        continue
                    # --- 二次验证: 在整张相机图上问 "是否真的有文字 text?" ---
                    # (用整图而不是 crop, 避免 crop 里其他文字混淆)
                    # 注意: high confidence 的候选可以跳过 verify, 让多帧投票决定
                    verify_claim = text or name_cn
                    skip_verify = (confidence == "high")
                    if (not skip_verify) and self.verifier and self.verifier.available:
                        ok = self.verifier.verify_text(img, verify_claim)
                        if not ok:
                            self.metrics.setdefault("plate_drops_verify", 0)
                            self.metrics["plate_drops_verify"] += 1
                            logger.debug(f"[DoorPlate-VERIFY-DROP] "
                                         f"fidx={fidx} cam={cam_id} "
                                         f"text='{text}' name_cn='{name_cn}'")
                            continue
                except Exception as e:
                    logger.debug(f"plate scan fail: {e}")
                    continue

                # 通过严格 prompt + 二次验证, 进入投票 + 追踪
                # 优先使用原始 text (避免 Qwen 把 DEEPROUTE.AI 翻译成 "店铺招牌"
                # 这种通用词丢失 brand identity)
                import re as _re2
                if text and _re2.fullmatch(r"[A-Za-z][A-Za-z0-9\.\- &']{3,30}", text):
                    vote_name = text
                else:
                    vote_name = name_cn or text
                self.plate_voter.add(NameVote(
                    name=vote_name, frame_idx=fidx,
                    camera=cam_id, area=float((x2 - x1) * (y2 - y1)),
                    confidence=confidence or "medium",
                ))
                # 记录每帧 hits, 给 keyframe 分类器使用
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
            self.writer.write_node(
                node_id=nid, timestamp=node.timestamp,
                cameras=node.cameras,
                position_name=names[nid]["cn"],
                position_name_eng=names[nid]["en"],
                next_positions=[],  # 先空, 后面 ConnectionBuilder 填
            )

        # 4. ConnectionBuilder: 真实 next_positions
        if self.cfg.enable_real_connections:
            try:
                from online_mapper.topology.connection_builder import ConnectionBuilder
                cb = ConnectionBuilder(
                    sim_threshold=self.cfg.connection_sim_threshold,
                    qwen_gpu=self.cfg.qwen_gpu, namer=self.namer)
                node_dirs = {nid: out_root / nid for nid in self.topo.nodes}
                for nid, node in self.topo.nodes.items():
                    nbs = [self.topo.nodes[nbid] for nbid in node.neighbors
                           if nbid in self.topo.nodes]
                    if not nbs:
                        continue
                    nps = cb.build_for_node(node, nbs, node_dirs[nid], node_dirs,
                                             self._all_frames_cache)
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

        logger.info(f"Online mapping complete: {self.metrics}")

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
        added_spatial = 0
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
        logger.info(f"[NameDedup] alias applied: removed {len(alias_map)} nodes "
                    f"({alias_map})")

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
