"""RegionClusterer — 三阶段把 topo nodes 聚类到 regions (R2 重构).

阶段:
  1. 时序切段: 按 timestamp gap > _SEG_GAP_MS 切轨迹段 (跨段不直接合并)
  2. 段内聚类: DINO fused_feature + 空间距离 affinity → 谱聚类 (eigengap 选 k)
  3. 跨段同 plate 合并: 不同段中含同一 confirmed plate (dominant) 的子段合并

不修改节点本身 (vln_mem 风格), 只**分组**到 region. 每 region 保留多个 node.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# 时序切段阈值. 在数据集内连续段无大 gap; 数据采集中断 / 轨迹分段时跨阈值.
_SEG_GAP_MS_DEFAULT = 60_000


@dataclass
class RegionInfo:
    region_id: str
    node_ids: List[str]
    anchor_node_id: str
    region_feature: np.ndarray  # 768d L2-normalized
    dominant_plate: Optional[str] = None
    dominant_category: Optional[str] = None
    confidence: float = 1.0


def extract_dino_fused_feature(node, dinov3_strategy) -> Optional[np.ndarray]:
    """对单个 node 的 4 cam jpg 提 DINOv3 image-level fused feature (mean of patch grid, 4-cam mean).

    Args:
        node: 含 cameras={"camera_1": path, ...} 的 TopoNode
        dinov3_strategy: memory_nav.sub_image_matcher.DINOv3Strategy 实例 (公开 extract_patch_grid(np.ndarray))
    Returns:
        np.ndarray (768,) L2-normalized, 或 None 如果 4 cam 都失败.
    """
    try:
        import cv2
    except ImportError:
        return None
    feats = []
    for cam in ("camera_1", "camera_2", "camera_3", "camera_4"):
        img_path = (node.cameras or {}).get(cam) if hasattr(node, "cameras") else None
        if not img_path:
            continue
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            # extract_patch_grid: np.ndarray → grid [n_ph, n_pw, dim] (L2-normalized per patch)
            grid = dinov3_strategy.extract_patch_grid(img)
        except Exception as e:
            logger.debug(f"[DINO-fused] cam {cam} extract failed: {e}")
            continue
        if grid is None:
            continue
        # tensor → ndarray
        if isinstance(grid, np.ndarray):
            arr = grid
        else:
            arr = grid.detach().cpu().numpy() if hasattr(grid, "detach") else np.asarray(grid)
        if arr.ndim == 3:
            v = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        elif arr.ndim == 2:
            v = arr.mean(axis=0)
        else:
            continue
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            feats.append(v / n)
    if not feats:
        return None
    fused = np.mean(np.stack(feats), axis=0)
    n = float(np.linalg.norm(fused))
    return fused / max(n, 1e-6)


class RegionClusterer:
    def __init__(self,
                 feature_weight: float = 0.5,
                 spatial_weight: float = 0.3,
                 plate_weight: float = 0.2,
                 k_max: int = 12,
                 min_nodes_per_region: int = 1,
                 seg_gap_ms: int = _SEG_GAP_MS_DEFAULT):
        self.fw = feature_weight
        self.sw = spatial_weight
        self.pw = plate_weight
        self.k_max = k_max
        self.min_nodes = min_nodes_per_region
        self.seg_gap_ms = seg_gap_ms

    def cluster(self,
                nodes: Dict,
                node_features: Dict[str, np.ndarray],
                pose_graph,
                area_id_prefix: str = "default__indoor") -> List[RegionInfo]:
        """
        Args:
            nodes: Dict[node_id -> TopoNode] (含 timestamp/cameras/name_struct)
            node_features: Dict[node_id -> np.ndarray (768d, L2-norm)] (DINO fused)
            pose_graph: 含 .nodes[node_id] 含 x/y
        """
        if not nodes:
            return []

        ordered_ids = sorted(nodes.keys(), key=lambda i: int(nodes[i].timestamp))
        segments = self._split_by_time(ordered_ids, nodes)
        logger.info(f"[RegionClusterer] {len(ordered_ids)} nodes → {len(segments)} time segments")

        all_clusters: List[Tuple[int, int, List[str]]] = []
        for seg_idx, seg_ids in enumerate(segments):
            sub = self._cluster_within_segment(seg_ids, nodes, node_features, pose_graph)
            for ci, nids in enumerate(sub):
                all_clusters.append((seg_idx, ci, nids))
        logger.info(f"[RegionClusterer] segments → {len(all_clusters)} sub-clusters before plate merge")

        merged = self._merge_across_segments_by_plate(all_clusters, nodes)
        logger.info(f"[RegionClusterer] sub-clusters → {len(merged)} regions after plate merge")

        regions: List[RegionInfo] = []
        for cluster_idx, node_ids in enumerate(merged):
            feat = self._region_feature(node_ids, node_features)
            anchor = self._pick_anchor(node_ids, node_features, feat)
            dom_plate, dom_cat = self._dominant_plate_and_category(node_ids, nodes)
            slug = self._canonical_slug(dom_plate or dom_cat or "")
            rid = f"{area_id_prefix}__r{cluster_idx:02d}"
            if slug:
                rid = f"{rid}__{slug}"
            regions.append(RegionInfo(
                region_id=rid,
                node_ids=sorted(node_ids, key=lambda i: int(nodes[i].timestamp)),
                anchor_node_id=anchor,
                region_feature=feat,
                dominant_plate=dom_plate,
                dominant_category=dom_cat,
            ))
        return regions

    # ----- Stage 1: 时序切段 -----
    def _split_by_time(self, ordered_ids: List[str], nodes: Dict) -> List[List[str]]:
        segs: List[List[str]] = []
        cur: List[str] = []
        prev_ts: Optional[int] = None
        for nid in ordered_ids:
            try:
                ts = int(nodes[nid].timestamp)
            except (TypeError, ValueError):
                ts = prev_ts or 0
            if prev_ts is not None and (ts - prev_ts) > self.seg_gap_ms:
                if cur:
                    segs.append(cur); cur = []
            cur.append(nid)
            prev_ts = ts
        if cur:
            segs.append(cur)
        return segs

    # ----- Stage 2: 段内聚类 -----
    def _cluster_within_segment(self, seg_ids, nodes, node_features, pose_graph) -> List[List[str]]:
        N = len(seg_ids)
        if N <= 1:
            return [list(seg_ids)]

        feats = []
        for nid in seg_ids:
            f = node_features.get(nid)
            if f is None:
                f = np.zeros(768)
            feats.append(f)
        feats = np.array(feats)

        f_sim = feats @ feats.T
        f_dist = 1.0 - np.clip(f_sim, -1.0, 1.0)

        positions = []
        for nid in seg_ids:
            pn = pose_graph.nodes.get(nid) if hasattr(pose_graph, "nodes") else None
            if pn is not None:
                positions.append([float(pn.x), float(pn.y)])
            else:
                positions.append([0.0, 0.0])
        positions = np.array(positions)
        s_dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        med = float(np.median(s_dist[s_dist > 0])) if (s_dist > 0).any() else 1.0
        s_dist = s_dist / max(med, 0.1)

        d = self.fw * f_dist + self.sw * s_dist
        sigma = float(np.median(d)) if d.size else 1.0
        W = np.exp(-d / max(sigma, 0.1))
        np.fill_diagonal(W, 0.0)

        k = self._pick_k_eigengap(W, k_max=min(self.k_max, N - 1))
        if k <= 1:
            return [list(seg_ids)]

        try:
            from sklearn.cluster import SpectralClustering
            labels = SpectralClustering(
                n_clusters=k, affinity="precomputed", random_state=42,
                assign_labels="kmeans"
            ).fit_predict(W)
        except Exception as e:
            logger.warning(f"[RegionClusterer] SpectralClustering failed: {e}, single cluster")
            return [list(seg_ids)]

        groups: Dict[int, List[str]] = {}
        for nid, lbl in zip(seg_ids, labels):
            groups.setdefault(int(lbl), []).append(nid)
        return list(groups.values())

    def _pick_k_eigengap(self, W: np.ndarray, k_max: int) -> int:
        """eigengap heuristic: 选第一个明显 spectral gap 之后的 cluster 数 (k≥2).

        归一化 Laplacian L = I - D^-1/2 W D^-1/2 的特征值排序后, eigvals[0] ≈ 0 (连通图 trivial),
        从 gaps[1] = eigvals[2]-eigvals[1] 开始考察, 选最大 gap 的位置 + 2 (因 gap 索引 1 对应 k=2).

        若图较大 (n>=10) 且无明显 gap, fallback k = max(2, round(sqrt(n)/2)).
        """
        n = W.shape[0]
        if n <= 2:
            return 1
        deg = W.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-6))
        L = np.eye(n) - (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
        try:
            eigvals = np.linalg.eigvalsh(L)
        except Exception:
            return max(2, int(round(n ** 0.5 / 2)))
        eigvals = np.sort(eigvals)
        upper = min(k_max + 1, len(eigvals))
        if upper < 3:
            return min(2, k_max)
        # gap[i] = eigvals[i+1] - eigvals[i]; 从 gap[1] 开始 (跳过 trivial eigvals[0]≈0)
        gaps = np.diff(eigvals[1:upper])
        if len(gaps) == 0:
            return min(2, k_max)
        k = int(np.argmax(gaps)) + 2  # gap index 0 对应 k=2
        # 若所有 gap 接近 (无明显 cluster boundary), fallback 启发式 k≈sqrt(n)/2
        gap_max = float(gaps.max())
        gap_med = float(np.median(gaps))
        if gap_med > 1e-9 and gap_max / gap_med < 1.5:
            k = max(2, int(round(n ** 0.5 / 2)))
        return min(max(k, 2), k_max)

    # ----- Stage 3: 跨段同 plate 合并 -----
    def _merge_across_segments_by_plate(self, all_clusters, nodes) -> List[List[str]]:
        cluster_dom_plate: List[Optional[str]] = []
        for _seg_idx, _lbl, nids in all_clusters:
            dom_plate, _ = self._dominant_plate_and_category(nids, nodes)
            cluster_dom_plate.append(dom_plate)

        used = [False] * len(all_clusters)
        result: List[List[str]] = []
        for i, (_si, _li, nids_i) in enumerate(all_clusters):
            if used[i]:
                continue
            merged = list(nids_i)
            used[i] = True
            dom_i = cluster_dom_plate[i]
            if dom_i:
                for j in range(i + 1, len(all_clusters)):
                    if used[j]:
                        continue
                    if cluster_dom_plate[j] == dom_i:
                        merged.extend(all_clusters[j][2])
                        used[j] = True
                        logger.info(f"[RegionClusterer] cross-segment merge by plate '{dom_i}': "
                                    f"seg{all_clusters[j][0]}.c{all_clusters[j][1]} → seg{_si}.c{_li}")
            result.append(merged)
        return result

    # ----- helpers -----
    def _dominant_plate_and_category(self, node_ids, nodes) -> Tuple[Optional[str], Optional[str]]:
        plate_count: Dict[str, int] = {}
        cat_count: Dict[str, int] = {}
        for nid in node_ids:
            n = nodes.get(nid)
            if not n:
                continue
            ns = getattr(n, "name_struct", None)
            if ns is not None:
                cat = (ns.category or "").strip()
                org = (ns.organization or "").strip()
                if org:
                    plate_count[org] = plate_count.get(org, 0) + 1
                if cat:
                    cat_count[cat] = cat_count.get(cat, 0) + 1
            pos_name = getattr(n, "position_name", "") or ""
            if pos_name and pos_name not in plate_count:
                # 用 position_name 兜底 (split '·' 取主体)
                head = pos_name.split("·")[0].strip()
                if head:
                    plate_count[head] = plate_count.get(head, 0) + 1
        dom_plate = max(plate_count.items(), key=lambda x: x[1])[0] if plate_count else None
        dom_cat = max(cat_count.items(), key=lambda x: x[1])[0] if cat_count else None
        return dom_plate, dom_cat

    def _region_feature(self, node_ids, node_features) -> np.ndarray:
        feats = [node_features[n] for n in node_ids if n in node_features and node_features[n] is not None]
        if not feats:
            return np.zeros(768)
        m = np.mean(np.stack(feats), axis=0)
        norm = float(np.linalg.norm(m))
        return m / max(norm, 1e-6)

    def _pick_anchor(self, node_ids, node_features, region_feature) -> str:
        best, best_sim = node_ids[0], -2.0
        for nid in node_ids:
            f = node_features.get(nid)
            if f is None:
                continue
            sim = float(f @ region_feature)
            if sim > best_sim:
                best_sim = sim; best = nid
        return best

    @staticmethod
    def _canonical_slug(name: str) -> str:
        """中文/英文 plate 转 ASCII snake_case slug. 中文用拼音首字母, 否则保留."""
        if not name:
            return ""
        # 简易拼音首字母 (无 pypinyin 依赖). 中文字符提其字符的 unicode hash 后缀.
        out = []
        for c in name:
            if c.isascii() and (c.isalnum() or c == "_"):
                out.append(c.lower())
            elif c.isspace() or c in "·/.":
                out.append("_")
            elif "一" <= c <= "鿿":
                out.append(f"u{ord(c):04x}"[:5])
            else:
                out.append("_")
        s = "".join(out)
        s = re.sub(r"_+", "_", s).strip("_")
        return s[:40]
