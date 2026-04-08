"""语义去重: 同 room/landmark 且 VPR 接近 -> 合并"""
import numpy as np


def cyclic_cosine(fa: dict, fb: dict) -> float:
    cams = ['camera_1', 'camera_2', 'camera_3', 'camera_4']
    if not all(c in fa and c in fb for c in cams):
        return 0.0
    best = -1.0
    for shift in range(4):
        sims = []
        for i, c in enumerate(cams):
            cb = cams[(i + shift) % 4]
            a = fa[c]; b = fb[cb]
            s = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            sims.append(s)
        best = max(best, sum(sims) / 4.0)
    return best


def find_merge_target(new_room: str, new_landmark: str, new_feats: dict,
                      existing_nodes: dict, existing_feats: dict,
                      vpr_threshold: float = 0.65) -> str:
    """返回应合并的旧 node_id, 否则 None"""
    if not new_room and not new_landmark:
        return None
    for nid, node in existing_nodes.items():
        same_room = (new_room and node.room == new_room)
        same_landmark = (new_landmark and node.landmark_name == new_landmark)
        if not (same_room or same_landmark):
            continue
        feats = existing_feats.get(nid)
        if feats is None:
            continue
        if cyclic_cosine(new_feats, feats) >= vpr_threshold:
            return nid
    return None
