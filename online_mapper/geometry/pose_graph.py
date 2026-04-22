"""轻量 2D pose graph (x, y, theta) — scipy least-squares 优化"""
import logging, numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PoseNode:
    node_id: str
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    # P3.18: motion-based body heading (atan2 of position delta to next keyframe).
    # Preferred over `theta` for cam azimuth matching when available,
    # because VGGT yaw estimate drifts (all 4 nodes estimated to -85°).
    motion_theta: Optional[float] = None


@dataclass
class PoseEdge:
    a: str
    b: str
    dx: float
    dy: float
    dtheta: float
    info: float = 1.0  # weight
    kind: str = "odom"  # odom | loop


class PoseGraph:
    def __init__(self):
        self.nodes: Dict[str, PoseNode] = {}
        self.edges: List[PoseEdge] = []

    def add_node(self, node_id: str, x=0.0, y=0.0, theta=0.0):
        self.nodes[node_id] = PoseNode(node_id, x, y, theta)

    def add_edge(self, a: str, b: str, dx: float, dy: float, dtheta: float,
                 info: float = 1.0, kind: str = "odom"):
        self.edges.append(PoseEdge(a, b, dx, dy, dtheta, info, kind))

    def to_dict(self):
        return {
            "nodes": [{"id": n.node_id, "x": n.x, "y": n.y, "theta": n.theta}
                      for n in self.nodes.values()],
            "edges": [{"a": e.a, "b": e.b, "dx": e.dx, "dy": e.dy,
                       "dtheta": e.dtheta, "info": e.info, "kind": e.kind}
                      for e in self.edges],
        }

    def optimize(self, iters: int = 30):
        """简单 Gauss-Newton-like 优化 (固定第一个节点)"""
        if len(self.nodes) < 2 or not self.edges:
            return
        try:
            from scipy.optimize import least_squares
        except Exception as e:
            logger.warning(f"scipy not available: {e}")
            return

        ids = list(self.nodes.keys())
        idx = {nid: i for i, nid in enumerate(ids)}
        x0 = np.zeros(3 * len(ids), dtype=np.float64)
        for i, nid in enumerate(ids):
            n = self.nodes[nid]
            x0[3 * i:3 * i + 3] = [n.x, n.y, n.theta]

        edges = self.edges
        first_id = ids[0]

        def residuals(x):
            res = []
            for e in edges:
                ia, ib = idx[e.a], idx[e.b]
                xa, ya, ta = x[3 * ia], x[3 * ia + 1], x[3 * ia + 2]
                xb, yb, tb = x[3 * ib], x[3 * ib + 1], x[3 * ib + 2]
                # predicted relative
                c, s = np.cos(ta), np.sin(ta)
                pdx = c * (xb - xa) + s * (yb - ya)
                pdy = -s * (xb - xa) + c * (yb - ya)
                pdth = tb - ta
                w = np.sqrt(e.info)
                res.extend([w * (pdx - e.dx), w * (pdy - e.dy), w * (pdth - e.dtheta)])
            # anchor first node
            i0 = idx[first_id]
            res.extend([1e3 * x[3 * i0], 1e3 * x[3 * i0 + 1], 1e3 * x[3 * i0 + 2]])
            return np.array(res)

        try:
            sol = least_squares(residuals, x0, max_nfev=iters * 10, method='lm')
            for i, nid in enumerate(ids):
                self.nodes[nid].x = float(sol.x[3 * i])
                self.nodes[nid].y = float(sol.x[3 * i + 1])
                self.nodes[nid].theta = float(sol.x[3 * i + 2])
            logger.info(f"PoseGraph optimized: cost={sol.cost:.4f} edges={len(edges)}")
        except Exception as e:
            logger.warning(f"PoseGraph opt failed: {e}")
