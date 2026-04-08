"""真实图结构 — 邻接表, 支持 >2 邻居"""
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class TopoNode:
    node_id: str
    timestamp: str
    frame_idx: int
    cameras: Dict[str, str]
    landmark_name: str = ""
    room: str = ""
    floor: str = "F1"
    neighbors: Set[str] = field(default_factory=set)
    semantic_objects: List[Dict] = field(default_factory=list)
    representative_score: float = 0.0  # 用于门牌帧回溯


class TopoGraph:
    def __init__(self):
        self.nodes: Dict[str, TopoNode] = {}
        self.edges: Set[Tuple[str, str]] = set()  # 无向

    def add_node(self, node: TopoNode):
        self.nodes[node.node_id] = node

    def add_edge(self, a: str, b: str):
        if a == b:
            return
        if a not in self.nodes or b not in self.nodes:
            return
        e = (min(a, b), max(a, b))
        if e in self.edges:
            return
        self.edges.add(e)
        self.nodes[a].neighbors.add(b)
        self.nodes[b].neighbors.add(a)

    def neighbors(self, node_id: str) -> List[str]:
        return list(self.nodes[node_id].neighbors) if node_id in self.nodes else []

    def __len__(self):
        return len(self.nodes)
