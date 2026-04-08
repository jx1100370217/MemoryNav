"""层次化场景图 floor → room → node → object"""
from typing import Dict, List
from dataclasses import dataclass, field
import json


@dataclass
class SceneObject:
    label: str
    bbox: list
    score: float
    camera: str
    plate_text: str = ""


@dataclass
class SceneNode:
    node_id: str
    room: str
    objects: List[SceneObject] = field(default_factory=list)


class SceneGraph:
    def __init__(self):
        self.floors: Dict[str, Dict[str, List[str]]] = {"F1": {}}  # floor -> room -> [node_ids]
        self.scene_nodes: Dict[str, SceneNode] = {}

    def add_node(self, node_id: str, room: str, floor: str = "F1", objects=None):
        self.scene_nodes[node_id] = SceneNode(node_id, room, list(objects or []))
        self.floors.setdefault(floor, {}).setdefault(room or "unknown", []).append(node_id)

    def to_dict(self):
        return {
            "floors": {
                fl: {rm: list(nids) for rm, nids in rooms.items()}
                for fl, rooms in self.floors.items()
            },
            "nodes": {
                nid: {
                    "room": sn.room,
                    "objects": [
                        {"label": o.label, "bbox": o.bbox, "score": o.score,
                         "camera": o.camera, "plate_text": o.plate_text}
                        for o in sn.objects
                    ],
                } for nid, sn in self.scene_nodes.items()
            },
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
