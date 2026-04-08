"""产出 merged_labeled_data 兼容目录"""
import os, json, shutil, cv2, logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class MergedDataWriter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_node(self, node_id: str, timestamp: str, cameras: Dict[str, str],
                   position_name: str, position_name_eng: str,
                   next_positions: List[Dict],
                   name_struct: Dict = None):
        node_dir = self.output_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "crops").mkdir(exist_ok=True)

        copied = {}
        for cam_id, src_path in cameras.items():
            if not os.path.exists(src_path):
                continue
            tgt = f"{timestamp}_{cam_id}.jpg"
            shutil.copy2(src_path, node_dir / tgt)
            copied[cam_id] = tgt

        # 处理 next_positions: 复制 crops
        out_nexts = []
        for i, np_info in enumerate(next_positions):
            new_np = dict(np_info)
            cps = {}
            for size in ('big', 'mid', 'small'):
                src_crop = np_info.get('_src_crops', {}).get(size)
                if src_crop and os.path.exists(src_crop):
                    fname = os.path.basename(src_crop)
                    shutil.copy2(src_crop, node_dir / "crops" / fname)
                    cps[size] = f"crops/{fname}"
            new_np['crop_image_paths'] = cps
            new_np['crop_image_path'] = cps.get('big', '')
            new_np['pixel_box'] = new_np.get('pixel_box', '')
            new_np.pop('_src_crops', None)
            # 必填字段保底
            for k in ('big_box', 'mid_box', 'small_box',
                      'landmark_name', 'landmark_name_eng',
                      'position_name', 'position_name_eng',
                      'camera_name', 'position_id'):
                new_np.setdefault(k, '')
            out_nexts.append(new_np)

        self_pos = {
            "position_id": node_id,
            "position_name": position_name,
            "position_name_eng": position_name_eng,
        }
        # 结构化命名字段 (来自 NodeName.to_dict)
        if name_struct:
            for k in ("category", "category_eng", "organization",
                      "nearby_plates", "nearby_landmarks", "instance_suffix"):
                if k in name_struct:
                    self_pos[k] = name_struct[k]
        self_pos.update({c: copied[c] for c in copied})
        info = {
            "self_position": self_pos,
            "next_positions": out_nexts,
        }
        with open(node_dir / "node_position_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    @staticmethod
    def make_dummy_crop(src_image_path: str, save_dir: Path, timestamp: str,
                       camera_name: str, next_idx: int):
        """从源图生成 big/mid/small 三个 crop, 中心 27%x34% 区域"""
        img = cv2.imread(src_image_path)
        if img is None:
            return None, None
        h, w = img.shape[:2]
        side = int(min(w * 0.27, h * 0.34))
        cx, cy = w // 2, int(h * 0.48)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_crops = {}
        out_boxes = {}
        cam_num = camera_name.split('_')[1]
        for size, scale in [('big', 1.0), ('mid', 0.87), ('small', 0.75)]:
            s = int(side * scale)
            x1 = max(0, min(w - s, cx - s // 2))
            y1 = max(0, min(h - s, cy - s // 2))
            x2 = min(w, x1 + s)
            y2 = min(h, y1 + s)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fname = f"{timestamp}_camera_{cam_num}__{next_idx}__{size}__{x1}_{y1}_{x2-x1}_{y2-y1}.jpg"
            fp = save_dir / fname
            cv2.imwrite(str(fp), crop)
            out_crops[size] = str(fp)
            out_boxes[size] = f"{x1/w:.6f},{y1/h:.6f},{x2/w:.6f},{y2/h:.6f}"
        return out_crops, out_boxes
