"""StreamLoader: 模拟在线流式输入"""
import os, glob, logging
from typing import Dict, Iterator, List
from pathlib import Path

logger = logging.getLogger(__name__)
CAMERA_IDS = ['camera_1', 'camera_2', 'camera_3', 'camera_4']


class StreamLoader:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(input_dir)
        files = glob.glob(str(self.input_dir / "*_camera_*.jpg"))
        timestamps = sorted({os.path.basename(f).split('_')[0] for f in files})
        self.frames: List[Dict] = []
        for ts in timestamps:
            cams = {}
            for cid in CAMERA_IDS:
                p = self.input_dir / f"{ts}_{cid}.jpg"
                if p.exists():
                    cams[cid] = str(p)
            if len(cams) == 4:
                self.frames.append({"timestamp": ts, "frame_idx": len(self.frames), "cameras": cams})
        logger.info(f"StreamLoader loaded {len(self.frames)} frames from {input_dir}")

    def __iter__(self) -> Iterator[Dict]:
        for f in self.frames:
            yield f

    def __len__(self):
        return len(self.frames)
