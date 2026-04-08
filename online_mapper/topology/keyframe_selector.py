"""多触发关键帧选择器"""
import logging
logger = logging.getLogger(__name__)


class KeyframeSelector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.acc_trans = 0.0
        self.acc_rot = 0.0
        self.acc_info = 0.0
        self.last_kf_frame_idx = -10**9

    def update_motion(self, dtrans: float, drot: float, dinfo: float):
        self.acc_trans += abs(dtrans)
        self.acc_rot += abs(drot)
        self.acc_info += max(0.0, dinfo)

    def reset(self, frame_idx: int):
        self.acc_trans = self.acc_rot = self.acc_info = 0.0
        self.last_kf_frame_idx = frame_idx

    def should_trigger(self, frame_idx: int, vpr_sim_to_last: float) -> (bool, str):
        if frame_idx - self.last_kf_frame_idx < self.cfg.min_keyframe_frame_interval:
            return False, "min_gap"
        # 1) VPR
        if vpr_sim_to_last is not None and vpr_sim_to_last < self.cfg.vpr_dissim_threshold:
            return True, f"vpr<{self.cfg.vpr_dissim_threshold}"
        # 2) 累积位移
        if self.acc_trans >= self.cfg.accumulated_translation:
            return True, "acc_trans"
        # 3) 累积旋转
        if self.acc_rot >= self.cfg.accumulated_rotation:
            return True, "acc_rot"
        # 4) 信息增益
        if self.acc_info >= self.cfg.info_gain_threshold:
            return True, "info_gain"
        return False, "none"
