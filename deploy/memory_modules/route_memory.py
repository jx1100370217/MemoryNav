#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 路线记忆管理模块 v2.0

管理导航路线的记录、保存和检索。

v2.0 新增功能:
1. 语义级别模糊匹配: 支持语义等价但文字不同的指令匹配
2. 关键词提取和匹配: 从指令中提取关键地点和动作
3. 相似度评分: 综合多维度评估指令相似性
4. 部分路线复用: 支持匹配路线的部分段落

Note: 此模块使用pickle进行序列化，仅用于内部生成的数据存储。
      不接受外部输入的pickle文件，避免安全风险。
      pickle仅用于保存系统自身生成的路线数据，不加载任何外部来源的pickle文件。
"""

import os
import time
import pickle
import logging
import re
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import cv2

from .config import MemoryNavigationConfig
from .models import RouteMemory

logger = logging.getLogger(__name__)


# ============================================================================
# 指令关键词提取和同义词映射
# ============================================================================
LOCATION_SYNONYMS = {
    '前台': ['reception', 'front desk', '接待处', '接待台', '前台接待', '服务台'],
    '沙发': ['sofa', 'couch', '长椅', '休息区', '休息沙发'],
    '办公室': ['office', '办公区', '工作室', '办公室区域'],
    '会议室': ['meeting room', '会客室', '议事厅', '会议区'],
    '走廊': ['corridor', 'hallway', '过道', '通道', '廊道'],
    '电梯': ['elevator', 'lift', '升降机'],
    '楼梯': ['stairs', 'stairway', '台阶'],
    '门口': ['entrance', 'door', '入口', '出口', '门'],
    '厨房': ['kitchen', '烹饪区', '餐厅'],
    '卫生间': ['bathroom', 'toilet', 'restroom', '洗手间', '厕所'],
    '起点': ['start', 'origin', '出发点', '起始位置', '当前位置'],
}

ACTION_SYNONYMS = {
    '去': ['go to', 'navigate to', 'walk to', '走到', '到达', '前往'],
    '返回': ['return', 'go back', 'come back', '回到', '回来'],
    '然后': ['then', 'after that', '接着', '之后', '再'],
}

# 构建反向索引
LOCATION_REVERSE = {}
for standard, synonyms in LOCATION_SYNONYMS.items():
    LOCATION_REVERSE[standard.lower()] = standard
    for syn in synonyms:
        LOCATION_REVERSE[syn.lower()] = standard

ACTION_REVERSE = {}
for standard, synonyms in ACTION_SYNONYMS.items():
    ACTION_REVERSE[standard.lower()] = standard
    for syn in synonyms:
        ACTION_REVERSE[syn.lower()] = standard


class RouteMemoryManager:
    """路线记忆管理器 v2.0 - 支持语义模糊匹配"""

    def __init__(self, config: MemoryNavigationConfig):
        self.config = config
        self.current_route: Optional[RouteMemory] = None
        self.saved_routes: Dict[str, RouteMemory] = {}
        self.frame_count = 0
        self.start_node_id: Optional[int] = None

        # [新增] 模糊匹配参数
        self.fuzzy_match_threshold = 0.65  # 模糊匹配阈值 (从0.7降低到0.65以提高召回率)
        self.enable_fuzzy_match = True  # 是否启用模糊匹配

        # [新增] 指令解析缓存
        self._instruction_cache: Dict[str, Dict] = {}

    def start_recording(self, instruction: str) -> str:
        """开始记录新路线"""
        route_id = f"route_{int(time.time())}"
        self.current_route = RouteMemory(
            route_id=route_id,
            start_instruction=instruction,
            start_timestamp=time.time(),
            visual_features=np.empty((0, self.config.feature_dim))
        )
        self.frame_count = 0
        self.start_node_id = None
        logger.info(f"开始记录路线: {route_id}, 指令: {instruction}")
        return route_id

    def record_step(self,
                    node_id: int,
                    visual_feature: np.ndarray,
                    action: List[int],
                    rgb_image: np.ndarray = None,
                    is_keyframe: bool = False):
        """
        记录一步

        Args:
            node_id: 拓扑图节点ID
            visual_feature: 视觉特征向量
            action: 动作序列
            rgb_image: RGB图像（用于关键帧保存）
            is_keyframe: 是否为关键帧（基于pixel_target是否为None判断）
        """
        if self.current_route is None:
            return

        # 记录起点
        if self.start_node_id is None:
            self.start_node_id = node_id

        self.current_route.node_sequence.append(node_id)
        self.current_route.action_history.append(action)

        # 添加视觉特征
        self.current_route.visual_features = np.vstack([
            self.current_route.visual_features,
            visual_feature.reshape(1, -1)
        ])

        # 关键帧保存 - 基于pixel_target判断而非固定间隔
        if is_keyframe:
            self.current_route.keyframe_indices.append(
                len(self.current_route.node_sequence) - 1
            )
            if rgb_image is not None:
                # 压缩图像以节省内存
                small_image = cv2.resize(rgb_image, (160, 120))
                self.current_route.keyframe_images.append(small_image)
            logger.info(f"[关键帧] 帧#{self.frame_count} 记录为关键帧，当前关键帧数: {len(self.current_route.keyframe_indices)}")

            # 实时保存路线到磁盘（每个关键帧都保存）
            self._save_route_to_disk(self.current_route)

        self.frame_count += 1

    def clear_current_route(self):
        """
        v2.5.2: 清除当前正在记录的路线，但不清空已保存的路线
        用于在多场景切换时保留拓扑图记忆
        """
        if self.current_route is not None:
            logger.info(f"清除当前路线: {self.current_route.route_id}")
        self.current_route = None
        self.frame_count = 0
        self.start_node_id = None

    def clear_all(self):
        """
        v2.1: 完全清空所有路线记忆
        """
        self.current_route = None
        self.saved_routes.clear()
        self.frame_count = 0
        self.start_node_id = None
        logger.info("[RouteMemory] 所有路线记忆已清空")

    def stop_recording(self) -> Optional[RouteMemory]:
        """停止记录并保存路线"""
        if self.current_route is None:
            return None

        self.current_route.is_complete = True
        self.current_route.end_timestamp = time.time()

        route = self.current_route
        self.saved_routes[route.route_id] = route

        logger.info(f"路线记录完成: {route.route_id}, "
                   f"节点数={len(route.node_sequence)}, "
                   f"关键帧数={len(route.keyframe_images)}")

        # 持久化保存到文件
        self._save_route_to_disk(route)

        self.current_route = None

        # 内存管理
        if len(self.saved_routes) > self.config.max_memory_routes:
            oldest_id = min(
                self.saved_routes.keys(),
                key=lambda k: self.saved_routes[k].start_timestamp
            )
            del self.saved_routes[oldest_id]
            logger.info(f"删除旧路线: {oldest_id}")

        return route

    def get_return_trajectory(self) -> List[Tuple[int, List[int]]]:
        """
        获取返回轨迹

        Returns:
            List[(node_id, reversed_action)]: 反向轨迹
        """
        if self.current_route is None or len(self.current_route.node_sequence) == 0:
            return []

        # 反转节点序列和动作
        reversed_nodes = list(reversed(self.current_route.node_sequence))
        reversed_actions = []

        for action in reversed(self.current_route.action_history):
            reversed_action = self._reverse_action(action)
            reversed_actions.append(reversed_action)

        return list(zip(reversed_nodes, reversed_actions))

    def get_start_node(self) -> Optional[int]:
        """获取起始节点ID"""
        return self.start_node_id

    def _reverse_action(self, action: List[int]) -> List[int]:
        """
        反转动作序列
        动作编码: 0=STOP, 1=前进, 2=左转, 3=右转, 5=向下看
        """
        action_map = {
            0: 0,
            1: 1,
            2: 3,
            3: 2,
            5: 5
        }
        return [action_map.get(a, a) for a in action]

    def is_recording(self) -> bool:
        """是否正在记录"""
        return self.current_route is not None

    def get_current_progress(self) -> Dict:
        """获取当前记录进度"""
        if self.current_route is None:
            return {"recording": False}

        return {
            "recording": True,
            "route_id": self.current_route.route_id,
            "frames": len(self.current_route.node_sequence),
            "keyframes": len(self.current_route.keyframe_images),
            "duration": time.time() - self.current_route.start_timestamp
        }

    def _save_route_to_disk(self, route: RouteMemory):
        """将路线保存到磁盘（内部生成数据，非外部输入）"""
        try:
            save_dir = self.config.memory_save_path
            os.makedirs(save_dir, exist_ok=True)

            # 保存路线元数据
            route_data = {
                'route_id': route.route_id,
                'start_instruction': route.start_instruction,
                'start_timestamp': route.start_timestamp,
                'end_timestamp': route.end_timestamp,
                'node_sequence': route.node_sequence,
                'action_history': route.action_history,
                'keyframe_indices': route.keyframe_indices,
                'is_complete': route.is_complete,
            }

            # 保存为pickle文件（内部数据，非外部来源）
            route_file = os.path.join(save_dir, f"{route.route_id}.pkl")
            with open(route_file, 'wb') as f:
                pickle.dump(route_data, f)
            logger.info(f"路线元数据已保存: {route_file}")

            # 保存视觉特征为numpy文件
            if route.visual_features is not None and len(route.visual_features) > 0:
                features_file = os.path.join(save_dir, f"{route.route_id}_features.npy")
                np.save(features_file, route.visual_features)
                logger.info(f"视觉特征已保存: {features_file} ({route.visual_features.shape})")

            # 保存关键帧图像
            if len(route.keyframe_images) > 0:
                keyframes_dir = os.path.join(save_dir, f"{route.route_id}_keyframes")
                os.makedirs(keyframes_dir, exist_ok=True)
                for i, img in enumerate(route.keyframe_images):
                    img_path = os.path.join(keyframes_dir, f"keyframe_{i:04d}.jpg")
                    cv2.imwrite(img_path, img)
                logger.info(f"关键帧图像已保存: {keyframes_dir}/ ({len(route.keyframe_images)}张)")

        except Exception as e:
            logger.error(f"保存路线失败: {e}")

    def find_matching_route(self, instruction: str) -> Optional[RouteMemory]:
        """
        根据指令查找匹配的已保存路线 - 增强版 v2.0

        支持:
        1. 精确匹配
        2. 语义模糊匹配

        Args:
            instruction: 导航指令

        Returns:
            匹配的路线，如果没有找到返回None
        """
        # 首先尝试精确匹配内存中的路线
        for route_id, route in self.saved_routes.items():
            if route.start_instruction == instruction and route.is_complete:
                logger.info(f"[精确匹配] 在内存中找到匹配路线: {route_id}")
                return route

        # 然后检查磁盘上的路线（精确匹配 - 内部生成的pickle文件）
        save_dir = self.config.memory_save_path
        exact_match = self._find_exact_match_on_disk(instruction, save_dir)
        if exact_match:
            return exact_match

        # [新增] 语义模糊匹配
        if self.enable_fuzzy_match:
            fuzzy_match = self._find_fuzzy_match(instruction)
            if fuzzy_match:
                return fuzzy_match

        return None

    def _find_exact_match_on_disk(self, instruction: str, save_dir: str) -> Optional[RouteMemory]:
        """在磁盘上查找精确匹配的路线"""
        if not os.path.exists(save_dir):
            return None

        for filename in os.listdir(save_dir):
            if filename.endswith('.pkl') and not filename.endswith('_features.pkl'):
                route_file = os.path.join(save_dir, filename)
                try:
                    # 仅加载系统自身生成的pickle文件
                    with open(route_file, 'rb') as f:
                        route_data = pickle.load(f)
                    if route_data.get('start_instruction') == instruction and route_data.get('is_complete', False):
                        route = self._rebuild_route_from_data(route_data, save_dir)
                        if route:
                            logger.info(f"[精确匹配] 从磁盘加载匹配路线: {route.route_id}")
                            return route
                except Exception as e:
                    logger.warning(f"加载路线文件失败 {route_file}: {e}")
                    continue
        return None

    def _rebuild_route_from_data(self, route_data: Dict, save_dir: str) -> Optional[RouteMemory]:
        """从数据字典重建RouteMemory对象"""
        try:
            route = RouteMemory(
                route_id=route_data['route_id'],
                start_instruction=route_data['start_instruction'],
                start_timestamp=route_data['start_timestamp']
            )
            route.end_timestamp = route_data['end_timestamp']
            route.node_sequence = route_data['node_sequence']
            route.action_history = route_data['action_history']
            route.keyframe_indices = route_data['keyframe_indices']
            route.is_complete = route_data['is_complete']

            # 加载视觉特征
            features_file = os.path.join(save_dir, f"{route_data['route_id']}_features.npy")
            if os.path.exists(features_file):
                route.visual_features = np.load(features_file)

            # 缓存到内存中
            self.saved_routes[route.route_id] = route
            return route
        except Exception as e:
            logger.warning(f"重建路线对象失败: {e}")
            return None

    def _find_fuzzy_match(self, instruction: str) -> Optional[RouteMemory]:
        """
        [新增] 语义模糊匹配 - 查找语义相似的路线

        Args:
            instruction: 导航指令

        Returns:
            最佳匹配的路线，如果没有满足阈值的匹配则返回None
        """
        query_parsed = self._parse_instruction(instruction)
        best_match = None
        best_score = 0.0

        # 检查所有已保存的路线
        all_routes = list(self.saved_routes.items())

        # 也从磁盘加载路线进行匹配
        save_dir = self.config.memory_save_path
        if os.path.exists(save_dir):
            for filename in os.listdir(save_dir):
                if filename.endswith('.pkl') and not filename.endswith('_features.pkl'):
                    route_file = os.path.join(save_dir, filename)
                    try:
                        with open(route_file, 'rb') as f:
                            route_data = pickle.load(f)
                        if route_data.get('is_complete', False):
                            route_id = route_data['route_id']
                            if route_id not in self.saved_routes:
                                route = self._rebuild_route_from_data(route_data, save_dir)
                                if route:
                                    all_routes.append((route_id, route))
                    except Exception:
                        continue

        for route_id, route in all_routes:
            if not route.is_complete:
                continue

            saved_parsed = self._parse_instruction(route.start_instruction)
            score = self._compute_instruction_similarity(query_parsed, saved_parsed)

            if score > best_score:
                best_score = score
                best_match = route

        if best_match and best_score >= self.fuzzy_match_threshold:
            logger.info(f"[模糊匹配] 找到语义相似路线: {best_match.route_id}, "
                       f"相似度={best_score:.3f}, "
                       f"原指令='{best_match.start_instruction}', "
                       f"查询指令='{instruction}'")
            return best_match

        return None

    def _parse_instruction(self, instruction: str) -> Dict:
        """
        [新增] 解析导航指令，提取关键信息

        Args:
            instruction: 导航指令

        Returns:
            解析结果字典，包含地点、动作等信息
        """
        # 检查缓存
        if instruction in self._instruction_cache:
            return self._instruction_cache[instruction]

        instruction_lower = instruction.lower()

        # 提取地点
        locations = []
        for word in re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z\s]+', instruction_lower):
            word = word.strip()
            if word in LOCATION_REVERSE:
                locations.append(LOCATION_REVERSE[word])
            elif len(word) > 1:
                # 检查是否包含地点关键词
                for loc_key, loc_standard in LOCATION_REVERSE.items():
                    if loc_key in word or word in loc_key:
                        if loc_standard not in locations:
                            locations.append(loc_standard)

        # 提取动作
        actions = []
        for word in re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z\s]+', instruction_lower):
            word = word.strip()
            if word in ACTION_REVERSE:
                actions.append(ACTION_REVERSE[word])

        # 判断是否包含"返回"动作
        has_return = '返回' in actions or 'return' in instruction_lower or '回' in instruction_lower

        result = {
            'locations': locations,
            'actions': actions,
            'has_return': has_return,
            'raw': instruction_lower
        }

        # 缓存结果
        self._instruction_cache[instruction] = result
        return result

    def _compute_instruction_similarity(self, parsed1: Dict, parsed2: Dict) -> float:
        """
        [新增] 计算两个解析后指令的相似度

        Args:
            parsed1: 第一个解析结果
            parsed2: 第二个解析结果

        Returns:
            相似度分数 (0-1)
        """
        score = 0.0
        max_score = 0.0

        # 地点匹配 (权重最高)
        loc1 = set(parsed1.get('locations', []))
        loc2 = set(parsed2.get('locations', []))
        if loc1 and loc2:
            intersection = loc1 & loc2
            union = loc1 | loc2
            loc_score = len(intersection) / len(union) if union else 0.0
            score += loc_score * 3.0
            max_score += 3.0
        elif not loc1 and not loc2:
            score += 1.0
            max_score += 3.0
        else:
            max_score += 3.0

        # 返回动作匹配
        if parsed1.get('has_return') == parsed2.get('has_return'):
            score += 1.0
        max_score += 1.0

        # 动作匹配
        act1 = set(parsed1.get('actions', []))
        act2 = set(parsed2.get('actions', []))
        if act1 and act2:
            intersection = act1 & act2
            union = act1 | act2
            act_score = len(intersection) / len(union) if union else 0.0
            score += act_score * 1.0
        elif not act1 and not act2:
            score += 0.5
        max_score += 1.0

        # 原始字符串相似度 (编辑距离)
        raw1 = parsed1.get('raw', '')
        raw2 = parsed2.get('raw', '')
        if raw1 and raw2:
            edit_sim = 1.0 - self._levenshtein_ratio(raw1, raw2)
            score += edit_sim * 0.5
        max_score += 0.5

        return score / max_score if max_score > 0 else 0.0

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """计算编辑距离比率"""
        if not s1 and not s2:
            return 0.0
        if not s1 or not s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 < len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        previous_row = list(range(len2 + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        distance = previous_row[-1]
        max_len = max(len1, len2)
        return distance / max_len if max_len > 0 else 0.0

    def find_best_matching_routes(self, instruction: str, k: int = 5) -> List[Tuple[RouteMemory, float]]:
        """
        [新增] 查找最佳匹配的多个路线

        Args:
            instruction: 导航指令
            k: 返回top-k结果

        Returns:
            [(route, score), ...] 匹配结果列表
        """
        query_parsed = self._parse_instruction(instruction)
        results = []

        # 收集所有路线
        all_routes = list(self.saved_routes.values())

        # 也从磁盘加载
        save_dir = self.config.memory_save_path
        if os.path.exists(save_dir):
            for filename in os.listdir(save_dir):
                if filename.endswith('.pkl') and not filename.endswith('_features.pkl'):
                    route_file = os.path.join(save_dir, filename)
                    try:
                        with open(route_file, 'rb') as f:
                            route_data = pickle.load(f)
                        if route_data.get('is_complete', False):
                            route_id = route_data['route_id']
                            if route_id not in self.saved_routes:
                                route = self._rebuild_route_from_data(route_data, save_dir)
                                if route:
                                    all_routes.append(route)
                    except Exception:
                        continue

        for route in all_routes:
            if not route.is_complete:
                continue

            saved_parsed = self._parse_instruction(route.start_instruction)
            score = self._compute_instruction_similarity(query_parsed, saved_parsed)
            results.append((route, score))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_action_at_step(self, route: RouteMemory, step: int) -> Optional[List[int]]:
        """
        获取指定路线在指定步骤的动作

        Args:
            route: 路线对象
            step: 步骤索引

        Returns:
            动作序列，如果超出范围返回None
        """
        if step < len(route.action_history):
            return route.action_history[step]
        return None

    # ========================================================================
    # v3.0 新增: 部分路线匹配
    # ========================================================================

    def find_partial_route_match(self, instruction: str,
                                  current_visual_feature: np.ndarray = None,
                                  min_segment_length: int = 5) -> Optional[Dict]:
        """
        [v3.0] 查找部分路线匹配

        当完整路线不匹配时，尝试找到可复用的路线段

        Args:
            instruction: 导航指令
            current_visual_feature: 当前视觉特征 (用于定位)
            min_segment_length: 最小段落长度

        Returns:
            匹配结果字典，包含路线、起始索引、结束索引等
        """
        query_parsed = self._parse_instruction(instruction)
        query_locations = set(query_parsed.get('locations', []))

        if not query_locations:
            return None

        best_match = None
        best_score = 0.0

        # 遍历所有路线
        for route_id, route in self.saved_routes.items():
            if not route.is_complete:
                continue

            route_parsed = self._parse_instruction(route.start_instruction)
            route_locations = set(route_parsed.get('locations', []))

            # 检查是否有共同的目的地
            common_locations = query_locations & route_locations

            if common_locations:
                # 计算部分匹配分数
                partial_score = len(common_locations) / max(len(query_locations), 1)

                # 如果查询包含"返回"但路线不包含，可以复用前半段
                if query_parsed.get('has_return') and not route_parsed.get('has_return'):
                    # 找到路线中间点
                    mid_point = len(route.node_sequence) // 2
                    if mid_point >= min_segment_length:
                        segment_score = partial_score * 0.8  # 部分匹配降权
                        if segment_score > best_score:
                            best_score = segment_score
                            best_match = {
                                'route': route,
                                'start_idx': 0,
                                'end_idx': mid_point,
                                'match_type': 'partial_forward',
                                'score': segment_score,
                                'matched_locations': list(common_locations)
                            }

                # 如果都有相同的目的地，可能可以复用整段
                elif partial_score > 0.5:
                    if partial_score > best_score:
                        best_score = partial_score
                        best_match = {
                            'route': route,
                            'start_idx': 0,
                            'end_idx': len(route.node_sequence),
                            'match_type': 'full_segment',
                            'score': partial_score,
                            'matched_locations': list(common_locations)
                        }

        if best_match and best_match['score'] >= 0.5:
            logger.info(f"[RouteMemory v3.0] 找到部分路线匹配: "
                       f"type={best_match['match_type']}, "
                       f"score={best_match['score']:.3f}, "
                       f"segment=[{best_match['start_idx']}:{best_match['end_idx']}]")
            return best_match

        return None

    def get_route_segment(self, route: RouteMemory,
                          start_idx: int, end_idx: int) -> Optional[Dict]:
        """
        [v3.0] 获取路线段落

        Args:
            route: 路线对象
            start_idx: 起始索引
            end_idx: 结束索引

        Returns:
            段落信息字典
        """
        if start_idx < 0 or end_idx > len(route.node_sequence) or start_idx >= end_idx:
            return None

        return {
            'node_sequence': route.node_sequence[start_idx:end_idx],
            'action_history': route.action_history[start_idx:end_idx],
            'visual_features': route.visual_features[start_idx:end_idx] if route.visual_features is not None else None,
            'length': end_idx - start_idx
        }

    def estimate_route_confidence(self, route: RouteMemory, instruction: str) -> float:
        """
        [v3.0] 估计路线匹配的置信度

        综合考虑指令相似度、路线完整性等因素

        Args:
            route: 路线对象
            instruction: 查询指令

        Returns:
            置信度 (0-1)
        """
        query_parsed = self._parse_instruction(instruction)
        route_parsed = self._parse_instruction(route.start_instruction)

        # 基础相似度
        base_score = self._compute_instruction_similarity(query_parsed, route_parsed)

        # 路线完整性加成
        completeness_bonus = 0.1 if route.is_complete else 0.0

        # 关键帧数量加成 (更多关键帧意味着更好的导航)
        keyframe_ratio = len(route.keyframe_indices) / max(len(route.node_sequence), 1)
        keyframe_bonus = min(0.1, keyframe_ratio * 0.5)

        # 路线长度惩罚 (过长的路线可能不太可靠)
        length_penalty = 0.0
        if len(route.node_sequence) > 200:
            length_penalty = 0.05

        confidence = min(1.0, base_score + completeness_bonus + keyframe_bonus - length_penalty)
        return confidence

    def get_route_statistics(self) -> Dict:
        """
        [v3.0] 获取路线统计信息

        Returns:
            统计信息字典
        """
        total_routes = len(self.saved_routes)
        complete_routes = sum(1 for r in self.saved_routes.values() if r.is_complete)
        total_steps = sum(len(r.node_sequence) for r in self.saved_routes.values())
        total_keyframes = sum(len(r.keyframe_indices) for r in self.saved_routes.values())

        return {
            'total_routes': total_routes,
            'complete_routes': complete_routes,
            'total_steps': total_steps,
            'total_keyframes': total_keyframes,
            'avg_route_length': total_steps / max(total_routes, 1),
            'avg_keyframes_per_route': total_keyframes / max(total_routes, 1),
            'fuzzy_match_enabled': self.enable_fuzzy_match,
            'fuzzy_threshold': self.fuzzy_match_threshold
        }
