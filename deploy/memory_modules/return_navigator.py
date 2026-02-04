#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 返回导航模块

执行返回起点的导航功能。
"""

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class ReturnNavigator:
    """返回导航器 - 执行返回起点的导航"""

    def __init__(self, topo_map, route_memory):
        """
        初始化返回导航器

        Args:
            topo_map: TopologicalMapManager 实例
            route_memory: RouteMemoryManager 实例
        """
        self.topo_map = topo_map
        self.route_memory = route_memory
        self.is_returning = False
        self.return_path: List[Tuple[int, List[int]]] = []
        self.current_return_index = 0

    def start_return(self) -> bool:
        """开始返回导航"""
        start_node = self.route_memory.get_start_node()
        if start_node is None:
            logger.warning("无法开始返回导航: 没有记录的起点")
            return False

        current_node = self.topo_map.last_node_id
        if current_node is None:
            logger.warning("无法开始返回导航: 当前位置未知")
            return False

        # 尝试使用拓扑图规划路径
        topo_path = self.topo_map.find_path(current_node, start_node)

        if topo_path and len(topo_path) > 1:
            # 使用拓扑图路径
            logger.info(f"使用拓扑图规划返回路径: {topo_path}")
            self.return_path = []
            for i in range(len(topo_path) - 1):
                actions = self.topo_map.get_edge_actions(topo_path[i], topo_path[i+1])
                self.return_path.append((topo_path[i+1], actions))
        else:
            # 使用轨迹回溯
            logger.info("拓扑图路径不可用，使用轨迹回溯")
            self.return_path = self.route_memory.get_return_trajectory()

        if not self.return_path:
            logger.warning("无法生成返回路径")
            return False

        self.is_returning = True
        self.current_return_index = 0
        logger.info(f"返回导航已启动: 路径长度={len(self.return_path)}")
        return True

    def get_next_return_action(self) -> Tuple[Optional[List[int]], bool]:
        """
        获取下一个返回动作

        Returns:
            (action, is_complete): 动作和是否完成
        """
        if not self.is_returning or self.current_return_index >= len(self.return_path):
            self.is_returning = False
            return None, True

        _, actions = self.return_path[self.current_return_index]
        self.current_return_index += 1

        is_complete = self.current_return_index >= len(self.return_path)
        if is_complete:
            self.is_returning = False
            logger.info("返回导航完成")

        return actions, is_complete

    def stop_return(self):
        """停止返回导航"""
        self.is_returning = False
        self.return_path = []
        self.current_return_index = 0

    def get_return_progress(self) -> dict:
        """获取返回导航进度"""
        if not self.is_returning:
            return {
                "is_returning": False,
                "progress": 0,
                "total": 0
            }
        return {
            "is_returning": True,
            "progress": self.current_return_index,
            "total": len(self.return_path)
        }
