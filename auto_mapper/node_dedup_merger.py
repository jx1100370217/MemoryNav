#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
节点去重与合并器 v1

功能:
1. 同帧合并: 同一 timestamp 的多个 node → 合并为一个，最近的为主名，其余为别名
2. VPR 去重: 所有 node 两两比较 VPR 相似度，超过阈值 → 合并

别名字段:
  self_position.aliases: ["摩尔会议室", ...]
  self_position.aliases_eng: ["Moore Meeting Room", ...]
"""

import logging
import shutil
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class NodeDedupMerger:
    """节点去重与合并器"""

    def __init__(self, vpr_dedup_threshold: float = 0.80):
        """
        Args:
            vpr_dedup_threshold: VPR 相似度超过此阈值的 node 对将被合并
        """
        self.vpr_dedup_threshold = vpr_dedup_threshold
        logger.info(f"NodeDedupMerger v1 initialized (vpr_threshold={vpr_dedup_threshold})")

    # ==================================================================
    # 同帧合并
    # ==================================================================
    def merge_same_frame_nodes(self, created_nodes: List[Dict],
                               output_dir: Path) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """将同一 timestamp 的多个 node 合并为一个

        策略: 同一帧检测到多个门牌/标识时，保留第一个作为主名，其余作为别名。
        (第一个 = 在 created_nodes 中排序靠前的，通常是更高优先级 camera 检测到的)

        Args:
            created_nodes: 当前所有 node 列表
            output_dir: merged_labeled_data 目录

        Returns:
            (merged_nodes, alias_map)
            - merged_nodes: 合并后的 node 列表
            - alias_map: {keeper_position_id: [removed_node_info, ...]}
        """
        # 按 timestamp 分组
        ts_groups = defaultdict(list)
        for node in created_nodes:
            ts_groups[node['timestamp']].append(node)

        merged_nodes = []
        alias_map = {}  # keeper_id -> [removed_nodes]

        for ts, group in sorted(ts_groups.items(), key=lambda x: x[1][0]['frame_index']):
            if len(group) == 1:
                merged_nodes.append(group[0])
                continue

            # 同帧多 node: 保留第一个，其余为别名
            keeper = group[0]
            removed = group[1:]

            logger.info(f"同帧合并: timestamp={ts}, "
                        f"保留 [{keeper['position_name']}], "
                        f"别名 {[n['position_name'] for n in removed]}")

            alias_map[keeper['position_id']] = removed

            # 删除被合并 node 的目录
            for node in removed:
                node_dir = Path(node['node_dir'])
                if node_dir.exists():
                    shutil.rmtree(node_dir)
                    logger.info(f"  删除目录: {node_dir.name}")

            merged_nodes.append(keeper)

        if alias_map:
            logger.info(f"同帧合并完成: {len(created_nodes)} → {len(merged_nodes)} 个 node")
        else:
            logger.info("同帧合并: 无需合并")

        return merged_nodes, alias_map

    # ==================================================================
    # VPR 去重
    # ==================================================================
    def dedup_by_vpr(self, created_nodes: List[Dict],
                     distance_estimator,
                     output_dir: Path) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """通过 VPR 特征相似度去重

        对所有 node 两两比较 VPR 相似度，如果超过阈值则合并。
        优先保留有语义名称(非 VPR 自动生成)的 node。

        Args:
            created_nodes: 当前所有 node 列表
            distance_estimator: NodeDistanceEstimator (含 node_features)
            output_dir: merged_labeled_data 目录

        Returns:
            (deduped_nodes, alias_map)
        """
        if len(created_nodes) < 2:
            logger.info("VPR 去重: 少于 2 个 node, 跳过")
            return created_nodes, {}

        n = len(created_nodes)
        # 计算所有 node 对的 VPR 相似度
        sim_matrix = {}
        for i in range(n):
            for j in range(i + 1, n):
                id_i = created_nodes[i]['position_id']
                id_j = created_nodes[j]['position_id']

                feat_i = distance_estimator.node_features.get(id_i)
                feat_j = distance_estimator.node_features.get(id_j)

                if feat_i is None or feat_j is None:
                    continue

                sim = distance_estimator.compute_similarity(feat_i, feat_j)
                sim_matrix[(i, j)] = sim

                if sim > self.vpr_dedup_threshold:
                    logger.info(f"  VPR 相似: node {id_i}({created_nodes[i]['position_name']}) "
                                f"<-> node {id_j}({created_nodes[j]['position_name']}) "
                                f"sim={sim:.4f} > {self.vpr_dedup_threshold}")

        # 找出需要合并的 node 对 (贪心: 按相似度降序处理)
        to_merge = [(i, j, sim) for (i, j), sim in sim_matrix.items()
                    if sim > self.vpr_dedup_threshold]
        to_merge.sort(key=lambda x: x[2], reverse=True)

        removed_indices = set()
        alias_map = {}  # keeper_id -> [removed_node_info]

        for i, j, sim in to_merge:
            if i in removed_indices or j in removed_indices:
                continue

            node_i = created_nodes[i]
            node_j = created_nodes[j]

            # 策略: 保留帧序靠前的 node (路径上先经过的)
            keeper, removed = node_i, node_j
            removed_idx = j

            logger.info(f"VPR 去重: 合并 [{removed['position_name']}] → [{keeper['position_name']}] "
                        f"(sim={sim:.4f})")

            removed_indices.add(removed_idx)

            if keeper['position_id'] not in alias_map:
                alias_map[keeper['position_id']] = []
            alias_map[keeper['position_id']].append(removed)

            # 删除被合并 node 的目录
            node_dir = Path(removed['node_dir'])
            if node_dir.exists():
                shutil.rmtree(node_dir)
                logger.info(f"  删除目录: {node_dir.name}")

            # 清理 distance_estimator 中的特征
            rem_id = removed['position_id']
            distance_estimator.node_features.pop(rem_id, None)
            distance_estimator.node_frames.pop(rem_id, None)

        deduped = [n for i, n in enumerate(created_nodes) if i not in removed_indices]

        if alias_map:
            logger.info(f"VPR 去重完成: {len(created_nodes)} → {len(deduped)} 个 node")
        else:
            logger.info("VPR 去重: 无需去重")

        return deduped, alias_map

    # ==================================================================
    # 写入别名到 node_position_info.json
    # ==================================================================
    @staticmethod
    def write_aliases(output_dir: Path, alias_map: Dict[str, List[Dict]]):
        """将别名信息写入 keeper node 的 node_position_info.json

        Args:
            output_dir: merged_labeled_data 目录
            alias_map: {keeper_position_id: [removed_node_info, ...]}
        """
        import json

        for keeper_id, removed_nodes in alias_map.items():
            info_file = output_dir / keeper_id / "node_position_info.json"
            if not info_file.exists():
                logger.warning(f"  node_position_info.json 不存在: {info_file}")
                continue

            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)

            # 收集别名
            aliases = []
            aliases_eng = []
            for removed in removed_nodes:
                name = removed.get('position_name', '')
                name_eng = removed.get('position_name_eng', '')
                if name:
                    aliases.append(name)
                if name_eng:
                    aliases_eng.append(name_eng)

            if aliases:
                # 追加到已有别名 (如果有的话)
                existing_aliases = info['self_position'].get('aliases', [])
                existing_aliases_eng = info['self_position'].get('aliases_eng', [])
                info['self_position']['aliases'] = existing_aliases + aliases
                info['self_position']['aliases_eng'] = existing_aliases_eng + aliases_eng

                with open(info_file, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)

                logger.info(f"  node {keeper_id}: 写入别名 {aliases}")

    # ==================================================================
    # 统一入口
    # ==================================================================
    def run(self, created_nodes: List[Dict],
            distance_estimator,
            output_dir: Path) -> List[Dict]:
        """执行完整的去重与合并流程

        顺序:
        1. 同帧合并 (处理同 timestamp 的 node)
        2. VPR 去重 (处理 VPR 相似度过高的 node)

        Args:
            created_nodes: 当前所有 node 列表
            distance_estimator: NodeDistanceEstimator
            output_dir: merged_labeled_data 目录

        Returns:
            去重合并后的 node 列表
        """
        logger.info("=" * 50)
        logger.info("Phase 1.6: 节点去重与合并")
        logger.info("=" * 50)

        all_alias_maps = {}

        # Step 1: 同帧合并
        nodes, alias_map_frame = self.merge_same_frame_nodes(created_nodes, output_dir)
        all_alias_maps.update(alias_map_frame)

        # Step 2: VPR 去重
        nodes, alias_map_vpr = self.dedup_by_vpr(nodes, distance_estimator, output_dir)
        # 合并别名 (VPR 去重可能涉及已有别名的 node)
        for k, v in alias_map_vpr.items():
            if k in all_alias_maps:
                all_alias_maps[k].extend(v)
            else:
                all_alias_maps[k] = v

        # Step 3: 写入别名
        if all_alias_maps:
            self.write_aliases(output_dir, all_alias_maps)

        logger.info(f"去重与合并完成: {len(created_nodes)} → {len(nodes)} 个 node")
        return nodes
