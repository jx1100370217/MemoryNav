#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - VLM场景描述生成模块 v5.0

使用 Qwen2.5-VL-7B-Instruct 生成场景描述、语义标签和节点名称。

v5.0 核心改进:
1. 添加环视相机布局信息到提示词，让VLM理解方位
2. 改进节点命名：基于场景特征生成有意义的名称而非"1号"、"2号"
3. 继承v4的抗幻觉机制

相机布局说明:
- camera_1: 头部前方偏右37.5度，鱼眼等角投影，HFOV=190度
- camera_2: 头部前方偏左37.5度，鱼眼等角投影，HFOV=190度
- camera_3: 头部后方偏左37.5度，鱼眼等角投影，HFOV=190度
- camera_4: 头部后方偏右37.5度，鱼眼等角投影，HFOV=190度
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

import torch

from .config import MemoryNavigationConfig

logger = logging.getLogger(__name__)

# 相机布局常量
CAMERA_LAYOUT = {
    'camera_1': {
        'direction': '前方偏右',
        'angle': 37.5,
        'type': '鱼眼等角投影',
        'hfov': 190,
        'description': '前方偏右37.5度（覆盖右前方和正右方）'
    },
    'camera_2': {
        'direction': '前方偏左',
        'angle': 37.5,
        'type': '鱼眼等角投影',
        'hfov': 190,
        'description': '前方偏左37.5度（覆盖左前方和正左方）'
    },
    'camera_3': {
        'direction': '后方偏左',
        'angle': 37.5,
        'type': '鱼眼等角投影',
        'hfov': 190,
        'description': '后方偏左37.5度（覆盖左后方和正后方）'
    },
    'camera_4': {
        'direction': '后方偏右',
        'angle': 37.5,
        'type': '鱼眼等角投影',
        'hfov': 190,
        'description': '后方偏右37.5度（覆盖右后方和正后方）'
    }
}


class SceneDescriptionGeneratorV5:
    """
    方位感知场景描述生成器 v5.0

    主要改进:
    1. 方位感知：提示词包含相机布局信息，生成准确的方位描述
    2. 智能命名：基于场景特征生成有意义的节点名称
    3. 继承v4抗幻觉：只描述100%确定可见的内容
    """

    def __init__(self, config: MemoryNavigationConfig):
        self.config = config
        self.is_available = False
        self.model = None
        self.processor = None

        # 已生成的描述缓存 (用于对比去重)
        self._description_cache = []
        self._max_cache_size = 50

        # 已生成的节点名称缓存（用于确保唯一性）
        self._node_name_cache = set()

        if not config.vlm_enabled:
            logger.info("VLM场景描述生成器已禁用")
            return

        try:
            logger.info(f"正在加载 Qwen2.5-VL 模型: {config.vlm_model_path}")
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                config.vlm_model_path,
                trust_remote_code=True
            )

            # 确定设备
            vlm_device = config.vlm_device
            if vlm_device.startswith("cuda:"):
                gpu_id = int(vlm_device.split(":")[1])
                device_map = {"": gpu_id}
            else:
                device_map = "auto"

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.vlm_model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True
            )

            if hasattr(self.model, 'set_default_attn_implementation'):
                try:
                    self.model.set_default_attn_implementation("flash_attention_2")
                except Exception:
                    pass

            self.is_available = True
            logger.info(f"Qwen2.5-VL 模型加载成功 (设备: {vlm_device})")

        except Exception as e:
            logger.warning(f"Qwen2.5-VL 加载失败: {e}，将使用简化描述")
            self.is_available = False

    def _get_camera_layout_prompt(self) -> str:
        """生成相机布局说明文本"""
        return """【相机布局说明 - 请据此描述方位】
这4张图来自机器人头部的4个环视相机，按顺序分别是：
- 图1 (camera_1): 前方偏右37.5度，覆盖右前方到正右方（HFOV=190度鱼眼）
- 图2 (camera_2): 前方偏左37.5度，覆盖左前方到正左方（HFOV=190度鱼眼）
- 图3 (camera_3): 后方偏左37.5度，覆盖左后方到正后方（HFOV=190度鱼眼）
- 图4 (camera_4): 后方偏右37.5度，覆盖右后方到正后方（HFOV=190度鱼眼）

鱼眼镜头特点：图像边缘有畸变，中心区域最清晰。"""

    def generate_description(self, images: Dict[str, np.ndarray]) -> str:
        """
        v5.0 方位感知场景描述生成

        核心原则：
        1. 根据相机布局准确描述各方位的内容
        2. 只描述100%确定看到的内容，不猜测
        """
        if not self.is_available or not images:
            return self._fallback_description(images)

        try:
            pil_images = self._prepare_images(images)
            if not pil_images:
                return self._fallback_description(images)

            # v5.0 核心改进: 带方位信息的提示词 (强调每张图不同)
            description_prompt = f"""{self._get_camera_layout_prompt()}

【任务】仔细观察这4张环视图，描述你当前所在的具体位置。

【重要规则】
1. 根据相机方位准确描述（前方、后方、左侧、右侧）
2. 只描述100%确定看到的东西，看不清就写"无"
3. 每个位置都不同，请仔细观察当前图片的实际内容
4. 禁止猜测或使用通用描述

【禁止的内容】
- 马桶、厕所、卫生间（常见幻觉）
- 公司logo、落地窗、电梯厅（除非真的清晰可见）

输出格式每行一条，如实描述：
【场景类型】：实际看到的场景
【正前方】：前方具体有什么
【右侧】：右边具体有什么
【左侧】：左边具体有什么
【正后方】：后方具体有什么
【显著特征】：最有辨识度的物体（带颜色）
【墙面地面】：墙和地面的实际颜色/图案

请根据当前图片如实描述："""

            # 构建消息
            content = []
            for pil_img in pil_images[:4]:
                content.append({"type": "image", "image": pil_img})
            content.append({"type": "text", "text": description_prompt})

            messages = [{"role": "user", "content": content}]

            # 处理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text,
                images=pil_images[:4],
                return_tensors="pt"
            ).to(self.model.device)

            # 适中的temperature：平衡准确性和多样性
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.1,  
                    top_p=0.9,
                    repetition_penalty=1.1
                )

            description = self.processor.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # 缓存描述用于去重检查
            self._update_cache(description)

            logger.info(f"[VLM v5] 生成场景描述: {description[:100]}...")
            return description

        except Exception as e:
            logger.warning(f"VLM生成描述失败: {e}")
            return self._fallback_description(images)


    def extract_semantic_labels(self, images: Dict[str, np.ndarray]) -> List[str]:
        """
        v5.0 方位感知语义标签提取

        核心改进：标签包含方位信息
        """
        if not self.is_available or not images:
            return self._fallback_labels(images)

        try:
            pil_images = self._prepare_images(images)
            if not pil_images:
                return self._fallback_labels(images)

            # v5.0 改进: 带方位信息的标签提取 (去掉具体示例防止复制)
            label_prompt = f"""{self._get_camera_layout_prompt()}

【任务】仔细观察这4张图，列出你100%确定看到的物体。

【要求】
1. 只写确定看到的，看不清就不写
2. 带颜色，如"红色灭火器"、"蓝色墙面"
3. 每个场景的物体都不同，请仔细观察当前图片
4. 禁止照抄其他场景的标签

【禁止的内容】
- 马桶、厕所、卫生间（除非真的看到）
- 公司logo、标语（除非能看清文字）


输出格式：逗号分隔，6-10个标签
注意：每张图片的内容不同，请根据实际画面描述！

当前图片中确定可见的物体："""

            content = []
            for pil_img in pil_images[:4]:
                content.append({"type": "image", "image": pil_img})
            content.append({"type": "text", "text": label_prompt})

            messages = [{"role": "user", "content": content}]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text,
                images=pil_images[:4],
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=True,
                    temperature=0.1,  
                    top_p=0.9
                )

            labels_text = self.processor.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # 解析标签：支持中英文逗号、顿号、句号分隔
            # 先统一替换分隔符
            labels_text = labels_text.replace('，', ',').replace('、', ',').replace('。', ',')
            labels_text = labels_text.replace('\n', ',').replace(';', ',').replace('；', ',')
            raw_labels = []
            for label in labels_text.split(','):
                label = label.strip().strip('。').strip('.').strip('-').strip('- ').strip()
                if label:
                    raw_labels.append(label)

            # 过滤幻觉标签
            filtered_labels = self._filter_hallucination_labels(raw_labels)
            unique_labels = self._deduplicate_labels(filtered_labels)

            logger.info(f"[VLM v5] 提取语义标签: {unique_labels}")
            return unique_labels[:12]

        except Exception as e:
            logger.warning(f"VLM提取标签失败: {e}")
            return self._fallback_labels(images)

    def _filter_hallucination_labels(self, labels: List[str]) -> List[str]:
        """过滤幻觉标签"""
        hallucination_labels = {
            '马桶', '厕所', '卫生间', '洗手台', '镜子',
            '公司logo', 'logo', '标语',
        }

        filtered = []
        for label in labels:
            label_lower = label.lower()
            is_hallucination = False
            for h in hallucination_labels:
                if h in label_lower:
                    is_hallucination = True
                    break
            if not is_hallucination:
                filtered.append(label)

        return filtered

    def generate_node_name(self, scene_description: str, semantic_labels: List[str],
                           images: Dict[str, np.ndarray] = None) -> str:
        """
        v5.0 智能节点命名

        核心改进：基于场景特征生成有意义的名称
        格式：场景类型-显著特征 或 显著特征-方位
        """
        if not scene_description and not semantic_labels:
            return "未知位置"

        # 优先使用VLM生成有意义的名称
        if self.is_available and images:
            try:
                vlm_name = self._generate_node_name_with_vlm(images, scene_description, semantic_labels)
                if vlm_name and vlm_name != "未知位置" and len(vlm_name) >= 3:
                    # 过滤幻觉名称
                    if not self._is_hallucination_name(vlm_name):
                        vlm_name = self._ensure_unique_name(vlm_name)
                        logger.info(f"[VLM v5] 生成节点名称: {vlm_name}")
                        return vlm_name
            except Exception as e:
                logger.warning(f"[VLM v5] 节点名称生成失败: {e}")

        # 回退方法
        return self._generate_node_name_fallback(scene_description, semantic_labels)

    def _is_hallucination_name(self, name: str) -> bool:
        """检查名称是否包含幻觉"""
        hallucination_keywords = ['卫生间', '厕所', '马桶', 'logo', '1号', '2号', '3号', '号位']
        name_lower = name.lower()
        for keyword in hallucination_keywords:
            if keyword in name_lower:
                return True
        return False

    def _generate_node_name_with_vlm(self, images: Dict[str, np.ndarray],
                                      scene_description: str = None,
                                      semantic_labels: List[str] = None) -> str:
        """使用VLM生成节点名称 v5.0"""
        pil_images = self._prepare_images(images)
        if not pil_images:
            return None

        # 构建上下文提示
        context = ""
        if semantic_labels:
            context = f"已确认的特征：{', '.join(semantic_labels[:6])}\n"
        if scene_description:
            # 提取场景描述的关键信息
            lines = scene_description.split('\n')[:4]
            context += f"场景概况：{'; '.join(lines)}\n"

        # v5.0 智能命名提示词
        name_prompt = f"""{context}
【任务】为这个位置起一个有意义的名字（5-12字），让人一看就知道是什么地方。

【命名规则】
1. 使用显著特征命名，如"红灭火器旁"、"六边形墙走廊"、"玻璃门前"
2. 如有明显地标，优先使用地标命名，如"出口标志处"、"休息椅旁"
3. 可组合场景和特征，如"走廊-灰座椅旁"、"大厅-玻璃门前"
4. 不要用"1号"、"2号"、"位置A"这种无意义编号

【禁止】
- 不要用"卫生间"、"厕所"等猜测性场景名

【好的例子】
- 六边形墙走廊
- 红灭火器拐角
- 玻璃隔断旁走廊
- 出口标志处
- 灰色座椅区

直接输出名字（5-12字）："""

        content = []
        # 使用前两张图（覆盖前方）
        for pil_img in pil_images[:2]:
            content.append({"type": "image", "image": pil_img})
        content.append({"type": "text", "text": name_prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=pil_images[:2],
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.1,  
                top_p=0.85
            )

        node_name = self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        # 清理名称
        node_name = node_name.split('\n')[0].strip()
        node_name = node_name.strip('"').strip("'").strip('「').strip('」')

        # 移除常见无意义前缀
        for prefix in ['名字：', '名称：', '节点名称：', '位置：', '名字是：', '答案：']:
            if node_name.startswith(prefix):
                node_name = node_name[len(prefix):].strip()

        # 限制长度
        if len(node_name) > 14:
            node_name = node_name[:14]

        return node_name if node_name else None

    def _generate_node_name_fallback(self, scene_description: str,
                                      semantic_labels: List[str]) -> str:
        """回退方法：基于解析规则生成名称 v5.0"""
        combined_text = (scene_description or "") + " " + " ".join(semantic_labels or [])

        # 1. 提取显著特征（按优先级）
        distinctive_feature = None

        # 颜色+物体组合（最有辨识度）
        color_patterns = [
            (r'红[色]?(灭火器|消防栓|箱)', '红灭火器'),
            (r'绿[色]?(植物?|盆栽?|出口标志?)', '绿色标志'),
            (r'蓝[灰]?[色]?(六边形|图案)', '六边形墙'),
            (r'灰[色]?(座椅|椅子?|沙发)', '灰座椅'),
            (r'玻璃(门|隔断|墙)', '玻璃隔断'),
            (r'(出口|安全出口)标志', '出口标志'),
        ]
        for pattern, short_name in color_patterns:
            if re.search(pattern, combined_text):
                distinctive_feature = short_name
                break

        # 2. 提取空间特征
        spatial_feature = None
        spatial_patterns = [
            (r'转角|拐角', '转角'),
            (r'尽头|末端', '尽头'),
            (r'入口|门口', '入口'),
            (r'中段|中间', '中段'),
            (r'交叉|十字', '交叉口'),
        ]
        for pattern, short_form in spatial_patterns:
            if re.search(pattern, combined_text):
                spatial_feature = short_form
                break

        # 3. 提取场景类型
        scene_type = "走廊"  # 默认
        scene_patterns = [
            (r'大厅|大堂', '大厅'),
            (r'办公', '办公区'),
            (r'休息', '休息区'),
            (r'走廊|通道', '走廊'),
        ]
        for pattern, scene_name in scene_patterns:
            if re.search(pattern, combined_text):
                scene_type = scene_name
                break

        # 4. 从标签中提取未被匹配的特征
        if not distinctive_feature and semantic_labels:
            generic_words = {'走廊', '场景', '位置', '区域', '空间', '墙', '地板', '天花板'}
            for label in semantic_labels:
                label_clean = label.strip()
                if len(label_clean) >= 2 and label_clean not in generic_words:
                    if any(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in label_clean):
                        distinctive_feature = label_clean[:6]  # 限制长度
                        break

        # 5. 组合名称
        if distinctive_feature and spatial_feature:
            node_name = f"{distinctive_feature}{spatial_feature}"
        elif distinctive_feature:
            node_name = f"{scene_type}-{distinctive_feature}"
        elif spatial_feature:
            node_name = f"{scene_type}{spatial_feature}"
        else:
            node_name = scene_type

        # 限制长度
        if len(node_name) > 14:
            node_name = node_name[:14]

        return self._ensure_unique_name(node_name)

    def _ensure_unique_name(self, name: str) -> str:
        """确保节点名称唯一"""
        if name not in self._node_name_cache:
            self._node_name_cache.add(name)
            return name

        # 名称已存在，添加后缀
        suffix = 2
        while f"{name}-{suffix}" in self._node_name_cache:
            suffix += 1

        unique_name = f"{name}-{suffix}"
        self._node_name_cache.add(unique_name)
        return unique_name

    def _prepare_images(self, images: Dict[str, np.ndarray]) -> List[Image.Image]:
        """准备PIL图像列表"""
        pil_images = []
        for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            if cam_id in images and images[cam_id] is not None:
                img = images[cam_id]
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img))
        return pil_images

    def _deduplicate_labels(self, labels: List[str]) -> List[str]:
        """标签去重并保持顺序"""
        unique = []
        seen = set()
        for label in labels:
            label_lower = label.lower().strip()
            if label_lower not in seen and len(label) >= 2:
                unique.append(label.strip())
                seen.add(label_lower)
        return unique

    def _update_cache(self, description: str):
        """更新描述缓存"""
        self._description_cache.append(description)
        if len(self._description_cache) > self._max_cache_size:
            self._description_cache.pop(0)

    def _fallback_description(self, images: Dict[str, np.ndarray]) -> str:
        """无VLM时的回退描述"""
        if not images:
            return "未知场景"
        cam_count = sum(1 for v in images.values() if v is not None)
        return f"导航场景 (包含{cam_count}个相机视角)"

    def _fallback_labels(self, images: Dict[str, np.ndarray]) -> List[str]:
        """无VLM时的回退标签"""
        return ["走廊"]

    def generate_complete_scene_info(self, images: Dict[str, np.ndarray]) -> Tuple[str, List[str], str]:
        """
        一次性生成完整的场景信息 v5.0

        Args:
            images: 环视相机图像字典

        Returns:
            (scene_description, semantic_labels, node_name) 三元组
        """
        scene_description = self.generate_description(images)
        semantic_labels = self.extract_semantic_labels(images)
        node_name = self.generate_node_name(scene_description, semantic_labels, images)

        return scene_description, semantic_labels, node_name

    def clear_name_cache(self):
        """清空节点名称缓存（新建地图时调用）"""
        self._node_name_cache.clear()
        logger.info("[VLM v5] 节点名称缓存已清空")
