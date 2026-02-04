# 场景描述生成器 v3.0 优化报告

## 问题背景

v2.x 版本的场景描述生成器存在严重的节点区分度问题：
- 不同位置生成的场景描述几乎完全相同
- 语义标签高度重复
- 节点名称无法区分不同位置

**v2.x 测试结果（10个样本）：**
| 指标 | 数值 | 问题 |
|------|------|------|
| 场景描述相似度 | 0.975 | 几乎完全相同 |
| 标签重叠率 | 0.628 | 过高 |
| 节点名称相似度 | 0.843 | 极高 |
| 唯一名称数 | 2/10 | 严重不足 |
| 唯一场景描述数 | 3/10 | 严重不足 |
| 唯一标签数 | 13 | 偏少 |

## 优化方案

基于学术研究最佳实践进行了以下优化：

### 1. 参考文献
- **SENT-Map**: Semantically Enhanced Topological Maps with Foundation Models
- **ROOT**: VLM-based System for Indoor Scene Understanding
- **Visual Landmark Sequence**: Indoor Topological Localization
- **HOV-SG**: Hierarchical Open-Vocabulary 3D Scene Graphs

### 2. 核心改进

#### 2.1 分层语义描述
将场景描述从单一文本改为结构化的多层描述：
- L1: 场景类型（走廊/房间/大厅等）
- L2: 空间结构（形状、方向、连通性）
- L3: 关键对象（具体物体及其属性）
- L4: 独特细节（只有此位置才有的特征）

#### 2.2 强制差异化提示词
在提示词中强调必须找出"此位置独有"的特征，避免生成通用描述。

#### 2.3 多层次标签提取
参考 Visual Landmark Sequence 的分层地标概念：
- 场景标签：精确的场景类型
- 组合地标：物体+位置组合
- 显著物体：带属性的具体物体

#### 2.4 增加生成多样性
- 使用 temperature=0.7（原为0.0）
- 使用 top_p=0.9
- 允许采样（do_sample=True）

## 优化效果

**v3.0 测试结果（10个样本）：**
| 指标 | v2.x | v3.0 | 改进 |
|------|------|------|------|
| 场景描述相似度 | 0.975 | 0.560 | ↓ 42.5% |
| 标签重叠率 | 0.628 | 0.278 | ↓ 55.7% |
| 节点名称相似度 | 0.843 | 0.400 | ↓ 52.5% |
| 唯一名称数 | 2/10 | 8/10 | ↑ 300% |
| 唯一场景描述数 | 3/10 | 10/10 | ↑ 233% |
| 唯一标签数 | 13 | 23 | ↑ 77% |

## 生成示例对比

### v2.x 生成结果（所有样本几乎相同）
```
场景描述：【位置类型】走廊
【空间布局】走廊向前延伸，左侧有房间，右侧是玻璃墙
【显著地标】绿色安全出口标志、玻璃隔断墙
【独特特征】墙面六边形图案

标签：走廊,玻璃隔断,六边形墙纸,标识牌,绿植
名称：走廊-绿安全出口
```

### v3.0 生成结果（各不相同）
**样本1:**
```
场景：办公室走廊中段
形状：直走廊
前方：尽头是墙
左侧：紧邻21号门
右侧：灰色沙发区
可见物体：红色灭火器1个，黑色办公椅3把
独特标记：墙上有公司logo

标签：走廊中段,玻璃门,紧急出口标志,黑座椅,灰墙,蓝地砖
名称：走廊-玻璃门前
```

**样本6:**
```
场景：休息区
形状：矩形
前方：远处有出口指示标志
左侧：有沙发和茶几
右侧：有植物和另一张桌子
可见物体：灰色沙发1个，白色茶几1张，绿色植物1株
独特标记：没有明显独特标记

标签：走廊中段,沙发区入口,红灭火器,绿植1盆,灰沙发,安全出口标志
名称：沙发区-红灭火器旁
```

## 使用方法

### 配置选择
在 `MemoryNavigationConfig` 中设置：
```python
config = MemoryNavigationConfig(
    vlm_enabled=True,
    vlm_version="v3"  # 使用v3版本（推荐）
    # vlm_version="v2"  # 使用旧版本
)
```

### 代码位置
- v3 实现：`deploy/memory_modules/scene_description_v3.py`
- v2 实现：`deploy/memory_modules/scene_description.py`

## 文件变更

1. **新增文件**
   - `deploy/memory_modules/scene_description_v3.py` - v3版本实现

2. **修改文件**
   - `deploy/memory_modules/config.py` - 添加 `vlm_version` 配置项
   - `deploy/memory_modules/__init__.py` - 导出 v3 类
   - `deploy/ws_proxy_with_memory.py` - 支持版本选择

3. **测试脚本**
   - `scripts/test_scene_description_quality.py` - 质量评估测试
   - `scripts/test_scene_description_v3.py` - v2 vs v3 对比测试
   - `scripts/test_topological_map_v3.py` - 集成测试

## 结论

v3.0 版本的场景描述生成器显著改善了节点区分度问题：
- 所有关键指标改进超过 40%
- 唯一名称比例从 20% 提升到 80%
- 唯一场景描述比例达到 100%

建议在生产环境中使用 v3.0 版本。
