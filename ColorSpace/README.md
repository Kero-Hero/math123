# 色域转换模型 (ColorSpace Mapping)

## 项目概述

本项目实现了一个基于感知均匀性的色域转换模型，目标是在控制色彩偏差（deltaE）的前提下，尽可能保留色彩的感知均匀性。

## 核心特性

- **感知均匀性保持**: 在CIELAB色彩空间中进行处理，确保色彩感知的一致性
- **DeltaE约束**: 通过损失函数控制色彩偏差上限
- **多色域支持**: 
  - BT 2020 → sRGB 映射
  - 四通道 → 五通道映射
- **三个实现版本**:
  - Apple MLX (苹果芯片优化)
  - PyTorch GPU (CUDA/MPS自动检测)
  - PyTorch CPU

## 技术原理

1. **色彩空间转换**: RGB → XYZ → CIELAB
2. **感知均匀映射**: 在CIELAB空间中训练神经网络
3. **约束优化**: 使用deltaE作为损失函数的约束项
4. **数据采样**: 从完整色域中随机采样训练数据

## 项目结构

```
ColorSpace/
├── core/               # 核心算法
│   ├── color_conversion.py    # 色彩空间转换
│   ├── gamut_mapping.py       # 色域映射模型
│   └── loss_functions.py      # 损失函数(含deltaE)
├── models/             # 神经网络模型
│   ├── mlx_model.py           # Apple MLX实现
│   ├── pytorch_model.py       # PyTorch实现
│   └── base_model.py          # 基础模型类
├── data/               # 数据处理
│   ├── sampler.py             # 色域采样器
│   └── dataset.py             # 数据集定义
├── training/           # 训练脚本
│   ├── train_mlx.py           # MLX训练
│   ├── train_pytorch.py       # PyTorch训练
│   └── utils.py               # 训练工具
├── examples/           # 使用示例
└── requirements/       # 依赖管理
```

## 安装要求

### 基础依赖
```bash
pip install -r requirements/base.txt
```

### 平台特定依赖
- MLX版本: `pip install -r requirements/mlx.txt`
- PyTorch GPU版本: `pip install -r requirements/pytorch_gpu.txt`  
- PyTorch CPU版本: `pip install -r requirements/pytorch_cpu.txt`

## 快速开始

### 基础使用示例

```python
from models.pytorch_model import create_pytorch_mapper
from data.sampler import create_sampler

# 创建BT.2020到sRGB映射模型
model = create_pytorch_mapper(
    source_gamut='bt2020',
    target_gamut='srgb',
    deltaE_threshold=3.0,
    device='auto'
)

# 准备训练数据
train_sampler = create_sampler('bt2020', 'srgb', 'perceptual')

# 训练模型
model.train(
    train_sampler=train_sampler,
    epochs=50,
    batch_size=1024,
    learning_rate=0.001
)

# 使用模型进行色域映射
import numpy as np
bt2020_colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
srgb_colors = model.transform(bt2020_colors)
```

### 运行示例

```bash
# 简单使用示例
python examples/simple_example.py

# 完整功能演示
python examples/quickstart.py

# 训练自定义模型
python training/train_pytorch.py --source_gamut bt2020 --target_gamut srgb --epochs 100
```

## API 文档

### 主要功能

#### 1. 创建色域映射器

```python
from models.pytorch_model import create_pytorch_mapper
from models.mlx_model import create_mlx_mapper

# PyTorch版本
pytorch_mapper = create_pytorch_mapper(
    source_gamut='bt2020',      # 源色域: 'bt2020', 'srgb', '4ch', '5ch'
    target_gamut='srgb',        # 目标色域
    network_type='standard',    # 网络类型: 'standard', 'deep', 'wide'
    deltaE_threshold=3.0,       # deltaE阈值
    device='auto'               # 设备: 'auto', 'cuda', 'mps', 'cpu'
)

# MLX版本 (Apple Silicon)
mlx_mapper = create_mlx_mapper(
    source_gamut='bt2020',
    target_gamut='srgb',
    deltaE_threshold=3.0
)
```

#### 2. 数据采样

```python
from data.sampler import create_sampler

# 创建采样器
sampler = create_sampler(
    source_gamut='bt2020',
    target_gamut='srgb', 
    strategy='perceptual',      # 'uniform', 'perceptual', 'boundary', 'stratified'
    adaptive=True               # 自适应采样
)

# 生成样本
source_colors, target_colors = sampler.sample(1000)
```

#### 3. 色彩空间转换

```python
from core.color_conversion import *
import numpy as np

# RGB到LAB转换
rgb = np.array([[1.0, 0.0, 0.0]])
lab = rgb_to_lab(rgb, 'srgb')

# 计算deltaE
deltaE = delta_e_cie94(lab1, lab2)

# BT.2020到sRGB直接转换
srgb = bt2020_to_srgb_direct(bt2020_rgb)
```

#### 4. 训练模型

```python
# 训练配置
history = model.train(
    train_sampler=train_sampler,
    validation_sampler=val_sampler,
    epochs=100,
    batch_size=1024,
    learning_rate=0.001,
    patience=15,
    verbose=True
)

# 评估模型
metrics = model.evaluate(test_source, test_target)
print(f"平均deltaE: {metrics['mean_deltaE']:.3f}")
```

#### 5. 保存和加载模型

```python
# 保存模型
model.save_model('color_mapper.pth')

# 加载模型
model.load_model('color_mapper.pth')
```

### 多通道映射

```python
# 四通道到五通道映射
mapper = create_pytorch_mapper(
    source_gamut='4ch',
    target_gamut='5ch'
)

# 训练和使用流程相同
```

### 损失函数配置

```python
from core.loss_functions import ColorMappingLoss

# 自定义损失函数
loss_fn = ColorMappingLoss(
    deltaE_threshold=3.0,       # deltaE阈值
    deltaE_weight=1.0,          # deltaE损失权重
    perceptual_weight=1.0,      # 感知损失权重
    gamut_weight=0.5,           # 色域损失权重
    deltaE_method='cie94'       # deltaE计算方法
)
```

### 数据集管理

```python
from data.dataset import create_dataset

# 创建数据集管理器
dataset = create_dataset('bt2020', 'srgb', cache_dir='./cache')

# 获取数据
train_data = dataset.get_train_data(50000, 'perceptual')
val_data = dataset.get_validation_data(5000, 'uniform')
test_data = dataset.get_test_data(2000, 'uniform')

# 批量数据加载
from data.dataset import BatchDataLoader
loader = BatchDataLoader(train_source, train_target, batch_size=512)

for batch_source, batch_target in loader:
    # 训练批次
    pass
```

## 模型架构

### 网络设计

- **编码器**: 将源色域编码到CIELAB空间
- **映射网络**: 在感知均匀空间中进行非线性变换
- **解码器**: 映射到目标色域
- **约束层**: 确保deltaE在可接受范围内

### 损失函数

```python
loss = perceptual_loss + lambda_deltaE * deltaE_constraint + gamut_loss
```

- `perceptual_loss`: CIELAB空间中的感知损失
- `deltaE_constraint`: 色彩偏差约束
- `gamut_loss`: 色域边界约束

## 训练指南

### 1. 基础训练

```bash
python training/train_pytorch.py \
    --source_gamut bt2020 \
    --target_gamut srgb \
    --epochs 100 \
    --batch_size 1024 \
    --learning_rate 0.001 \
    --deltaE_threshold 3.0
```

### 2. 高级配置

```bash
python training/train_pytorch.py \
    --source_gamut bt2020 \
    --target_gamut srgb \
    --network_type deep \
    --sampling_strategy perceptual \
    --epochs 200 \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --device cuda \
    --output_dir ./models
```

### 3. 多通道训练

```bash
python training/train_pytorch.py \
    --source_gamut 4ch \
    --target_gamut 5ch \
    --epochs 150 \
    --batch_size 512
```

## 性能优化

### Apple Silicon (MLX)

```python
from models.mlx_model import create_mlx_mapper, benchmark_mlx_performance

# 创建MLX模型
model = create_mlx_mapper('bt2020', 'srgb')

# 性能测试
metrics = benchmark_mlx_performance(model, n_samples=10000)
print(f"吞吐量: {metrics['throughput_samples_per_sec']:.0f} 样本/秒")
```

### GPU加速 (PyTorch)

```python
# 自动检测最佳设备
model = create_pytorch_mapper(device='auto')  # CUDA > MPS > CPU

# 手动指定设备
model = create_pytorch_mapper(device='cuda')  # 强制使用CUDA
```

## 评估指标

### 色彩质量指标

- **deltaE**: 色彩感知差异 (≤3.0 理想, ≤5.0 可接受)
- **RGB MSE**: RGB空间均方误差
- **色域覆盖率**: 映射结果在目标色域内的比例
- **感知保真度**: 相对距离保持程度

### 性能指标

- **推理速度**: 样本/秒
- **内存使用**: MB
- **训练时间**: 秒/轮次

## 常见问题

### Q: 如何选择deltaE阈值？

A: 根据应用场景：
- **严格要求**: deltaE ≤ 1.0 (专业显示)
- **一般应用**: deltaE ≤ 3.0 (消费电子)
- **宽松应用**: deltaE ≤ 5.0 (网络图像)

### Q: 不同采样策略的区别？

A: 
- **uniform**: 均匀随机采样，训练稳定
- **perceptual**: 在LAB空间均匀分布，更符合感知
- **boundary**: 重点采样色域边界，处理边缘情况
- **stratified**: 分层采样，确保各区域代表性

### Q: 如何处理超出色域的颜色？

A: 模型自动处理：
1. 训练时过滤无效样本
2. 推理时软约束到[0,1]范围
3. 使用gamut_loss引导网络学习边界

### Q: 多通道映射的原理？

A: 
- 输入/输出通道数可以不同
- 使用线性插值扩展或平均降维
- 适用于光谱重建、打印校色等场景

## 贡献指南

欢迎贡献代码！请：

1. Fork本仓库
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或联系开发团队。

---

**⚡ 快速链接**:
- [简单示例](examples/simple_example.py) - 5分钟上手
- [完整演示](examples/quickstart.py) - 功能展示  
- [训练脚本](training/train_pytorch.py) - 自定义训练
- [API文档](#api-文档) - 详细接口 