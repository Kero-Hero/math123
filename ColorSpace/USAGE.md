# 使用指南 - Windows兼容版本

## 项目概述

本项目实现了基于深度学习的色域转换模型，主要特点是在CIELAB感知均匀色彩空间中进行处理，并通过deltaE约束来控制色彩偏差。

**注意：此版本已针对Windows系统优化，移除了MLX依赖，仅使用PyTorch。**

## 环境配置

### 1. 安装依赖

```bash
# 基础依赖
pip install -r requirements/base.txt

# 根据你的平台选择一个：
pip install -r requirements/pytorch_gpu.txt   # PyTorch GPU版本 (CUDA)
pip install -r requirements/pytorch_cpu.txt   # PyTorch CPU版本
# pip install -r requirements/mlx.txt         # Apple MLX版本 - Windows版本已禁用
```

### 2. 验证安装

```python
import sys
sys.path.append('.')

# 测试基础功能
from core.color_conversion import rgb_to_lab
import numpy as np

test_rgb = np.array([[1.0, 0.0, 0.0]])
test_lab = rgb_to_lab(test_rgb, 'srgb')
print(f"测试成功: RGB {test_rgb[0]} -> LAB {test_lab[0]}")

# 测试模型创建
from models.pytorch_model import create_pytorch_mapper
model = create_pytorch_mapper()
print("PyTorch模型创建成功")

# MLX测试已禁用 (Windows版本不支持)
# try:
#     from models.mlx_model import create_mlx_mapper
#     mlx_model = create_mlx_mapper()
#     print("MLX模型创建成功")
# except ImportError:
#     print("MLX不可用 (仅支持Apple Silicon)")
print("MLX已在Windows版本中禁用")
```

## 快速开始

### 5分钟入门

运行简单示例，了解基本用法：

```bash
python examples/simple_example.py
```

这个示例会：
1. 创建BT.2020到sRGB的映射模型
2. 快速训练20轮 (演示用)
3. 测试映射效果
4. 与传统方法对比
5. 评估色彩质量

### 完整功能演示

运行完整演示，了解所有功能：

```bash
python examples/quickstart.py
```

这个演示包含：
- PyTorch GPU/CPU性能对比
- 多通道映射 (4ch → 5ch)
- deltaE阈值影响测试
- 综合性能评估
<!-- - 多平台性能对比 (PyTorch vs MLX) - Windows版本已禁用 -->

## 核心功能详解

### 1. 色彩空间转换

项目支持精确的色彩空间转换：

```python
from core.color_conversion import *
import numpy as np

# RGB到LAB转换 (感知均匀空间)
rgb = np.array([[0.8, 0.2, 0.4]])
lab_srgb = rgb_to_lab(rgb, 'srgb')      # sRGB空间
lab_bt2020 = rgb_to_lab(rgb, 'bt2020')  # BT.2020空间

# LAB到RGB转换
rgb_back = lab_to_rgb(lab_srgb, 'srgb')

# 计算色彩差异 (deltaE)
deltaE = delta_e_cie94(lab_srgb, lab_bt2020)
print(f"色彩差异: {deltaE[0]:.3f}")

# 直接色域转换
bt2020_rgb = np.array([[1.0, 0.0, 0.0]])  # BT.2020红色
srgb_rgb = bt2020_to_srgb_direct(bt2020_rgb)  # 转为sRGB
```

### 2. 神经网络模型

#### PyTorch版本 (推荐)

```python
from models.pytorch_model import create_pytorch_mapper

# 创建模型
model = create_pytorch_mapper(
    source_gamut='bt2020',     # 源色域
    target_gamut='srgb',       # 目标色域
    network_type='standard',   # 网络复杂度: standard/deep/wide
    deltaE_threshold=3.0,      # 允许的最大色彩偏差
    device='auto'              # 自动选择设备: CUDA > CPU
)

# 检查设备
print(f"使用设备: {model.device}")
print(f"模型参数: {sum(p.numel() for p in model.model.parameters()):,}")
```

<!-- #### Apple MLX版本 (Windows版本已禁用)

```python
from models.mlx_model import create_mlx_mapper

# MLX版本 (专为Apple Silicon优化)
mlx_model = create_mlx_mapper(
    source_gamut='bt2020',
    target_gamut='srgb',
    deltaE_threshold=3.0
)

# MLX性能测试
from models.mlx_model import benchmark_mlx_performance
perf_metrics = benchmark_mlx_performance(mlx_model, n_samples=10000)
print(f"MLX推理速度: {perf_metrics['throughput_samples_per_sec']:.0f} 样本/秒")
``` -->

### 3. 数据采样策略

不同的采样策略适用于不同的场景：

```python
from data.sampler import create_sampler

# 均匀采样 (训练稳定)
uniform_sampler = create_sampler('bt2020', 'srgb', 'uniform')
source, target = uniform_sampler.sample(1000)

# 感知采样 (LAB空间均匀分布)
perceptual_sampler = create_sampler('bt2020', 'srgb', 'perceptual')
source, target = perceptual_sampler.sample(1000)

# 边界采样 (重点处理色域边界)
boundary_sampler = create_sampler('bt2020', 'srgb', 'boundary')
source, target = boundary_sampler.sample(1000)

# 分层采样 (确保各区域代表性)
stratified_sampler = create_sampler('bt2020', 'srgb', 'stratified')
source, target = stratified_sampler.sample(1000)

# 自适应采样 (根据训练进度调整策略)
adaptive_sampler = create_sampler('bt2020', 'srgb', 'perceptual', adaptive=True)
```

### 4. 模型训练

#### 基础训练

```python
from data.sampler import create_sampler

# 准备数据
train_sampler = create_sampler('bt2020', 'srgb', 'perceptual', adaptive=True)
val_sampler = create_sampler('bt2020', 'srgb', 'uniform')

# 训练模型
history = model.train(
    train_sampler=train_sampler,
    validation_sampler=val_sampler,
    epochs=100,
    batch_size=1024,
    learning_rate=0.001,
    patience=15,        # 早停耐心值
    verbose=True
)

# 查看训练历史
print("训练历史:")
for epoch in range(0, len(history['train_loss']), 10):
    print(f"Epoch {epoch}: Loss={history['train_loss'][epoch]:.6f}, "
          f"DeltaE={history['deltaE_loss'][epoch]:.6f}")
```

#### 高级训练配置

```python
# 自定义损失函数
from core.loss_functions import ColorMappingLoss

custom_loss = ColorMappingLoss(
    deltaE_threshold=2.0,      # 更严格的deltaE要求
    deltaE_weight=2.0,         # 增加deltaE损失权重
    perceptual_weight=1.0,     # 感知损失权重
    gamut_weight=0.5,          # 色域边界损失权重
    deltaE_method='cie94'      # 使用CIE94 deltaE
)

# 应用自定义损失
model.loss_fn = custom_loss

# 带配置的训练
history = model.train(
    train_sampler=train_sampler,
    validation_sampler=val_sampler,
    epochs=200,
    batch_size=2048,           # 更大批次
    learning_rate=0.0005,      # 更小学习率
    patience=20,
    min_delta=1e-6,           # 最小改进量
    verbose=True
)
```

### 5. 模型评估

```python
# 生成测试数据
test_sampler = create_sampler('bt2020', 'srgb', 'uniform')
test_source, test_target = test_sampler.sample(2000)

# 评估模型
metrics = model.evaluate(test_source, test_target)

print("模型性能评估:")
print(f"  平均deltaE: {metrics['mean_deltaE']:.3f}")
print(f"  最大deltaE: {metrics['max_deltaE']:.3f}")
print(f"  RGB MSE: {metrics['mse']:.6f}")
print(f"  色域内比例: {metrics['in_gamut_ratio']:.3f}")
print(f"  deltaE≤3满足率: {metrics['deltaE_satisfied_ratio']:.3f}")
```

### 6. 模型保存和加载

```python
# 保存模型
model.save_model('models/bt2020_to_srgb.pth')

# 加载模型
from models.pytorch_model import create_pytorch_mapper
loaded_model = create_pytorch_mapper('bt2020', 'srgb')
loaded_model.load_model('models/bt2020_to_srgb.pth')

# 验证加载成功
test_input = np.array([[1.0, 0.0, 0.0]])
original_output = model.transform(test_input)
loaded_output = loaded_model.transform(test_input)
print(f"加载验证: {np.allclose(original_output, loaded_output)}")
```

## 高级应用

### 1. 多通道映射

适用于光谱重建、打印校色等场景：

```python
# 创建多通道映射模型
multi_model = create_pytorch_mapper(
    source_gamut='4ch',    # 4通道输入
    target_gamut='5ch',    # 5通道输出
    deltaE_threshold=5.0
)

# 多通道数据采样
multi_sampler = create_sampler('4ch', '5ch', 'uniform')
source_4ch, target_5ch = multi_sampler.sample(1000)

print(f"输入形状: {source_4ch.shape}")
print(f"输出形状: {target_5ch.shape}")

# 训练多通道模型
multi_model.train(
    train_sampler=multi_sampler,
    epochs=100,
    batch_size=512,
    learning_rate=0.003
)

# 测试多通道映射
test_4ch = np.random.rand(10, 4)
mapped_5ch = multi_model.transform(test_4ch)
print(f"映射结果: {test_4ch.shape} -> {mapped_5ch.shape}")
```

### 2. 批量图像处理

处理大量图像像素：

```python
import time

# 模拟图像数据 (1920x1080像素)
image_height, image_width = 1080, 1920
total_pixels = image_height * image_width
image_data = np.random.rand(total_pixels, 3)

print(f"处理图像: {image_width}x{image_height} = {total_pixels:,} 像素")

# 批量处理
batch_size = 10000
start_time = time.time()

processed_pixels = []
for i in range(0, total_pixels, batch_size):
    batch = image_data[i:i+batch_size]
    mapped_batch = model.transform(batch)
    processed_pixels.append(mapped_batch)

processed_image = np.vstack(processed_pixels)
processing_time = time.time() - start_time

print(f"处理时间: {processing_time:.2f} 秒")
print(f"处理速度: {total_pixels/processing_time:.0f} 像素/秒")
print(f"吞吐量: {total_pixels/processing_time/1000000:.2f} M像素/秒")
```

### 3. 实时性能优化

```python
# 模型预热 (消除首次推理的开销)
warmup_data = np.random.rand(100, 3)
_ = model.transform(warmup_data)

# 性能基准测试
def benchmark_model(model, n_samples=10000, n_runs=5):
    """模型性能基准测试"""
    test_data = np.random.rand(n_samples, 3)
    times = []
    
    for _ in range(n_runs):
        start = time.time()
        _ = model.transform(test_data)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = n_samples / avg_time
    
    return {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_samples_per_sec': throughput,
        'samples': n_samples
    }

# 运行基准测试
benchmark_results = benchmark_model(model)
print("性能基准测试结果:")
for key, value in benchmark_results.items():
    print(f"  {key}: {value:.2f}")
```

## 命令行工具

### 训练脚本

```bash
# 基础训练
python training/train_pytorch.py \
    --source_gamut bt2020 \
    --target_gamut srgb \
    --epochs 100

# 高级训练配置
python training/train_pytorch.py \
    --source_gamut bt2020 \
    --target_gamut srgb \
    --network_type deep \
    --deltaE_threshold 2.0 \
    --epochs 200 \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --sampling_strategy perceptual \
    --device cuda \
    --output_dir ./trained_models

# 多通道训练
python training/train_pytorch.py \
    --source_gamut 4ch \
    --target_gamut 5ch \
    --epochs 150 \
    --batch_size 512 \
    --deltaE_threshold 5.0
```

## 最佳实践

### 1. deltaE阈值选择

- **严格应用 (专业显示)**: deltaE ≤ 1.0
- **一般应用 (消费电子)**: deltaE ≤ 3.0  
- **宽松应用 (网络图像)**: deltaE ≤ 5.0

### 2. 采样策略选择

- **训练初期**: `perceptual` - 在感知空间均匀分布
- **训练中期**: `boundary` - 重点处理色域边界
- **训练后期**: `uniform` - 稳定收敛

### 3. 网络架构选择

- **标准应用**: `standard` - 平衡性能和精度
- **高精度需求**: `deep` - 更深的网络，更高精度
- **实时应用**: `standard` 配合较小batch_size

### 4. 设备选择优化

```python
# 自动选择最佳设备
def get_optimal_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# 创建模型时使用
model = create_pytorch_mapper(device=get_optimal_device())
```

## 故障排除

### 常见问题

1. **训练损失不收敛**
   - 降低学习率 (0.001 → 0.0005)
   - 增加批次大小
   - 检查数据质量

2. **deltaE损失过高**
   - 降低deltaE阈值权重
   - 增加训练轮次
   - 使用感知采样策略

3. **色域超出范围**
   - 增加gamut_loss权重
   - 使用边界采样策略
   - 检查数据预处理

4. **Windows系统优化**
   - 确保安装了CUDA驱动 (如果有NVIDIA GPU)
   - 使用PyTorch GPU版本获得最佳性能
   - CPU版本可在任何Windows系统上运行

### 性能优化建议

1. **内存优化**
   - 适当降低batch_size
   - 使用梯度累积
   - 清理中间变量

2. **速度优化**
   - 使用GPU加速 (CUDA)
   - 预编译模型
   - 批量处理数据

3. **精度优化**
   - 增加训练数据量
   - 使用更深的网络
   - 调整损失函数权重

## Windows系统特别说明

### 支持的设备
- ✅ **CUDA GPU**: 首选，性能最佳
- ✅ **CPU**: 备选，兼容性最好
- ❌ **MPS**: 不支持，仅限Apple设备

### 推荐配置
- **GPU用户**: 安装PyTorch GPU版本 + CUDA驱动
- **CPU用户**: 安装PyTorch CPU版本，减少依赖

### 设备自动检测
```python
# 创建模型时自动选择最佳设备
model = create_pytorch_mapper(device='auto')
# Windows上会自动选择: CUDA (如果可用) > CPU
```

## 扩展开发

### 添加新的色域类型

1. 在`core/color_conversion.py`中添加变换矩阵
2. 更新`data/sampler.py`中的支持列表
3. 测试新色域的转换精度

### 添加新的损失函数

1. 在`core/loss_functions.py`中定义新损失
2. 继承`ColorMappingLoss`类
3. 在训练脚本中集成测试

### 添加新的采样策略

1. 在`data/sampler.py`中添加新方法
2. 更新`sample()`方法的分支
3. 进行充分的测试验证

---

这个使用指南涵盖了项目的主要功能和使用方法。此Windows兼容版本已完全移除MLX依赖，确保在Windows系统上的稳定运行。如果遇到问题，请参考README.md或提交Issue。 