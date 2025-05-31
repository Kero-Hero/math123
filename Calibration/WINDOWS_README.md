# Windows系统使用说明

## 修改内容

本项目已针对Windows系统进行了优化，主要修改包括：

### 1. 移除MLX依赖
- **requirements.txt**: 注释掉了 `mlx>=0.0.6` 依赖
- **train_calibration.py**: 注释掉了所有MLX相关的导入和代码
- **README.md**: 更新了功能描述，移除MLX相关内容

### 2. 保留的功能
- ✅ **PyTorch支持**: 完整保留PyTorch GPU/CPU加速功能
- ✅ **CUDA支持**: 自动检测NVIDIA GPU并使用CUDA加速
- ✅ **CPU后备**: 当GPU不可用时自动使用CPU计算
- ✅ **NumPy支持**: 完整的NumPy后备计算功能

## 安装步骤

1. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

2. **验证安装**:
   ```bash
   python -c "import train_calibration; print('框架:', train_calibration.framework, '设备:', train_calibration.device)"
   ```

## 使用方法

### 基本训练
```bash
python train_calibration.py --data_dir 数据集 --epochs 1000
```

### GPU加速训练（如果有NVIDIA GPU）
```bash
python train_calibration.py --data_dir 数据集 --epochs 2000 --learning_rate 0.005
```

### 快速开始（需要数据集文件夹）
```bash
python quick_start.py
```

## 性能说明

- **NVIDIA GPU**: 自动使用CUDA加速，训练速度最快
- **CPU**: 使用PyTorch CPU版本，速度较慢但稳定
- **内存使用**: 64×64像素矩阵约需要1-2GB内存

## 数据集要求

需要在项目目录下创建"数据集"文件夹，包含以下9个CSV文件：
```
数据集/
├── R_R.csv  # 红色输出时R通道响应
├── R_G.csv  # 红色输出时G通道响应  
├── R_B.csv  # 红色输出时B通道响应
├── G_R.csv  # 绿色输出时R通道响应
├── G_G.csv  # 绿色输出时G通道响应
├── G_B.csv  # 绿色输出时B通道响应
├── B_R.csv  # 蓝色输出时R通道响应
├── B_G.csv  # 蓝色输出时G通道响应
└── B_B.csv  # 蓝色输出时B通道响应
```

## 输出文件

训练完成后会生成：
- `calibration.csv`: 校准矩阵
- `calibrated_*.csv`: 9个校准后的输出文件
- `calibration_evaluation.txt`: 详细评估报告

## 故障排除

### 常见问题
1. **"未找到数据集文件夹"**: 确保在项目目录下有"数据集"文件夹
2. **内存不足**: 减少epochs数量或使用更小的学习率
3. **训练缓慢**: 检查是否正确安装了PyTorch GPU版本

### 检查GPU支持
```bash
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('设备数量:', torch.cuda.device_count())"
```

## 与原版差异

- ❌ **移除**: Apple MLX框架支持（仅适用于Apple Silicon Mac）
- ✅ **保留**: 所有核心校准功能
- ✅ **保留**: PyTorch GPU加速
- ✅ **保留**: 完整的损失函数和优化算法
- ✅ **增强**: Windows系统兼容性

## 技术支持

如果遇到问题，请检查：
1. Python版本 >= 3.8
2. PyTorch版本 >= 2.0.0
3. 数据集文件格式正确
4. 系统内存充足（建议8GB+） 