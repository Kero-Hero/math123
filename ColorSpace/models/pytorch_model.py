"""
PyTorch实现的色域映射模型
支持GPU/CPU自动检测和感知均匀性映射
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseColorMapper, PerceptualMappingNetwork
from core.color_conversion import rgb_to_lab, lab_to_rgb


class PerceptualMappingNet(nn.Module):
    """感知映射神经网络"""
    
    def __init__(self, 
                 input_dim: int = 3,
                 output_dim: int = 3,
                 hidden_dims: List[int] = [128, 256, 512, 256, 128],
                 use_batch_norm: bool = True,
                 dropout_rate: float = 0.1,
                 use_skip_connections: bool = False):
        """
        初始化网络
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dims: 隐藏层维度列表
            use_batch_norm: 是否使用批标准化
            dropout_rate: Dropout比率
            use_skip_connections: 是否使用跳跃连接
        """
        super(PerceptualMappingNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_skip_connections = use_skip_connections
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # 线性层
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # 批标准化
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # 激活函数
            if i < len(hidden_dims) - 1:
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Tanh())  # 最后一层使用Tanh
            
            # Dropout
            if dropout_rate > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # 输出层也使用Tanh确保在[-1,1]范围
        
        self.network = nn.Sequential(*layers)
        
        # 跳跃连接层
        if use_skip_connections and input_dim == output_dim:
            self.skip_weight = nn.Parameter(torch.tensor(0.1))
        else:
            self.skip_weight = None
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        out = self.network(x)
        
        # 跳跃连接
        if (self.skip_weight is not None and 
            x.shape[-1] == out.shape[-1]):
            out = out + self.skip_weight * x
            
        return out


class PyTorchColorMapper(BaseColorMapper):
    """PyTorch实现的色域映射器"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 output_channels: int = 3,
                 hidden_dims: List[int] = [128, 256, 512, 256, 128],
                 source_gamut: str = 'bt2020',
                 target_gamut: str = 'srgb',
                 deltaE_threshold: float = 3.0,
                 network_type: str = 'standard',
                 device: str = 'auto'):
        """
        初始化PyTorch色域映射器
        
        Args:
            input_channels: 输入通道数
            output_channels: 输出通道数
            hidden_dims: 隐藏层维度
            source_gamut: 源色域
            target_gamut: 目标色域
            deltaE_threshold: deltaE阈值
            network_type: 网络类型
            device: 设备类型 ('auto', 'cuda', 'mps', 'cpu')
        """
        self.network_type = network_type
        self.device = self._get_device(device)
        
        # 调用父类初始化
        super().__init__(input_channels, output_channels, hidden_dims,
                        source_gamut, target_gamut, deltaE_threshold)
        
        # 优化器
        self.optimizer = None
        self.scheduler = None
        
    def _get_device(self, device: str) -> torch.device:
        """自动检测或设置设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"检测到CUDA，使用GPU: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                print("检测到MPS，使用Apple Silicon GPU")
            else:
                device = 'cpu'
                print("使用CPU")
        
        return torch.device(device)
    
    def _build_network(self):
        """构建神经网络"""
        # 获取网络配置
        config = PerceptualMappingNetwork.get_network_config(
            self.input_channels, 
            self.output_channels, 
            self.network_type
        )
        
        # 创建网络
        self.model = PerceptualMappingNet(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            hidden_dims=config['hidden_dims'],
            use_batch_norm=config['use_batch_norm'],
            dropout_rate=config['dropout_rates'][0],  # 使用第一个dropout率
            use_skip_connections=config['use_skip_connections']
        ).to(self.device)
        
        # 确保模型使用float32类型（兼容MPS）
        if self.device.type in ['mps', 'cuda']:
            self.model = self.model.float()
        
        print(f"模型已创建，参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 转换为tensor - 确保float32类型
        x_tensor = torch.from_numpy(x.astype(np.float32)).to(self.device)
        
        # 前向传播
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_tensor)
        
        # 转换回numpy
        return output.cpu().numpy().astype(np.float32)
    
    def train_step(self, 
                   source_data: np.ndarray, 
                   target_data: np.ndarray) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 数据预处理
        source_rgb, source_processed = self._preprocess_data(source_data, self.source_gamut)
        target_rgb, target_processed = self._preprocess_data(target_data, self.target_gamut)
        
        # 转换为tensor - 确保使用float32类型以兼容MPS
        input_tensor = torch.from_numpy(source_processed.astype(np.float32)).to(self.device)
        target_tensor = torch.from_numpy(target_processed.astype(np.float32)).to(self.device)
        
        # 前向传播
        predicted_output = self.model(input_tensor)
        
        # 后处理得到RGB (对于RGB色域) 或直接使用预测输出 (对于多通道)
        if self.target_gamut in ['srgb', 'bt2020']:
            predicted_output_np = predicted_output.detach().cpu().numpy()
            predicted_rgb = self._postprocess_data(predicted_output_np, self.target_gamut)
        else:
            # 多通道映射：直接使用预测输出
            predicted_rgb = predicted_output.detach().cpu().numpy()
        
        # 计算损失 (使用numpy版本)
        if self.source_gamut in ['srgb', 'bt2020'] and self.target_gamut in ['srgb', 'bt2020']:
            # RGB色域映射：使用LAB空间计算损失
            source_lab = rgb_to_lab(source_rgb, self.source_gamut)
            target_lab = rgb_to_lab(target_rgb, self.target_gamut)
            predicted_lab_np = predicted_output.detach().cpu().numpy()
            predicted_lab_denorm = self._denormalize_lab(predicted_lab_np)
            
            total_loss, loss_details = self.loss_fn.total_loss(
                source_lab, target_lab, predicted_lab_denorm, predicted_rgb, self.target_gamut
            )
        else:
            # 多通道映射：使用简化的损失函数
            total_loss = np.mean((target_data - predicted_rgb) ** 2)
            loss_details = {
                'total_loss': total_loss,
                'perceptual_loss': total_loss,
                'deltaE_loss': 0.0,
                'gamut_loss': 0.0
            }
        
        # 如果有梯度计算，使用PyTorch的损失函数
        if self.optimizer is not None:
            # 使用PyTorch内置损失 - 确保float32类型
            mse_loss = F.mse_loss(predicted_output, target_tensor)
            
            # 根据映射类型计算特定损失
            if self.source_gamut in ['srgb', 'bt2020'] and self.target_gamut in ['srgb', 'bt2020']:
                # RGB色域映射：计算deltaE和色域损失
                source_lab_tensor = torch.from_numpy(
                    self._normalize_lab(rgb_to_lab(source_rgb, self.source_gamut)).astype(np.float32)
                ).to(self.device)
                deltaE_loss_tensor = self._compute_deltaE_loss_torch(source_lab_tensor, predicted_output)
                
                predicted_rgb_tensor = torch.from_numpy(predicted_rgb.astype(np.float32)).to(self.device)
                gamut_loss_tensor = self._compute_gamut_loss_torch(predicted_rgb_tensor)
                
                # 总损失 - 确保所有损失项都是float32
                total_loss_tensor = (self.loss_fn.perceptual_weight * mse_loss + 
                                   self.loss_fn.deltaE_weight * deltaE_loss_tensor +
                                   self.loss_fn.gamut_weight * gamut_loss_tensor)
            else:
                # 多通道映射：仅使用MSE损失
                total_loss_tensor = mse_loss
            
            # 确保损失是float32标量
            total_loss_tensor = total_loss_tensor.float()
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss_tensor.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 更新学习率 - 需要传入损失值给ReduceLROnPlateau
            if self.scheduler is not None:
                # ReduceLROnPlateau需要传入损失值，转换为Python float
                loss_value = float(total_loss_tensor.detach().cpu().item())
                self.scheduler.step(loss_value)
        
        return loss_details
    
    def _compute_deltaE_loss_torch(self, 
                                  source_lab: torch.Tensor, 
                                  predicted_lab: torch.Tensor) -> torch.Tensor:
        """计算PyTorch版本的deltaE损失"""
        # 反标准化LAB
        source_lab_denorm = self._denormalize_lab_torch(source_lab)
        predicted_lab_denorm = self._denormalize_lab_torch(predicted_lab)
        
        # 计算deltaE (CIE76简化版本) - 确保float32类型
        diff = source_lab_denorm - predicted_lab_denorm
        deltaE = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        
        # 超过阈值的部分 - 确保阈值是float32类型
        threshold = torch.tensor(self.deltaE_threshold, dtype=torch.float32, device=deltaE.device)
        excess_deltaE = torch.clamp(deltaE - threshold, min=0)
        
        return torch.mean(excess_deltaE ** 2)
    
    def _denormalize_lab_torch(self, normalized_lab: torch.Tensor) -> torch.Tensor:
        """PyTorch版本的LAB反标准化"""
        lab_data = normalized_lab.clone()
        # 确保所有常数都是float32类型
        lab_data[..., 0] = (normalized_lab[..., 0] + 1.0) * 50.0
        lab_data[..., 1] = normalized_lab[..., 1] * 128.0
        lab_data[..., 2] = normalized_lab[..., 2] * 128.0
        return lab_data
    
    def _compute_gamut_loss_torch(self, predicted_rgb: torch.Tensor) -> torch.Tensor:
        """计算PyTorch版本的色域损失"""
        # 确保边界值是float32类型
        zeros = torch.zeros_like(predicted_rgb)
        ones = torch.ones_like(predicted_rgb)
        
        out_of_gamut_low = torch.clamp(-predicted_rgb, min=0)
        out_of_gamut_high = torch.clamp(predicted_rgb - ones, min=0)
        
        return torch.mean(out_of_gamut_low ** 2) + torch.mean(out_of_gamut_high ** 2)
    
    def setup_optimizer(self, 
                       learning_rate: float = 0.001,
                       weight_decay: float = 1e-4,
                       optimizer_type: str = 'adam'):
        """设置优化器"""
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 学习率调度器 - 兼容不同PyTorch版本
        try:
            # 新版本PyTorch支持verbose参数
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5,
                verbose=True
            )
        except TypeError:
            # 旧版本PyTorch不支持verbose参数
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5
            )
    
    def save_model(self, filepath: str):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': {
                'input_channels': self.input_channels,
                'output_channels': self.output_channels,
                'hidden_dims': self.hidden_dims,
                'source_gamut': self.source_gamut,
                'target_gamut': self.target_gamut,
                'deltaE_threshold': self.deltaE_threshold,
                'network_type': self.network_type
            },
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        torch.save(checkpoint, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 重建网络 (如果配置不同)
        config = checkpoint['config']
        if (config['input_channels'] != self.input_channels or 
            config['output_channels'] != self.output_channels):
            self.__init__(
                input_channels=config['input_channels'],
                output_channels=config['output_channels'],
                hidden_dims=config['hidden_dims'],
                source_gamut=config['source_gamut'],
                target_gamut=config['target_gamut'],
                deltaE_threshold=config['deltaE_threshold'],
                network_type=config['network_type']
            )
        
        # 加载状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_history = checkpoint.get('training_history', [])
        self.is_trained = checkpoint.get('is_trained', False)
        
        print(f"模型已从 {filepath} 加载")
    
    def train(self, *args, **kwargs):
        """训练模型 (重写父类方法以设置优化器)"""
        # 设置优化器
        learning_rate = kwargs.get('learning_rate', 0.001)
        self.setup_optimizer(learning_rate=learning_rate)
        
        # 调用父类训练方法
        return super().train(*args, **kwargs)


# 便捷函数
def create_pytorch_mapper(source_gamut: str = 'bt2020',
                         target_gamut: str = 'srgb',
                         network_type: str = 'standard',
                         deltaE_threshold: float = 3.0,
                         device: str = 'auto') -> PyTorchColorMapper:
    """
    创建PyTorch色域映射器
    
    Args:
        source_gamut: 源色域
        target_gamut: 目标色域
        network_type: 网络类型
        deltaE_threshold: deltaE阈值
        device: 设备类型
        
    Returns:
        PyTorch色域映射器实例
    """
    # 确定通道数
    if source_gamut in ['srgb', 'bt2020']:
        input_channels = 3  # RGB色域转换为LAB处理，仍然是3维
    elif source_gamut == '4ch':
        input_channels = 4
    elif source_gamut == '5ch':
        input_channels = 5
    else:
        raise ValueError(f"不支持的源色域: {source_gamut}")
    
    if target_gamut in ['srgb', 'bt2020']:
        output_channels = 3  # RGB色域输出LAB然后转换回RGB，仍然是3维
    elif target_gamut == '4ch':
        output_channels = 4
    elif target_gamut == '5ch':
        output_channels = 5
    else:
        raise ValueError(f"不支持的目标色域: {target_gamut}")
    
    return PyTorchColorMapper(
        input_channels=input_channels,
        output_channels=output_channels,
        source_gamut=source_gamut,
        target_gamut=target_gamut,
        deltaE_threshold=deltaE_threshold,
        network_type=network_type,
        device=device
    ) 