"""
色域映射基础模型类
定义通用的网络架构和训练接口
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.color_conversion import rgb_to_lab, lab_to_rgb
from core.loss_functions import ColorMappingLoss, AdaptiveLossScheduler


class BaseColorMapper(ABC):
    """色域映射基础模型类"""
    
    def __init__(self,
                 input_channels: int = 3,
                 output_channels: int = 3,
                 hidden_dims: List[int] = [128, 256, 512, 256, 128],
                 source_gamut: str = 'bt2020',
                 target_gamut: str = 'srgb',
                 deltaE_threshold: float = 3.0):
        """
        初始化基础模型
        
        Args:
            input_channels: 输入通道数
            output_channels: 输出通道数
            hidden_dims: 隐藏层维度列表
            source_gamut: 源色域
            target_gamut: 目标色域
            deltaE_threshold: deltaE阈值
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dims = hidden_dims
        self.source_gamut = source_gamut
        self.target_gamut = target_gamut
        self.deltaE_threshold = deltaE_threshold
        
        # 训练相关
        self.training_history = []
        self.is_trained = False
        
        # 损失函数
        self.loss_fn = ColorMappingLoss(deltaE_threshold=deltaE_threshold)
        self.loss_scheduler = AdaptiveLossScheduler()
        
        # 初始化网络
        self._build_network()
    
    @abstractmethod
    def _build_network(self):
        """构建神经网络 - 由子类实现"""
        pass
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播 - 由子类实现"""
        pass
    
    @abstractmethod
    def train_step(self, 
                   source_data: np.ndarray, 
                   target_data: np.ndarray) -> Dict[str, float]:
        """单步训练 - 由子类实现"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """保存模型 - 由子类实现"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """加载模型 - 由子类实现"""
        pass
    
    def _preprocess_data(self, 
                        rgb_data: np.ndarray, 
                        gamut: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理：RGB -> LAB 或多通道数据标准化
        
        Args:
            rgb_data: RGB数据或多通道数据
            gamut: 色域类型
            
        Returns:
            rgb_normalized: 标准化的RGB数据
            lab_data: LAB数据或标准化的多通道数据
        """
        # 确保数据在[0,1]范围
        rgb_normalized = np.clip(rgb_data, 0, 1)
        
        # 根据色域类型处理
        if gamut in ['srgb', 'bt2020']:
            # RGB色域：转换到LAB空间 (感知均匀空间)
            lab_data = rgb_to_lab(rgb_normalized, gamut)
            # LAB数据标准化
            lab_normalized = self._normalize_lab(lab_data)
            return rgb_normalized, lab_normalized
        else:
            # 多通道数据：直接标准化，不进行色彩空间转换
            # 对于多通道映射，输入和输出都使用原始数据
            return rgb_normalized, rgb_normalized
    
    def _normalize_lab(self, lab_data: np.ndarray) -> np.ndarray:
        """
        标准化LAB数据到[-1, 1]范围
        
        Args:
            lab_data: LAB数据 (L: 0-100, a/b: -128~127)
            
        Returns:
            标准化的LAB数据
        """
        normalized_lab = np.array(lab_data)  # 创建副本
        # L通道: 0-100 -> [-1, 1]
        normalized_lab[..., 0] = (lab_data[..., 0] / 50.0) - 1.0
        # a,b通道: -128~127 -> [-1, 1]  
        normalized_lab[..., 1] = lab_data[..., 1] / 128.0
        normalized_lab[..., 2] = lab_data[..., 2] / 128.0
        
        return normalized_lab
    
    def _denormalize_lab(self, normalized_lab: np.ndarray) -> np.ndarray:
        """
        反标准化LAB数据
        
        Args:
            normalized_lab: 标准化的LAB数据
            
        Returns:
            原始LAB数据
        """
        lab_data = np.array(normalized_lab)  # 创建副本
        # L通道: [-1, 1] -> 0-100
        lab_data[..., 0] = (normalized_lab[..., 0] + 1.0) * 50.0
        # a,b通道: [-1, 1] -> -128~127
        lab_data[..., 1] = normalized_lab[..., 1] * 128.0
        lab_data[..., 2] = normalized_lab[..., 2] * 128.0
        
        return lab_data
    
    def _postprocess_data(self, 
                         predicted_data: np.ndarray, 
                         gamut: str) -> np.ndarray:
        """
        数据后处理：LAB -> RGB 或多通道数据反标准化
        
        Args:
            predicted_data: 预测的LAB数据 (标准化的) 或多通道数据
            gamut: 目标色域
            
        Returns:
            RGB数据或多通道数据
        """
        if gamut in ['srgb', 'bt2020']:
            # RGB色域：LAB转换到RGB
            # 反标准化LAB
            lab_data = self._denormalize_lab(predicted_data)
            # 转换到RGB
            rgb_data = lab_to_rgb(lab_data, gamut)
            # 裁剪到[0,1]范围
            return np.clip(rgb_data, 0, 1)
        else:
            # 多通道数据：直接裁剪到[0,1]范围
            return np.clip(predicted_data, 0, 1)
    
    def transform(self, source_data: np.ndarray) -> np.ndarray:
        """
        执行色域映射变换
        
        Args:
            source_data: 源色彩数据 (RGB格式或多通道数据)
            
        Returns:
            目标色彩数据 (RGB格式或多通道数据)
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train()方法")
        
        # 预处理
        rgb_norm, processed_data = self._preprocess_data(source_data, self.source_gamut)
        
        # 前向传播
        predicted_data = self.forward(processed_data)
        
        # 后处理
        target_data = self._postprocess_data(predicted_data, self.target_gamut)
        
        return target_data
    
    def evaluate(self, 
                source_data: np.ndarray, 
                target_data: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            source_data: 源数据
            target_data: 目标数据
            
        Returns:
            评估指标字典
        """
        # 获取预测结果
        predicted_data = self.transform(source_data)
        
        # 转换到LAB空间计算误差
        if self.target_gamut in ['srgb', 'bt2020']:
            target_lab = rgb_to_lab(target_data, self.target_gamut)
            predicted_lab = rgb_to_lab(predicted_data, self.target_gamut)
            source_lab = rgb_to_lab(source_data, self.source_gamut)
            
            # 计算各种指标
            mse = np.mean((predicted_data - target_data) ** 2)
            mae = np.mean(np.abs(predicted_data - target_data))
            deltaE = np.mean(self.loss_fn.deltaE_func(source_lab, predicted_lab))
            max_deltaE = np.max(self.loss_fn.deltaE_func(source_lab, predicted_lab))
            
            # 在范围内的预测比例
            in_gamut_ratio = np.mean(np.all((predicted_data >= 0) & (predicted_data <= 1), axis=1))
            
            # deltaE在阈值内的比例
            deltaE_values = self.loss_fn.deltaE_func(source_lab, predicted_lab)
            deltaE_satisfied_ratio = np.mean(deltaE_values <= self.deltaE_threshold)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'mean_deltaE': deltaE,
                'max_deltaE': max_deltaE,
                'in_gamut_ratio': in_gamut_ratio,
                'deltaE_satisfied_ratio': deltaE_satisfied_ratio
            }
        else:
            # 多通道评估
            mse = np.mean((predicted_data - target_data) ** 2)
            mae = np.mean(np.abs(predicted_data - target_data))
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'in_gamut_ratio': np.mean(np.all((predicted_data >= 0) & (predicted_data <= 1), axis=1))
            }
        
        return metrics
    
    def train(self,
             train_sampler,
             validation_sampler=None,
             epochs: int = 100,
             batch_size: int = 1024,
             learning_rate: float = 0.001,
             patience: int = 10,
             min_delta: float = 1e-6,
             verbose: bool = True) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_sampler: 训练数据采样器
            validation_sampler: 验证数据采样器
            epochs: 训练轮次
            batch_size: 批次大小
            learning_rate: 学习率
            patience: 早停耐心值
            min_delta: 最小改进量
            verbose: 是否显示训练过程
            
        Returns:
            训练历史字典
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'deltaE_loss': [],
            'perceptual_loss': [],
            'gamut_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            epoch_losses = []
            n_batches = max(1, 5000 // batch_size)  # 每轮使用约5000个样本
            
            for batch in range(n_batches):
                # 采样训练数据
                source_batch, target_batch = train_sampler.sample(batch_size)
                
                if len(source_batch) == 0:
                    continue
                
                # 训练步骤
                loss_dict = self.train_step(source_batch, target_batch)
                epoch_losses.append(loss_dict)
            
            if epoch_losses:
                # 计算平均损失
                avg_losses = {}
                for key in epoch_losses[0].keys():
                    avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
                
                history['train_loss'].append(avg_losses['total_loss'])
                history['deltaE_loss'].append(avg_losses['deltaE_loss'])
                history['perceptual_loss'].append(avg_losses['perceptual_loss'])
                history['gamut_loss'].append(avg_losses['gamut_loss'])
                
                # 更新损失权重
                self.loss_scheduler.update_weights(avg_losses['deltaE_loss'], self.loss_fn)
                
                # 验证阶段
                if validation_sampler:
                    val_source, val_target = validation_sampler.sample(batch_size)
                    if len(val_source) > 0:
                        val_metrics = self.evaluate(val_source, val_target)
                        val_loss = val_metrics.get('mse', 0) + val_metrics.get('mean_deltaE', 0)
                        history['val_loss'].append(val_loss)
                        
                        # 早停检查
                        if val_loss < best_val_loss - min_delta:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        # 更新学习率调度器 (如果存在)
                        if hasattr(self, 'scheduler') and self.scheduler is not None:
                            try:
                                self.scheduler.step(val_loss)
                            except:
                                # 某些调度器不需要参数
                                pass
                else:
                    history['val_loss'].append(avg_losses['total_loss'])
                    
                    # 没有验证数据时使用训练损失更新调度器
                    if hasattr(self, 'scheduler') and self.scheduler is not None:
                        try:
                            self.scheduler.step(avg_losses['total_loss'])
                        except:
                            pass
                
                # 输出进度
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs}")
                    print(f"  训练损失: {avg_losses['total_loss']:.6f}")
                    print(f"  deltaE损失: {avg_losses['deltaE_loss']:.6f}")
                    print(f"  感知损失: {avg_losses['perceptual_loss']:.6f}")
                    print(f"  色域损失: {avg_losses['gamut_loss']:.6f}")
                    if validation_sampler:
                        print(f"  验证损失: {history['val_loss'][-1]:.6f}")
                        print(f"  平均deltaE: {val_metrics.get('mean_deltaE', 0):.3f}")
                        print(f"  deltaE满足率: {val_metrics.get('deltaE_satisfied_ratio', 0):.3f}")
                    print()
                
                # 早停
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
        
        self.is_trained = True
        self.training_history = history
        
        return history


class PerceptualMappingNetwork:
    """感知映射网络架构"""
    
    @staticmethod
    def get_network_config(input_dim: int, 
                          output_dim: int, 
                          network_type: str = 'standard') -> Dict[str, Any]:
        """
        获取网络配置
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            network_type: 网络类型 ('standard', 'deep', 'wide')
            
        Returns:
            网络配置字典
        """
        if network_type == 'standard':
            config = {
                'hidden_dims': [128, 256, 512, 256, 128],
                'activations': ['relu', 'relu', 'relu', 'relu', 'tanh'],
                'dropout_rates': [0.1, 0.1, 0.2, 0.1, 0.0],
                'use_batch_norm': True,
                'use_skip_connections': False
            }
        elif network_type == 'deep':
            config = {
                'hidden_dims': [128, 256, 512, 512, 512, 256, 128],
                'activations': ['relu'] * 6 + ['tanh'],
                'dropout_rates': [0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.0],
                'use_batch_norm': True,
                'use_skip_connections': True
            }
        elif network_type == 'wide':
            config = {
                'hidden_dims': [256, 512, 1024, 512, 256],
                'activations': ['relu', 'relu', 'relu', 'relu', 'tanh'],
                'dropout_rates': [0.1, 0.2, 0.3, 0.2, 0.0],
                'use_batch_norm': True,
                'use_skip_connections': False
            }
        else:
            raise ValueError(f"不支持的网络类型: {network_type}")
        
        config['input_dim'] = input_dim
        config['output_dim'] = output_dim
        
        return config 