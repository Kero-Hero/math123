"""
色域数据采样器
从完整色域中智能采样训练数据，确保采样的多样性和代表性
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.color_conversion import (
    rgb_to_lab, lab_to_rgb, 
    bt2020_to_srgb_direct, srgb_to_bt2020_direct,
    generate_color_space_volume
)


class ColorGamutSampler:
    """色域采样器基类"""
    
    def __init__(self, 
                 source_gamut: str = 'bt2020',
                 target_gamut: str = 'srgb',
                 sampling_strategy: str = 'uniform'):
        """
        初始化采样器
        
        Args:
            source_gamut: 源色域类型
            target_gamut: 目标色域类型  
            sampling_strategy: 采样策略 ('uniform', 'perceptual', 'boundary')
        """
        self.source_gamut = source_gamut
        self.target_gamut = target_gamut
        self.sampling_strategy = sampling_strategy
        
        # 验证色域支持
        supported_gamuts = ['srgb', 'bt2020', '4ch', '5ch']
        if source_gamut not in supported_gamuts:
            raise ValueError(f"不支持的源色域: {source_gamut}")
        if target_gamut not in supported_gamuts:
            raise ValueError(f"不支持的目标色域: {target_gamut}")
    
    def uniform_sampling(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        均匀随机采样
        
        Args:
            n_samples: 采样数量
            
        Returns:
            源色彩数组, 目标色彩数组
        """
        if self.source_gamut in ['srgb', 'bt2020']:
            # RGB色域采样
            source_rgb = np.random.rand(n_samples, 3)
            
            # 转换到目标色域
            if self.source_gamut == 'bt2020' and self.target_gamut == 'srgb':
                target_rgb = bt2020_to_srgb_direct(source_rgb)
                # 裁剪超出目标色域的颜色
                valid_mask = np.all((target_rgb >= 0) & (target_rgb <= 1), axis=1)
                source_rgb = source_rgb[valid_mask]
                target_rgb = target_rgb[valid_mask]
                
            elif self.source_gamut == 'srgb' and self.target_gamut == 'bt2020':
                target_rgb = srgb_to_bt2020_direct(source_rgb)
                valid_mask = np.all((target_rgb >= 0) & (target_rgb <= 1), axis=1)
                source_rgb = source_rgb[valid_mask]
                target_rgb = target_rgb[valid_mask]
                
            else:
                target_rgb = source_rgb.copy()  # 同色域映射
            
            return source_rgb, target_rgb
            
        else:
            # 多通道采样
            source_channels = self._get_channel_count(self.source_gamut)
            target_channels = self._get_channel_count(self.target_gamut)
            
            source_data = np.random.rand(n_samples, source_channels)
            target_data = self._multi_channel_transform(source_data)
            
            return source_data, target_data
    
    def perceptual_sampling(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        感知均匀采样 (在CIELAB空间中均匀分布)
        
        Args:
            n_samples: 采样数量
            
        Returns:
            源色彩数组, 目标色彩数组
        """
        if self.source_gamut in ['srgb', 'bt2020']:
            # 在LAB空间中采样
            # L: 0-100, a: -100-100, b: -100-100
            L = np.random.uniform(0, 100, n_samples)
            a = np.random.uniform(-100, 100, n_samples)
            b = np.random.uniform(-100, 100, n_samples)
            
            source_lab = np.stack([L, a, b], axis=1)
            
            # 转换到RGB，过滤无效颜色
            source_rgb = lab_to_rgb(source_lab, self.source_gamut)
            valid_mask = np.all((source_rgb >= 0) & (source_rgb <= 1), axis=1)
            
            source_rgb = source_rgb[valid_mask]
            source_lab = source_lab[valid_mask]
            
            # 转换到目标色域
            if self.source_gamut == 'bt2020' and self.target_gamut == 'srgb':
                target_rgb = bt2020_to_srgb_direct(source_rgb)
            elif self.source_gamut == 'srgb' and self.target_gamut == 'bt2020':
                target_rgb = srgb_to_bt2020_direct(source_rgb)
            else:
                target_rgb = source_rgb.copy()
            
            # 再次过滤目标色域边界
            valid_mask = np.all((target_rgb >= 0) & (target_rgb <= 1), axis=1)
            source_rgb = source_rgb[valid_mask]
            target_rgb = target_rgb[valid_mask]
            
            return source_rgb, target_rgb
        else:
            return self.uniform_sampling(n_samples)
    
    def boundary_sampling(self, n_samples: int, boundary_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        边界重点采样 (重点采样色域边界附近的颜色)
        
        Args:
            n_samples: 采样数量
            boundary_ratio: 边界采样的比例
            
        Returns:
            源色彩数组, 目标色彩数组
        """
        n_boundary = int(n_samples * boundary_ratio)
        n_uniform = n_samples - n_boundary
        
        # 边界采样
        boundary_source, boundary_target = self._sample_boundary_colors(n_boundary)
        
        # 均匀采样
        uniform_source, uniform_target = self.uniform_sampling(n_uniform)
        
        # 合并
        source_colors = np.vstack([boundary_source, uniform_source])
        target_colors = np.vstack([boundary_target, uniform_target])
        
        # 随机打乱
        indices = np.random.permutation(len(source_colors))
        return source_colors[indices], target_colors[indices]
    
    def _sample_boundary_colors(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """采样色域边界颜色"""
        if self.source_gamut in ['srgb', 'bt2020']:
            # RGB立方体的边界采样
            boundary_colors = []
            
            # 6个面的采样
            for _ in range(n_samples):
                # 随机选择一个面 (R=0, R=1, G=0, G=1, B=0, B=1)
                face = np.random.randint(0, 6)
                rgb = np.random.rand(3)
                
                if face == 0:    # R=0面
                    rgb[0] = 0
                elif face == 1:  # R=1面
                    rgb[0] = 1
                elif face == 2:  # G=0面
                    rgb[1] = 0
                elif face == 3:  # G=1面
                    rgb[1] = 1
                elif face == 4:  # B=0面
                    rgb[2] = 0
                elif face == 5:  # B=1面
                    rgb[2] = 1
                
                boundary_colors.append(rgb)
            
            source_rgb = np.array(boundary_colors)
            
            # 转换到目标色域
            if self.source_gamut == 'bt2020' and self.target_gamut == 'srgb':
                target_rgb = bt2020_to_srgb_direct(source_rgb)
            elif self.source_gamut == 'srgb' and self.target_gamut == 'bt2020':
                target_rgb = srgb_to_bt2020_direct(source_rgb)
            else:
                target_rgb = source_rgb.copy()
            
            return source_rgb, target_rgb
        else:
            return self.uniform_sampling(n_samples)
    
    def stratified_sampling(self, n_samples: int, n_strata: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        分层采样 (确保各个颜色区域都有代表)
        
        Args:
            n_samples: 总采样数量
            n_strata: 分层数量 (每个维度)
            
        Returns:
            源色彩数组, 目标色彩数组
        """
        if self.source_gamut in ['srgb', 'bt2020']:
            samples_per_stratum = n_samples // (n_strata ** 3)
            source_colors = []
            target_colors = []
            
            # 在每个分层中采样
            for i in range(n_strata):
                for j in range(n_strata):
                    for k in range(n_strata):
                        # 当前分层的边界
                        r_min, r_max = i/n_strata, (i+1)/n_strata
                        g_min, g_max = j/n_strata, (j+1)/n_strata
                        b_min, b_max = k/n_strata, (k+1)/n_strata
                        
                        # 在当前分层中采样
                        for _ in range(samples_per_stratum):
                            r = np.random.uniform(r_min, r_max)
                            g = np.random.uniform(g_min, g_max)
                            b = np.random.uniform(b_min, b_max)
                            
                            source_colors.append([r, g, b])
            
            source_rgb = np.array(source_colors)
            
            # 转换到目标色域
            if self.source_gamut == 'bt2020' and self.target_gamut == 'srgb':
                target_rgb = bt2020_to_srgb_direct(source_rgb)
            elif self.source_gamut == 'srgb' and self.target_gamut == 'bt2020':
                target_rgb = srgb_to_bt2020_direct(source_rgb)
            else:
                target_rgb = source_rgb.copy()
            
            return source_rgb, target_rgb
        else:
            return self.uniform_sampling(n_samples)
    
    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据采样策略采样数据
        
        Args:
            n_samples: 采样数量
            
        Returns:
            源色彩数组, 目标色彩数组
        """
        if self.sampling_strategy == 'uniform':
            return self.uniform_sampling(n_samples)
        elif self.sampling_strategy == 'perceptual':
            return self.perceptual_sampling(n_samples)
        elif self.sampling_strategy == 'boundary':
            return self.boundary_sampling(n_samples)
        elif self.sampling_strategy == 'stratified':
            return self.stratified_sampling(n_samples)
        else:
            raise ValueError(f"不支持的采样策略: {self.sampling_strategy}")
    
    def _get_channel_count(self, gamut: str) -> int:
        """获取色域通道数"""
        if gamut in ['srgb', 'bt2020']:
            return 3
        elif gamut == '4ch':
            return 4
        elif gamut == '5ch':
            return 5
        else:
            raise ValueError(f"未知色域: {gamut}")
    
    def _multi_channel_transform(self, source_data: np.ndarray) -> np.ndarray:
        """多通道数据变换 (这里使用简单的线性变换作为示例)"""
        source_channels = source_data.shape[1]
        target_channels = self._get_channel_count(self.target_gamut)
        
        if target_channels > source_channels:
            # 扩展通道 (通过插值)
            target_data = np.zeros((len(source_data), target_channels))
            target_data[:, :source_channels] = source_data
            
            # 简单的插值填充额外通道
            for i in range(source_channels, target_channels):
                alpha = (i - source_channels + 1) / (target_channels - source_channels + 1)
                target_data[:, i] = (1 - alpha) * source_data[:, -1] + alpha * source_data[:, 0]
                
        elif target_channels < source_channels:
            # 降维通道 (通过平均)
            target_data = np.zeros((len(source_data), target_channels))
            for i in range(target_channels):
                start_idx = int(i * source_channels / target_channels)
                end_idx = int((i + 1) * source_channels / target_channels)
                target_data[:, i] = np.mean(source_data[:, start_idx:end_idx], axis=1)
        else:
            target_data = source_data.copy()
        
        return target_data


class AdaptiveSampler:
    """自适应采样器 - 根据训练进度调整采样策略"""
    
    def __init__(self, base_sampler: ColorGamutSampler):
        """
        初始化自适应采样器
        
        Args:
            base_sampler: 基础采样器
        """
        self.base_sampler = base_sampler
        self.training_history = []
        self.current_strategy = base_sampler.sampling_strategy
    
    def update_strategy(self, loss_history: List[float], epoch: int):
        """
        根据训练历史更新采样策略
        
        Args:
            loss_history: 损失历史
            epoch: 当前轮次
        """
        if len(loss_history) < 5:
            return
        
        # 分析损失趋势
        recent_losses = loss_history[-5:]
        loss_trend = np.polyfit(range(5), recent_losses, 1)[0]
        
        # 如果损失停滞，切换到边界采样
        if abs(loss_trend) < 0.001 and self.current_strategy != 'boundary':
            self.current_strategy = 'boundary'
            self.base_sampler.sampling_strategy = 'boundary'
            print(f"轮次 {epoch}: 切换到边界采样策略")
        
        # 如果是训练早期，使用感知采样
        elif epoch < 50 and self.current_strategy != 'perceptual':
            self.current_strategy = 'perceptual'
            self.base_sampler.sampling_strategy = 'perceptual'
            print(f"轮次 {epoch}: 使用感知采样策略")
    
    def sample(self, n_samples: int, epoch: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        自适应采样
        
        Args:
            n_samples: 采样数量
            epoch: 当前轮次
            
        Returns:
            源色彩数组, 目标色彩数组
        """
        return self.base_sampler.sample(n_samples)


def create_sampler(source_gamut: str,
                  target_gamut: str, 
                  strategy: str = 'uniform',
                  adaptive: bool = False) -> Union[ColorGamutSampler, AdaptiveSampler]:
    """
    创建采样器
    
    Args:
        source_gamut: 源色域
        target_gamut: 目标色域
        strategy: 采样策略
        adaptive: 是否使用自适应采样
        
    Returns:
        采样器实例
    """
    base_sampler = ColorGamutSampler(source_gamut, target_gamut, strategy)
    
    if adaptive:
        return AdaptiveSampler(base_sampler)
    else:
        return base_sampler 