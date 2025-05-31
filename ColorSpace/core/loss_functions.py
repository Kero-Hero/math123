"""
色域映射损失函数模块
包含感知均匀性损失、deltaE约束损失和色域边界损失
"""

import numpy as np
from typing import Union, Tuple, Callable
from .color_conversion import delta_e_cie76, delta_e_cie94, lab_to_rgb, rgb_to_lab


class ColorMappingLoss:
    """色域映射损失函数类"""
    
    def __init__(self, 
                 deltaE_threshold: float = 3.0,
                 deltaE_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 gamut_weight: float = 0.5,
                 deltaE_method: str = 'cie94'):
        """
        初始化损失函数
        
        Args:
            deltaE_threshold: deltaE的阈值上限
            deltaE_weight: deltaE损失的权重
            perceptual_weight: 感知损失的权重
            gamut_weight: 色域边界损失的权重
            deltaE_method: deltaE计算方法 ('cie76' 或 'cie94')
        """
        self.deltaE_threshold = deltaE_threshold
        self.deltaE_weight = deltaE_weight
        self.perceptual_weight = perceptual_weight
        self.gamut_weight = gamut_weight
        
        # 选择deltaE计算方法
        if deltaE_method == 'cie76':
            self.deltaE_func = delta_e_cie76
        elif deltaE_method == 'cie94':
            self.deltaE_func = delta_e_cie94
        else:
            raise ValueError(f"不支持的deltaE方法: {deltaE_method}")
    
    def perceptual_loss(self, 
                       source_lab: np.ndarray, 
                       target_lab: np.ndarray,
                       predicted_lab: np.ndarray) -> float:
        """
        感知均匀性损失
        在CIELAB空间中计算，保持色彩间的相对距离关系
        
        Args:
            source_lab: 源色彩的LAB值
            target_lab: 目标色彩的LAB值  
            predicted_lab: 预测的LAB值
            
        Returns:
            感知损失值
        """
        # 基础L2损失
        l2_loss = np.mean((predicted_lab - target_lab) ** 2)
        
        # 相对距离保持损失
        n_samples = len(source_lab)
        if n_samples > 1:
            # 计算源色彩间的距离
            source_distances = []
            target_distances = []
            predicted_distances = []
            
            # 采样部分距离对来避免计算量过大
            sample_indices = np.random.choice(n_samples, 
                                            min(1000, n_samples), 
                                            replace=False)
            
            for i in range(0, len(sample_indices), 2):
                if i + 1 < len(sample_indices):
                    idx1, idx2 = sample_indices[i], sample_indices[i+1]
                    
                    source_dist = np.linalg.norm(source_lab[idx1] - source_lab[idx2])
                    target_dist = np.linalg.norm(target_lab[idx1] - target_lab[idx2])
                    pred_dist = np.linalg.norm(predicted_lab[idx1] - predicted_lab[idx2])
                    
                    source_distances.append(source_dist)
                    target_distances.append(target_dist)
                    predicted_distances.append(pred_dist)
            
            if source_distances:
                source_distances = np.array(source_distances)
                target_distances = np.array(target_distances)
                predicted_distances = np.array(predicted_distances)
                
                # 距离比例损失
                distance_loss = np.mean((predicted_distances - target_distances) ** 2)
                
                return l2_loss + 0.1 * distance_loss
        
        return l2_loss
    
    def deltaE_constraint_loss(self, 
                              source_lab: np.ndarray, 
                              predicted_lab: np.ndarray) -> float:
        """
        DeltaE约束损失
        确保预测颜色与源颜色的色彩差异在可接受范围内
        
        Args:
            source_lab: 源色彩的LAB值
            predicted_lab: 预测的LAB值
            
        Returns:
            deltaE约束损失
        """
        deltaE_values = self.deltaE_func(source_lab, predicted_lab)
        
        # 超过阈值的部分进行惩罚
        excess_deltaE = np.maximum(0, deltaE_values - self.deltaE_threshold)
        
        # 使用平方损失增强惩罚
        deltaE_loss = np.mean(excess_deltaE ** 2)
        
        return deltaE_loss
    
    def gamut_boundary_loss(self, 
                           predicted_rgb: np.ndarray,
                           target_gamut: str = 'srgb') -> float:
        """
        色域边界损失
        确保预测的RGB值在目标色域范围内
        
        Args:
            predicted_rgb: 预测的RGB值
            target_gamut: 目标色域类型
            
        Returns:
            边界损失
        """
        # RGB值应该在[0, 1]范围内
        out_of_gamut_low = np.maximum(0, -predicted_rgb)
        out_of_gamut_high = np.maximum(0, predicted_rgb - 1)
        
        gamut_loss = np.mean(out_of_gamut_low ** 2) + np.mean(out_of_gamut_high ** 2)
        
        return gamut_loss
    
    def total_loss(self, 
                   source_lab: np.ndarray,
                   target_lab: np.ndarray, 
                   predicted_lab: np.ndarray,
                   predicted_rgb: np.ndarray,
                   target_gamut: str = 'srgb') -> Tuple[float, dict]:
        """
        计算总损失
        
        Args:
            source_lab: 源色彩LAB值
            target_lab: 目标色彩LAB值
            predicted_lab: 预测LAB值
            predicted_rgb: 预测RGB值
            target_gamut: 目标色域
            
        Returns:
            总损失值和各项损失的详细信息
        """
        # 各项损失计算
        perc_loss = self.perceptual_loss(source_lab, target_lab, predicted_lab)
        deltaE_loss = self.deltaE_constraint_loss(source_lab, predicted_lab)
        gamut_loss = self.gamut_boundary_loss(predicted_rgb, target_gamut)
        
        # 加权总损失
        total = (self.perceptual_weight * perc_loss + 
                self.deltaE_weight * deltaE_loss + 
                self.gamut_weight * gamut_loss)
        
        loss_details = {
            'total_loss': total,
            'perceptual_loss': perc_loss,
            'deltaE_loss': deltaE_loss,
            'gamut_loss': gamut_loss,
            'deltaE_weight': self.deltaE_weight,
            'perceptual_weight': self.perceptual_weight,
            'gamut_weight': self.gamut_weight
        }
        
        return total, loss_details


class AdaptiveLossScheduler:
    """自适应损失权重调度器"""
    
    def __init__(self, 
                 initial_deltaE_weight: float = 1.0,
                 max_deltaE_weight: float = 10.0,
                 adaptation_factor: float = 1.1):
        """
        初始化调度器
        
        Args:
            initial_deltaE_weight: 初始deltaE权重
            max_deltaE_weight: 最大deltaE权重
            adaptation_factor: 适应因子
        """
        self.current_deltaE_weight = initial_deltaE_weight
        self.max_deltaE_weight = max_deltaE_weight
        self.adaptation_factor = adaptation_factor
        self.best_deltaE_loss = float('inf')
    
    def update_weights(self, current_deltaE_loss: float, loss_fn: ColorMappingLoss):
        """
        根据当前deltaE损失更新权重
        
        Args:
            current_deltaE_loss: 当前deltaE损失
            loss_fn: 损失函数对象
        """
        if current_deltaE_loss > self.best_deltaE_loss:
            # 如果deltaE损失增加，提高其权重
            self.current_deltaE_weight = min(
                self.current_deltaE_weight * self.adaptation_factor,
                self.max_deltaE_weight
            )
        else:
            self.best_deltaE_loss = current_deltaE_loss
        
        # 更新损失函数的权重
        loss_fn.deltaE_weight = self.current_deltaE_weight


def multi_channel_loss(source_channels: np.ndarray, 
                      target_channels: np.ndarray,
                      predicted_channels: np.ndarray,
                      channel_weights: np.ndarray = None) -> float:
    """
    多通道映射损失 (用于四通道->五通道等场景)
    
    Args:
        source_channels: 源通道值 (N, source_channels)
        target_channels: 目标通道值 (N, target_channels)
        predicted_channels: 预测通道值 (N, target_channels)
        channel_weights: 各通道的权重
        
    Returns:
        多通道损失
    """
    if channel_weights is None:
        channel_weights = np.ones(target_channels.shape[1])
    
    # 按通道计算加权损失
    channel_losses = []
    for i in range(target_channels.shape[1]):
        channel_loss = np.mean((predicted_channels[:, i] - target_channels[:, i]) ** 2)
        weighted_loss = channel_loss * channel_weights[i]
        channel_losses.append(weighted_loss)
    
    return np.sum(channel_losses)


def spectral_fidelity_loss(source_spectrum: np.ndarray,
                          predicted_spectrum: np.ndarray,
                          wavelengths: np.ndarray = None) -> float:
    """
    光谱保真度损失 (用于光谱映射)
    
    Args:
        source_spectrum: 源光谱 (N, wavelengths)
        predicted_spectrum: 预测光谱 (N, wavelengths)
        wavelengths: 波长数组
        
    Returns:
        光谱保真度损失
    """
    # 基础L2损失
    l2_loss = np.mean((source_spectrum - predicted_spectrum) ** 2)
    
    # 光谱形状保持损失
    if wavelengths is not None:
        # 计算一阶导数 (光谱斜率)
        source_grad = np.gradient(source_spectrum, wavelengths, axis=1)
        pred_grad = np.gradient(predicted_spectrum, wavelengths, axis=1)
        grad_loss = np.mean((source_grad - pred_grad) ** 2)
        
        return l2_loss + 0.1 * grad_loss
    
    return l2_loss 