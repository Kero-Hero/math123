"""
色域转换模型包
基于感知均匀性的色域映射，支持deltaE约束
"""

__version__ = "1.0.0"
__author__ = "ColorSpace Team"
# __description__ = "感知均匀性色域转换模型，支持PyTorch和Apple MLX"
__description__ = "感知均匀性色域转换模型，支持PyTorch (Windows兼容版本)"

# 核心功能导入
from core.color_conversion import (
    rgb_to_lab, lab_to_rgb, rgb_to_xyz, xyz_to_rgb,
    bt2020_to_srgb_direct, srgb_to_bt2020_direct,
    delta_e_cie76, delta_e_cie94
)

from core.loss_functions import (
    ColorMappingLoss, AdaptiveLossScheduler
)

from data.sampler import (
    ColorGamutSampler, AdaptiveSampler, create_sampler
)

# 模型导入
from models.pytorch_model import create_pytorch_mapper

# MLX导入已注释掉，在Windows系统上不支持
# try:
#     from models.mlx_model import create_mlx_mapper
#     MLX_AVAILABLE = True
# except ImportError:
#     MLX_AVAILABLE = False

# Windows系统禁用MLX
MLX_AVAILABLE = False

# 便捷函数
def create_mapper(framework='pytorch', **kwargs):
    """
    创建色域映射器的便捷函数
    
    Args:
        framework: 框架类型 ('pytorch')，Windows版本不支持'mlx'
        **kwargs: 其他参数
        
    Returns:
        色域映射器实例
    """
    if framework == 'pytorch':
        return create_pytorch_mapper(**kwargs)
    # elif framework == 'mlx':
    #     if not MLX_AVAILABLE:
    #         raise ImportError("MLX不可用，请安装MLX或使用pytorch框架")
    #     return create_mlx_mapper(**kwargs)
    elif framework == 'mlx':
        raise ImportError("Windows系统不支持MLX，请使用pytorch框架")
    else:
        raise ValueError(f"不支持的框架: {framework}")

# 主要类和函数
__all__ = [
    # 色彩转换
    'rgb_to_lab', 'lab_to_rgb', 'rgb_to_xyz', 'xyz_to_rgb',
    'bt2020_to_srgb_direct', 'srgb_to_bt2020_direct',
    'delta_e_cie76', 'delta_e_cie94',
    
    # 损失函数
    'ColorMappingLoss', 'AdaptiveLossScheduler',
    
    # 数据采样
    'ColorGamutSampler', 'AdaptiveSampler', 'create_sampler',
    
    # 模型创建
    'create_pytorch_mapper', 'create_mapper',
    
    # 常量
    'MLX_AVAILABLE', '__version__'
]

# MLX相关导出已注释掉
# if MLX_AVAILABLE:
#     __all__.append('create_mlx_mapper') 