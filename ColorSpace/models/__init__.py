"""
神经网络模型模块
包含PyTorch实现 (Windows兼容版本，MLX已禁用)
"""

from .base_model import BaseColorMapper, PerceptualMappingNetwork
from .pytorch_model import create_pytorch_mapper, PyTorchColorMapper

# MLX导入已注释掉，Windows系统不支持
# try:
#     from .mlx_model import create_mlx_mapper, MLXColorMapper
#     MLX_AVAILABLE = True
# except ImportError:
#     MLX_AVAILABLE = False

# Windows系统禁用MLX
MLX_AVAILABLE = False

__all__ = [
    'BaseColorMapper', 'PerceptualMappingNetwork',
    'PyTorchColorMapper', 'create_pytorch_mapper',
    'MLX_AVAILABLE'
]

# MLX相关导出已注释掉
# if MLX_AVAILABLE:
#     __all__.extend(['MLXColorMapper', 'create_mlx_mapper']) 