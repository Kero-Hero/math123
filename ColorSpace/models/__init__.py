"""
神经网络模型模块
包含PyTorch和MLX实现
"""

from .base_model import BaseColorMapper, PerceptualMappingNetwork
from .pytorch_model import create_pytorch_mapper, PyTorchColorMapper

try:
    from .mlx_model import create_mlx_mapper, MLXColorMapper
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

__all__ = [
    'BaseColorMapper', 'PerceptualMappingNetwork',
    'PyTorchColorMapper', 'create_pytorch_mapper',
    'MLX_AVAILABLE'
]

if MLX_AVAILABLE:
    __all__.extend(['MLXColorMapper', 'create_mlx_mapper']) 