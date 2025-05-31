"""
色彩空间转换模块
实现RGB、XYZ、CIELAB等色彩空间之间的转换
"""

import numpy as np
from typing import Union, Tuple, Optional


# 标准光源D65白点
D65_WHITEPOINT = np.array([95.047, 100.000, 108.883])

# 不同RGB色域的转换矩阵
# sRGB -> XYZ (D65)
SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

# BT.2020 -> XYZ (D65)  
BT2020_TO_XYZ = np.array([
    [0.6369580, 0.1446169, 0.1688809],
    [0.2627045, 0.6779981, 0.0593017],
    [0.0000000, 0.0280727, 1.0609851]
])

# 对应的逆矩阵
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)
XYZ_TO_BT2020 = np.linalg.inv(BT2020_TO_XYZ)


def gamma_encode_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """sRGB gamma编码"""
    return np.where(linear_rgb <= 0.0031308,
                    12.92 * linear_rgb,
                    1.055 * np.power(linear_rgb, 1/2.4) - 0.055)


def gamma_decode_srgb(srgb: np.ndarray) -> np.ndarray:
    """sRGB gamma解码"""
    return np.where(srgb <= 0.04045,
                    srgb / 12.92,
                    np.power((srgb + 0.055) / 1.055, 2.4))


def gamma_encode_bt2020(linear_rgb: np.ndarray) -> np.ndarray:
    """BT.2020 gamma编码"""
    alpha = 1.09929682680944
    beta = 0.018053968510807
    return np.where(linear_rgb < beta,
                    4.5 * linear_rgb,
                    alpha * np.power(linear_rgb, 0.45) - (alpha - 1))


def gamma_decode_bt2020(bt2020: np.ndarray) -> np.ndarray:
    """BT.2020 gamma解码"""
    alpha = 1.09929682680944
    beta = 0.018053968510807 * 4.5
    return np.where(bt2020 < beta,
                    bt2020 / 4.5,
                    np.power((bt2020 + alpha - 1) / alpha, 1/0.45))


def rgb_to_xyz(rgb: np.ndarray, rgb_space: str = 'srgb') -> np.ndarray:
    """
    RGB转XYZ
    
    Args:
        rgb: RGB值 (0-1范围)
        rgb_space: RGB色域类型 ('srgb' 或 'bt2020')
    
    Returns:
        XYZ值
    """
    # Gamma解码
    if rgb_space == 'srgb':
        linear_rgb = gamma_decode_srgb(rgb)
        transform_matrix = SRGB_TO_XYZ
    elif rgb_space == 'bt2020':
        linear_rgb = gamma_decode_bt2020(rgb)
        transform_matrix = BT2020_TO_XYZ
    else:
        raise ValueError(f"不支持的RGB色域: {rgb_space}")
    
    # 转换到XYZ
    if rgb.ndim == 1:
        return transform_matrix @ linear_rgb
    else:
        return (transform_matrix @ linear_rgb.T).T


def xyz_to_rgb(xyz: np.ndarray, rgb_space: str = 'srgb') -> np.ndarray:
    """
    XYZ转RGB
    
    Args:
        xyz: XYZ值
        rgb_space: 目标RGB色域类型
        
    Returns:
        RGB值 (0-1范围)
    """
    # XYZ转线性RGB
    if rgb_space == 'srgb':
        transform_matrix = XYZ_TO_SRGB
    elif rgb_space == 'bt2020':
        transform_matrix = XYZ_TO_BT2020
    else:
        raise ValueError(f"不支持的RGB色域: {rgb_space}")
    
    if xyz.ndim == 1:
        linear_rgb = transform_matrix @ xyz
    else:
        linear_rgb = (transform_matrix @ xyz.T).T
    
    # 裁剪负值
    linear_rgb = np.clip(linear_rgb, 0, 1)
    
    # Gamma编码
    if rgb_space == 'srgb':
        return gamma_encode_srgb(linear_rgb)
    elif rgb_space == 'bt2020':
        return gamma_encode_bt2020(linear_rgb)


def xyz_to_lab(xyz: np.ndarray, whitepoint: np.ndarray = D65_WHITEPOINT) -> np.ndarray:
    """
    XYZ转CIELAB
    
    Args:
        xyz: XYZ值
        whitepoint: 参考白点
        
    Returns:
        LAB值 (L: 0-100, a/b: 约-127至127)
    """
    # 标准化到白点
    xyz_normalized = xyz / whitepoint
    
    # f函数
    def f(t):
        delta = 6/29
        return np.where(t > delta**3,
                       np.power(t, 1/3),
                       t / (3 * delta**2) + 4/29)
    
    fx = f(xyz_normalized[..., 0])
    fy = f(xyz_normalized[..., 1])
    fz = f(xyz_normalized[..., 2])
    
    # LAB计算
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: np.ndarray, whitepoint: np.ndarray = D65_WHITEPOINT) -> np.ndarray:
    """
    CIELAB转XYZ
    
    Args:
        lab: LAB值
        whitepoint: 参考白点
        
    Returns:
        XYZ值
    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    
    # 中间值计算
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    # 逆f函数
    def f_inv(t):
        delta = 6/29
        return np.where(t > delta,
                       np.power(t, 3),
                       3 * delta**2 * (t - 4/29))
    
    x = f_inv(fx)
    y = f_inv(fy)
    z = f_inv(fz)
    
    xyz = np.stack([x, y, z], axis=-1)
    return xyz * whitepoint


def rgb_to_lab(rgb: np.ndarray, rgb_space: str = 'srgb') -> np.ndarray:
    """RGB直接转CIELAB"""
    xyz = rgb_to_xyz(rgb, rgb_space)
    return xyz_to_lab(xyz)


def lab_to_rgb(lab: np.ndarray, rgb_space: str = 'srgb') -> np.ndarray:
    """CIELAB直接转RGB"""
    xyz = lab_to_xyz(lab)
    return xyz_to_rgb(xyz, rgb_space)


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    计算CIE76 Delta E色彩差异
    
    Args:
        lab1, lab2: LAB色彩值
        
    Returns:
        Delta E值
    """
    diff = lab1 - lab2
    return np.sqrt(np.sum(diff**2, axis=-1))


def delta_e_cie94(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    计算CIE94 Delta E色彩差异 (更精确)
    
    Args:
        lab1, lab2: LAB色彩值
        
    Returns:
        Delta E值
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    
    dL = L1 - L2
    da = a1 - a2
    db = b1 - b2
    
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C1 - C2
    
    dH_squared = da**2 + db**2 - dC**2
    dH = np.sqrt(np.maximum(0, dH_squared))
    
    # 织物应用的权重参数
    kL, kC, kH = 1, 1, 1
    K1, K2 = 0.045, 0.015
    
    SL = 1
    SC = 1 + K1 * C1
    SH = 1 + K2 * C1
    
    delta_e = np.sqrt((dL/(kL*SL))**2 + (dC/(kC*SC))**2 + (dH/(kH*SH))**2)
    return delta_e


def bt2020_to_srgb_direct(bt2020_rgb: np.ndarray) -> np.ndarray:
    """BT.2020直接转换到sRGB (通过XYZ)"""
    xyz = rgb_to_xyz(bt2020_rgb, 'bt2020')
    return xyz_to_rgb(xyz, 'srgb')


def srgb_to_bt2020_direct(srgb_rgb: np.ndarray) -> np.ndarray:
    """sRGB直接转换到BT.2020 (通过XYZ)"""
    xyz = rgb_to_xyz(srgb_rgb, 'srgb')
    return xyz_to_rgb(xyz, 'bt2020')


def generate_color_space_volume(rgb_space: str = 'srgb', 
                               resolution: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成色域的完整体积采样
    
    Args:
        rgb_space: RGB色域类型
        resolution: 每个维度的分辨率
        
    Returns:
        rgb_colors: RGB颜色数组
        lab_colors: 对应的LAB颜色数组
    """
    # 创建RGB网格
    r = np.linspace(0, 1, resolution)
    g = np.linspace(0, 1, resolution)
    b = np.linspace(0, 1, resolution)
    
    R, G, B = np.meshgrid(r, g, b, indexing='ij')
    rgb_colors = np.stack([R.ravel(), G.ravel(), B.ravel()], axis=1)
    
    # 转换到LAB
    lab_colors = rgb_to_lab(rgb_colors, rgb_space)
    
    return rgb_colors, lab_colors 