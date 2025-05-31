#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示器逐像素点校准矩阵训练器
支持 Apple MLX 和 PyTorch 自动GPU加速
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict
import argparse

# 尝试导入不同的深度学习框架
framework = None
device = None

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    framework = "mlx"
    print("使用 Apple MLX 框架")
except ImportError:
    try:
        import torch
        import torch.nn as nn_torch
        import torch.optim as optim_torch
        framework = "pytorch"
        # 检测设备
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("使用 Apple Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("使用 NVIDIA CUDA")
        else:
            device = torch.device("cpu")
            print("使用 CPU")
    except ImportError:
        print("警告：未找到 MLX 或 PyTorch，将使用 NumPy 进行计算")
        framework = "numpy"

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DisplayCalibrator:
    """显示器校准器类"""
    
    def __init__(self, data_dir: str = "数据集", target_brightness: float = 220.0):
        """
        初始化校准器
        
        Args:
            data_dir: 数据集目录
            target_brightness: 目标亮度值
        """
        self.data_dir = data_dir
        self.target_brightness = target_brightness
        self.height = 64
        self.width = 64
        self.channels = 3  # R, G, B
        
        # 加载数据
        self.data = self.load_data()
        logger.info(f"数据加载完成，形状: {self.data.shape}")
        
        # 初始化校准矩阵 (64, 64, 3, 3) - 每个像素点都有一个3x3变换矩阵
        # 使用单位矩阵作为初始值，加上小的随机扰动
        if framework == "mlx":
            # MLX使用32位浮点数
            identity = mx.eye(3, dtype=mx.float32)
            identity_expanded = mx.broadcast_to(identity[None, None, :, :], (self.height, self.width, 3, 3))
            noise = mx.random.normal((self.height, self.width, 3, 3), dtype=mx.float32) * 0.01
            self.calibration_matrix = identity_expanded + noise
        elif framework == "pytorch":
            identity = torch.eye(3, device=device, dtype=torch.float32)
            identity_expanded = identity.unsqueeze(0).unsqueeze(0).expand(self.height, self.width, -1, -1)
            noise = torch.randn((self.height, self.width, 3, 3), device=device, dtype=torch.float32) * 0.01
            self.calibration_matrix = (identity_expanded + noise).requires_grad_(True)
        else:
            identity = np.eye(3, dtype=np.float32)
            identity_expanded = np.tile(identity[None, None, :, :], (self.height, self.width, 1, 1))
            noise = np.random.normal(0, 0.01, (self.height, self.width, 3, 3)).astype(np.float32)
            self.calibration_matrix = identity_expanded + noise
            
    def load_data(self) -> np.ndarray:
        """
        加载所有数据文件
        
        Returns:
            形状为 (3, 3, 64, 64) 的数组，表示 [输出颜色, 测量通道, 高度, 宽度]
        """
        data = np.zeros((3, 3, self.height, self.width), dtype=np.float32)
        
        color_map = {'R': 0, 'G': 1, 'B': 2}
        
        for output_color in ['R', 'G', 'B']:
            for channel in ['R', 'G', 'B']:
                filename = f"{output_color}_{channel}.csv"
                filepath = os.path.join(self.data_dir, filename)
                
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath, header=None)
                    # 只获取前64行64列的数据，并转换为float32
                    csv_data = df.values[:self.height, :self.width].astype(np.float32)
                    data[color_map[output_color], color_map[channel]] = csv_data
                    logger.info(f"加载文件: {filename}, 尺寸: {csv_data.shape}")
                else:
                    logger.warning(f"文件不存在: {filename}")
                    
        return data
    
    def apply_calibration(self, input_rgb):
        """
        应用校准矩阵
        
        Args:
            input_rgb: 输入RGB值，形状为 (64, 64, 3)
            
        Returns:
            校准后的RGB值（确保非负）
        """
        if framework == "mlx":
            # 使用 MLX 进行批量矩阵乘法
            input_rgb_expanded = input_rgb[:, :, :, None]  # (64, 64, 3, 1)
            calibrated = mx.matmul(self.calibration_matrix, input_rgb_expanded).squeeze(-1)
            # 确保输出非负
            calibrated = mx.maximum(calibrated, 0.0)
        elif framework == "pytorch":
            # 使用 PyTorch 进行批量矩阵乘法
            input_rgb_expanded = input_rgb.unsqueeze(-1)  # (64, 64, 3, 1)
            calibrated = torch.matmul(self.calibration_matrix, input_rgb_expanded).squeeze(-1)
            # 确保输出非负
            calibrated = torch.clamp(calibrated, min=0.0)
        else:
            # 使用 NumPy
            calibrated = np.zeros_like(input_rgb)
            for i in range(self.height):
                for j in range(self.width):
                    calibrated[i, j] = np.dot(self.calibration_matrix[i, j], input_rgb[i, j])
            # 确保输出非负
            calibrated = np.maximum(calibrated, 0.0)
                    
        return calibrated
    
    def pixel_loss(self, predicted_rgb, target_rgb):
        """
        计算每个像素点的损失函数
        
        Args:
            predicted_rgb: 预测的RGB值 (64, 64, 3)
            target_rgb: 目标RGB值 (64, 64, 3)
            
        Returns:
            损失值
        """
        if framework == "mlx":
            # 1. 亮度差损失 - 各通道与目标亮度的平方差
            brightness_loss = mx.sum((predicted_rgb - self.target_brightness) ** 2)
            
            # 2. 颜色纯度损失 - 惩罚不纯的颜色
            r, g, b = predicted_rgb[:, :, 0], predicted_rgb[:, :, 1], predicted_rgb[:, :, 2]
            
            # 计算通道间差异
            cross_channel_diff = (r - b) * (r - g) * (g - r) * (g - b) * (b - r) * (b - g)
            # 避免log(0)或log(负数)
            cross_channel_diff = mx.maximum(cross_channel_diff, 1e-8)
            purity_loss = mx.sum(mx.log(mx.abs(cross_channel_diff) + 1e-8))
            
            # 3. 高亮度差异惩罚减少
            high_brightness_factor = mx.log((r * g * b) / (256**3) + 1e-8)
            high_brightness_loss = mx.sum(high_brightness_factor)
            
        elif framework == "pytorch":
            # 1. 亮度差损失
            brightness_loss = torch.sum((predicted_rgb - self.target_brightness) ** 2)
            
            # 2. 颜色纯度损失
            r, g, b = predicted_rgb[:, :, 0], predicted_rgb[:, :, 1], predicted_rgb[:, :, 2]
            
            cross_channel_diff = (r - b) * (r - g) * (g - r) * (g - b) * (b - r) * (b - g)
            cross_channel_diff = torch.clamp(cross_channel_diff, min=1e-8)
            purity_loss = torch.sum(torch.log(torch.abs(cross_channel_diff) + 1e-8))
            
            # 3. 高亮度差异惩罚减少
            high_brightness_factor = torch.log((r * g * b) / (256**3) + 1e-8)
            high_brightness_loss = torch.sum(high_brightness_factor)
            
        else:
            # NumPy 版本
            brightness_loss = np.sum((predicted_rgb - self.target_brightness) ** 2)
            
            r, g, b = predicted_rgb[:, :, 0], predicted_rgb[:, :, 1], predicted_rgb[:, :, 2]
            
            cross_channel_diff = (r - b) * (r - g) * (g - r) * (g - b) * (b - r) * (b - g)
            cross_channel_diff = np.maximum(cross_channel_diff, 1e-8)
            purity_loss = np.sum(np.log(np.abs(cross_channel_diff) + 1e-8))
            
            high_brightness_factor = np.log((r * g * b) / (256**3) + 1e-8)
            high_brightness_loss = np.sum(high_brightness_factor)
        
        return brightness_loss + 0.1 * purity_loss + 0.05 * high_brightness_loss
    
    def global_loss(self, predicted_rgb):
        """
        计算全局损失函数
        
        Args:
            predicted_rgb: 预测的RGB值 (3, 64, 64, 3) 对应R/G/B输出
            
        Returns:
            全局损失值
        """
        total_loss = 0
        
        for color_idx in range(3):  # R, G, B 输出
            rgb_output = predicted_rgb[color_idx]  # (64, 64, 3)
            
            if framework == "mlx":
                # 1. 最小化极差
                max_val = mx.max(rgb_output)
                min_val = mx.min(rgb_output)
                range_loss = (max_val - min_val) ** 2
                
                # 2. 最小化相邻像素点的亮度差
                # 水平方向
                h_diff = mx.sum((rgb_output[1:, :, :] - rgb_output[:-1, :, :]) ** 2)
                # 垂直方向
                v_diff = mx.sum((rgb_output[:, 1:, :] - rgb_output[:, :-1, :]) ** 2)
                neighbor_loss = h_diff + v_diff
                
                # 3. 均值与目标亮度的差异
                mean_brightness = mx.mean(rgb_output)
                mean_loss = (mean_brightness - self.target_brightness) ** 2
                
            elif framework == "pytorch":
                # 1. 最小化极差
                max_val = torch.max(rgb_output)
                min_val = torch.min(rgb_output)
                range_loss = (max_val - min_val) ** 2
                
                # 2. 最小化相邻像素点的亮度差
                h_diff = torch.sum((rgb_output[1:, :, :] - rgb_output[:-1, :, :]) ** 2)
                v_diff = torch.sum((rgb_output[:, 1:, :] - rgb_output[:, :-1, :]) ** 2)
                neighbor_loss = h_diff + v_diff
                
                # 3. 均值与目标亮度的差异
                mean_brightness = torch.mean(rgb_output)
                mean_loss = (mean_brightness - self.target_brightness) ** 2
                
            else:
                # NumPy 版本
                max_val = np.max(rgb_output)
                min_val = np.min(rgb_output)
                range_loss = (max_val - min_val) ** 2
                
                h_diff = np.sum((rgb_output[1:, :, :] - rgb_output[:-1, :, :]) ** 2)
                v_diff = np.sum((rgb_output[:, 1:, :] - rgb_output[:, :-1, :]) ** 2)
                neighbor_loss = h_diff + v_diff
                
                mean_brightness = np.mean(rgb_output)
                mean_loss = (mean_brightness - self.target_brightness) ** 2
            
            total_loss += range_loss + 0.1 * neighbor_loss + mean_loss
            
        return total_loss
    
    def total_loss(self, predicted_outputs):
        """
        计算总损失函数
        
        Args:
            predicted_outputs: 预测输出 (3, 3, 64, 64) [输出颜色, 测量通道, 高度, 宽度]
            
        Returns:
            总损失
        """
        # 转换为目标格式
        if framework == "mlx":
            target_data = mx.array(self.data)
            predicted_rgb = mx.zeros((3, self.height, self.width, 3))
        elif framework == "pytorch":
            target_data = torch.tensor(self.data, device=device, dtype=torch.float32)
            predicted_rgb = torch.zeros((3, self.height, self.width, 3), device=device)
        else:
            target_data = self.data
            predicted_rgb = np.zeros((3, self.height, self.width, 3))
        
        # 重新排列预测输出格式
        for color_idx in range(3):
            for i in range(self.height):
                for j in range(self.width):
                    for channel in range(3):
                        predicted_rgb[color_idx, i, j, channel] = predicted_outputs[color_idx, channel, i, j]
        
        # 计算像素损失
        pixel_loss_total = 0
        for color_idx in range(3):
            target_rgb = target_data[color_idx].transpose(1, 2, 0)  # (64, 64, 3)
            pred_rgb = predicted_rgb[color_idx]  # (64, 64, 3)
            pixel_loss_total += self.pixel_loss(pred_rgb, target_rgb)
        
        # 计算全局损失
        global_loss_total = self.global_loss(predicted_rgb)
        
        return pixel_loss_total + 0.5 * global_loss_total
    
    def train(self, epochs: int = 1000, learning_rate: float = 0.01):
        """
        训练校准矩阵
        
        Args:
            epochs: 训练轮数
            learning_rate: 学习率
        """
        logger.info("开始训练校准矩阵...")
        
        if framework == "mlx":
            # MLX 使用手动梯度下降
            def loss_fn(calibration_matrix):
                total_loss = 0
                
                # 转换数据为MLX格式
                target_data = mx.array(self.data.astype(np.float32))
                
                for color_idx in range(3):
                    # 创建目标输入（用户想要显示的颜色）
                    if color_idx == 0:  # 想要显示红色
                        target_input = mx.stack([
                            mx.full((self.height, self.width), self.target_brightness, dtype=mx.float32),
                            mx.zeros((self.height, self.width), dtype=mx.float32),
                            mx.zeros((self.height, self.width), dtype=mx.float32)
                        ], axis=2)
                    elif color_idx == 1:  # 想要显示绿色
                        target_input = mx.stack([
                            mx.zeros((self.height, self.width), dtype=mx.float32),
                            mx.full((self.height, self.width), self.target_brightness, dtype=mx.float32),
                            mx.zeros((self.height, self.width), dtype=mx.float32)
                        ], axis=2)
                    else:  # 想要显示蓝色
                        target_input = mx.stack([
                            mx.zeros((self.height, self.width), dtype=mx.float32),
                            mx.zeros((self.height, self.width), dtype=mx.float32),
                            mx.full((self.height, self.width), self.target_brightness, dtype=mx.float32)
                        ], axis=2)
                    
                    # 应用校准矩阵
                    input_expanded = target_input[:, :, :, None]  # (64, 64, 3, 1)
                    calibrated_input = mx.matmul(calibration_matrix, input_expanded).squeeze(-1)  # (64, 64, 3)
                    
                    # 强制非负约束（在损失计算中）
                    negative_penalty = mx.sum(mx.maximum(0, -calibrated_input) ** 2) * 1000.0  # 对负值强烈惩罚
                    calibrated_input_clipped = mx.maximum(calibrated_input, 0.0)  # 用于后续计算的非负版本
                    
                    # 实际测量数据（当前输入220时的实际输出）
                    actual_output = target_data[color_idx].transpose(1, 2, 0)  # (64, 64, 3)
                    
                    # 理想输出（我们希望得到的纯色输出）
                    if color_idx == 0:  # 应该输出纯红色
                        ideal_output = mx.stack([
                            mx.full((self.height, self.width), self.target_brightness, dtype=mx.float32),
                            mx.zeros((self.height, self.width), dtype=mx.float32),
                            mx.zeros((self.height, self.width), dtype=mx.float32)
                        ], axis=2)
                    elif color_idx == 1:  # 应该输出纯绿色
                        ideal_output = mx.stack([
                            mx.zeros((self.height, self.width), dtype=mx.float32),
                            mx.full((self.height, self.width), self.target_brightness, dtype=mx.float32),
                            mx.zeros((self.height, self.width), dtype=mx.float32)
                        ], axis=2)
                    else:  # 应该输出纯蓝色
                        ideal_output = mx.stack([
                            mx.zeros((self.height, self.width), dtype=mx.float32),
                            mx.zeros((self.height, self.width), dtype=mx.float32),
                            mx.full((self.height, self.width), self.target_brightness, dtype=mx.float32)
                        ], axis=2)
                    
                    # 建立简单的线性显示器模型：output = gain * input + offset
                    # 假设校准后的输入通过相同的显示器特性应该产生理想输出
                    
                    # 计算显示器的响应特性（基于原始测量）
                    # 原始输入是220的纯色，实际输出是actual_output
                    original_input = mx.stack([
                        mx.full((self.height, self.width), self.target_brightness if color_idx == 0 else 0.0, dtype=mx.float32),
                        mx.full((self.height, self.width), self.target_brightness if color_idx == 1 else 0.0, dtype=mx.float32),
                        mx.full((self.height, self.width), self.target_brightness if color_idx == 2 else 0.0, dtype=mx.float32)
                    ], axis=2)
                    
                    # 估计显示器的增益（避免除零）
                    gain = actual_output / (original_input + 1e-8)
                    
                    # 使用校准后的输入预测输出
                    predicted_output = gain * calibrated_input_clipped
                    
                    # 主要损失：预测输出应该接近理想输出
                    output_loss = mx.sum((predicted_output - ideal_output) ** 2)
                    
                    # 辅助损失1：校准输入应该在合理范围内
                    range_loss = mx.sum(mx.maximum(0, calibrated_input_clipped - 255) ** 2) + \
                                mx.sum(mx.maximum(0, -calibrated_input_clipped) ** 2)
                    
                    # 辅助损失2：相邻像素的校准应该平滑
                    smooth_loss = mx.sum((calibrated_input_clipped[1:, :, :] - calibrated_input_clipped[:-1, :, :]) ** 2) + \
                                 mx.sum((calibrated_input_clipped[:, 1:, :] - calibrated_input_clipped[:, :-1, :]) ** 2)
                    
                    total_loss += output_loss + 0.01 * range_loss + 0.001 * smooth_loss + negative_penalty
                
                return total_loss
            
            # 创建梯度计算函数
            grad_fn = mx.value_and_grad(loss_fn)
            
            # 训练循环 - 手动梯度下降
            for epoch in range(epochs):
                loss_val, grads = grad_fn(self.calibration_matrix)
                
                # 检查NaN
                if mx.isnan(loss_val):
                    logger.warning(f"损失变成NaN在epoch {epoch}，停止训练")
                    break
                
                # 梯度裁剪防止梯度爆炸
                grad_norm = mx.sqrt(mx.sum(grads ** 2))
                if grad_norm > 1.0:
                    grads = grads / grad_norm
                
                # 手动更新参数，使用更小的学习率
                self.calibration_matrix = self.calibration_matrix - (learning_rate * 0.1) * grads
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {float(loss_val):.6f}, Grad Norm: {float(grad_norm):.6f}")
                    
        elif framework == "pytorch":
            # PyTorch 优化器
            optimizer = optim_torch.Adam([self.calibration_matrix], lr=learning_rate)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                # 应用校准矩阵得到预测输出
                predicted_outputs = torch.zeros((3, 3, self.height, self.width), device=device)
                
                for color_idx in range(3):
                    # 创建纯色输入
                    input_rgb = torch.zeros((self.height, self.width, 3), device=device)
                    input_rgb[:, :, color_idx] = self.target_brightness
                    
                    # 应用校准
                    calibrated = self.apply_calibration(input_rgb)
                    
                    # 存储到预测输出
                    for channel in range(3):
                        predicted_outputs[color_idx, channel] = calibrated[:, :, channel]
                
                loss = self.total_loss(predicted_outputs)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                    
        else:
            # NumPy 简单梯度下降
            for epoch in range(epochs):
                # 简化版本：随机扰动并检查损失
                current_loss = self._numpy_forward()
                
                # 简单的随机搜索优化
                noise = np.random.normal(0, 0.001, self.calibration_matrix.shape)
                self.calibration_matrix += noise
                new_loss = self._numpy_forward()
                
                if new_loss > current_loss:
                    self.calibration_matrix -= noise  # 回退
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {current_loss:.6f}")
    
    def _numpy_forward(self):
        """NumPy 前向传播"""
        predicted_outputs = np.zeros((3, 3, self.height, self.width))
        
        for color_idx in range(3):
            input_rgb = np.zeros((self.height, self.width, 3))
            input_rgb[:, :, color_idx] = self.target_brightness
            
            calibrated = self.apply_calibration(input_rgb)
            
            for channel in range(3):
                predicted_outputs[color_idx, channel] = calibrated[:, :, channel]
        
        return self.total_loss(predicted_outputs)
    
    def save_calibration_matrix(self, filename: str = "calibration.csv"):
        """保存校准矩阵"""
        if framework == "mlx":
            matrix_np = np.array(self.calibration_matrix)
        elif framework == "pytorch":
            matrix_np = self.calibration_matrix.detach().cpu().numpy()
        else:
            matrix_np = self.calibration_matrix
        
        # 将 (64, 64, 3, 3) 重塑为可保存的格式
        reshaped = matrix_np.reshape(self.height * self.width, 9)
        
        # 创建列名
        columns = [f'{i}_{j}' for i in range(3) for j in range(3)]
        
        df = pd.DataFrame(reshaped, columns=columns)
        df.to_csv(filename, index=False)
        logger.info(f"校准矩阵已保存到: {filename}")
    
    def generate_calibrated_outputs(self):
        """生成校准后的输出并保存"""
        logger.info("生成校准后的输出...")
        
        color_names = ['R', 'G', 'B']
        channel_names = ['R', 'G', 'B']
        
        results = {}
        
        for color_idx, color in enumerate(color_names):
            # 创建纯色输入
            if framework == "mlx":
                # MLX版本 - 使用正确的数组创建方式
                if color_idx == 0:  # R通道
                    r_channel = mx.full((self.height, self.width), self.target_brightness, dtype=mx.float32)
                    zeros = mx.zeros((self.height, self.width), dtype=mx.float32)
                    input_rgb = mx.stack([r_channel, zeros, zeros], axis=2)
                elif color_idx == 1:  # G通道
                    g_channel = mx.full((self.height, self.width), self.target_brightness, dtype=mx.float32)
                    zeros = mx.zeros((self.height, self.width), dtype=mx.float32)
                    input_rgb = mx.stack([zeros, g_channel, zeros], axis=2)
                else:  # B通道
                    b_channel = mx.full((self.height, self.width), self.target_brightness, dtype=mx.float32)
                    zeros = mx.zeros((self.height, self.width), dtype=mx.float32)
                    input_rgb = mx.stack([zeros, zeros, b_channel], axis=2)
            elif framework == "pytorch":
                input_rgb = torch.zeros((self.height, self.width, 3), device=device, dtype=torch.float32)
                input_rgb[:, :, color_idx] = self.target_brightness
            else:
                input_rgb = np.zeros((self.height, self.width, 3), dtype=np.float32)
                input_rgb[:, :, color_idx] = self.target_brightness
            
            # 应用校准
            calibrated = self.apply_calibration(input_rgb)
            
            # 转换为NumPy格式
            if framework == "mlx":
                calibrated_np = np.array(calibrated)
            elif framework == "pytorch":
                calibrated_np = calibrated.detach().cpu().numpy()
            else:
                calibrated_np = calibrated
            
            # 确保输出非负且在合理范围内
            calibrated_np = np.clip(calibrated_np, 0.0, 255.0)
            
            # 保存每个通道
            for channel_idx, channel in enumerate(channel_names):
                filename = f"calibrated_{color}_{channel}.csv"
                channel_data = calibrated_np[:, :, channel_idx]
                
                df = pd.DataFrame(channel_data)
                df.to_csv(filename, index=False, header=False)
                
                results[f"{color}_{channel}"] = channel_data
                logger.info(f"已保存: {filename}")
        
        return results
    
    def evaluate_results(self, calibrated_outputs: Dict, save_path: str = "calibration_evaluation.txt"):
        """评估校准结果"""
        logger.info("评估校准结果...")
        
        evaluation_text = []
        evaluation_text.append("=== 显示器校准结果评估 ===\n")
        
        # 整体统计
        evaluation_text.append("1. 整体亮度统计:")
        
        color_names = ['R', 'G', 'B']
        for color_idx, color in enumerate(color_names):
            evaluation_text.append(f"\n  目标{color}色时的输出效果:")
            
            # 获取该目标颜色的所有通道输出
            target_r = calibrated_outputs[f"{color}_R"]
            target_g = calibrated_outputs[f"{color}_G"] 
            target_b = calibrated_outputs[f"{color}_B"]
            
            # 主通道统计（应该接近目标亮度220）
            if color_idx == 0:  # 目标红色
                main_channel_data = target_r
                other_channels_data = [target_g, target_b]
                main_channel_name = "R"
                other_channels_names = ["G", "B"]
            elif color_idx == 1:  # 目标绿色
                main_channel_data = target_g
                other_channels_data = [target_r, target_b]
                main_channel_name = "G"
                other_channels_names = ["R", "B"]
            else:  # 目标蓝色
                main_channel_data = target_b
                other_channels_data = [target_r, target_g]
                main_channel_name = "B"
                other_channels_names = ["R", "G"]
            
            # 主通道分析
            main_mean = np.mean(main_channel_data)
            main_std = np.std(main_channel_data)
            main_min = np.min(main_channel_data)
            main_max = np.max(main_channel_data)
            main_range = main_max - main_min
            
            evaluation_text.append(f"    主通道{main_channel_name}: (目标: {self.target_brightness})")
            evaluation_text.append(f"      平均值: {main_mean:.2f}")
            evaluation_text.append(f"      标准差: {main_std:.2f}")
            evaluation_text.append(f"      范围: {main_min:.2f} - {main_max:.2f} (极差: {main_range:.2f})")
            evaluation_text.append(f"      与目标差异: {abs(main_mean - self.target_brightness):.2f}")
            
            # 其他通道分析（应该接近0）
            for i, (other_data, other_name) in enumerate(zip(other_channels_data, other_channels_names)):
                other_mean = np.mean(other_data)
                other_std = np.std(other_data)
                other_min = np.min(other_data)
                other_max = np.max(other_data)
                
                evaluation_text.append(f"    副通道{other_name}: (目标: 0)")
                evaluation_text.append(f"      平均值: {other_mean:.2f}")
                evaluation_text.append(f"      标准差: {other_std:.2f}")
                evaluation_text.append(f"      范围: {other_min:.2f} - {other_max:.2f}")
                evaluation_text.append(f"      串扰程度: {other_mean:.2f}")
            
            # 颜色纯度计算
            other_mean_total = np.mean([np.mean(data) for data in other_channels_data])
            purity_ratio = main_mean / (other_mean_total + 1e-8)
            evaluation_text.append(f"    颜色纯度比: {purity_ratio:.2f} (越大越好)")
            
            # 均匀性评估
            main_h_diff = np.mean((main_channel_data[1:, :] - main_channel_data[:-1, :]) ** 2)
            main_v_diff = np.mean((main_channel_data[:, 1:] - main_channel_data[:, :-1]) ** 2)
            main_uniformity = main_h_diff + main_v_diff
            evaluation_text.append(f"    主通道均匀性: {main_uniformity:.2f} (越小越好)")
        
        # 颜色纯度评估
        evaluation_text.append("\n2. 颜色纯度评估:")
        
        for color_idx, color in enumerate(['R', 'G', 'B']):
            main_channel = calibrated_outputs[f"{color}_{color}"]  # 主通道
            other_channels = []
            
            for channel_idx, channel in enumerate(['R', 'G', 'B']):
                if channel_idx != color_idx:
                    other_channels.append(calibrated_outputs[f"{color}_{channel}"])
            
            main_mean = np.mean(main_channel)
            other_mean = np.mean(other_channels)
            purity_ratio = main_mean / (other_mean + 1e-8)
            
            evaluation_text.append(f"  目标{color}色纯度:")
            evaluation_text.append(f"    主通道{color}平均值: {main_mean:.2f}")
            evaluation_text.append(f"    其他通道平均值: {other_mean:.2f}")
            evaluation_text.append(f"    纯度比例: {purity_ratio:.2f}")
        
        # 均匀性评估
        evaluation_text.append("\n3. 显示均匀性评估:")
        
        for color_idx, color in enumerate(['R', 'G', 'B']):
            main_channel = calibrated_outputs[f"{color}_{color}"]
            
            # 计算主通道的均匀性
            h_diff = np.mean((main_channel[1:, :] - main_channel[:-1, :]) ** 2)
            v_diff = np.mean((main_channel[:, 1:] - main_channel[:, :-1]) ** 2)
            uniformity = h_diff + v_diff
            
            evaluation_text.append(f"  目标{color}色主通道均匀性: {uniformity:.2f} (越小越好)")
        
        # 原始数据对比
        evaluation_text.append("\n4. 校准前后对比:")
        
        for color_idx, color in enumerate(['R', 'G', 'B']):
            evaluation_text.append(f"\n  目标{color}色对比:")
            
            # 原始数据：当输入目标颜色时的实际输出
            original_main_channel = self.data[color_idx, color_idx]  # 主通道的原始输出
            original_other_channels = []
            for channel_idx in range(3):
                if channel_idx != color_idx:
                    original_other_channels.append(self.data[color_idx, channel_idx])
            
            original_main_mean = np.mean(original_main_channel)
            original_other_mean = np.mean(original_other_channels)
            original_purity = original_main_mean / (original_other_mean + 1e-8)
            
            # 校准后数据
            calibrated_main_channel = calibrated_outputs[f"{color}_{color}"]
            calibrated_other_channels = []
            for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
                if channel_idx != color_idx:
                    calibrated_other_channels.append(calibrated_outputs[f"{color}_{channel_name}"])
            
            calibrated_main_mean = np.mean(calibrated_main_channel)
            calibrated_other_mean = np.mean(calibrated_other_channels)
            calibrated_purity = calibrated_main_mean / (calibrated_other_mean + 1e-8)
            
            evaluation_text.append(f"    主通道{color}:")
            evaluation_text.append(f"      校准前均值: {original_main_mean:.2f} (与目标差异: {abs(original_main_mean - self.target_brightness):.2f})")
            evaluation_text.append(f"      校准后均值: {calibrated_main_mean:.2f} (与目标差异: {abs(calibrated_main_mean - self.target_brightness):.2f})")
            evaluation_text.append(f"      主通道改善: {abs(original_main_mean - self.target_brightness) - abs(calibrated_main_mean - self.target_brightness):.2f}")
            
            evaluation_text.append(f"    其他通道串扰:")
            evaluation_text.append(f"      校准前均值: {original_other_mean:.2f}")
            evaluation_text.append(f"      校准后均值: {calibrated_other_mean:.2f}")
            evaluation_text.append(f"      串扰改善: {original_other_mean - calibrated_other_mean:.2f}")
            
            evaluation_text.append(f"    颜色纯度:")
            evaluation_text.append(f"      校准前纯度比: {original_purity:.2f}")
            evaluation_text.append(f"      校准后纯度比: {calibrated_purity:.2f}")
            evaluation_text.append(f"      纯度改善: {calibrated_purity - original_purity:.2f}")
        
        # 校准优化建议
        evaluation_text.append("\n5. 输出信号约束检查:")
        
        # 检查是否有负值
        has_negative = False
        negative_count = 0
        for color in ['R', 'G', 'B']:
            for channel in ['R', 'G', 'B']:
                data = calibrated_outputs[f"{color}_{channel}"]
                neg_pixels = np.sum(data < 0)
                if neg_pixels > 0:
                    has_negative = True
                    negative_count += neg_pixels
                    evaluation_text.append(f"  警告: {color}_{channel} 有 {neg_pixels} 个负值像素")
        
        if not has_negative:
            evaluation_text.append("  ✓ 所有输出信号均为非负值")
        else:
            evaluation_text.append(f"  警告: 总共有 {negative_count} 个负值像素已被裁剪为0")
        
        # 检查是否超出正常范围
        out_of_range_count = 0
        for color in ['R', 'G', 'B']:
            for channel in ['R', 'G', 'B']:
                data = calibrated_outputs[f"{color}_{channel}"]
                over_255 = np.sum(data > 255)
                if over_255 > 0:
                    out_of_range_count += over_255
                    evaluation_text.append(f"  注意: {color}_{channel} 有 {over_255} 个像素超过255")
        
        if out_of_range_count == 0:
            evaluation_text.append("  ✓ 所有输出信号均在0-255范围内")
        else:
            evaluation_text.append(f"  注意: 总共有 {out_of_range_count} 个像素超过255已被裁剪")
        
        evaluation_text.append("\n6. 优化建议:")
        evaluation_text.append("  - 增加更多的测量点以提高校准精度")
        evaluation_text.append("  - 考虑添加温度补偿机制")
        evaluation_text.append("  - 实施自适应亮度调整")
        evaluation_text.append("  - 定期重新校准以保持性能")
        evaluation_text.append("  - 考虑使用更高分辨率的校准矩阵")
        evaluation_text.append("  - 使用非负矩阵分解(NMF)方法确保物理约束")
        evaluation_text.append("  - 添加更强的正则化项以减少过校准")
        evaluation_text.append("  - 考虑分段线性校准以提高精度")
        
        # 保存评估结果
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(evaluation_text))
        
        logger.info(f"评估结果已保存到: {save_path}")
        
        # 打印简要结果
        print("\n=== 校准结果简要 ===")
        for line in evaluation_text[:15]:  # 打印前15行
            print(line)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='显示器校准训练器')
    parser.add_argument('--data_dir', default='数据集', help='数据集目录')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--target_brightness', type=float, default=220.0, help='目标亮度')
    
    args = parser.parse_args()
    
    # 创建校准器
    calibrator = DisplayCalibrator(
        data_dir=args.data_dir,
        target_brightness=args.target_brightness
    )
    
    # 训练
    calibrator.train(epochs=args.epochs, learning_rate=args.learning_rate)
    
    # 保存校准矩阵
    calibrator.save_calibration_matrix("calibration.csv")
    
    # 生成校准后的输出
    calibrated_outputs = calibrator.generate_calibrated_outputs()
    
    # 评估结果
    calibrator.evaluate_results(calibrated_outputs)
    
    logger.info("校准训练完成！")

if __name__ == "__main__":
    main() 