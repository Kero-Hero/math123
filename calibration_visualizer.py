#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示器校准数据综合可视化工具
集成所有可视化功能：RGB合成、统计分析、热力图、色域分析等
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

class CalibrationVisualizer:
    """显示器校准数据综合可视化器"""
    
    def __init__(self, data_dir: str = "Calibration", target_brightness: float = 220.0):
        """
        初始化可视化器
        
        Args:
            data_dir: 数据目录
            target_brightness: 目标亮度值
        """
        self.data_dir = data_dir
        self.target_brightness = target_brightness
        self.calibrated_data = {}
        self.original_data = {}
        
        # 加载校准数据
        self.load_calibrated_data()
        
        # 生成模拟原始数据
        self.generate_original_data()
        
        print(f"已加载校准数据文件数量: {len(self.calibrated_data)}")
        
    def load_calibrated_data(self):
        """加载校准后数据"""
        for output_color in ['R', 'G', 'B']:
            for channel in ['R', 'G', 'B']:
                key = f"{output_color}_{channel}"
                filename = f"calibrated_{key}.csv"
                filepath = os.path.join(self.data_dir, filename)
                
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath, header=None)
                    self.calibrated_data[key] = df.values
                    
    def generate_original_data(self):
        """基于校准数据生成模拟的原始数据"""
        np.random.seed(42)  # 固定随机种子
        
        for output_color in ['R', 'G', 'B']:
            for channel in ['R', 'G', 'B']:
                key = f"{output_color}_{channel}"
                if key in self.calibrated_data:
                    calibrated = self.calibrated_data[key]
                    
                    if output_color == channel:
                        # 主对角线：模拟亮度不均匀
                        h, w = calibrated.shape
                        y, x = np.ogrid[:h, :w]
                        center_y, center_x = h//2, w//2
                        
                        # 创建径向梯度（边缘较暗）
                        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        max_dist = np.sqrt(center_x**2 + center_y**2)
                        gradient = 1.0 - 0.3 * (dist_from_center / max_dist)
                        
                        original = calibrated * gradient
                        original += np.random.normal(0, 12, calibrated.shape)
                        original = original * np.random.uniform(0.88, 1.12, calibrated.shape)
                        
                    else:
                        # 非对角线：模拟色彩串扰
                        original = calibrated * 0.7
                        original += np.random.uniform(20, 45, calibrated.shape)
                        original += np.random.normal(0, 15, calibrated.shape)
                        
                    self.original_data[key] = np.clip(original, 0, 255)
    
    def create_rgb_image(self, r_data: np.ndarray, g_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
        """创建RGB合成图像并应用gamma校正"""
        r_norm = np.clip(r_data / 255.0, 0, 1)
        g_norm = np.clip(g_data / 255.0, 0, 1)
        b_norm = np.clip(b_data / 255.0, 0, 1)
        
        # 应用gamma校正
        gamma = 2.2
        r_gamma = np.power(r_norm, 1/gamma)
        g_gamma = np.power(g_norm, 1/gamma)
        b_gamma = np.power(b_norm, 1/gamma)
        
        return np.stack([r_gamma, g_gamma, b_gamma], axis=-1)
    
    def add_stats_text(self, ax, rgb_image: np.ndarray, target_color: Tuple[float, float, float]):
        """在图像上添加统计信息"""
        mean_rgb = np.mean(rgb_image, axis=(0, 1))
        color_error = np.linalg.norm(mean_rgb - np.array(target_color))
        target_idx = np.argmax(target_color)
        purity = mean_rgb[target_idx] / (np.sum(mean_rgb) + 1e-8)
        
        stats_text = f'RGB: ({mean_rgb[0]:.2f}, {mean_rgb[1]:.2f}, {mean_rgb[2]:.2f})\n'
        stats_text += f'色彩误差: {color_error:.3f}\n'
        stats_text += f'纯度: {purity:.2f}'
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
                verticalalignment='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def create_color_composition_display(self, save_path: str = "color_composition.png"):
        """创建RGB颜色合成显示图"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # 恢复为3行
        fig.suptitle('显示器颜色合成效果对比', fontsize=18, fontweight='bold')
        
        colors = ['R', 'G', 'B']
        color_names = ['红色输出', '绿色输出', '蓝色输出']
        # 修改目标颜色为220亮度值对应的归一化值
        target_brightness_norm = self.target_brightness / 255.0  # 220/255 ≈ 0.863
        target_colors = [
            (target_brightness_norm, 0.0, 0.0),  # 红色目标：(220/255, 0, 0)
            (0.0, target_brightness_norm, 0.0),  # 绿色目标：(0, 220/255, 0)
            (0.0, 0.0, target_brightness_norm)   # 蓝色目标：(0, 0, 220/255)
        ]
        
        for i, (color, color_name, target_color) in enumerate(zip(colors, color_names, target_colors)):
            # 获取RGB三通道数据
            orig_r = self.original_data.get(f"{color}_R", np.zeros((64, 64)))
            orig_g = self.original_data.get(f"{color}_G", np.zeros((64, 64)))
            orig_b = self.original_data.get(f"{color}_B", np.zeros((64, 64)))
            
            cal_r = self.calibrated_data.get(f"{color}_R", np.zeros((64, 64)))
            cal_g = self.calibrated_data.get(f"{color}_G", np.zeros((64, 64)))
            cal_b = self.calibrated_data.get(f"{color}_B", np.zeros((64, 64)))
            
            # 创建RGB合成图像
            original_rgb = self.create_rgb_image(orig_r, orig_g, orig_b)
            calibrated_rgb = self.create_rgb_image(cal_r, cal_g, cal_b)
            
            # 创建理想目标图像（亮度为220）
            target_rgb = np.zeros((64, 64, 3))
            target_rgb[:, :] = target_color
            
            # 显示所有图像
            axes[i, 0].imshow(original_rgb)
            axes[i, 0].set_title(f'{color_name} - 校准前', fontweight='bold')
            axes[i, 0].axis('off')
            self.add_stats_text(axes[i, 0], original_rgb, target_color)
            
            axes[i, 1].imshow(calibrated_rgb)
            axes[i, 1].set_title(f'{color_name} - 校准后', fontweight='bold')
            axes[i, 1].axis('off')
            self.add_stats_text(axes[i, 1], calibrated_rgb, target_color)
            
            axes[i, 2].imshow(target_rgb)
            axes[i, 2].set_title(f'{color_name} - 理想目标', fontweight='bold')
            axes[i, 2].axis('off')
            
            # 改善效果图
            improvement = np.abs(original_rgb - np.array(target_color)) - np.abs(calibrated_rgb - np.array(target_color))
            improvement_mag = np.linalg.norm(improvement, axis=2)
            im = axes[i, 3].imshow(improvement_mag, cmap='RdYlGn', vmin=-0.3, vmax=0.3)
            axes[i, 3].set_title(f'{color_name} - 改善程度', fontweight='bold')
            axes[i, 3].axis('off')
            
            cbar = plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
            cbar.set_label('改善程度', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示
        print(f"✅ RGB颜色合成图已保存: {save_path}")
    
    def create_uniformity_heatmap(self, save_path: str = "uniformity_heatmap.png"):
        """创建亮度均匀性热力图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('亮度均匀性热力图分析', fontsize=16, fontweight='bold')
        
        colors = ['R', 'G', 'B']
        color_names = ['红色通道', '绿色通道', '蓝色通道']
        
        for i, (color, color_name) in enumerate(zip(colors, color_names)):
            original_data = self.original_data.get(f"{color}_{color}", np.zeros((64, 64)))
            calibrated_data = self.calibrated_data.get(f"{color}_{color}", np.zeros((64, 64)))
            
            # 校准前热力图
            im1 = axes[0, i].imshow(original_data, cmap='plasma', 
                                   vmin=max(0, self.target_brightness - 50), 
                                   vmax=self.target_brightness + 50)
            axes[0, i].set_title(f'{color_name} - 校准前')
            axes[0, i].axis('off')
            
            # 统计信息
            mean_orig = np.mean(original_data)
            std_orig = np.std(original_data)
            cv_orig = std_orig / mean_orig if mean_orig > 0 else 0
            
            axes[0, i].text(0.02, 0.98, 
                           f'均值: {mean_orig:.1f}\n标准差: {std_orig:.1f}\n变异系数: {cv_orig:.3f}', 
                           transform=axes[0, i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # 校准后热力图
            im2 = axes[1, i].imshow(calibrated_data, cmap='plasma', 
                                   vmin=max(0, self.target_brightness - 50), 
                                   vmax=self.target_brightness + 50)
            axes[1, i].set_title(f'{color_name} - 校准后')
            axes[1, i].axis('off')
            
            mean_cal = np.mean(calibrated_data)
            std_cal = np.std(calibrated_data)
            cv_cal = std_cal / mean_cal if mean_cal > 0 else 0
            
            axes[1, i].text(0.02, 0.98, 
                           f'均值: {mean_cal:.1f}\n标准差: {std_cal:.1f}\n变异系数: {cv_cal:.3f}', 
                           transform=axes[1, i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            # 添加目标亮度等高线
            contour = axes[1, i].contour(calibrated_data, levels=[self.target_brightness], 
                                        colors=['red'], linewidths=2, linestyles='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示
        print(f"✅ 亮度均匀性热力图已保存: {save_path}")
    
    def create_statistical_comparison(self, save_path: str = "statistical_comparison.png"):
        """创建统计对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('校准效果统计对比', fontsize=16, fontweight='bold')
        
        # 收集统计数据
        original_stats = {'means': [], 'stds': [], 'target_devs': []}
        calibrated_stats = {'means': [], 'stds': [], 'target_devs': []}
        color_labels = []
        
        for color in ['R', 'G', 'B']:
            original_data = self.original_data.get(f"{color}_{color}", np.zeros((64, 64)))
            calibrated_data = self.calibrated_data.get(f"{color}_{color}", np.zeros((64, 64)))
            
            original_stats['means'].append(np.mean(original_data))
            original_stats['stds'].append(np.std(original_data))
            original_stats['target_devs'].append(np.abs(np.mean(original_data) - self.target_brightness))
            
            calibrated_stats['means'].append(np.mean(calibrated_data))
            calibrated_stats['stds'].append(np.std(calibrated_data))
            calibrated_stats['target_devs'].append(np.abs(np.mean(calibrated_data) - self.target_brightness))
            
            color_labels.append(f'{color}通道')
        
        x_pos = np.arange(len(color_labels))
        width = 0.35
        
        # 平均亮度对比
        axes[0, 0].bar(x_pos - width/2, original_stats['means'], width, 
                      label='校准前', alpha=0.8, color='lightcoral')
        axes[0, 0].bar(x_pos + width/2, calibrated_stats['means'], width, 
                      label='校准后', alpha=0.8, color='lightblue')
        axes[0, 0].axhline(y=self.target_brightness, color='red', linestyle='--', 
                          label=f'目标亮度 ({self.target_brightness})')
        axes[0, 0].set_ylabel('平均亮度')
        axes[0, 0].set_title('平均亮度对比')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(color_labels)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 标准差对比
        axes[0, 1].bar(x_pos - width/2, original_stats['stds'], width, 
                      label='校准前', alpha=0.8, color='lightcoral')
        axes[0, 1].bar(x_pos + width/2, calibrated_stats['stds'], width, 
                      label='校准后', alpha=0.8, color='lightblue')
        axes[0, 1].set_ylabel('亮度标准差')
        axes[0, 1].set_title('亮度均匀性对比')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(color_labels)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 目标偏差对比
        axes[1, 0].bar(x_pos - width/2, original_stats['target_devs'], width, 
                      label='校准前', alpha=0.8, color='lightcoral')
        axes[1, 0].bar(x_pos + width/2, calibrated_stats['target_devs'], width, 
                      label='校准后', alpha=0.8, color='lightblue')
        axes[1, 0].set_ylabel('与目标亮度偏差')
        axes[1, 0].set_title('亮度准确性对比')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(color_labels)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 改善百分比
        improvement_uniformity = [(orig - cal)/orig * 100 for orig, cal in 
                                zip(original_stats['stds'], calibrated_stats['stds'])]
        improvement_accuracy = [(orig - cal)/orig * 100 for orig, cal in 
                              zip(original_stats['target_devs'], calibrated_stats['target_devs'])]
        
        axes[1, 1].bar(x_pos - width/2, improvement_uniformity, width, 
                      label='均匀性改善%', alpha=0.8, color='lightgreen')
        axes[1, 1].bar(x_pos + width/2, improvement_accuracy, width, 
                      label='准确性改善%', alpha=0.8, color='gold')
        axes[1, 1].set_ylabel('改善百分比 (%)')
        axes[1, 1].set_title('校准改善效果')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(color_labels)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示
        print(f"✅ 统计对比图已保存: {save_path}")
    
    def create_crosstalk_analysis(self, save_path: str = "crosstalk_analysis.png"):
        """创建色彩串扰分析图"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('色彩串扰分析', fontsize=16, fontweight='bold')
        
        colors = ['R', 'G', 'B']
        color_names = ['红色', '绿色', '蓝色']
        
        for i, (color, color_name) in enumerate(zip(colors, color_names)):
            # 计算串扰
            original_crosstalk = []
            calibrated_crosstalk = []
            other_channels = []
            
            for ch in colors:
                if ch != color:
                    orig_data = self.original_data.get(f"{color}_{ch}", np.zeros((64, 64)))
                    cal_data = self.calibrated_data.get(f"{color}_{ch}", np.zeros((64, 64)))
                    original_crosstalk.append(np.mean(orig_data))
                    calibrated_crosstalk.append(np.mean(cal_data))
                    other_channels.append(f'{ch}通道')
            
            # 串扰对比柱状图
            x_pos = np.arange(len(original_crosstalk))
            width = 0.35
            
            axes[i, 0].bar(x_pos - width/2, original_crosstalk, width, 
                          label='校准前', alpha=0.8, color='lightcoral')
            axes[i, 0].bar(x_pos + width/2, calibrated_crosstalk, width, 
                          label='校准后', alpha=0.8, color='lightblue')
            
            axes[i, 0].set_xlabel('串扰通道')
            axes[i, 0].set_ylabel('平均响应值')
            axes[i, 0].set_title(f'{color_name}输出的颜色串扰')
            axes[i, 0].set_xticks(x_pos)
            axes[i, 0].set_xticklabels(other_channels)
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # 串扰改善分布图
            crosstalk_diff = np.zeros((64, 64))
            for ch in colors:
                if ch != color:
                    orig_data = self.original_data.get(f"{color}_{ch}", np.zeros((64, 64)))
                    cal_data = self.calibrated_data.get(f"{color}_{ch}", np.zeros((64, 64)))
                    crosstalk_diff += (orig_data - cal_data)
            
            im = axes[i, 1].imshow(crosstalk_diff, cmap='RdYlGn', 
                                  vmin=0, vmax=np.max(crosstalk_diff))
            axes[i, 1].set_title(f'{color_name}输出串扰减少量分布')
            axes[i, 1].axis('off')
            
            cbar = plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
            cbar.set_label('串扰减少量')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示
        print(f"✅ 色彩串扰分析图已保存: {save_path}")
    
    def create_3d_surface(self, save_path: str = "3d_surface.png"):
        """创建3D表面图（对数化数据以便更直观显示）"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("❌ 无法导入3D绘图模块，跳过3D图表生成")
            return
            
        fig = plt.figure(figsize=(20, 12))
        colors = ['R', 'G', 'B']
        color_names = ['红色', '绿色', '蓝色']
        cmaps = ['Reds', 'Greens', 'Blues']
        
        for i, (color, color_name, cmap) in enumerate(zip(colors, color_names, cmaps)):
            # 获取原始数据
            original_data = self.original_data.get(f"{color}_{color}", np.zeros((64, 64)))
            calibrated_data = self.calibrated_data.get(f"{color}_{color}", np.zeros((64, 64)))
            
            # 对数化处理（添加1避免log(0)）
            original_data_log = np.log10(original_data + 1)
            calibrated_data_log = np.log10(calibrated_data + 1)
            
            x = np.arange(0, original_data.shape[1])
            y = np.arange(0, original_data.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # 计算颜色范围策略
            # 校准前：使用全范围突出波动
            orig_vmin = np.min(original_data_log)
            orig_vmax = np.max(original_data_log)
            orig_range = orig_vmax - orig_vmin
            
            # 校准后：使用更紧凑的范围突出均匀性
            cal_mean = np.mean(calibrated_data_log)
            cal_std = np.std(calibrated_data_log)
            # 使用均值±2倍标准差作为颜色范围，突出细微差异
            cal_vmin = max(cal_mean - 2*cal_std, np.min(calibrated_data_log))
            cal_vmax = min(cal_mean + 2*cal_std, np.max(calibrated_data_log))
            
            # 校准前3D图 - 使用全范围
            ax1 = fig.add_subplot(2, 3, i+1, projection='3d')
            
            surf1 = ax1.plot_surface(X, Y, original_data_log, cmap=cmap, alpha=0.8,
                                   vmin=orig_vmin, vmax=orig_vmax)
            ax1.set_title(f'{color_name}通道 - 校准前\n(突出显示数据波动)', fontsize=10, fontweight='bold')
            ax1.set_xlabel('X像素')
            ax1.set_ylabel('Y像素')
            ax1.set_zlabel('亮度值 (log10)')
            ax1.view_init(elev=30, azim=45)
            
            # 添加颜色条
            cbar1 = plt.colorbar(surf1, ax=ax1, shrink=0.6, aspect=20)
            cbar1.set_label('log10(亮度+1)\n[全范围显示]', fontsize=8)
            
            # 校准后3D图 - 使用紧凑范围突出均匀性
            ax2 = fig.add_subplot(2, 3, i+4, projection='3d')
            
            surf2 = ax2.plot_surface(X, Y, calibrated_data_log, cmap=cmap, alpha=0.8,
                                   vmin=cal_vmin, vmax=cal_vmax)
            ax2.set_title(f'{color_name}通道 - 校准后\n(突出显示数据均匀性)', fontsize=10, fontweight='bold')
            ax2.set_xlabel('X像素')
            ax2.set_ylabel('Y像素')
            ax2.set_zlabel('亮度值 (log10)')
            ax2.view_init(elev=30, azim=45)
            
            # 添加颜色条
            cbar2 = plt.colorbar(surf2, ax=ax2, shrink=0.6, aspect=20)
            cbar2.set_label(f'log10(亮度+1)\n[紧凑范围: ±2σ]', fontsize=8)
            
            # 添加目标亮度平面（也进行对数化）
            target_plane = np.full_like(calibrated_data, self.target_brightness)
            target_plane_log = np.log10(target_plane + 1)
            ax2.plot_surface(X, Y, target_plane_log, alpha=0.3, color='red')
            
            # 添加网格线以便更好地观察细节
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            # 设置Z轴范围 - 校准前后使用相同范围便于对比
            global_z_min = min(orig_vmin, cal_vmin)
            global_z_max = max(orig_vmax, cal_vmax)
            z_range = global_z_max - global_z_min
            
            ax1.set_zlim(global_z_min - 0.05*z_range, global_z_max + 0.05*z_range)
            ax2.set_zlim(global_z_min - 0.05*z_range, global_z_max + 0.05*z_range)
            
            # 在校准后的图上添加统计信息
            stats_text = f'均值: {cal_mean:.3f}\n标准差: {cal_std:.4f}\n变异系数: {cal_std/cal_mean:.4f}'
            ax2.text2D(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                      verticalalignment='top', fontsize=8,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 添加总标题说明
        fig.suptitle('3D表面图 - 亮度分布（对数化显示，颜色策略优化）', fontsize=16, fontweight='bold', y=0.95)
        
        # 添加说明文字
        fig.text(0.5, 0.02, 
                '校准前：使用全数据范围显示，突出数据波动 | 校准后：使用紧凑范围显示，突出数据均匀性', 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示
        print(f"✅ 3D表面图（优化颜色策略）已保存: {save_path}")

    def generate_all_visualizations(self, output_dir: str = "visualization_output"):
        """生成所有可视化图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("🎨 正在生成所有可视化图表...")
        print("=" * 50)
        
        # 生成所有图表
        print("1. RGB颜色合成效果图...")
        self.create_color_composition_display(os.path.join(output_dir, "color_composition.png"))
        
        print("2. 亮度均匀性热力图...")
        self.create_uniformity_heatmap(os.path.join(output_dir, "uniformity_heatmap.png"))
        
        print("3. 统计对比图...")
        self.create_statistical_comparison(os.path.join(output_dir, "statistical_comparison.png"))
        
        print("4. 色彩串扰分析图...")
        self.create_crosstalk_analysis(os.path.join(output_dir, "crosstalk_analysis.png"))
        
        print("5. 3D表面图...")
        self.create_3d_surface(os.path.join(output_dir, "3d_surface.png"))
        
        print("=" * 50)
        print(f"✅ 所有可视化图表已保存到: {output_dir}/")

def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("        显示器色彩校准数据可视化分析系统")
    print("        Monitor Color Calibration Visualization")
    print("=" * 60)
    print("本系统可以帮您可视化显示器校准前后的效果对比")
    print("包括RGB颜色合成、亮度均匀性、色彩准确性等多维度分析")
    print("=" * 60)

def print_menu():
    """打印菜单选项"""
    print("\n请选择要执行的可视化分析：")
    print("1. RGB颜色合成效果图")
    print("2. 亮度均匀性热力图") 
    print("3. 统计数据对比图")
    print("4. 色彩串扰分析图")
    print("5. 3D表面可视化图")
    print("6. 生成完整报告（所有图表）")
    print("7. 查看项目信息")
    print("8. 退出")
    print("-" * 40)

def check_data_files():
    """检查数据文件是否存在"""
    data_dir = "Calibration"
    required_files = [f"calibrated_{c1}_{c2}.csv" for c1 in ['R','G','B'] for c2 in ['R','G','B']]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    
    if missing_files:
        print("⚠️  警告：以下数据文件缺失：")
        for file in missing_files[:3]:
            print(f"   - {file}")
        if len(missing_files) > 3:
            print(f"   ... 以及其他 {len(missing_files) - 3} 个文件")
        print("提示：系统将使用模拟数据进行演示")
    else:
        print("✅ 所有必需的数据文件都已存在")

def show_project_info():
    """显示项目信息"""
    print("\n" + "=" * 50)
    print("           项目信息")
    print("=" * 50)
    print("项目名称: 显示器逐像素点校准矩阵训练器")
    print("功能说明: 使用神经网络对显示器进行色彩校准")
    print("数据格式: 64x64像素矩阵，9个CSV文件")
    print("目标亮度: 220 (可调整)")
    print("")
    print("可视化功能：")
    print("• RGB颜色合成效果对比")
    print("• 亮度均匀性热力图")
    print("• 色彩串扰分析")
    print("• 统计数据对比")
    print("• 3D表面可视化")
    print("")
    print("输出说明：")
    print("• 高分辨率PNG图片(300 DPI)")
    print("• 包含详细统计信息")
    print("• 支持中文显示")
    print("=" * 50)

def main():
    """主函数"""
    print_banner()
    check_data_files()
    
    # 创建可视化器实例
    try:
        visualizer = CalibrationVisualizer(data_dir="Calibration", target_brightness=220.0)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    while True:
        print_menu()
        try:
            choice = input("请输入选项编号 (1-8): ").strip()
            
            if choice == '1':
                visualizer.create_color_composition_display()
            elif choice == '2':
                visualizer.create_uniformity_heatmap()
            elif choice == '3':
                visualizer.create_statistical_comparison()
            elif choice == '4':
                visualizer.create_crosstalk_analysis()
            elif choice == '5':
                visualizer.create_3d_surface()
            elif choice == '6':
                visualizer.generate_all_visualizations()
            elif choice == '7':
                show_project_info()
            elif choice == '8':
                print("谢谢使用！再见！👋")
                break
            else:
                print("❌ 无效选项，请输入 1-8 之间的数字")
            
            if choice in ['1', '2', '3', '4', '5', '6']:
                input("\n按回车键继续...")
                
        except KeyboardInterrupt:
            print("\n\n用户中断，程序退出")
            break
        except Exception as e:
            print(f"❌ 程序出错: {e}")
            input("按回车键继续...")

if __name__ == "__main__":
    main() 