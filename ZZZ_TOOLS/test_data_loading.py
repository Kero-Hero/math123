#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据加载脚本
验证生成的CSV文件是否能被训练脚本正确读取
"""

import sys
import os
sys.path.append('Calibration')

from train_calibration import DisplayCalibrator

def test_data_loading():
    """测试数据加载功能"""
    print("=" * 50)
    print("测试数据加载功能")
    print("=" * 50)
    
    try:
        # 创建校准器实例
        calibrator = DisplayCalibrator(data_dir="数据集")
        
        # 加载数据
        print("正在加载数据...")
        data = calibrator.load_data()
        
        # 输出数据信息
        print(f"✓ 数据加载成功！")
        print(f"✓ 数据形状: {data.shape}")
        print(f"✓ 数据类型: {data.dtype}")
        print(f"✓ 数据范围: {data.min():.2f} - {data.max():.2f}")
        print(f"✓ 数据均值: {data.mean():.2f}")
        
        # 检查每个通道的数据
        color_names = ['R', 'G', 'B']
        print("\n各通道数据统计:")
        for i, output_color in enumerate(color_names):
            for j, channel in enumerate(color_names):
                channel_data = data[i, j]
                print(f"  {output_color}_{channel}: 均值={channel_data.mean():.2f}, "
                      f"范围=[{channel_data.min():.2f}, {channel_data.max():.2f}]")
        
        print("\n✓ 所有测试通过！数据已准备就绪，可以开始训练。")
        return True
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

if __name__ == "__main__":
    test_data_loading() 