#!/usr/bin/env python3
"""
简单使用示例：基本的色域映射
演示最基础的BT.2020到sRGB映射
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入主要功能
from models.pytorch_model import create_pytorch_mapper
from data.sampler import create_sampler
from core.color_conversion import bt2020_to_srgb_direct, delta_e_cie94, rgb_to_lab


def main():
    """简单示例主函数"""
    print("🎨 色域映射简单示例")
    print("=" * 50)
    
    # 1. 创建模型
    print("1. 创建BT.2020到sRGB映射模型...")
    model = create_pytorch_mapper(
        source_gamut='bt2020',
        target_gamut='srgb',
        deltaE_threshold=3.0,  # 允许的最大色彩偏差
        device='auto'  # 自动检测GPU
    )
    
    # 2. 准备训练数据
    print("2. 准备训练数据...")
    train_sampler = create_sampler('bt2020', 'srgb', 'perceptual')
    val_sampler = create_sampler('bt2020', 'srgb', 'uniform')
    
    # 3. 训练模型
    print("3. 开始训练 (快速演示，仅20轮)...")
    history = model.train(
        train_sampler=train_sampler,
        validation_sampler=val_sampler,
        epochs=20,
        batch_size=512,
        learning_rate=0.005,
        verbose=True
    )
    
    # 4. 测试模型
    print("\n4. 测试模型性能...")
    
    # 生成测试颜色
    test_colors_bt2020 = np.array([
        [1.0, 0.0, 0.0],  # 纯红
        [0.0, 1.0, 0.0],  # 纯绿
        [0.0, 0.0, 1.0],  # 纯蓝
        [1.0, 1.0, 0.0],  # 黄色
        [1.0, 0.0, 1.0],  # 品红
        [0.0, 1.0, 1.0],  # 青色
        [0.8, 0.2, 0.4],  # 自定义颜色1
        [0.3, 0.7, 0.9],  # 自定义颜色2
    ])
    
    # 使用模型进行映射
    mapped_colors_srgb = model.transform(test_colors_bt2020)
    
    # 对比传统的直接转换
    direct_colors_srgb = bt2020_to_srgb_direct(test_colors_bt2020)
    
    print("\n映射结果对比:")
    print("BT.2020 原色  →  神经网络映射  →  直接数学转换")
    print("-" * 60)
    
    for i, (bt2020, nn_mapped, direct) in enumerate(
        zip(test_colors_bt2020, mapped_colors_srgb, direct_colors_srgb)
    ):
        print(f"颜色 {i+1}: {bt2020} → {nn_mapped} → {direct}")
    
    # 5. 评估色彩质量
    print("\n5. 色彩质量评估...")
    
    # 计算deltaE
    bt2020_lab = rgb_to_lab(test_colors_bt2020, 'bt2020')
    mapped_lab = rgb_to_lab(mapped_colors_srgb, 'srgb')
    direct_lab = rgb_to_lab(direct_colors_srgb, 'srgb')
    
    deltaE_nn = delta_e_cie94(bt2020_lab, mapped_lab)
    deltaE_direct = delta_e_cie94(bt2020_lab, direct_lab)
    
    print(f"神经网络映射平均deltaE: {np.mean(deltaE_nn):.3f}")
    print(f"直接转换平均deltaE: {np.mean(deltaE_direct):.3f}")
    print(f"神经网络最大deltaE: {np.max(deltaE_nn):.3f}")
    print(f"直接转换最大deltaE: {np.max(deltaE_direct):.3f}")
    
    # 检查色域覆盖
    nn_in_gamut = np.all((mapped_colors_srgb >= 0) & (mapped_colors_srgb <= 1), axis=1)
    direct_in_gamut = np.all((direct_colors_srgb >= 0) & (direct_colors_srgb <= 1), axis=1)
    
    print(f"神经网络映射色域内比例: {np.mean(nn_in_gamut):.3f}")
    print(f"直接转换色域内比例: {np.mean(direct_in_gamut):.3f}")
    
    # 6. 保存模型 (可选)
    print("\n6. 保存模型...")
    output_dir = './simple_example_output'
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'bt2020_to_srgb_model.pth')
    model.save_model(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 7. 实际使用演示
    print("\n7. 实际使用演示 - 批量处理图像像素...")
    
    # 模拟一些图像像素 (随机BT.2020颜色)
    np.random.seed(42)
    image_pixels = np.random.rand(1000, 3)  # 1000个像素
    
    # 使用模型进行映射
    import time
    start_time = time.time()
    mapped_pixels = model.transform(image_pixels)
    processing_time = time.time() - start_time
    
    print(f"处理了 {len(image_pixels)} 个像素")
    print(f"处理时间: {processing_time*1000:.2f} ms")
    print(f"处理速度: {len(image_pixels)/processing_time:.0f} 像素/秒")
    
    # 统计信息
    print(f"映射后像素范围: [{mapped_pixels.min():.3f}, {mapped_pixels.max():.3f}]")
    
    print("\n" + "=" * 50)
    print("✅ 简单示例完成!")
    print("🔍 主要特性:")
    print("   • 感知均匀性保持 (CIELAB空间处理)")
    print("   • deltaE约束控制")
    print("   • 高效的批量处理")
    print("   • 保持色域边界")
    
    print("\n📝 下一步:")
    print("   • 尝试 examples/quickstart.py 查看完整功能")
    print("   • 使用 training/train_pytorch.py 训练自定义模型")
    print("   • 探索多通道映射功能")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n出现错误: {e}")
        import traceback
        traceback.print_exc() 