#!/usr/bin/env python3
"""
色域转换模型快速入门示例
演示如何使用三个版本的实现：PyTorch GPU、PyTorch CPU、Apple MLX
"""

import os
import sys
import numpy as np
import time
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pytorch_model import create_pytorch_mapper
from data.sampler import create_sampler
from core.color_conversion import bt2020_to_srgb_direct, delta_e_cie94, rgb_to_lab


def print_section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def benchmark_model(model, test_data: np.ndarray, model_name: str) -> Dict[str, Any]:
    """
    性能测试
    
    Args:
        model: 模型实例
        test_data: 测试数据
        model_name: 模型名称
        
    Returns:
        性能指标字典
    """
    print(f"\n测试 {model_name} 性能...")
    
    # 预热
    _ = model.transform(test_data[:100])
    
    # 测试推理速度
    start_time = time.time()
    result = model.transform(test_data)
    inference_time = time.time() - start_time
    
    throughput = len(test_data) / inference_time
    
    metrics = {
        'model_name': model_name,
        'samples': len(test_data),
        'inference_time_ms': inference_time * 1000,
        'throughput_samples_per_sec': throughput,
        'memory_usage_mb': result.nbytes / (1024 * 1024)
    }
    
    print(f"  推理时间: {metrics['inference_time_ms']:.2f} ms")
    print(f"  吞吐量: {metrics['throughput_samples_per_sec']:.0f} 样本/秒")
    print(f"  内存使用: {metrics['memory_usage_mb']:.2f} MB")
    
    return metrics


def evaluate_color_quality(source_colors: np.ndarray, 
                          mapped_colors: np.ndarray,
                          target_colors: np.ndarray,
                          source_gamut: str = 'bt2020',
                          target_gamut: str = 'srgb') -> Dict[str, float]:
    """
    评估色彩质量
    
    Args:
        source_colors: 源颜色
        mapped_colors: 映射后颜色
        target_colors: 目标颜色 (Ground Truth)
        source_gamut: 源色域
        target_gamut: 目标色域
        
    Returns:
        质量指标
    """
    # 转换到LAB空间
    source_lab = rgb_to_lab(source_colors, source_gamut)
    mapped_lab = rgb_to_lab(mapped_colors, target_gamut)
    target_lab = rgb_to_lab(target_colors, target_gamut)
    
    # 计算deltaE
    deltaE_source_mapped = delta_e_cie94(source_lab, mapped_lab)
    deltaE_mapped_target = delta_e_cie94(mapped_lab, target_lab)
    
    # 计算RGB误差
    rgb_mse = np.mean((mapped_colors - target_colors) ** 2)
    rgb_mae = np.mean(np.abs(mapped_colors - target_colors))
    
    # 色域覆盖率
    in_gamut_ratio = np.mean(np.all((mapped_colors >= 0) & (mapped_colors <= 1), axis=1))
    
    metrics = {
        'mean_deltaE_source_mapped': np.mean(deltaE_source_mapped),
        'max_deltaE_source_mapped': np.max(deltaE_source_mapped),
        'mean_deltaE_mapped_target': np.mean(deltaE_mapped_target),
        'rgb_mse': rgb_mse,
        'rgb_mae': rgb_mae,
        'in_gamut_ratio': in_gamut_ratio,
        'deltaE_threshold_3_satisfied': np.mean(deltaE_source_mapped <= 3.0),
        'deltaE_threshold_5_satisfied': np.mean(deltaE_source_mapped <= 5.0)
    }
    
    return metrics


def demo_bt2020_to_srgb():
    """演示BT.2020到sRGB的映射"""
    print_section("BT.2020 → sRGB 色域映射演示")
    
    # 创建测试数据
    print("生成测试数据...")
    sampler = create_sampler('bt2020', 'srgb', 'uniform')
    source_colors, target_colors = sampler.sample(1000)
    
    print(f"测试样本数量: {len(source_colors)}")
    print("样本颜色范围:")
    print(f"  源颜色 (BT.2020): [{source_colors.min():.3f}, {source_colors.max():.3f}]")
    print(f"  目标颜色 (sRGB): [{target_colors.min():.3f}, {target_colors.max():.3f}]")
    
    # 创建和训练模型
    models = {}
    
    # PyTorch GPU/MPS版本
    try:
        print("\n创建PyTorch模型 (自动检测GPU)...")
        pytorch_model = create_pytorch_mapper(
            source_gamut='bt2020',
            target_gamut='srgb',
            network_type='standard',
            device='auto'
        )
        
        # 快速训练 (演示用)
        print("快速训练模型 (10轮)...")
        train_sampler = create_sampler('bt2020', 'srgb', 'perceptual')
        
        pytorch_model.train(
            train_sampler=train_sampler,
            epochs=10,
            batch_size=512,
            learning_rate=0.005,
            verbose=False
        )
        
        models['PyTorch GPU/CPU'] = pytorch_model
        print("PyTorch模型训练完成")
        
    except Exception as e:
        print(f"PyTorch模型创建失败: {e}")
    
    # MLX版本 (如果可用)
    try:
        from models.mlx_model import create_mlx_mapper
        
        print("\n创建MLX模型 (Apple Silicon)...")
        mlx_model = create_mlx_mapper(
            source_gamut='bt2020',
            target_gamut='srgb',
            network_type='standard'
        )
        
        # 快速训练
        mlx_model.train(
            train_sampler=train_sampler,
            epochs=10,
            batch_size=512,
            learning_rate=0.005,
            verbose=False
        )
        
        models['Apple MLX'] = mlx_model
        print("MLX模型训练完成")
        
    except Exception as e:
        print(f"MLX模型创建失败: {e}")
    
    # 性能对比
    if models:
        print_section("性能对比")
        
        performance_results = []
        quality_results = []
        
        for name, model in models.items():
            # 性能测试
            perf_metrics = benchmark_model(model, source_colors, name)
            performance_results.append(perf_metrics)
            
            # 质量评估
            mapped_colors = model.transform(source_colors)
            quality_metrics = evaluate_color_quality(
                source_colors, mapped_colors, target_colors, 'bt2020', 'srgb'
            )
            quality_metrics['model_name'] = name
            quality_results.append(quality_metrics)
        
        # 打印性能对比
        print("\n性能对比摘要:")
        print(f"{'模型':<15} {'推理时间(ms)':<12} {'吞吐量(样本/秒)':<15} {'内存(MB)':<10}")
        print("-" * 60)
        for result in performance_results:
            print(f"{result['model_name']:<15} "
                  f"{result['inference_time_ms']:<12.2f} "
                  f"{result['throughput_samples_per_sec']:<15.0f} "
                  f"{result['memory_usage_mb']:<10.2f}")
        
        # 打印质量对比
        print("\n质量对比摘要:")
        print(f"{'模型':<15} {'平均ΔE':<8} {'最大ΔE':<8} {'RGB MSE':<10} {'色域内比例':<10}")
        print("-" * 60)
        for result in quality_results:
            print(f"{result['model_name']:<15} "
                  f"{result['mean_deltaE_source_mapped']:<8.3f} "
                  f"{result['max_deltaE_source_mapped']:<8.3f} "
                  f"{result['rgb_mse']:<10.6f} "
                  f"{result['in_gamut_ratio']:<10.3f}")


def demo_multichannel_mapping():
    """演示多通道映射 (4ch → 5ch)"""
    print_section("多通道映射演示 (4通道 → 5通道)")
    
    try:
        # 创建多通道模型
        print("创建4通道到5通道映射模型...")
        model = create_pytorch_mapper(
            source_gamut='4ch',
            target_gamut='5ch',
            network_type='standard',
            device='auto'
        )
        
        # 生成测试数据
        sampler = create_sampler('4ch', '5ch', 'uniform')
        source_data, target_data = sampler.sample(500)
        
        print(f"源数据形状: {source_data.shape}")
        print(f"目标数据形状: {target_data.shape}")
        
        # 快速训练
        print("训练多通道映射模型...")
        train_sampler = create_sampler('4ch', '5ch', 'uniform')
        
        model.train(
            train_sampler=train_sampler,
            epochs=15,
            batch_size=256,
            learning_rate=0.003,
            verbose=False
        )
        
        # 测试映射
        mapped_data = model.transform(source_data)
        
        # 评估
        mse = np.mean((mapped_data - target_data) ** 2)
        mae = np.mean(np.abs(mapped_data - target_data))
        
        print("\n多通道映射结果:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  输出范围: [{mapped_data.min():.3f}, {mapped_data.max():.3f}]")
        
        # 显示几个映射样例
        print("\n映射样例:")
        for i in range(min(5, len(source_data))):
            print(f"  源: {source_data[i]} → 映射: {mapped_data[i]}")
            print(f"     目标: {target_data[i]}")
            print()
        
    except Exception as e:
        print(f"多通道映射演示失败: {e}")


def demo_color_accuracy():
    """演示色彩精度控制"""
    print_section("色彩精度控制演示")
    
    # 测试不同deltaE阈值的效果
    deltaE_thresholds = [1.0, 3.0, 5.0, 10.0]
    
    print("测试不同deltaE阈值对映射质量的影响...")
    
    results = []
    
    for threshold in deltaE_thresholds:
        print(f"\n训练模型 (deltaE阈值 = {threshold})...")
        
        try:
            model = create_pytorch_mapper(
                source_gamut='bt2020',
                target_gamut='srgb',
                deltaE_threshold=threshold,
                device='auto'
            )
            
            # 训练
            sampler = create_sampler('bt2020', 'srgb', 'perceptual')
            model.train(
                train_sampler=sampler,
                epochs=10,
                batch_size=512,
                learning_rate=0.005,
                verbose=False
            )
            
            # 评估
            test_source, test_target = sampler.sample(500)
            mapped_colors = model.transform(test_source)
            
            quality_metrics = evaluate_color_quality(
                test_source, mapped_colors, test_target, 'bt2020', 'srgb'
            )
            quality_metrics['deltaE_threshold'] = threshold
            results.append(quality_metrics)
            
        except Exception as e:
            print(f"  失败: {e}")
            continue
    
    # 显示结果
    if results:
        print("\ndeltaE阈值对比:")
        print(f"{'阈值':<6} {'平均ΔE':<8} {'ΔE≤3满足率':<12} {'ΔE≤5满足率':<12} {'RGB MSE':<10}")
        print("-" * 55)
        
        for result in results:
            print(f"{result['deltaE_threshold']:<6.1f} "
                  f"{result['mean_deltaE_source_mapped']:<8.3f} "
                  f"{result['deltaE_threshold_3_satisfied']:<12.3f} "
                  f"{result['deltaE_threshold_5_satisfied']:<12.3f} "
                  f"{result['rgb_mse']:<10.6f}")


def main():
    """主演示函数"""
    print("🎨 色域转换模型快速入门演示")
    print("本演示将展示:")
    print("1. BT.2020 → sRGB 映射")
    print("2. 多通道映射 (4ch → 5ch)")  
    print("3. 色彩精度控制")
    print("4. 不同实现版本的性能对比")
    
    try:
        # 演示1: BT.2020到sRGB映射
        demo_bt2020_to_srgb()
        
        # 演示2: 多通道映射
        demo_multichannel_mapping()
        
        # 演示3: 色彩精度控制
        demo_color_accuracy()
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print_section("演示完成")
    print("🎉 感谢使用色域转换模型!")
    print("\n项目特性:")
    print("✅ 感知均匀性保持 (CIELAB空间处理)")
    print("✅ DeltaE约束控制")
    print("✅ 多平台支持 (PyTorch GPU/CPU + Apple MLX)")
    print("✅ 多种色域映射 (BT.2020↔sRGB, 多通道)")
    print("✅ 智能数据采样")
    print("✅ 自适应训练策略")
    print("\n详细文档请参考 README.md")


if __name__ == "__main__":
    main() 