"""
PyTorch色域映射模型训练脚本
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pytorch_model import create_pytorch_mapper
from data.sampler import create_sampler
from core.color_conversion import bt2020_to_srgb_direct


def setup_training_data(source_gamut: str, 
                       target_gamut: str,
                       train_samples: int = 50000,
                       val_samples: int = 5000,
                       strategy: str = 'perceptual') -> Tuple:
    """
    设置训练和验证数据采样器
    
    Args:
        source_gamut: 源色域
        target_gamut: 目标色域
        train_samples: 训练样本数
        val_samples: 验证样本数
        strategy: 采样策略
        
    Returns:
        训练采样器, 验证采样器, 测试数据
    """
    print(f"设置数据采样器: {source_gamut} -> {target_gamut}")
    print(f"采样策略: {strategy}")
    
    # 创建采样器
    train_sampler = create_sampler(source_gamut, target_gamut, strategy, adaptive=True)
    val_sampler = create_sampler(source_gamut, target_gamut, 'uniform', adaptive=False)
    
    # 生成固定的测试数据集
    test_source, test_target = val_sampler.sample(val_samples)
    
    print(f"训练采样器设置完成")
    print(f"验证数据: {len(test_source)} 样本")
    
    return train_sampler, val_sampler, (test_source, test_target)


def train_model(model, 
               train_sampler, 
               val_sampler,
               test_data: Tuple,
               config: Dict) -> Dict:
    """
    训练模型
    
    Args:
        model: 色域映射模型
        train_sampler: 训练数据采样器
        val_sampler: 验证数据采样器
        test_data: 测试数据
        config: 训练配置
        
    Returns:
        训练历史
    """
    print("开始训练...")
    print(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    start_time = time.time()
    
    # 训练模型
    history = model.train(
        train_sampler=train_sampler,
        validation_sampler=val_sampler,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    print(f"\n训练完成! 总时间: {training_time:.2f}秒")
    
    # 最终评估
    test_source, test_target = test_data
    final_metrics = model.evaluate(test_source, test_target)
    
    print("\n最终评估结果:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # 添加训练信息到历史
    history['training_time'] = training_time
    history['final_metrics'] = final_metrics
    history['config'] = config
    
    return history


def plot_training_history(history: Dict, save_path: str = None):
    """
    绘制训练历史
    
    Args:
        history: 训练历史
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('色域映射模型训练历史', fontsize=16)
    
    # 总损失
    axes[0, 0].plot(history['train_loss'], label='训练损失', color='blue')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='验证损失', color='orange')
    axes[0, 0].set_title('总损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # deltaE损失
    axes[0, 1].plot(history['deltaE_loss'], label='DeltaE损失', color='red')
    axes[0, 1].set_title('DeltaE约束损失')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('DeltaE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 感知损失
    axes[1, 0].plot(history['perceptual_loss'], label='感知损失', color='green')
    axes[1, 0].set_title('感知均匀性损失')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Perceptual Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 色域损失
    axes[1, 1].plot(history['gamut_loss'], label='色域损失', color='purple')
    axes[1, 1].set_title('色域边界损失')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gamut Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图表已保存到: {save_path}")
    
    plt.show()


def save_results(model, history: Dict, config: Dict, output_dir: str):
    """
    保存训练结果
    
    Args:
        model: 训练好的模型
        history: 训练历史
        config: 训练配置
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(output_dir, 'color_mapper_pytorch.pth')
    model.save_model(model_path)
    
    # 保存训练历史
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python类型
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            elif isinstance(value, dict):
                history_serializable[key] = {k: float(v) if hasattr(v, 'item') else v 
                                           for k, v in value.items()}
            else:
                history_serializable[key] = value
        
        json.dump(history_serializable, f, indent=2, ensure_ascii=False)
    
    # 保存训练图表
    plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # 保存配置
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"所有结果已保存到: {output_dir}")


def demonstrate_mapping(model, source_gamut: str, target_gamut: str, output_dir: str):
    """
    演示色域映射效果
    
    Args:
        model: 训练好的模型
        source_gamut: 源色域
        target_gamut: 目标色域
        output_dir: 输出目录
    """
    print(f"\n演示色域映射: {source_gamut} -> {target_gamut}")
    
    # 生成测试颜色
    test_colors = np.array([
        [1.0, 0.0, 0.0],  # 红色
        [0.0, 1.0, 0.0],  # 绿色
        [0.0, 0.0, 1.0],  # 蓝色
        [1.0, 1.0, 0.0],  # 黄色
        [1.0, 0.0, 1.0],  # 品红
        [0.0, 1.0, 1.0],  # 青色
        [0.5, 0.5, 0.5],  # 灰色
        [1.0, 1.0, 1.0],  # 白色
    ])
    
    # 执行映射
    mapped_colors = model.transform(test_colors)
    
    # 对比结果
    print("颜色映射对比:")
    print("源颜色 (RGB) -> 映射颜色 (RGB)")
    for i, (source, mapped) in enumerate(zip(test_colors, mapped_colors)):
        print(f"  {source} -> {mapped}")
    
    # 可视化对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 源颜色
    ax1.imshow([test_colors], aspect='auto')
    ax1.set_title(f'源颜色 ({source_gamut})')
    ax1.set_xticks(range(len(test_colors)))
    ax1.set_xticklabels(['红', '绿', '蓝', '黄', '品红', '青', '灰', '白'])
    ax1.set_yticks([])
    
    # 映射后颜色
    ax2.imshow([np.clip(mapped_colors, 0, 1)], aspect='auto')
    ax2.set_title(f'映射颜色 ({target_gamut})')
    ax2.set_xticks(range(len(mapped_colors)))
    ax2.set_xticklabels(['红', '绿', '蓝', '黄', '品红', '青', '灰', '白'])
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    demo_path = os.path.join(output_dir, 'color_mapping_demo.png')
    plt.savefig(demo_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"色域映射演示图已保存到: {demo_path}")


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='训练PyTorch色域映射模型')
    parser.add_argument('--source_gamut', type=str, default='bt2020', 
                       choices=['bt2020', 'srgb', '4ch', '5ch'],
                       help='源色域')
    parser.add_argument('--target_gamut', type=str, default='srgb',
                       choices=['bt2020', 'srgb', '4ch', '5ch'],
                       help='目标色域')
    parser.add_argument('--network_type', type=str, default='standard',
                       choices=['standard', 'deep', 'wide'],
                       help='网络类型')
    parser.add_argument('--deltaE_threshold', type=float, default=3.0,
                       help='deltaE阈值')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--sampling_strategy', type=str, default='perceptual',
                       choices=['uniform', 'perceptual', 'boundary', 'stratified'],
                       help='采样策略')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='计算设备')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 训练配置
    config = {
        'source_gamut': args.source_gamut,
        'target_gamut': args.target_gamut,
        'network_type': args.network_type,
        'deltaE_threshold': args.deltaE_threshold,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'sampling_strategy': args.sampling_strategy,
        'device': args.device,
        'patience': 15,
        'train_samples': 50000,
        'val_samples': 5000
    }
    
    print("=" * 60)
    print("PyTorch色域映射模型训练")
    print("=" * 60)
    
    # 创建模型
    print("创建模型...")
    model = create_pytorch_mapper(
        source_gamut=config['source_gamut'],
        target_gamut=config['target_gamut'],
        network_type=config['network_type'],
        deltaE_threshold=config['deltaE_threshold'],
        device=config['device']
    )
    
    # 设置数据
    train_sampler, val_sampler, test_data = setup_training_data(
        config['source_gamut'],
        config['target_gamut'],
        config['train_samples'],
        config['val_samples'],
        config['sampling_strategy']
    )
    
    # 训练模型
    history = train_model(model, train_sampler, val_sampler, test_data, config)
    
    # 保存结果
    output_dir = os.path.join(args.output_dir, 
                             f"pytorch_{config['source_gamut']}_to_{config['target_gamut']}")
    save_results(model, history, config, output_dir)
    
    # 演示映射效果
    demonstrate_mapping(model, config['source_gamut'], config['target_gamut'], output_dir)
    
    print("\n训练完成!")
    print(f"模型性能摘要:")
    final_metrics = history['final_metrics']
    print(f"  平均deltaE: {final_metrics.get('mean_deltaE', 0):.3f}")
    print(f"  deltaE满足率: {final_metrics.get('deltaE_satisfied_ratio', 0):.3f}")
    print(f"  色域内比例: {final_metrics.get('in_gamut_ratio', 0):.3f}")
    print(f"  MSE: {final_metrics.get('mse', 0):.6f}")


if __name__ == "__main__":
    main() 