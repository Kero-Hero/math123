#!/usr/bin/env python3
"""
测试MPS兼容性修复
"""

import sys
import numpy as np

def test_basic_import():
    """测试基础导入"""
    try:
        from models.pytorch_model import create_pytorch_mapper
        print("✓ PyTorch模型导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    try:
        from models.pytorch_model import create_pytorch_mapper
        
        model = create_pytorch_mapper(
            source_gamut='bt2020',
            target_gamut='srgb',
            deltaE_threshold=3.0,
            device='auto'
        )
        print("✓ 模型创建成功")
        print(f"  设备: {model.device}")
        return model
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return None

def test_forward_pass(model):
    """测试前向传播"""
    try:
        test_data = np.random.rand(5, 3).astype(np.float32)
        result = model.forward(test_data)
        
        print(f"✓ 前向传播成功: {test_data.shape} -> {result.shape}")
        print(f"  输入类型: {test_data.dtype}")
        print(f"  输出类型: {result.dtype}")
        print(f"  输出范围: [{result.min():.3f}, {result.max():.3f}]")
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False

def test_training_step(model):
    """测试训练步骤"""
    try:
        from data.sampler import create_sampler
        
        # 创建小量测试数据
        sampler = create_sampler('bt2020', 'srgb', 'uniform')
        source_data, target_data = sampler.sample(10)
        
        # 设置优化器
        model.setup_optimizer(learning_rate=0.001)
        
        # 执行一个训练步骤
        loss_dict = model.train_step(source_data, target_data)
        
        print("✓ 训练步骤成功")
        print(f"  总损失: {loss_dict['total_loss']:.6f}")
        print(f"  deltaE损失: {loss_dict['deltaE_loss']:.6f}")
        print(f"  感知损失: {loss_dict['perceptual_loss']:.6f}")
        return True
    except Exception as e:
        print(f"✗ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 MPS兼容性修复测试")
    print("=" * 50)
    
    # 测试1: 基础导入
    if not test_basic_import():
        print("\n❌ 基础导入失败，请检查PyTorch安装")
        return
    
    # 测试2: 模型创建
    model = test_model_creation()
    if model is None:
        print("\n❌ 模型创建失败")
        return
    
    # 测试3: 前向传播
    if not test_forward_pass(model):
        print("\n❌ 前向传播失败")
        return
    
    # 测试4: 训练步骤
    if not test_training_step(model):
        print("\n❌ 训练步骤失败")
        return
    
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！MPS兼容性修复成功")
    print("\n可以安全地运行:")
    print("  python examples/simple_example.py")
    print("  python examples/quickstart.py")

if __name__ == "__main__":
    main() 