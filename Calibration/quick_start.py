#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本 - 显示器校准训练器演示
"""

import os
import sys

def check_environment():
    """检查运行环境"""
    print("=== 显示器校准训练器 ===")
    print("正在检查环境...")
    
    # 检查数据集
    if not os.path.exists("数据集"):
        print("错误: 未找到数据集文件夹")
        return False
    
    # 检查数据集文件
    required_files = [
        "R_R.csv", "R_G.csv", "R_B.csv",
        "G_R.csv", "G_G.csv", "G_B.csv", 
        "B_R.csv", "B_G.csv", "B_B.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join("数据集", file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"错误: 缺少数据文件: {missing_files}")
        return False
    
    print("✓ 数据集检查完成")
    return True

def run_training():
    """运行训练"""
    print("开始训练校准矩阵...")
    
    from train_calibration import DisplayCalibrator
    
    # 创建校准器（使用较小的参数进行快速演示）
    calibrator = DisplayCalibrator(
        data_dir="数据集",
        target_brightness=220.0
    )
    
    print("✓ 校准器初始化完成")
    
    # 训练（使用更多轮数进行充分训练）
    print("开始训练，这可能需要几分钟时间...")
    calibrator.train(epochs=2000, learning_rate=0.001)
    print("✓ 训练完成")
    
    # 保存校准矩阵
    calibrator.save_calibration_matrix("calibration.csv")
    print("✓ 校准矩阵已保存")
    
    # 生成校准后的输出
    calibrated_outputs = calibrator.generate_calibrated_outputs()
    print("✓ 校准输出已生成")
    
    # 评估结果
    calibrator.evaluate_results(calibrated_outputs)
    print("✓ 评估报告已生成")
    
    return True

def main():
    """主函数"""
    try:
        # 检查环境
        if not check_environment():
            sys.exit(1)
        
        # 运行训练
        if run_training():
            print("\n=== 训练完成 ===")
            print("生成的文件:")
            print("- calibration.csv (校准矩阵)")
            print("- calibrated_*.csv (9个校准后输出文件)")
            print("- calibration_evaluation.txt (评估报告)")
            print("\n您可以查看这些文件来了解校准效果。")
        
    except ImportError as e:
        print(f"错误: 缺少必要的Python包: {e}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"运行时错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 