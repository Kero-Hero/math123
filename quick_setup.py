#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本
一键完成Excel转换、数据验证和训练准备
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_dependencies():
    """检查依赖包"""
    print_header("检查依赖包")
    
    required_packages = ['pandas', 'openpyxl', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n正在安装缺失的包: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✓ 依赖包安装完成")
        except subprocess.CalledProcessError:
            print("✗ 依赖包安装失败，请手动安装")
            return False
    
    return True

def check_excel_file():
    """检查Excel文件是否存在"""
    print_header("检查Excel文件")
    
    excel_file = "副本B题附件：RGB数值1.xlsx"
    if os.path.exists(excel_file):
        print(f"✓ 找到Excel文件: {excel_file}")
        return True
    else:
        print(f"✗ 未找到Excel文件: {excel_file}")
        print("请确保Excel文件在当前目录中")
        return False

def run_conversion():
    """运行Excel到CSV转换"""
    print_header("运行Excel到CSV转换")
    
    try:
        from excel_to_csv_converter import ExcelToCSVConverter
        
        converter = ExcelToCSVConverter()
        success = converter.convert_all()
        
        if success:
            print("✓ Excel转换完成")
            converter.verify_output()
            return True
        else:
            print("✗ Excel转换失败")
            return False
            
    except Exception as e:
        print(f"✗ 转换过程出错: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print_header("测试数据加载")
    
    try:
        sys.path.append('Calibration')
        from train_calibration import DisplayCalibrator
        
        calibrator = DisplayCalibrator(data_dir="数据集")
        data = calibrator.load_data()
        
        print(f"✓ 数据加载成功")
        print(f"✓ 数据形状: {data.shape}")
        print(f"✓ 数据范围: {data.min():.2f} - {data.max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        return False

def show_next_steps():
    """显示后续步骤"""
    print_header("后续步骤")
    
    print("数据准备完成！您现在可以：")
    print()
    print("1. 开始训练:")
    print("   cd Calibration")
    print("   python train_calibration.py")
    print()
    print("2. 查看生成的CSV文件:")
    print("   ls 数据集/")
    print()
    print("3. 查看详细使用说明:")
    print("   cat README_数据转换工具.md")
    print()

def main():
    """主函数"""
    print_header("Excel到CSV转换工具 - 快速启动")
    
    # 检查依赖
    if not check_dependencies():
        return False
    
    # 检查Excel文件
    if not check_excel_file():
        return False
    
    # 运行转换
    if not run_conversion():
        return False
    
    # 测试数据加载
    if not test_data_loading():
        return False
    
    # 显示后续步骤
    show_next_steps()
    
    print_header("设置完成")
    print("✓ 所有步骤完成！数据已准备就绪。")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n设置过程中遇到错误，请检查上述信息。")
        sys.exit(1) 