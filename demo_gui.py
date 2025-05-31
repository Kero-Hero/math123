#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI演示脚本
用于快速测试和展示GUI界面功能
"""

import os
import sys

def create_demo_data():
    """创建演示用的数据文件"""
    import numpy as np
    
    # 创建Calibration目录
    os.makedirs("Calibration", exist_ok=True)
    
    print("🎨 正在创建演示数据...")
    
    # 生成9个CSV文件
    colors = ['R', 'G', 'B']
    for output_color in colors:
        for channel in colors:
            filename = f"calibrated_{output_color}_{channel}.csv"
            filepath = os.path.join("Calibration", filename)
            
            # 生成64x64的随机数据
            if output_color == channel:
                # 主通道：较高的值
                data = np.random.normal(220, 15, (64, 64))
            else:
                # 串扰通道：较低的值
                data = np.random.normal(30, 8, (64, 64))
            
            # 确保数据在合理范围内
            data = np.clip(data, 0, 255)
            
            # 保存为CSV
            np.savetxt(filepath, data, delimiter=',', fmt='%.1f')
    
    print("✅ 演示数据创建完成！")
    print(f"📁 数据保存在: {os.path.abspath('Calibration')}")

def check_dependencies():
    """检查依赖包"""
    print("🔍 正在检查依赖包...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'ttkbootstrap'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 缺失")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有依赖包都已安装")
        return True

def main():
    """主演示函数"""
    print("=" * 60)
    print("      显示器校准可视化GUI演示")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        input("\n按回车键退出...")
        return
    
    print("\n" + "=" * 40)
    print("选择演示模式:")
    print("1. 创建演示数据并启动GUI")
    print("2. 直接启动GUI (使用现有数据)")
    print("3. 仅创建演示数据")
    print("4. 退出")
    print("=" * 40)
    
    while True:
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == '1':
            create_demo_data()
            print("\n🚀 正在启动GUI...")
            launch_gui()
            break
        elif choice == '2':
            print("\n🚀 正在启动GUI...")
            launch_gui()
            break
        elif choice == '3':
            create_demo_data()
            print("\n✅ 演示数据创建完成")
            print("您现在可以运行 'python calibration_gui.py' 启动GUI")
            break
        elif choice == '4':
            print("👋 再见！")
            break
        else:
            print("❌ 无效选择，请输入 1-4")

def launch_gui():
    """启动GUI界面"""
    try:
        from calibration_gui import CalibrationGUI
        app = CalibrationGUI()
        app.run()
    except ImportError as e:
        print(f"❌ 导入GUI模块失败: {e}")
        print("请确保 calibration_gui.py 文件存在")
    except Exception as e:
        print(f"❌ GUI启动失败: {e}")

if __name__ == "__main__":
    main() 