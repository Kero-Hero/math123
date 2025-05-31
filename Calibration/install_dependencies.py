#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖包安装脚本
自动安装可视化工具所需的Python包
"""

import subprocess
import sys
import importlib

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """安装Python包"""
    try:
        print(f"正在安装 {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package_name} 安装失败")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("    显示器校准可视化工具 - 依赖包安装")
    print("=" * 50)
    
    # 定义所需的包
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("opencv-python", "cv2"),
        ("scipy", "scipy"),
    ]
    
    # 检查已安装的包
    print("检查已安装的包...")
    installed = []
    to_install = []
    
    for package_name, import_name in required_packages:
        if check_package(package_name, import_name):
            print(f"✅ {package_name} 已安装")
            installed.append(package_name)
        else:
            print(f"❌ {package_name} 未安装")
            to_install.append(package_name)
    
    print(f"\n已安装: {len(installed)} 个包")
    print(f"需要安装: {len(to_install)} 个包")
    
    if not to_install:
        print("\n🎉 所有依赖包都已安装！可以直接运行可视化工具。")
        return
    
    # 询问是否安装
    print(f"\n需要安装以下包: {', '.join(to_install)}")
    response = input("是否现在安装这些包？(y/n): ").strip().lower()
    
    if response not in ['y', 'yes', '是']:
        print("跳过安装。您可以手动安装这些包：")
        for package in to_install:
            print(f"  pip install {package}")
        return
    
    # 安装包
    print("\n开始安装...")
    failed = []
    
    for package in to_install:
        if not install_package(package):
            failed.append(package)
    
    # 安装结果
    print("\n" + "=" * 30)
    print("安装结果:")
    
    if not failed:
        print("🎉 所有包都安装成功！")
        print("现在可以运行可视化工具了。")
    else:
        print(f"❌ 以下包安装失败: {', '.join(failed)}")
        print("请手动安装这些包：")
        for package in failed:
            print(f"  pip install {package}")
    
    print("=" * 30)

if __name__ == "__main__":
    main() 