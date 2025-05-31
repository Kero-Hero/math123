# 📦 显示器校准可视化GUI系统安装指南

## 🎯 快速安装（推荐）

### 一键安装方式
```bash
# Windows用户 - 双击运行
启动GUI.bat

# 该脚本会自动：
# 1. 检查Python环境
# 2. 安装所需依赖包
# 3. 启动GUI界面
```

## 🛠️ 手动安装

### 步骤1：环境准备
确保您的系统已安装Python 3.7或更高版本：

```bash
# 检查Python版本
python --version

# 如果显示类似 "Python 3.8.x" 则表示安装正确
```

**如果没有Python：**
- 访问 [python.org](https://python.org) 下载安装
- 安装时勾选 "Add Python to PATH"

### 步骤2：安装依赖包

#### 方法A：使用requirements.txt（推荐）
```bash
pip install -r requirements.txt
```

#### 方法B：手动安装各个包
```bash
pip install numpy pandas matplotlib seaborn ttkbootstrap pillow
```

### 步骤3：启动应用
```bash
python calibration_gui.py
```

## 🎮 演示模式

如果您想快速体验GUI功能，可以使用演示模式：

```bash
python demo_gui.py
```

演示脚本提供：
- 自动创建测试数据
- 依赖包检查
- 一键启动GUI

## 📁 文件结构

安装完成后，您的目录应该包含：

```
📦 项目目录/
├── 📄 calibration_visualizer.py    # 原始可视化模块
├── 📄 calibration_gui.py           # GUI主程序
├── 📄 demo_gui.py                  # 演示脚本
├── 📄 requirements.txt             # 依赖列表
├── 📄 启动GUI.bat                  # Windows启动脚本
├── 📄 README_GUI.md                # GUI使用说明
├── 📄 INSTALL_GUIDE.md             # 本安装指南
└── 📁 Calibration/                 # 数据目录（可选）
    ├── calibrated_R_R.csv
    ├── calibrated_R_G.csv
    └── ... (9个CSV文件)
```

## 🔧 故障排除

### 问题1：Python未找到
**错误信息：** `'python' 不是内部或外部命令`

**解决方案：**
1. 重新安装Python，确保勾选"Add to PATH"
2. 或尝试使用 `py` 代替 `python`

### 问题2：pip安装失败
**错误信息：** `pip install 失败`

**解决方案：**
```bash
# 升级pip
python -m pip install --upgrade pip

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 问题3：ttkbootstrap安装失败
**错误信息：** `Failed building wheel for ttkbootstrap`

**解决方案：**
```bash
# 先安装构建工具
pip install wheel setuptools

# 再安装ttkbootstrap
pip install ttkbootstrap
```

### 问题4：中文字体显示异常
**现象：** GUI中文字符显示为方块

**解决方案：**
1. 确保系统安装了中文字体（如微软雅黑）
2. 重启应用程序
3. 如仍有问题，修改 `calibration_gui.py` 中的字体设置

### 问题5：GUI界面无法启动
**错误信息：** 各种启动错误

**排查步骤：**
```bash
# 1. 检查依赖包
python -c "import ttkbootstrap; print('ttkbootstrap OK')"

# 2. 检查matplotlib
python -c "import matplotlib; print('matplotlib OK')"

# 3. 运行演示脚本
python demo_gui.py
```

## 🌟 高级配置

### 自定义主题
如果您想更改界面主题，编辑 `calibration_gui.py` 第36行：

```python
# 可选主题：cosmo, flatly, journal, litera, lumen, minty, 
#          pulse, sandstone, united, yeti, morph, simplex,
#          cerculean, solar, superhero, darkly, cyborg, vapor
themename="superhero"  # 更改为您喜欢的主题
```

### 性能优化
如果系统性能有限，可以调整以下设置：

```python
# 在 calibration_visualizer.py 中
plt.rcParams['figure.dpi'] = 72      # 降低DPI (默认100)
plt.rcParams['savefig.dpi'] = 150    # 降低保存DPI (默认300)
```

## 📋 系统要求

### 最低要求
- **操作系统：** Windows 7+ / macOS 10.12+ / Ubuntu 16.04+
- **Python：** 3.7或更高版本  
- **内存：** 2GB RAM
- **磁盘空间：** 500MB可用空间

### 推荐配置
- **操作系统：** Windows 10/11
- **Python：** 3.8或更高版本
- **内存：** 4GB+ RAM
- **磁盘空间：** 1GB+ 可用空间
- **显示器：** 1920x1080或更高分辨率

## 🎯 验证安装

安装完成后，运行以下命令验证：

```bash
# 1. 检查Python版本
python --version

# 2. 检查关键依赖
python -c "import numpy, pandas, matplotlib, ttkbootstrap; print('All dependencies OK!')"

# 3. 启动演示
python demo_gui.py
```

如果所有命令都成功执行，说明安装完成！

## 📞 获取帮助

如果安装过程中遇到问题：

1. **查看错误日志：** 记录完整的错误信息
2. **检查环境：** 确认Python版本和系统环境
3. **重新安装：** 尝试重新安装相关组件
4. **查阅文档：** 参考 `README_GUI.md` 获取更多信息

---

**祝您使用愉快！** 🎉 