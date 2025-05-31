#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示器校准数据可视化GUI界面
现代化的交互式界面，使用ttkbootstrap组件库
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from pathlib import Path

try:
    import ttkbootstrap as ttk_boot
    from ttkbootstrap.constants import *
    from ttkbootstrap.tooltip import ToolTip
    from ttkbootstrap.dialogs import Messagebox
except ImportError:
    print("请安装ttkbootstrap: pip install ttkbootstrap")
    sys.exit(1)

# 导入原有的可视化类
from calibration_visualizer import CalibrationVisualizer, check_data_files

class CalibrationGUI:
    """显示器校准可视化GUI主界面"""
    
    def __init__(self):
        """初始化GUI界面"""
        # 创建主窗口
        self.root = ttk_boot.Window(
            title="显示器色彩校准数据可视化分析系统",
            themename="superhero",  # 使用现代暗色主题
            size=(1200, 800),
            position=(100, 50),
            minsize=(1000, 600),
            maxsize=(1600, 1000)
        )
        
        # 设置窗口图标
        self.root.iconbitmap(default=None)
        
        # 初始化变量
        self.visualizer = None
        self.data_dir = tk.StringVar(value="Calibration")
        self.target_brightness = tk.DoubleVar(value=220.0)
        self.output_dir = tk.StringVar(value="visualization_output")
        self.progress_var = tk.DoubleVar()
        self.status_text = tk.StringVar(value="就绪")
        
        # 创建界面
        self.create_widgets()
        self.setup_layout()
        self.bind_events()
        
        # 检查数据文件
        self.check_initial_data()
        
    def create_widgets(self):
        """创建所有界面组件"""
        # 创建主容器
        self.main_container = ttk_boot.Frame(self.root, padding=10)
        
        # 1. 标题区域
        self.create_header()
        
        # 2. 设置面板
        self.create_settings_panel()
        
        # 3. 功能按钮区域
        self.create_function_buttons()
        
        # 4. 进度和状态区域
        self.create_progress_status()
        
        # 5. 日志显示区域
        self.create_log_area()
        
    def create_header(self):
        """创建标题区域"""
        self.header_frame = ttk_boot.Frame(self.main_container)
        
        # 主标题
        title_label = ttk_boot.Label(
            self.header_frame,
            text="显示器色彩校准数据可视化分析系统",
            font=("Microsoft YaHei", 20, "bold"),
            bootstyle="primary"
        )
        title_label.pack(pady=(0, 5))
        
        # 副标题
        subtitle_label = ttk_boot.Label(
            self.header_frame,
            text="Monitor Color Calibration Visualization System",
            font=("Arial", 12),
            bootstyle="secondary"
        )
        subtitle_label.pack(pady=(0, 10))
        
        # 分隔线
        separator = ttk_boot.Separator(self.header_frame, orient='horizontal')
        separator.pack(fill='x', pady=10)
        
    def create_settings_panel(self):
        """创建设置面板"""
        settings_frame = ttk_boot.LabelFrame(
            self.main_container,
            text="📁 配置设置",
            padding=15,
            bootstyle="info"
        )
        
        # 数据目录设置
        data_dir_frame = ttk_boot.Frame(settings_frame)
        ttk_boot.Label(
            data_dir_frame,
            text="数据目录:",
            font=("Microsoft YaHei", 10, "bold")
        ).pack(side='left', padx=(0, 10))
        
        data_dir_entry = ttk_boot.Entry(
            data_dir_frame,
            textvariable=self.data_dir,
            width=40,
            bootstyle="info"
        )
        data_dir_entry.pack(side='left', padx=(0, 10), fill='x', expand=True)
        
        browse_btn = ttk_boot.Button(
            data_dir_frame,
            text="📂 浏览",
            command=self.browse_data_dir,
            bootstyle="outline-info"
        )
        browse_btn.pack(side='right')
        ToolTip(browse_btn, text="选择包含校准数据CSV文件的目录")
        
        data_dir_frame.pack(fill='x', pady=(0, 10))
        
        # 设置参数行
        params_frame = ttk_boot.Frame(settings_frame)
        
        # 目标亮度设置
        brightness_frame = ttk_boot.Frame(params_frame)
        ttk_boot.Label(
            brightness_frame,
            text="目标亮度:",
            font=("Microsoft YaHei", 10, "bold")
        ).pack(side='left')
        
        brightness_spin = ttk_boot.Spinbox(
            brightness_frame,
            from_=100,
            to=400,
            increment=10,
            textvariable=self.target_brightness,
            width=10,
            bootstyle="info"
        )
        brightness_spin.pack(side='left', padx=(10, 5))
        ttk_boot.Label(brightness_frame, text="cd/m²").pack(side='left')
        
        brightness_frame.pack(side='left', padx=(0, 30))
        
        # 输出目录设置
        output_frame = ttk_boot.Frame(params_frame)
        ttk_boot.Label(
            output_frame,
            text="输出目录:",
            font=("Microsoft YaHei", 10, "bold")
        ).pack(side='left')
        
        output_entry = ttk_boot.Entry(
            output_frame,
            textvariable=self.output_dir,
            width=25,
            bootstyle="info"
        )
        output_entry.pack(side='left', padx=(10, 10))
        
        output_browse_btn = ttk_boot.Button(
            output_frame,
            text="📁",
            command=self.browse_output_dir,
            bootstyle="outline-info",
            width=3
        )
        output_browse_btn.pack(side='left')
        ToolTip(output_browse_btn, text="选择输出目录")
        
        output_frame.pack(side='right')
        
        params_frame.pack(fill='x', pady=(0, 10))
        
        # 初始化按钮
        init_btn = ttk_boot.Button(
            settings_frame,
            text="🔄 重新加载数据",
            command=self.initialize_visualizer,
            bootstyle="success-outline",
            width=20
        )
        init_btn.pack(pady=(10, 0))
        ToolTip(init_btn, text="根据当前设置重新加载校准数据")
        
        self.settings_frame = settings_frame
        
    def create_function_buttons(self):
        """创建功能按钮区域"""
        func_frame = ttk_boot.LabelFrame(
            self.main_container,
            text="🎨 可视化功能",
            padding=15,
            bootstyle="primary"
        )
        
        # 创建按钮网格
        buttons_container = ttk_boot.Frame(func_frame)
        
        # 定义按钮配置
        button_configs = [
            {
                "text": "🌈 RGB颜色合成",
                "desc": "显示RGB三通道的颜色合成效果对比",
                "command": lambda: self.run_visualization("color_composition"),
                "style": "primary"
            },
            {
                "text": "🔥 亮度热力图",
                "desc": "显示亮度均匀性的热力图分析",
                "command": lambda: self.run_visualization("uniformity_heatmap"),
                "style": "info"
            },
            {
                "text": "📊 统计对比",
                "desc": "校准前后的统计数据对比分析",
                "command": lambda: self.run_visualization("statistical_comparison"),
                "style": "success"
            },
            {
                "text": "🎭 色彩串扰",
                "desc": "分析和显示色彩通道间的串扰情况",
                "command": lambda: self.run_visualization("crosstalk_analysis"),
                "style": "warning"
            },
            {
                "text": "🏔️ 3D表面图",
                "desc": "三维立体显示亮度分布情况",
                "command": lambda: self.run_visualization("3d_surface"),
                "style": "danger"
            },
            {
                "text": "📋 完整报告",
                "desc": "生成包含所有图表的完整分析报告",
                "command": lambda: self.run_visualization("full_report"),
                "style": "dark"
            }
        ]
        
        # 创建按钮 (2x3 布局)
        for i, config in enumerate(button_configs):
            row = i // 3
            col = i % 3
            
            btn_frame = ttk_boot.Frame(buttons_container)
            btn_frame.grid(row=row, column=col, padx=10, pady=10, sticky='ew')
            
            btn = ttk_boot.Button(
                btn_frame,
                text=config["text"],
                command=config["command"],
                bootstyle=config["style"],
                width=20
            )
            btn.pack(pady=(0, 5))
            
            desc_label = ttk_boot.Label(
                btn_frame,
                text=config["desc"],
                font=("Microsoft YaHei", 8),
                bootstyle="secondary",
                wraplength=150
            )
            desc_label.pack()
            
            ToolTip(btn, text=config["desc"])
        
        # 配置列权重
        for i in range(3):
            buttons_container.columnconfigure(i, weight=1)
        
        buttons_container.pack(fill='both', expand=True)
        
        self.func_frame = func_frame
        
    def create_progress_status(self):
        """创建进度和状态区域"""
        progress_frame = ttk_boot.LabelFrame(
            self.main_container,
            text="📈 进度状态",
            padding=10,
            bootstyle="secondary"
        )
        
        # 状态文本
        status_label = ttk_boot.Label(
            progress_frame,
            textvariable=self.status_text,
            font=("Microsoft YaHei", 10),
            bootstyle="info"
        )
        status_label.pack(anchor='w', pady=(0, 5))
        
        # 进度条
        self.progress_bar = ttk_boot.Progressbar(
            progress_frame,
            variable=self.progress_var,
            mode='determinate',
            bootstyle="success-striped",
            length=400
        )
        self.progress_bar.pack(fill='x', pady=(0, 5))
        
        # 操作按钮
        action_frame = ttk_boot.Frame(progress_frame)
        
        self.cancel_btn = ttk_boot.Button(
            action_frame,
            text="⏹️ 取消",
            command=self.cancel_operation,
            bootstyle="danger-outline",
            state='disabled'
        )
        self.cancel_btn.pack(side='left', padx=(0, 10))
        
        open_output_btn = ttk_boot.Button(
            action_frame,
            text="📁 打开输出目录",
            command=self.open_output_directory,
            bootstyle="info-outline"
        )
        open_output_btn.pack(side='left')
        
        action_frame.pack(anchor='w', pady=(5, 0))
        
        self.progress_frame = progress_frame
        
    def create_log_area(self):
        """创建日志显示区域"""
        log_frame = ttk_boot.LabelFrame(
            self.main_container,
            text="📝 操作日志",
            padding=10,
            bootstyle="dark"
        )
        
        # 创建文本框和滚动条
        text_frame = ttk_boot.Frame(log_frame)
        
        self.log_text = tk.Text(
            text_frame,
            height=8,
            wrap='word',
            font=("Consolas", 9),
            bg="#2b3e50",
            fg="#ecf0f1",
            insertbackground="#ecf0f1",
            selectbackground="#3498db"
        )
        
        scrollbar = ttk_boot.Scrollbar(
            text_frame,
            orient='vertical',
            command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        text_frame.pack(fill='both', expand=True)
        
        # 日志控制按钮
        log_control_frame = ttk_boot.Frame(log_frame)
        
        clear_log_btn = ttk_boot.Button(
            log_control_frame,
            text="🗑️ 清空日志",
            command=self.clear_log,
            bootstyle="outline-secondary",
            width=12
        )
        clear_log_btn.pack(side='right')
        
        log_control_frame.pack(fill='x', pady=(10, 0))
        
        self.log_frame = log_frame
        
        # 添加欢迎信息
        self.log_message("🎉 欢迎使用显示器色彩校准数据可视化分析系统！", "INFO")
        self.log_message("💡 请先设置数据目录，然后选择需要的可视化功能。", "INFO")
        
    def setup_layout(self):
        """设置布局"""
        self.main_container.pack(fill='both', expand=True)
        
        self.header_frame.pack(fill='x', pady=(0, 10))
        self.settings_frame.pack(fill='x', pady=(0, 10))
        self.func_frame.pack(fill='x', pady=(0, 10))
        self.progress_frame.pack(fill='x', pady=(0, 10))
        self.log_frame.pack(fill='both', expand=True)
        
    def bind_events(self):
        """绑定事件"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 绑定数据目录变化事件
        self.data_dir.trace('w', self.on_data_dir_changed)
        
    def log_message(self, message: str, level: str = "INFO"):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        level_colors = {
            "INFO": "#3498db",
            "SUCCESS": "#2ecc71", 
            "WARNING": "#f39c12",
            "ERROR": "#e74c3c",
            "DEBUG": "#9b59b6"
        }
        
        self.log_text.configure(state='normal')
        
        # 插入时间戳
        self.log_text.insert('end', f"[{timestamp}] ", 'timestamp')
        
        # 插入级别标签
        self.log_text.insert('end', f"[{level}] ", level.lower())
        
        # 插入消息
        self.log_text.insert('end', f"{message}\n")
        
        # 配置标签颜色
        self.log_text.tag_configure('timestamp', foreground="#95a5a6")
        self.log_text.tag_configure(level.lower(), foreground=level_colors.get(level, "#ecf0f1"))
        
        self.log_text.configure(state='disabled')
        self.log_text.see('end')
        
    def clear_log(self):
        """清空日志"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, 'end')
        self.log_text.configure(state='disabled')
        self.log_message("📝 日志已清空", "INFO")
        
    def browse_data_dir(self):
        """浏览数据目录"""
        directory = filedialog.askdirectory(
            title="选择包含校准数据的目录",
            initialdir=self.data_dir.get()
        )
        if directory:
            self.data_dir.set(directory)
            self.log_message(f"📁 数据目录已设置为: {directory}", "INFO")
            
    def browse_output_dir(self):
        """浏览输出目录"""
        directory = filedialog.askdirectory(
            title="选择输出目录",
            initialdir=self.output_dir.get()
        )
        if directory:
            self.output_dir.set(directory)
            self.log_message(f"📁 输出目录已设置为: {directory}", "INFO")
            
    def on_data_dir_changed(self, *args):
        """数据目录变化时的回调"""
        if os.path.exists(self.data_dir.get()):
            self.check_data_files_in_dir(self.data_dir.get())
            
    def check_initial_data(self):
        """检查初始数据"""
        self.check_data_files_in_dir(self.data_dir.get())
        
    def check_data_files_in_dir(self, directory: str):
        """检查指定目录中的数据文件"""
        if not os.path.exists(directory):
            self.log_message(f"❌ 目录不存在: {directory}", "ERROR")
            return False
            
        required_files = [f"calibrated_{c1}_{c2}.csv" for c1 in ['R','G','B'] for c2 in ['R','G','B']]
        existing_files = [f for f in required_files if os.path.exists(os.path.join(directory, f))]
        missing_files = [f for f in required_files if f not in existing_files]
        
        if missing_files:
            self.log_message(f"⚠️ 发现 {len(missing_files)} 个缺失的数据文件", "WARNING")
            self.log_message("💡 系统将使用模拟数据进行演示", "INFO")
        else:
            self.log_message("✅ 所有必需的数据文件都已存在", "SUCCESS")
            
        return len(missing_files) == 0
        
    def initialize_visualizer(self):
        """初始化可视化器"""
        def init_task():
            try:
                self.root.after(0, lambda: self.update_status("正在初始化可视化器..."))
                self.root.after(0, lambda: self.progress_var.set(30))
                
                self.visualizer = CalibrationVisualizer(
                    data_dir=self.data_dir.get(),
                    target_brightness=self.target_brightness.get()
                )
                
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.update_status("可视化器初始化完成"))
                self.root.after(0, lambda: self.log_message("✅ 可视化器初始化成功", "SUCCESS"))
                
            except Exception as e:
                # 在主线程中显示错误
                error_msg = f"❌ 初始化失败: {str(e)}"
                self.root.after(0, lambda: self.log_message(error_msg, "ERROR"))
                
                def show_error():
                    Messagebox.show_error("初始化失败", f"无法初始化可视化器:\n{str(e)}")
                
                self.root.after(0, show_error)
            finally:
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.cancel_btn.configure(state='disabled'))
                
        self.cancel_btn.configure(state='normal')
        thread = threading.Thread(target=init_task, daemon=True)
        thread.start()
        
    def run_visualization(self, viz_type: str):
        """运行可视化任务"""
        if not self.visualizer:
            if Messagebox.show_question("需要初始化", "请先初始化可视化器，是否现在初始化？") == "Yes":
                self.initialize_visualizer()
                return
            else:
                return
                
        def viz_task():
            try:
                output_dir = self.output_dir.get()
                os.makedirs(output_dir, exist_ok=True)
                
                # 在主线程中更新UI
                self.root.after(0, lambda: self.cancel_btn.configure(state='normal'))
                
                if viz_type == "color_composition":
                    self.root.after(0, lambda: self.update_status("正在生成RGB颜色合成图..."))
                    self.root.after(0, lambda: self.progress_var.set(20))
                    file_path = os.path.join(output_dir, "color_composition.png")
                    self.visualizer.create_color_composition_display(file_path)
                    
                elif viz_type == "uniformity_heatmap":
                    self.root.after(0, lambda: self.update_status("正在生成亮度均匀性热力图..."))
                    self.root.after(0, lambda: self.progress_var.set(20))
                    file_path = os.path.join(output_dir, "uniformity_heatmap.png")
                    self.visualizer.create_uniformity_heatmap(file_path)
                    
                elif viz_type == "statistical_comparison":
                    self.root.after(0, lambda: self.update_status("正在生成统计对比图..."))
                    self.root.after(0, lambda: self.progress_var.set(20))
                    file_path = os.path.join(output_dir, "statistical_comparison.png")
                    self.visualizer.create_statistical_comparison(file_path)
                    
                elif viz_type == "crosstalk_analysis":
                    self.root.after(0, lambda: self.update_status("正在生成色彩串扰分析图..."))
                    self.root.after(0, lambda: self.progress_var.set(20))
                    file_path = os.path.join(output_dir, "crosstalk_analysis.png")
                    self.visualizer.create_crosstalk_analysis(file_path)
                    
                elif viz_type == "3d_surface":
                    self.root.after(0, lambda: self.update_status("正在生成3D表面图..."))
                    self.root.after(0, lambda: self.progress_var.set(20))
                    file_path = os.path.join(output_dir, "3d_surface.png")
                    self.visualizer.create_3d_surface(file_path)
                    
                elif viz_type == "full_report":
                    self.root.after(0, lambda: self.update_status("正在生成完整报告..."))
                    self.root.after(0, lambda: self.progress_var.set(10))
                    self.visualizer.generate_all_visualizations(output_dir)
                    
                # 在主线程中更新完成状态
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.update_status(f"可视化完成: {viz_type}"))
                self.root.after(0, lambda: self.log_message(f"✅ {viz_type} 生成完成", "SUCCESS"))
                
                # 询问是否打开输出目录 - 在主线程中执行
                def ask_open_dir():
                    if Messagebox.show_question("生成完成", "可视化图表已生成完成！\n是否打开输出目录？") == "Yes":
                        self.open_output_directory()
                
                self.root.after(0, ask_open_dir)
                    
            except Exception as e:
                # 在主线程中显示错误
                error_msg = f"❌ 生成失败: {str(e)}"
                self.root.after(0, lambda: self.log_message(error_msg, "ERROR"))
                
                def show_error():
                    Messagebox.show_error("生成失败", f"可视化生成失败:\n{str(e)}")
                
                self.root.after(0, show_error)
            finally:
                # 在主线程中重置状态
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.cancel_btn.configure(state='disabled'))
                
        thread = threading.Thread(target=viz_task, daemon=True)
        thread.start()
        
    def update_status(self, status: str):
        """更新状态文本"""
        self.status_text.set(status)
        self.root.update_idletasks()
        
    def cancel_operation(self):
        """取消当前操作"""
        self.log_message("⏹️ 用户取消了当前操作", "WARNING")
        self.update_status("操作已取消")
        self.progress_var.set(0)
        self.cancel_btn.configure(state='disabled')
        
    def open_output_directory(self):
        """打开输出目录"""
        output_dir = self.output_dir.get()
        if os.path.exists(output_dir):
            import subprocess
            import platform
            
            try:
                if platform.system() == "Windows":
                    os.startfile(output_dir)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", output_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", output_dir])
                    
                self.log_message(f"📁 已打开输出目录: {output_dir}", "INFO")
            except Exception as e:
                self.log_message(f"❌ 无法打开目录: {str(e)}", "ERROR")
        else:
            self.log_message("❌ 输出目录不存在", "ERROR")
            
    def on_closing(self):
        """关闭窗口时的处理"""
        if Messagebox.show_question("确认退出", "确定要退出应用程序吗？") == "Yes":
            self.root.quit()
            
    def run(self):
        """运行GUI应用"""
        self.root.mainloop()

def main():
    """主函数"""
    try:
        app = CalibrationGUI()
        app.run()
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        input("按回车键退出...")

if __name__ == "__main__":
    main() 