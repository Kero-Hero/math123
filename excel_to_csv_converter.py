#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel到CSV转换工具
用于将RGB数值Excel文件转换为训练脚本所需的CSV格式
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExcelToCSVConverter:
    def __init__(self, excel_file: str = "副本B题附件：RGB数值1.xlsx", output_dir: str = "数据集"):
        """
        初始化转换器
        
        Args:
            excel_file: Excel文件路径
            output_dir: 输出CSV文件的目录
        """
        self.excel_file = excel_file
        self.output_dir = output_dir
        
        # 创建输出目录
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # 定义需要读取的工作表名称
        self.sheet_names = [
            'R_R', 'R_G', 'R_B',
            'G_R', 'G_G', 'G_B', 
            'B_R', 'B_G', 'B_B'
        ]
        
    def read_excel_sheets(self):
        """
        读取Excel文件中的所有相关工作表
        
        Returns:
            dict: 包含所有工作表数据的字典
        """
        logger.info(f"开始读取Excel文件: {self.excel_file}")
        
        try:
            # 读取Excel文件的所有工作表
            excel_data = pd.read_excel(self.excel_file, sheet_name=None)
            logger.info(f"Excel文件包含的工作表: {list(excel_data.keys())}")
            
            # 筛选出我们需要的工作表
            filtered_data = {}
            for sheet_name in self.sheet_names:
                if sheet_name in excel_data:
                    filtered_data[sheet_name] = excel_data[sheet_name]
                    logger.info(f"找到工作表: {sheet_name}, 尺寸: {excel_data[sheet_name].shape}")
                else:
                    logger.warning(f"未找到工作表: {sheet_name}")
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"读取Excel文件时出错: {e}")
            return {}
    
    def process_sheet_data(self, data: pd.DataFrame, sheet_name: str):
        """
        处理单个工作表的数据
        
        Args:
            data: 工作表数据
            sheet_name: 工作表名称
            
        Returns:
            np.ndarray: 处理后的数据
        """
        logger.info(f"处理工作表: {sheet_name}")
        
        # 移除可能的标题行和列
        # 假设数据从第二行第二列开始（跳过标题）
        if data.shape[0] > 1 and data.shape[1] > 1:
            # 尝试找到数值数据的起始位置
            numeric_data = data.iloc[1:, 1:].copy()
            
            # 转换为数值类型，非数值设为NaN
            numeric_data = pd.to_numeric(numeric_data.stack(), errors='coerce').unstack()
            
            # 填充NaN值为0
            numeric_data = numeric_data.fillna(0)
            
            logger.info(f"处理后的数据尺寸: {numeric_data.shape}")
            return numeric_data.values
        else:
            logger.warning(f"工作表 {sheet_name} 数据不足")
            return np.zeros((64, 64))  # 返回默认尺寸的零矩阵
    
    def save_as_csv(self, data: np.ndarray, filename: str):
        """
        将数据保存为CSV文件
        
        Args:
            data: 要保存的数据
            filename: 文件名
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # 确保数据是64x64的尺寸
        if data.shape[0] >= 64 and data.shape[1] >= 64:
            # 取前64x64的数据
            data_64x64 = data[:64, :64]
        else:
            # 如果数据不足64x64，用零填充
            data_64x64 = np.zeros((64, 64))
            min_rows = min(data.shape[0], 64)
            min_cols = min(data.shape[1], 64)
            data_64x64[:min_rows, :min_cols] = data[:min_rows, :min_cols]
        
        # 保存为CSV（不包含行列索引）
        pd.DataFrame(data_64x64).to_csv(output_path, header=False, index=False)
        logger.info(f"保存CSV文件: {output_path}, 尺寸: {data_64x64.shape}")
    
    def convert_all(self):
        """
        转换所有工作表为CSV文件
        """
        logger.info("开始转换过程...")
        
        # 读取Excel数据
        excel_data = self.read_excel_sheets()
        
        if not excel_data:
            logger.error("没有读取到任何数据，转换失败")
            return False
        
        # 处理每个工作表
        for sheet_name, sheet_data in excel_data.items():
            try:
                # 处理数据
                processed_data = self.process_sheet_data(sheet_data, sheet_name)
                
                # 生成CSV文件名
                csv_filename = f"{sheet_name}.csv"
                
                # 保存为CSV
                self.save_as_csv(processed_data, csv_filename)
                
            except Exception as e:
                logger.error(f"处理工作表 {sheet_name} 时出错: {e}")
                continue
        
        logger.info("转换完成！")
        return True
    
    def verify_output(self):
        """
        验证输出文件
        """
        logger.info("验证输出文件...")
        
        expected_files = [f"{sheet}.csv" for sheet in self.sheet_names]
        
        for filename in expected_files:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                # 读取并检查文件
                try:
                    data = pd.read_csv(filepath, header=None)
                    logger.info(f"✓ {filename}: 尺寸 {data.shape}")
                except Exception as e:
                    logger.error(f"✗ {filename}: 读取失败 - {e}")
            else:
                logger.warning(f"✗ {filename}: 文件不存在")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("Excel到CSV转换工具")
    print("=" * 60)
    
    # 创建转换器
    converter = ExcelToCSVConverter()
    
    # 执行转换
    success = converter.convert_all()
    
    if success:
        # 验证输出
        converter.verify_output()
        print("\n转换完成！CSV文件已保存到 '数据集' 目录中。")
    else:
        print("\n转换失败，请检查Excel文件是否存在且格式正确。")

if __name__ == "__main__":
    main() 