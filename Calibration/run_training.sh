#!/bin/bash

echo "=== 显示器校准训练器 ==="
echo "正在检查环境..."

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 检查数据集目录
if [ ! -d "数据集" ]; then
    echo "错误: 未找到数据集文件夹"
    exit 1
fi

echo "环境检查完成，开始训练..."

# 运行训练
python3 train_calibration.py \
    --data_dir 数据集 \
    --epochs 500 \
    --learning_rate 0.01 \
    --target_brightness 220.0

echo "训练完成！"
echo "生成的文件："
echo "- calibration.csv (校准矩阵)"
echo "- calibrated_*.csv (校准后的输出)"
echo "- calibration_evaluation.txt (评估报告)" 