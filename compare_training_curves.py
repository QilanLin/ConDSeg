import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from matplotlib.ticker import MultipleLocator
import argparse


def extract_model_name(folder_path):
    """从文件夹路径中提取模型名称"""
    folder_name = os.path.basename(folder_path)
    if folder_name.startswith('UNet_'):
        return 'U-Net'
    elif folder_name.startswith('UNETPP_'):
        return 'U-Net++'
    elif folder_name.startswith('TGANET_'):
        return 'TGANet'
    elif folder_name.startswith('XBOUNDFORMER_'):
        return 'XBoundFormer'
    elif folder_name.startswith('stage1_'):
        return 'Ours(One-Stage)'
    elif folder_name.startswith('Kvasir-SEG_'):
        return 'Ours(Two-Stage)'
    else:
        return folder_name


def load_training_logs(base_dir='run_files/Kvasir-SEG'):
    """加载所有模型的训练日志"""
    model_data = {}
    folders = glob.glob(os.path.join(base_dir, '*/'))
    
    for folder in folders:
        model_name = extract_model_name(folder)
        log_path = os.path.join(folder, 'train_log.csv')
        
        if not os.path.exists(log_path):
            continue
        
        try:
            df = pd.read_csv(log_path)
            # 确保数据从epoch 0开始
            if df['epoch'].min() > 0:
                # 创建epoch 0的初始值（通常接近0）
                zero_epoch = pd.DataFrame({
                    'epoch': [0],
                    'train_mIoU': [0.01],
                    'train_f1': [0.02], 
                    'valid_mIoU': [0.01],
                    'valid_f1': [0.02]
                })
                df = pd.concat([zero_epoch, df]).reset_index(drop=True)
            
            # 如果日志超过100个epoch，只保留前100个
            if df['epoch'].max() > 100:
                df = df[df['epoch'] <= 100]
                
            model_data[model_name] = df
        except Exception as e:
            print(f"无法加载{log_path}：{e}")
    
    return model_data


def plot_convergence_curves(model_data, save_path='figures/comparison_curves.jpg'):
    """绘制所有模型的收敛曲线对比图"""
    if not model_data:
        print("没有找到任何训练数据")
        return
    
    # 设置颜色映射
    colors = {
        'U-Net': 'magenta',
        'U-Net++': 'green',
        'TGANet': 'yellow',
        'XBoundFormer': 'orange',
        'Ours(One-Stage)': 'blue',
        'Ours(Two-Stage)': 'red'
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制mIoU曲线
    ax1 = axes[0]
    ax1.set_title('mIoU vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mIoU')
    ax1.grid(True)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 0.9)
    
    # 绘制mDSC (F1)曲线
    ax2 = axes[1]
    ax2.set_title('mDSC vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mDSC')
    ax2.grid(True)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 0.9)
    
    # 为每个模型绘制曲线
    for model_name, df in model_data.items():
        color = colors.get(model_name, 'gray')  # 如果没有预定义颜色则使用灰色
        
        # 平滑处理（可选）
        window_size = 3
        if len(df) > window_size:
            valid_miou = df['valid_mIoU'].rolling(window=window_size, center=True).mean()
            valid_f1 = df['valid_f1'].rolling(window=window_size, center=True).mean()
            # 填充NaN值
            valid_miou.iloc[0:window_size//2] = df['valid_mIoU'].iloc[0:window_size//2]
            valid_miou.iloc[-window_size//2:] = df['valid_mIoU'].iloc[-window_size//2:]
            valid_f1.iloc[0:window_size//2] = df['valid_f1'].iloc[0:window_size//2]
            valid_f1.iloc[-window_size//2:] = df['valid_f1'].iloc[-window_size//2:]
        else:
            valid_miou = df['valid_mIoU']
            valid_f1 = df['valid_f1']
        
        # 绘制mIoU曲线
        ax1.plot(df['epoch'], valid_miou, label=model_name, color=color)
        
        # 绘制mDSC曲线
        ax2.plot(df['epoch'], valid_f1, label=model_name, color=color)
    
    # 添加图例
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    
    # 设置网格线
    ax1.xaxis.set_major_locator(MultipleLocator(20))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2.xaxis.set_major_locator(MultipleLocator(20))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 添加总标题
    fig.text(0.5, 0.01, 'Figure: Comparison of convergence curves with other methods in the training process on the Kvasir-SEG dataset.', 
             ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 保存带标题的图像
    plt.savefig(save_path.replace('.jpg', '_with_title.jpg'), dpi=300, bbox_inches='tight')
    
    print(f"收敛曲线已保存至 {save_path}")
    
    # 显示图像
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='绘制模型训练收敛曲线对比图')
    parser.add_argument('--log_dir', type=str, default='run_files/Kvasir-SEG', help='训练日志目录')
    parser.add_argument('--output', type=str, default='figures/comparison_curves.jpg', help='输出图像路径')
    args = parser.parse_args()
    
    # 加载所有训练日志
    model_data = load_training_logs(args.log_dir)
    
    # 绘制收敛曲线
    plot_convergence_curves(model_data, args.output)


if __name__ == "__main__":
    main() 