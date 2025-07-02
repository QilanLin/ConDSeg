import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# 1. 指定每个模型对应的 CSV 文件路径
model_files = {
    'U-Net':            'run_files/Kvasir-SEG/UNet_Kvasir-SEG_None_lr0.0001_20250625-172240/train_log.csv',
    'U-Net++':          'run_files/Kvasir-SEG/UNETPP_Kvasir-SEG_None_lr0.0001_20250626-200244/train_log.csv',
    'TGANet':           'run_files/Kvasir-SEG/TGANET_Kvasir-SEG_None_lr0.0001_20250626-040151/train_log.csv',
    'XBoundFormer':     'run_files/Kvasir-SEG/XBOUNDFORMER_Kvasir-SEG_None_lr0.0001_20250627-145618/train_log.csv',
    'Ours(One-Stage)':  'run_files/Kvasir-SEG/stage1_Kvasir-SEG_None_lr0.0001_20250621-180142/train_log.txt',
    'Ours(Two-Stage)':  'run_files/Kvasir-SEG/Kvasir-SEG_None_lr0.0001_20250623-183057/train_log.csv',
}

# 为每个模型指定固定颜色，确保包含紫色
model_colors = {
    'U-Net': 'blue',
    'U-Net++': 'green',
    'TGANet': 'red',
    'XBoundFormer': 'purple',  # 紫色
    'Ours(One-Stage)': 'orange',
    'Ours(Two-Stage)': 'brown',
}

# 2. 读入所有数据
histories = {}
for name, path in model_files.items():
    print(f"处理 {name} 的数据文件: {path}")
    if not os.path.isfile(path):
        print(f"警告: 找不到文件 {path}，跳过此模型")
        continue
    
    # 根据文件扩展名选择不同的处理方式
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        print(f"CSV文件: {path} 的列名: {df.columns.tolist()}")
        
        # 特殊处理XBoundFormer的数据，去除重复的epoch
        if name == 'XBoundFormer':
            # 确保epoch是连续的，删除重复的epoch
            df = df.drop_duplicates(subset=['epoch'], keep='first')
            # 重置索引
            df = df.reset_index(drop=True)
            
    elif path.endswith('.txt'):
        # 处理txt文件
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"TXT文件: {path} 已读取")
        
        # 提取训练数据
        epochs = []
        miou_values = []
        mdsc_values = []
        
        # 使用更精确的正则表达式匹配
        # 查找格式如 "Epoch: 01 | Epoch Time: 2m 18s" 的行
        epoch_lines = re.findall(r'Epoch: (\d+).*?Val\. Loss: [\d\.]+ - mIoU: ([\d\.]+).*?F1: ([\d\.]+)', content, re.DOTALL)
        
        for match in epoch_lines:
            epoch = int(match[0])
            miou = float(match[1])
            mdsc = float(match[2])
            
            epochs.append(epoch)
            miou_values.append(miou)
            mdsc_values.append(mdsc)
            
            # 打印前10行匹配的数据，用于调试
            if len(epochs) <= 10:
                print(f"  匹配到数据: Epoch={epoch}, mIoU={miou}, F1={mdsc}")
        
        # 创建DataFrame
        df = pd.DataFrame({
            'epoch': epochs,
            'miou': miou_values,
            'mdsc': mdsc_values
        })
        print(f"从TXT提取的数据: 找到 {len(epochs)} 个epoch的数据")
        
        # 如果是One-Stage，打印更多数据以便调试
        if name == 'Ours(One-Stage)':
            print(f"One-Stage数据前10行:")
            print(df.head(10))
    else:
        print(f"不支持的文件格式: {path}，跳过此模型")
        continue
    
    # 确保有这三列：epoch, miou, mdsc
    if 'epoch' not in df.columns or 'miou' not in df.columns or 'mdsc' not in df.columns:
        print(f"警告: {name} 的数据缺少必要的列。现有列: {df.columns.tolist()}")
        # 尝试查找替代列名
        if 'epoch' not in df.columns and 'Epoch' in df.columns:
            df['epoch'] = df['Epoch']
        if 'miou' not in df.columns and 'valid_mIoU' in df.columns:
            df['miou'] = df['valid_mIoU']
        if 'mdsc' not in df.columns and 'valid_f1' in df.columns:
            df['mdsc'] = df['valid_f1']
        print(f"转换后的列: {df.columns.tolist()}")
    
    # 确保数据正确加载
    if len(df) == 0:
        print(f"警告: {name} 的数据为空，跳过此模型")
        continue
        
    histories[name] = df

print("\n开始绘图...")
print("已加载的模型数据:")
for name, df in histories.items():
    print(f"  {name}: {len(df)}行数据")

# 检查是否有数据可绘制
if not histories:
    print("错误: 没有可绘制的数据，请检查数据文件")
    exit(1)

# 3. 开始绘图
plt.rcParams['font.sans-serif'] = ['SimHei']       # 支持中文
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# mIoU vs Epoch
for name, df in histories.items():
    print(f"绘制 {name} 的mIoU曲线, 数据点数: {len(df)}")
    color = model_colors.get(name, 'black')  # 使用指定颜色，如果没有则默认黑色
    ax1.plot(df['epoch'], df['miou'], label=name, linewidth=1, color=color)
ax1.set_title('mIoU vs Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('mIoU')
ax1.grid(True)
ax1.legend(loc='lower right')

# mDSC vs Epoch
for name, df in histories.items():
    print(f"绘制 {name} 的mDSC曲线, 数据点数: {len(df)}")
    color = model_colors.get(name, 'black')  # 使用指定颜色，如果没有则默认黑色
    ax2.plot(df['epoch'], df['mdsc'], label=name, linewidth=1, color=color)
ax2.set_title('mDSC vs Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('mDSC')
ax2.grid(True)
# 只在右侧子图显示图例
ax2.legend(loc='lower right')

plt.tight_layout()
print(f"保存图像到 comparison.png")
plt.savefig('comparison.png', dpi=300)
print(f"图像已保存")
plt.show()
print("程序执行完毕")