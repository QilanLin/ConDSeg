import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob
from tqdm import tqdm
import sys
import random
from matplotlib import font_manager

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 确保network模块可以被正确导入
sys.path.append(os.getcwd())
from network.model import ConDSeg

def get_random_samples(image_dir, mask_dir, n_samples=5):
    """随机选择n_samples个样本"""
    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))
    
    # 确保图像和掩码数量一致
    assert len(images) == len(masks), f"图像数量({len(images)})与掩码数量({len(masks)})不匹配"
    
    # 随机选择索引
    if len(images) <= n_samples:
        indices = list(range(len(images)))
    else:
        indices = np.random.choice(len(images), n_samples, replace=False)
    
    selected_images = [images[i] for i in indices]
    selected_masks = [masks[i] for i in indices]
    
    return selected_images, selected_masks

def process_image(image_path, size=(256, 256)):
    """处理图像用于模型输入"""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, size)
    display_image = image.copy()  # 用于显示的图像副本
    
    # 模型输入格式化
    image = np.transpose(image, (2, 0, 1))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    
    return image, display_image

def process_mask(mask_path, size=(256, 256)):
    """处理掩码用于模型输入和显示"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, size)
    display_mask = mask.copy()  # 用于显示的掩码副本
    
    # 模型输入格式化
    mask = np.expand_dims(mask, axis=0)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=0)
    mask = mask.astype(np.float32)
    mask = torch.from_numpy(mask)
    
    return mask, display_mask

def overlay_uncertain_region(image, uncertain_region, alpha=0.7, color=(255, 0, 0)):
    """在原图上叠加不确定区域，使用红色表示不确定区域，只标记真正不确定的区域"""
    overlay = image.copy()
    
    # 确保uncertain_region是二值掩码（只有0和1）
    # 我们期望uncertain_region已经是处理过的二值掩码，其中1表示不确定区域，0表示确定区域
    if np.max(uncertain_region) > 1.0:
        # 对于值范围为0-255的掩码，只将255视为不确定区域
        uncertain_mask = uncertain_region >= 128
    else:
        # 对于值范围为0-1的掩码，只将1视为不确定区域
        uncertain_mask = uncertain_region >= 0.5
    
    # 只为不确定区域着色
    uncertain_pixels = np.sum(uncertain_mask)
    if uncertain_pixels > 0:
        overlay[uncertain_mask] = overlay[uncertain_mask] * (1 - alpha) + np.array(color) * alpha
    
    return overlay, uncertain_pixels

def create_uncertainty_heatmap(uncertain_region):
    """创建不确定性热力图"""
    # 将不确定性区域转换为热力图 (红色越深表示不确定性越高)
    heatmap = cv2.applyColorMap((uncertain_region * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap

def get_layered_predictions(pred, thresholds=[0.4, 0.5, 0.6]):
    """获取分层的预测结果可视化"""
    # 创建彩色图像
    colored_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    
    # 应用不同的阈值，并用不同颜色表示
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 蓝、绿、红
    
    for i, thresh in enumerate(thresholds):
        mask = pred > thresh
        colored_pred[mask] = colors[i]
    
    return colored_pred

def enhance_uncertainty_region(uncertain_region, enhancement_factor=10000):
    """增强不确定性区域的对比度，使用线性映射而不是百分位数筛选"""
    # 简单线性映射，确保所有不确定性值都被保留
    # 将值映射到0-1范围
    if np.max(uncertain_region) > np.min(uncertain_region):
        enhanced = (uncertain_region - np.min(uncertain_region)) / (np.max(uncertain_region) - np.min(uncertain_region))
    else:
        enhanced = np.zeros_like(uncertain_region)
    
    return enhanced

def create_uncertainty_mask(mask_uc, threshold=0.01):
    """创建单独的不确定性掩码图像，使用灰色表示不确定区域"""
    # 将不确定性区域转换为二值掩码
    uc_mask = (mask_uc > threshold).astype(np.uint8) * 255
    
    # 创建灰度图像
    gray_mask = np.zeros((mask_uc.shape[0], mask_uc.shape[1], 3), dtype=np.uint8)
    gray_mask[uc_mask > 0] = [128, 128, 128]  # 灰色
    
    return gray_mask

def generate_visualization(model_path, data_path, output_path, n_samples=5, thresholds=[0.4, 0.5, 0.6]):
    """生成可视化效果"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置日志文件
    log_file = os.path.join(output_path, "analysis_log.txt")
    os.makedirs(output_path, exist_ok=True)
    
    # 创建日志函数，同时打印到控制台和写入文件
    def log_message(message):
        print(message)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    
    log_message(f"开始分析，时间: {np.datetime64('now')}")
    log_message(f"使用设备: {device}")
    
    # 加载模型
    log_message(f"加载two-stage模型: {model_path}")
    model = ConDSeg(256, 256).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        log_message("模型加载成功")
    except Exception as e:
        log_message(f"加载模型时出错: {e}")
        return
    
    # 获取随机样本
    image_dir = os.path.join(data_path, "images")
    mask_dir = os.path.join(data_path, "masks")
    
    try:
        selected_images, selected_masks = get_random_samples(image_dir, mask_dir, n_samples)
        log_message(f"成功选择了{len(selected_images)}个样本")
    except Exception as e:
        log_message(f"选择样本时出错: {e}")
        return
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 为每个样本生成可视化
    for idx, (img_path, mask_path) in enumerate(zip(selected_images, selected_masks)):
        # 获取样本名称
        sample_name = os.path.basename(img_path).split('.')[0]
        log_message(f"\n\n===== 处理样本 {idx+1}/{len(selected_images)}: {sample_name} =====")
        
        # 处理图像和掩码
        img_tensor, orig_img = process_image(img_path)
        mask_tensor, orig_mask = process_mask(mask_path)
        
        img_tensor = img_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        
        with torch.no_grad():
            # 模型预测
            try:
                outputs = model(img_tensor)
                log_message(f"模型预测成功，输出类型: {type(outputs)}")
                
                if isinstance(outputs, tuple):
                    log_message(f"模型输出是tuple，长度: {len(outputs)}")
                    if len(outputs) >= 4:
                        pred = outputs[0]
                        mask_fg = outputs[1]
                        mask_bg = outputs[2]
                        mask_uc = outputs[3]
                        log_message(f"不确定性区域最大值: {torch.max(mask_uc).item()}, 最小值: {torch.min(mask_uc).item()}")
                        
                        # 计算不确定性区域中大于0的像素点数量
                        uc_nonzero_count = torch.sum(mask_uc > 0).item()
                        uc_total_pixels = mask_uc.numel()
                        log_message(f"不确定性区域中非零像素点比例: {uc_nonzero_count/uc_total_pixels:.6f} ({uc_nonzero_count}/{uc_total_pixels})")
                        
                        # 分析不确定性值的分布
                        mask_uc_flat = mask_uc.flatten()
                        percentiles = [50, 75, 90, 95, 99]
                        percentile_values = {}
                        for p in percentiles:
                            value = torch.quantile(mask_uc_flat, p/100).item()
                            percentile_values[p] = value
                            log_message(f"不确定性区域 {p}% 分位数: {value:.8f}")
                    elif len(outputs) > 0:
                        pred = outputs[0]
                        mask_uc = torch.zeros_like(pred)
                    else:
                        log_message("错误: outputs是空元组")
                        continue
                else:
                    pred = outputs
                    mask_uc = torch.zeros_like(pred)
            
            except Exception as e:
                log_message(f"模型预测时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 转换为numpy用于可视化
        try:
            pred_np = pred.cpu().numpy()
            if len(pred_np.shape) == 4:  # 如果是[B, C, H, W]格式
                pred_np = pred_np[0, 0]
                
            uncertain_region_np = mask_uc.cpu().numpy()
            if len(uncertain_region_np.shape) == 4:  # 如果是[B, C, H, W]格式
                uncertain_region_np = uncertain_region_np[0, 0]
                
            log_message(f"不确定性区域numpy最大值: {np.max(uncertain_region_np)}, 最小值: {np.min(uncertain_region_np)}")
            # 打印不确定性区域的直方图统计
            hist, bin_edges = np.histogram(uncertain_region_np, bins=10)
            log_message("不确定性直方图:")
            for i in range(len(hist)):
                log_message(f"  [{bin_edges[i]:.8f}, {bin_edges[i+1]:.8f}): {hist[i]} 像素点")
                
            # 分析预测logits的分布
            log_message(f"预测logits最大值: {np.max(pred_np)}, 最小值: {np.min(pred_np)}")
            
            # 检查预测logits是否在0-1之间
            if np.max(pred_np) <= 1.0 and np.min(pred_np) >= 0.0:
                log_message("预测logits在0-1范围内")
            else:
                log_message("警告：预测logits不在0-1范围内")
            
            # 分析不同区间的占比
            intervals = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), 
                         (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
            total_pixels = pred_np.size
            log_message("预测logits在不同区间的占比:")
            for low, high in intervals:
                count = np.sum((pred_np >= low) & (pred_np < high))
                percentage = count / total_pixels * 100
                log_message(f"  [{low:.1f}, {high:.1f}): {count} 像素点 ({percentage:.2f}%)")
            
            # 特别分析高值区域
            high_intervals = [(0.95, 0.96), (0.96, 0.97), (0.97, 0.98), (0.98, 0.99), (0.99, 1.0)]
            log_message("预测logits在高值区域的细分占比:")
            for low, high in high_intervals:
                count = np.sum((pred_np >= low) & (pred_np < high))
                percentage = count / total_pixels * 100
                log_message(f"  [{low:.2f}, {high:.2f}): {count} 像素点 ({percentage:.2f}%)")
            
            # 分析预测logits与真实标签的关系
            # 将真实标签转换为numpy数组
            true_mask_np = mask_tensor.cpu().numpy()
            if len(true_mask_np.shape) == 4:  # 如果是[B, C, H, W]格式
                true_mask_np = true_mask_np[0, 0]
            
            # 计算真实标签为1（前景）的区域中，预测logits的分布
            fg_mask = true_mask_np > 0.5
            if np.sum(fg_mask) > 0:
                fg_logits = pred_np[fg_mask]
                log_message(f"前景区域的预测logits: 最大值={np.max(fg_logits)}, 最小值={np.min(fg_logits)}, 平均值={np.mean(fg_logits):.4f}")
                
                # 分析前景区域中不同区间的占比
                log_message("前景区域中预测logits在不同区间的占比:")
                for low, high in intervals:
                    count = np.sum((fg_logits >= low) & (fg_logits < high))
                    percentage = count / fg_logits.size * 100
                    log_message(f"  [{low:.1f}, {high:.1f}): {count} 像素点 ({percentage:.2f}%)")
                
                # 特别分析前景区域中的高值区域
                log_message("前景区域中预测logits在高值区域的细分占比:")
                for low, high in high_intervals:
                    count = np.sum((fg_logits >= low) & (fg_logits < high))
                    percentage = count / fg_logits.size * 100
                    log_message(f"  [{low:.2f}, {high:.2f}): {count} 像素点 ({percentage:.2f}%)")
            
            # 计算真实标签为0（背景）的区域中，预测logits的分布
            bg_mask = true_mask_np <= 0.5
            if np.sum(bg_mask) > 0:
                bg_logits = pred_np[bg_mask]
                log_message(f"背景区域的预测logits: 最大值={np.max(bg_logits)}, 最小值={np.min(bg_logits)}, 平均值={np.mean(bg_logits):.4f}")
                
                # 分析背景区域中不同区间的占比
                log_message("背景区域中预测logits在不同区间的占比:")
                for low, high in intervals:
                    count = np.sum((bg_logits >= low) & (bg_logits < high))
                    percentage = count / bg_logits.size * 100
                    log_message(f"  [{low:.1f}, {high:.1f}): {count} 像素点 ({percentage:.2f}%)")
                
        except Exception as e:
            log_message(f"转换预测结果为numpy时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 增强不确定性区域，使其更加可见
        enhanced_uncertain_region = enhance_uncertainty_region(uncertain_region_np, 100000)
        log_message(f"增强后不确定性区域最大值: {np.max(enhanced_uncertain_region)}, 最小值: {np.min(enhanced_uncertain_region)}")
        
        # 使用中等阈值(0.5)，与可视化部分保持一致
        uncertainty_threshold = 0.5
        log_message(f"使用中等阈值: {uncertainty_threshold:.4f}")
        
        # 生成不确定区域的二值掩码
        uncertain_mask = (enhanced_uncertain_region > uncertainty_threshold)
        
        # 打印不确定区域像素数量
        uncertain_pixels = np.sum(uncertain_mask)
        total_pixels = uncertain_mask.size
        log_message(f"阈值{uncertainty_threshold}下的不确定区域像素比例: {uncertain_pixels/total_pixels*100:.2f}% ({uncertain_pixels}/{total_pixels})")
        
        # 生成带有不确定区域的原图
        overlay_img, _ = overlay_uncertain_region(orig_img, uncertain_mask * 255)
        
        # 生成不确定性热力图 (使用增强后的不确定性区域)
        uncertainty_heatmap = create_uncertainty_heatmap(enhanced_uncertain_region)
        
        # 创建单独的不确定性掩码图像
        uncertainty_mask = create_uncertainty_mask(enhanced_uncertain_region, threshold=uncertainty_threshold)
        
        # 生成分层预测图像
        layered_pred = get_layered_predictions(pred_np, thresholds=thresholds)
        
        # 创建可视化图像
        plt.figure(figsize=(20, 8))
        
        # 第1列：原始图像
        plt.subplot(151)
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.title("原始图像", fontsize=14)
        plt.axis('off')
        
        # 第2列：真实标签
        plt.subplot(152)
        plt.imshow(orig_mask, cmap='gray')
        plt.title("真实标签", fontsize=14)
        plt.axis('off')
        
        # 第3列：预测logits转换为0-255范围的热力图
        # 将预测分数（logits）转换为0-255范围
        pred_255 = (pred_np * 255).astype(np.uint8)
        plt.subplot(153)
        plt.imshow(pred_255, cmap='jet')
        plt.title("预测logits (0-255)", fontsize=14)
        plt.axis('off')
        plt.colorbar(shrink=0.7)
        
        # 第4列：不确定性区域转换为0-255范围的热力图
        # 将不确定性值转换为0-255范围
        uc_255 = ((uncertain_region_np - np.min(uncertain_region_np)) / 
                 (np.max(uncertain_region_np) - np.min(uncertain_region_np) + 1e-10) * 255).astype(np.uint8)
        plt.subplot(154)
        plt.imshow(uc_255, cmap='jet')
        plt.title("不确定性区域 (0-255)", fontsize=14)
        plt.axis('off')
        plt.colorbar(shrink=0.7)
        
        # 第5列：二值化的预测结果
        binary_pred = (pred_np > 0.5).astype(np.uint8) * 255
        plt.subplot(155)
        plt.imshow(binary_pred, cmap='gray')
        plt.title("二值化预测 (阈值=0.5)", fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{sample_name}_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()
        log_message(f"已保存样本{sample_name}的可视化结果")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成模型分割可视化效果")
    parser.add_argument("--model_path", type=str, 
                       default="run_files/Kvasir-SEG/Kvasir-SEG_None_lr0.0001_20250623-183057/checkpoint.pth", 
                       help="模型检查点路径")
    parser.add_argument("--data_path", type=str, 
                       default="data/Kvasir-SEG", 
                       help="数据集路径")
    parser.add_argument("--output_path", type=str, 
                       default="visualization_results", 
                       help="输出路径")
    parser.add_argument("--n_samples", type=int, 
                       default=5, 
                       help="要生成的样本数量")
    parser.add_argument("--thresholds", type=float, nargs='+',
                       default=[0.4, 0.5, 0.6],
                       help="分层预测的阈值列表")
    
    args = parser.parse_args()
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    generate_visualization(
        args.model_path,
        args.data_path,
        args.output_path,
        args.n_samples,
        args.thresholds
    )
    
    print(f"已生成{args.n_samples}个样本的可视化结果，保存在{args.output_path}目录中") 