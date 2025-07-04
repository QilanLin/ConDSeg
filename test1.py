import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
import cv2
from tqdm import tqdm
import torch
from network.model import ConDSeg
from utils.utils import create_dir, seeding
from utils.utils import calculate_metrics
from utils.run_engine import load_data


def extract_uncertainty_masks(outputs):
    """
    处理ConDSeg模型的预测掩码，提取前景不确定性和背景不确定性区域
    
    前景不确定性：概率在0.5-0.7之间
    背景不确定性：概率在0.3-0.4之间
    """
    # 主要分割掩码
    mask = outputs[0]
    
    # 转换为NumPy数组
    mask = mask.cpu().numpy()
    
    # 确保是2D数组，移除所有长度为1的维度
    mask = np.squeeze(mask)
    
    # 从主掩码中提取前景和背景不确定性区域
    # 前景不确定性区域 (0.5 <= prob < 0.7)
    fg_uncertainty_mask = (mask >= 0.5) & (mask < 0.7)
    
    # 背景不确定性区域 (0.3 <= prob < 0.4)
    bg_uncertainty_mask = (mask >= 0.3) & (mask < 0.4)
    
    # 完整不确定性区域 (0.3 <= prob < 0.4) 或 (0.5 <= prob < 0.7)
    full_uncertainty_mask = ((mask >= 0.3) & (mask < 0.4)) | ((mask >= 0.5) & (mask < 0.7))
    
    # 前景不确定性掩码
    fg_uncertainty_mask_img = fg_uncertainty_mask.astype(np.int32) * 255
    fg_uncertainty_mask_img = np.array(fg_uncertainty_mask_img, dtype=np.uint8)
    fg_uncertainty_mask_img = np.expand_dims(fg_uncertainty_mask_img, axis=-1)
    fg_uncertainty_mask_img = np.concatenate([fg_uncertainty_mask_img, fg_uncertainty_mask_img, fg_uncertainty_mask_img], axis=2)
    
    # 背景不确定性掩码
    bg_uncertainty_mask_img = bg_uncertainty_mask.astype(np.int32) * 255
    bg_uncertainty_mask_img = np.array(bg_uncertainty_mask_img, dtype=np.uint8)
    bg_uncertainty_mask_img = np.expand_dims(bg_uncertainty_mask_img, axis=-1)
    bg_uncertainty_mask_img = np.concatenate([bg_uncertainty_mask_img, bg_uncertainty_mask_img, bg_uncertainty_mask_img], axis=2)
    
    # 完整不确定性掩码
    full_uncertainty_mask_img = full_uncertainty_mask.astype(np.int32) * 255
    full_uncertainty_mask_img = np.array(full_uncertainty_mask_img, dtype=np.uint8)
    full_uncertainty_mask_img = np.expand_dims(full_uncertainty_mask_img, axis=-1)
    full_uncertainty_mask_img = np.concatenate([full_uncertainty_mask_img, full_uncertainty_mask_img, full_uncertainty_mask_img], axis=2)
    
    return fg_uncertainty_mask_img, bg_uncertainty_mask_img, full_uncertainty_mask_img, mask


def create_uncertainty_visualization(image, mask, fg_uncertainty_mask, bg_uncertainty_mask, raw_pred):
    """
    创建更明显的不确定性可视化，包含原始图像、mask和区分前景/背景不确定性的叠加图
    使用更粗的线条链接不确定性区域，并确保颜色正确显示
    
    Args:
        image: 原始图像
        mask: 分割掩码
        fg_uncertainty_mask: 前景不确定性掩码
        bg_uncertainty_mask: 背景不确定性掩码
        raw_pred: 原始预测概率
    
    Returns:
        visualization: 拼接后的图像
    """
    h, w = image.shape[:2]
    
    # 定义更鲜明的颜色 - BGR格式
    FG_UNCERTAINTY_COLOR = (128, 0, 255)  # 紫色（偏红）
    BG_UNCERTAINTY_COLOR = (0, 255, 255)  # 黄色
    FG_CONTOUR_COLOR = (0, 0, 255)        # 红色
    BG_CONTOUR_COLOR = (255, 0, 0)        # 蓝色
    
    # 创建不确定性遮罩
    # 前景不确定性用紫色
    fg_mask = np.zeros_like(image)
    # 确保索引方式兼容
    if len(fg_uncertainty_mask.shape) == 3:
        fg_mask[fg_uncertainty_mask[:,:,0] > 0] = FG_UNCERTAINTY_COLOR
    else:
        # 如果掩码是2D的
        fg_mask[fg_uncertainty_mask > 0] = FG_UNCERTAINTY_COLOR
    
    # 背景不确定性用黄色
    bg_mask = np.zeros_like(image)
    # 确保索引方式兼容
    if len(bg_uncertainty_mask.shape) == 3:
        bg_mask[bg_uncertainty_mask[:,:,0] > 0] = BG_UNCERTAINTY_COLOR
    else:
        # 如果掩码是2D的
        bg_mask[bg_uncertainty_mask > 0] = BG_UNCERTAINTY_COLOR
    
    # 将不确定性区域叠加到原图上（先叠加背景不确定性，再叠加前景不确定性）
    overlay_img = image.copy()
    # 增加透明度使不确定性区域更加明显
    overlay_img = cv2.addWeighted(overlay_img, 1.0, bg_mask, 0.7, 0)
    overlay_img = cv2.addWeighted(overlay_img, 1.0, fg_mask, 0.7, 0)
    
    # 用形态学操作提取前景和背景不确定性区域的边界
    kernel = np.ones((3, 3), np.uint8)
    
    # 对前景不确定性区域进行膨胀，使连接更粗
    if len(fg_uncertainty_mask.shape) == 3:
        fg_dilated = cv2.dilate(fg_uncertainty_mask[:,:,0], kernel, iterations=1)
    else:
        fg_dilated = cv2.dilate(fg_uncertainty_mask, kernel, iterations=1)
    fg_edges = cv2.Canny(fg_dilated, 50, 150)
    
    # 对背景不确定性区域进行膨胀，使连接更粗
    if len(bg_uncertainty_mask.shape) == 3:
        bg_dilated = cv2.dilate(bg_uncertainty_mask[:,:,0], kernel, iterations=1)
    else:
        bg_dilated = cv2.dilate(bg_uncertainty_mask, kernel, iterations=1)
    bg_edges = cv2.Canny(bg_dilated, 50, 150)
    
    # 在叠加图上绘制边界线条（更粗的线条）
    overlay_img[fg_edges > 0] = FG_CONTOUR_COLOR
    overlay_img[bg_edges > 0] = BG_CONTOUR_COLOR
    
    # 创建单独的前景和背景不确定性可视化
    fg_overlay = image.copy()
    bg_overlay = image.copy()
    
    # 前景不确定性单独显示（更明显的颜色和更粗的边界）
    fg_overlay = cv2.addWeighted(fg_overlay, 1.0, fg_mask, 0.7, 0)
    fg_overlay[fg_edges > 0] = FG_CONTOUR_COLOR
    
    # 背景不确定性单独显示（更明显的颜色和更粗的边界）
    bg_overlay = cv2.addWeighted(bg_overlay, 1.0, bg_mask, 0.7, 0)
    bg_overlay[bg_edges > 0] = BG_CONTOUR_COLOR
    
    # 创建不确定性概率热力图 - 前景不确定性用红色系，背景不确定性用蓝色系
    uncertainty_heatmap = np.zeros_like(image)
    
    # 创建热力图版本（使用概率值）
    fg_prob_mask = np.zeros_like(raw_pred)
    fg_prob_mask[(raw_pred >= 0.5) & (raw_pred < 0.7)] = raw_pred[(raw_pred >= 0.5) & (raw_pred < 0.7)]
    
    bg_prob_mask = np.zeros_like(raw_pred)
    bg_prob_mask[(raw_pred >= 0.3) & (raw_pred < 0.4)] = raw_pred[(raw_pred >= 0.3) & (raw_pred < 0.4)]
    
    # 将前景不确定性映射到红色通道（修复normalize调用）
    fg_normalized = ((fg_prob_mask - 0.5) / 0.2 * 255).astype(np.uint8)
    uncertainty_heatmap[:,:,2] = fg_normalized
    
    # 将背景不确定性映射到蓝色通道（修复normalize调用）
    bg_normalized = (((0.4 - bg_prob_mask) / 0.1) * 255).astype(np.uint8)
    uncertainty_heatmap[:,:,0] = bg_normalized
    
    # 将原始预测转换为可视化掩码
    pred_mask = (raw_pred > 0.5).astype(np.uint8) * 255
    pred_mask = np.expand_dims(pred_mask, axis=-1)
    pred_mask = np.concatenate([pred_mask, pred_mask, pred_mask], axis=2)
    
    # 创建拼接图像 - 上排：原图、标准mask、叠加了所有不确定性区域的图像
    # 下排：前景不确定性、背景不确定性、预测mask
    visualization = np.zeros((h*2, w*3, 3), dtype=np.uint8)
    
    # 上排
    visualization[:h, :w] = image  # 原图
    visualization[:h, w:2*w] = mask  # 真实mask
    visualization[:h, 2*w:] = overlay_img  # 叠加了所有不确定性区域的原图
    
    # 下排
    visualization[h:, :w] = fg_overlay  # 前景不确定性
    visualization[h:, w:2*w] = bg_overlay  # 背景不确定性
    visualization[h:, 2*w:] = pred_mask  # 预测mask
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(visualization, "Original", (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(visualization, "Mask", (w+10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(visualization, "All Uncertainty", (2*w+10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(visualization, "FG Uncertainty", (10, h+20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(visualization, "BG Uncertainty", (w+10, h+20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(visualization, "Predicted Mask", (2*w+10, h+20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 添加缩小的不确定性图例（放在右上角）
    legend_x = 2*w + 10
    legend_y = 40
    legend_font_size = 0.35
    
    # 在右上角添加缩小版的热力图图例
    cv2.rectangle(visualization, (legend_x, legend_y), (legend_x+15, legend_y+15), FG_UNCERTAINTY_COLOR, -1)
    cv2.putText(visualization, "FG (0.5-0.7)", (legend_x+20, legend_y+12), font, legend_font_size, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.rectangle(visualization, (legend_x, legend_y+20), (legend_x+15, legend_y+35), BG_UNCERTAINTY_COLOR, -1)
    cv2.putText(visualization, "BG (0.3-0.4)", (legend_x+20, legend_y+32), font, legend_font_size, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.rectangle(visualization, (legend_x, legend_y+40), (legend_x+15, legend_y+55), FG_CONTOUR_COLOR, -1)
    cv2.putText(visualization, "FG Contour", (legend_x+20, legend_y+52), font, legend_font_size, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.rectangle(visualization, (legend_x, legend_y+60), (legend_x+15, legend_y+75), BG_CONTOUR_COLOR, -1)
    cv2.putText(visualization, "BG Contour", (legend_x+20, legend_y+72), font, legend_font_size, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 添加分割线
    cv2.line(visualization, (w, 0), (w, h*2), (255, 255, 255), 1)
    cv2.line(visualization, (2*w, 0), (2*w, h*2), (255, 255, 255), 1)
    cv2.line(visualization, (0, h), (w*3, h), (255, 255, 255), 1)
    
    return visualization


def print_score(metrics_score):
    jaccard = metrics_score[0] / len(test_x)  #
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    f2 = metrics_score[5] / len(test_x)

    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")


def evaluate(model, save_path, test_x, test_y, size):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # 创建输出目录
    create_dir("./test_exp_condseg")  # 原始不确定性掩码目录
    create_dir("./enhanced_uncertainty_condseg")  # 增强的不确定性可视化目录

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]
        x = '.'+ x.split("..")[1] 
        y = '.'+ y.split("..")[1] 
        """ Image """

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image_for_model = np.transpose(image, (2, 0, 1))
        image_for_model = image_for_model / 255.0
        image_for_model = np.expand_dims(image_for_model, axis=0)
        image_for_model = image_for_model.astype(np.float32)
        image_for_model = torch.from_numpy(image_for_model)
        image_for_model = image_for_model.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask_tensor = np.expand_dims(mask, axis=0)
        mask_tensor = mask_tensor / 255.0
        mask_tensor = np.expand_dims(mask_tensor, axis=0)
        mask_tensor = mask_tensor.astype(np.float32)
        mask_tensor = torch.from_numpy(mask_tensor)
        mask_tensor = mask_tensor.to(device)

        with torch.no_grad():
            # ConDSeg模型返回多个输出(mask, mask_fg, mask_bg, mask_uc)
            outputs = model(image_for_model)
            mask_pred = outputs[0]  # 主要分割掩码

            """ Evaluation metrics """
            score_1 = calculate_metrics(mask_tensor, mask_pred)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))
            
            # 提取前景和背景不确定性掩码
            fg_uncertainty_mask, bg_uncertainty_mask, full_uncertainty_mask, raw_pred = extract_uncertainty_masks(outputs)
            
            # 创建预测掩码
            mask_pred_np = mask_pred.cpu().numpy()
            mask_pred_np = np.squeeze(mask_pred_np)
            
            # 创建增强的不确定性可视化
            enhanced_visualization = create_uncertainty_visualization(
                save_img, save_mask, fg_uncertainty_mask, bg_uncertainty_mask, mask_pred_np
            )
            
        name_new = name.split("\\")[-1] if "\\" in name else name
        
        # 保存原始不确定性掩码到test_exp_condseg目录
        cv2.imwrite(f"./test_exp_condseg/{name_new}.jpg", full_uncertainty_mask)
        
        # 保存增强的不确定性可视化
        cv2.imwrite(f"./enhanced_uncertainty_condseg/{name_new}_enhanced.png", enhanced_visualization)
        
        # 保存预测掩码到results目录
        pred_mask = (mask_pred_np > 0.5).astype(np.uint8) * 255
        cv2.imwrite(f"{save_path}/mask/{name_new}.png", pred_mask)

    print_score(metrics_score_1)

    with open(f"{save_path}/result.txt", "w") as file:
        file.write(f"Jaccard: {metrics_score_1[0] / len(test_x):1.4f}\n")
        file.write(f"F1: {metrics_score_1[1] / len(test_x):1.4f}\n")
        file.write(f"Recall: {metrics_score_1[2] / len(test_x):1.4f}\n")
        file.write(f"Precision: {metrics_score_1[3] / len(test_x):1.4f}\n")
        file.write(f"Acc: {metrics_score_1[4] / len(test_x):1.4f}\n")
        file.write(f"F2: {metrics_score_1[5] / len(test_x):1.4f}\n")


if __name__ == "__main__":
    """ Seeding """

    dataset_name = 'Kvasir-SEG'

    seeding(42)
    size = (256, 256)
    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConDSeg()
    model = model.to(device)
    checkpoint_path = r"C:\Users\Administrator\PycharmProjects\ConDSeg\run_files\Kvasir-SEG\Kvasir-SEG_None_lr0.0001_20250623-183057\checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    path = "../data/{}/".format(dataset_name)
    (train_x, train_y), (test_x, test_y) = load_data(path)

    # 创建输出目录
    create_dir("./test_exp_condseg")  # 原始不确定性掩码目录
    create_dir("./enhanced_uncertainty_condseg")  # 增强的不确定性可视化目录
    
    save_path = f"results/{dataset_name}/ConDSeg"

    create_dir(f"{save_path}/mask")
    evaluate(model, save_path, test_x, test_y, size) 