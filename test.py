import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
import cv2
from tqdm import tqdm
import torch
from network.model import ConDSeg
from network.unet import UNet
from network.unetpp import UNetPlusPlus
from network.tganet import TGANet
from network.xboundformer import XBoundFormer
from utils.utils import create_dir, seeding
from utils.utils import calculate_metrics
from utils.run_engine import load_data


def process_mask(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def process_edge(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)

    y_pred = y_pred > 0.001
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def print_score(metrics_score, test_x_length):
    jaccard = metrics_score[0] / test_x_length  #
    f1 = metrics_score[1] / test_x_length
    recall = metrics_score[2] / test_x_length
    precision = metrics_score[3] / test_x_length
    acc = metrics_score[4] / test_x_length
    f2 = metrics_score[5] / test_x_length

    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")
    
    return jaccard, f1, recall, precision, acc, f2


def evaluate_model(model, save_path, test_x, test_y, size, device):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            # 处理不同模型的输出格式
            if isinstance(model, ConDSeg):
                mask_pred, fg_pred, bg_pred, uc_pred = model(image)
                p1 = mask_pred
            else:
                outputs = model(image)
                if isinstance(outputs, tuple):
                    p1 = outputs[0]  # 对于返回多个输出的模型，取第一个
                else:
                    p1 = outputs
                    
            """ Evaluation metrics """
            score_1 = calculate_metrics(mask, p1)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))
            p1 = process_mask(p1)

        cv2.imwrite(f"{save_path}/mask/{name}.jpg", p1)

    # 计算并输出评估指标
    jaccard, f1, recall, precision, acc, f2 = print_score(metrics_score_1, len(test_x))

    with open(f"{save_path}/result.txt", "w") as file:
        file.write(f"Jaccard: {jaccard:1.4f}\n")
        file.write(f"F1: {f1:1.4f}\n")
        file.write(f"Recall: {recall:1.4f}\n")
        file.write(f"Precision: {precision:1.4f}\n")
        file.write(f"Acc: {acc:1.4f}\n")
        file.write(f"F2: {f2:1.4f}\n")
    
    return jaccard, f1


def test_all_models(dataset_name='Kvasir-SEG', size=(256, 256)):
    """测试所有模型并保存结果"""
    # 设置种子以确保可重现性
    seeding(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载测试数据
    path = f"data/{dataset_name}/"
    (train_x, train_y), (test_x, test_y) = load_data(path)
    
    # 模型配置
    models_config = [
        {
            "name": "ConDSeg",
            "model": ConDSeg(256, 256),
            "checkpoint": "run_files/Kvasir-SEG/Kvasir-SEG_None_lr0.0001_20250623-183057/checkpoint.pth",
            "save_dir": f"results/{dataset_name}/ConDSeg"
        },
        {
            "name": "UNet",
            "model": UNet(n_channels=3, n_classes=1),
            "checkpoint": "run_files/Kvasir-SEG/UNet_Kvasir-SEG_None_lr0.0001_20250625-172240/checkpoint.pth",
            "save_dir": f"results/{dataset_name}/UNet"
        },
        {
            "name": "UNet++",
            "model": UNetPlusPlus(n_channels=3, n_classes=1),
            "checkpoint": "run_files/Kvasir-SEG/UNETPP_Kvasir-SEG_None_lr0.0001_*/checkpoint.pth",
            "save_dir": f"results/{dataset_name}/UNetPP"
        },
        {
            "name": "TGANet",
            "model": TGANet(n_channels=3, n_classes=1),
            "checkpoint": "run_files/Kvasir-SEG/TGANET_Kvasir-SEG_None_lr0.0001_*/checkpoint.pth",
            "save_dir": f"results/{dataset_name}/TGANet"
        },
        {
            "name": "XBoundFormer",
            "model": XBoundFormer(n_channels=3, n_classes=1),
            "checkpoint": "run_files/Kvasir-SEG/XBOUNDFORMER_Kvasir-SEG_None_lr0.0001_*/checkpoint.pth",
            "save_dir": f"results/{dataset_name}/XBoundFormer"
        }
    ]
    
    # 用于存储所有模型的性能结果
    results = {}
    
    # 测试每个模型
    for config in models_config:
        model_name = config["name"]
        model = config["model"].to(device)
        checkpoint_pattern = config["checkpoint"]
        save_dir = config["save_dir"]
        
        # 查找最新的检查点文件
        if "*" in checkpoint_pattern:
            import glob
            checkpoint_files = glob.glob(checkpoint_pattern)
            if not checkpoint_files:
                print(f"找不到{model_name}的检查点文件，跳过测试")
                continue
            checkpoint_path = sorted(checkpoint_files)[-1]  # 使用最新的检查点
        else:
            checkpoint_path = checkpoint_pattern
        
        # 检查检查点文件是否存在
        if not os.path.exists(checkpoint_path):
            print(f"找不到检查点文件: {checkpoint_path}，跳过{model_name}测试")
            continue
        
        print(f"\n== 测试 {model_name} ==")
        print(f"使用检查点: {checkpoint_path}")
        
        # 加载模型权重
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
        except Exception as e:
            print(f"加载{model_name}模型权重失败: {e}")
            continue
        
        # 创建保存目录
        create_dir(f"{save_dir}/mask")
        
        # 评估模型
        jaccard, f1 = evaluate_model(model, save_dir, test_x, test_y, size, device)
        
        # 存储结果
        results[model_name] = {"jaccard": jaccard, "f1": f1}
    
    # 输出所有模型的结果比较
    print("\n=== 模型性能比较 ===")
    print(f"{'模型名称':<15} {'Jaccard (mIoU)':<15} {'F1 Score (Dice)':<15}")
    print("-" * 45)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['jaccard']:<15.4f} {metrics['f1']:<15.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试医学图像分割模型")
    parser.add_argument("--dataset", type=str, default="Kvasir-SEG", help="数据集名称")
    parser.add_argument("--size", type=int, default=256, help="图像大小")
    parser.add_argument("--specific_model", type=str, default=None, help="仅测试特定模型 (ConDSeg, UNet, UNetPP, TGANet, XBoundFormer)")
    
    args = parser.parse_args()
    
    if args.specific_model:
        print(f"仅测试 {args.specific_model} 模型")
        # 这里可以添加针对特定模型的测试逻辑
        # TODO: 实现特定模型测试
    else:
        test_all_models(args.dataset, (args.size, args.size))
