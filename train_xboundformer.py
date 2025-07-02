import os
import random
import time
import datetime
import warnings

# 忽略所有警告，特别是来自albumentations的版本检查警告
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import DataLoader

from utils.utils import print_and_save, shuffling, epoch_time
from network.xboundformer import XBoundFormer
from utils.metrics import DiceBCELoss

#
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from utils.run_engine_xboundformer import load_data, train, evaluate, DATASET


def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    # dataset
    dataset_name = 'Kvasir-SEG'
    val_name = None

    seed = 0
    my_seeding(seed)

    # hyperparameters
    image_size = 256
    size = (image_size, image_size)
    batch_size = 8  # 降低批次大小，因为XBoundFormer需要更多显存
    num_epochs = 300
    lr = 1e-4
    early_stopping_patience = 100

    resume_path = "run_files/Kvasir-SEG/XBOUNDFORMER_Kvasir-SEG_None_lr0.0001_20250627-145618/checkpoint.pth"

    # make a folder
    if resume_path:
        # 从已有检查点路径中提取文件夹路径
        save_dir = os.path.dirname(resume_path)
        folder_name = os.path.basename(os.path.dirname(resume_path))
    else:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"XBOUNDFORMER_{dataset_name}_{val_name}_lr{lr}_{current_time}"
        # Directories
        base_dir = "data"
        data_path = os.path.join(base_dir, dataset_name)
        save_dir = os.path.join("run_files", dataset_name, folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # 无论是否是恢复训练，都需要设置这些路径
    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    # 如果是新训练则创建新日志文件，否则追加到现有文件
    if not resume_path:
        train_log = open(train_log_path, "w")
        train_log.write("\n")
        train_log.close()
    else:
        # 在现有日志文件末尾添加恢复训练信息
        with open(train_log_path, "a") as train_log:
            train_log.write("\n\n" + "="*50 + "\n")
            train_log.write(f"恢复训练，从检查点: {resume_path}\n")
            train_log.write("="*50 + "\n\n")

    # 数据路径设置
    base_dir = "data"
    data_path = os.path.join(base_dir, dataset_name)

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    hyperparameters_str = f"Model: XBoundFormer\nImage Size: {image_size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    hyperparameters_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    hyperparameters_str += f"Seed: {seed}\n"
    print_and_save(train_log_path, hyperparameters_str)

    """ Data augmentation: Transforms """
    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3)
    ])

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y) = load_data(data_path, val_name)
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ Model """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = XBoundFormer(n_channels=3, n_classes=1, bilinear=True)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params / 1000000:.6f}M\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """

    # 检查是否存在训练日志CSV文件，如果存在且是恢复训练，则从中读取最佳指标
    csv_log_path = os.path.join(save_dir, "train_log.csv")
    if resume_path and os.path.exists(csv_log_path):
        try:
            df = pd.read_csv(csv_log_path)
            if not df.empty:
                # 获取之前训练的最佳验证mIoU
                best_valid_metrics = df['valid_mIoU'].max()
                # 获取上次训练的最后一个epoch
                last_epoch = int(df['epoch'].max())
                print_and_save(train_log_path, f"从训练日志恢复最佳指标: {best_valid_metrics:.4f}, 从epoch {last_epoch} 继续训练")
            else:
                best_valid_metrics = 0.0
                last_epoch = 0
        except Exception as e:
            print_and_save(train_log_path, f"读取训练日志出错: {str(e)}，重置指标为0")
            best_valid_metrics = 0.0
            last_epoch = 0
    else:
        best_valid_metrics = 0.0
        last_epoch = 0
        with open(csv_log_path, "w") as f:
            f.write(
                "epoch,train_loss,train_mIoU,train_f1,train_recall,train_precision,valid_loss,valid_mIoU,valid_f1,valid_recall,valid_precision\n")

    early_stopping_count = 0

    for epoch in range(last_epoch, num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[0] > best_valid_metrics:
            data_str = f"Valid mIoU improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0


        elif valid_metrics[0] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - mIoU: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - mIoU: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        with open(os.path.join(save_dir, "train_log.csv"), "a") as f:
            f.write(
                f"{epoch + 1},{train_loss},{train_metrics[0]},{train_metrics[1]},{train_metrics[2]},{train_metrics[3]},{valid_loss},{valid_metrics[0]},{valid_metrics[1]},{valid_metrics[2]},{valid_metrics[3]}\n")

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break 