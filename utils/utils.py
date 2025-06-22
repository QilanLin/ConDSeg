import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops, find_contours
from sklearn.utils import shuffle
from utils.metrics import precision, recall, F2, dice_score, jac_score
from sklearn.metrics import accuracy_score

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Only set CUDA seeds and cudnn deterministic for CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Shuffle the dataset. """
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")


""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]


def fold_mps(x: torch.Tensor, output_size: tuple, kernel_size: int,
             padding: int = 0, stride: int = 1, dilation: int = 1) -> torch.Tensor:
    """Emulate :func:`torch.nn.functional.fold` on devices that do not support it.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``(B, C * kernel_size**2, L)``.
    output_size : tuple
        The spatial size ``(H, W)`` of the folded output.
    kernel_size : int
        Size of each unfolding kernel.
    padding : int, optional
        Padding used during unfolding.
    stride : int, optional
        Stride used during unfolding.
    dilation : int, optional
        Dilation used during unfolding.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, C, H, W)`` placed on the same device as ``x``.
    """

    B, ck2, L = x.shape
    C = ck2 // (kernel_size * kernel_size)
    H, W = output_size

    # Calculate the spatial size of the unfolded feature map
    unfold_h = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    unfold_w = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    x = x.view(B, ck2, unfold_h, unfold_w)

    weight = x.new_zeros(kernel_size * kernel_size, 1, kernel_size, kernel_size)
    for idx in range(kernel_size * kernel_size):
        r = idx // kernel_size
        c = idx % kernel_size
        weight[idx, 0, r, c] = 1.0
    weight = weight.repeat(C, 1, 1, 1)

    out = F.conv_transpose2d(x, weight, bias=None, stride=stride,
                             padding=padding, dilation=dilation, groups=C)
    return out




