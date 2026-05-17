import torch
import torch.nn.functional as F
import numpy as np
from PIL import ImageFilter
from torchvision import transforms
from utils.metrics_utils import calculate_au_pro
from sklearn.metrics import roc_auc_score

def smooth_with_conv(cos_comb, device):
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device) / (w_l ** 2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device) / (w_u ** 2)

    for _ in range(5):
        cos_comb = F.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
    for _ in range(3):
        cos_comb = F.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
    return cos_comb

def smooth_with_gaussian_blur(cos_comb, device):
    map_max = cos_comb.max()
    if map_max > 0:
        map_max_val = float(map_max.detach().cpu().item())
        cos_cpu = (cos_comb.detach().cpu().squeeze(0) / map_max_val)
        cos_pil = transforms.ToPILImage()(cos_cpu)
        cos_blur = cos_pil.filter(ImageFilter.GaussianBlur(radius=4))
        cos_tensor = transforms.ToTensor()(cos_blur).unsqueeze(0)
        cos_comb = (cos_tensor * map_max_val).to(device)
    return cos_comb

def compute_anomaly_scores_and_metrics(cos_comb, gt, label, device):
    cos_combi = smooth_with_conv(cos_comb, device)

    cos_comb = smooth_with_gaussian_blur(cos_comb, device)

    cos_comb = cos_comb.reshape(224, 224)
    
    gt_np = gt.squeeze().cpu().detach().numpy()
    pred_np = (cos_comb / (cos_comb[cos_comb != 0].mean() + 1e-8)).cpu().detach().numpy()
    
    # Image level and pixel level anomaly scores
    image_pred = (cos_combi / (torch.sqrt(cos_combi[cos_combi != 0].mean()) + 1e-8)).cpu().detach().numpy().max()
    pixel_preds = (cos_comb / (torch.sqrt(cos_comb.mean()) + 1e-8)).flatten().cpu().detach().numpy()
    pixel_labels = gt.flatten().cpu().detach().numpy()

    return {
        'gt_np': gt_np,
        'pred_np': pred_np,
        'image_label': label,
        'image_pred': image_pred,
        'pixel_preds': pixel_preds,
        'pixel_labels': pixel_labels,
        'cos_comb_smoothed': cos_comb
    }

def calculate_all_metrics(gts, predictions, image_labels, image_preds, pixel_labels, pixel_preds):
    au_pros, _ = calculate_au_pro(gts, predictions)
    pixel_rocauc = roc_auc_score(np.concatenate(pixel_labels), np.concatenate(pixel_preds))
    image_rocauc = roc_auc_score(np.array(image_labels), np.array(image_preds))
    return au_pros, pixel_rocauc, image_rocauc