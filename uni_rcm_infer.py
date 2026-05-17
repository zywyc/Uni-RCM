import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.quantizer import build_quantizer
from models.metrics import compute_anomaly_scores_and_metrics, calculate_all_metrics
from uni_rcm_train import UniRCM_Model


RGB_DIM = 768
XYZ_DIM = 1152
FEAT_H = 56
FEAT_W = 56


def set_seeds(sid=41):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def load_feature_orq(args, device):
    if not args.orq_checkpoint:
        return None

    print(f"[Uni-RCM] Loading ORQ checkpoint: {args.orq_checkpoint}")
    checkpoint = torch.load(args.orq_checkpoint, map_location='cpu')

    vq_layers = checkpoint.get('vq_layers', args.vq_layers)
    num_embeddings_rgb = checkpoint.get('num_embeddings_rgb', args.num_embeddings_rgb)
    num_embeddings_xyz = checkpoint.get('num_embeddings_xyz', args.num_embeddings_xyz)

    rgb_orq = build_quantizer(
        num_embeddings=num_embeddings_rgb,
        embedding_dim=RGB_DIM,
        vq_layers=vq_layers,
    )
    xyz_orq = build_quantizer(
        num_embeddings=num_embeddings_xyz,
        embedding_dim=XYZ_DIM,
        vq_layers=vq_layers,
    )

    rgb_orq.load_state_dict(checkpoint['rgb_codebook'])
    xyz_orq.load_state_dict(checkpoint['xyz_codebook'])

    rgb_orq = rgb_orq.to(device).eval()
    xyz_orq = xyz_orq.to(device).eval()

    print(
        '[Uni-RCM] ORQ meta: '
        f"Krgb={num_embeddings_rgb}, Kxyz={num_embeddings_xyz}, layers={vq_layers}, "
        f"rgb_util={checkpoint.get('rgb_utilization', 0.0):.3f}, "
        f"xyz_util={checkpoint.get('xyz_utilization', 0.0):.3f}"
    )

    return {
        'rgb': rgb_orq,
        'xyz': xyz_orq,
        'meta': checkpoint,
    }


def infer_UniRCM(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_loader = get_data_loader(
        "test",
        class_name=args.class_name,
        img_size=224,
        dataset_path=args.dataset_path,
    )

    feature_extractor = MultimodalFeatures()
    feature_extractor.to(device)

    model = UniRCM_Model(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    )

    checkpoint_class_name = getattr(args, 'checkpoint_class_name', 'combine')
    checkpoint_path = os.path.join(
        args.checkpoint_folder, checkpoint_class_name,
        f'UniRCM_{checkpoint_class_name}_{args.epochs_no}ep_{args.train_batch_size}bs'
        f'_h{args.hidden_dim}_L{args.num_blocks}.pth'
    )
    print(f"[Uni-RCM] Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.to(device)
    model.eval()

    orq = load_feature_orq(args, device)


    predictions, gts = [], []
    image_labels, pixel_labels = [], []
    image_preds, pixel_preds = [], []

    for (rgb, pc, depth), gt, label, rgb_path in tqdm(test_loader, desc=f'Uni-RCM Inference: {args.class_name}'):
        rgb, pc, depth = rgb.to(device), pc.to(device), depth.to(device)

        with torch.no_grad():
            rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)

            if rgb_patch.dim() == 2:
                rgb_patch = rgb_patch.unsqueeze(0)
                xyz_patch = xyz_patch.unsqueeze(0)

            xyz_pred = model.net_2d3d(rgb_patch)
            rgb_pred = model.net_3d2d(xyz_patch)

            xyz_pred = xyz_pred.squeeze(0)
            rgb_pred = rgb_pred.squeeze(0)
            rgb_patch_2d = rgb_patch.squeeze(0)
            xyz_patch_2d = xyz_patch.squeeze(0)

            xyz_mask = (xyz_patch_2d.sum(dim=-1) == 0)

            cos_3d = (
                F.normalize(xyz_pred, dim=-1) - F.normalize(xyz_patch_2d, dim=-1)
            ).pow(2).sum(-1).sqrt()
            cos_3d[xyz_mask] = 0.
            cos_3d = cos_3d.reshape(FEAT_H, FEAT_W)

            cos_2d = (
                F.normalize(rgb_pred, dim=-1) - F.normalize(rgb_patch_2d, dim=-1)
            ).pow(2).sum(-1).sqrt()
            cos_2d[xyz_mask] = 0.
            cos_2d = cos_2d.reshape(FEAT_H, FEAT_W)

            cos_comb = cos_2d * cos_3d
            cos_comb.reshape(-1)[xyz_mask] = 0.

            if orq is not None:
                rgb_vq_input = rgb_patch_2d
                xyz_vq_input = xyz_patch_2d

                _, _, qe_xyz = orq['xyz'].quantize(xyz_vq_input)
                _, _, qe_rgb = orq['rgb'].quantize(rgb_vq_input)

                qe_3d = qe_xyz.clone()
                qe_2d = qe_rgb.clone()
                qe_3d[xyz_mask] = 0.
                qe_2d[xyz_mask] = 0.

                qe_3d = qe_3d.reshape(FEAT_H, FEAT_W)
                qe_2d = qe_2d.reshape(FEAT_H, FEAT_W)

                if qe_3d.max() > 0:
                    qe_3d = qe_3d / qe_3d.max()
                if qe_2d.max() > 0:
                    qe_2d = qe_2d / qe_2d.max()

                signal_orq = qe_3d
                signal_orq.reshape(-1)[xyz_mask] = 0.
                cos_comb = (cos_2d + args.vq_beta_2d * qe_2d) * (cos_3d + args.vq_beta_3d * qe_3d)
                cos_comb.reshape(-1)[xyz_mask] = 0.

            cos_comb = cos_comb.reshape(1, 1, FEAT_H, FEAT_W)
            cos_comb = F.interpolate(
                cos_comb, size=(224, 224), mode='bilinear', align_corners=False)

            metrics_out = compute_anomaly_scores_and_metrics(cos_comb, gt, label, device)
            
            cos_comb = metrics_out['cos_comb_smoothed']
            gts.append(metrics_out['gt_np'])
            predictions.append(metrics_out['pred_np'])
            image_labels.append(metrics_out['image_label'])
            image_preds.append(metrics_out['image_pred'])
            pixel_labels.append(metrics_out['pixel_labels'])
            pixel_preds.append(metrics_out['pixel_preds'])

            if args.produce_qualitatives:
                _save_qualitatives(
                    rgb, depth, gt, cos_2d, cos_3d, cos_comb,
                    rgb_path, args,
                )

    au_pros, pixel_rocauc, image_rocauc = calculate_all_metrics(
        gts, predictions, image_labels, image_preds, pixel_labels, pixel_preds
    )
    
    result_file_name = os.path.join(
        args.quantitative_folder,
        f'{args.class_name}_uni_rcm_h{args.hidden_dim}_L{args.num_blocks}'
        f'_{args.epochs_no}ep_{args.train_batch_size}bs'
    )
    os.makedirs(args.quantitative_folder, exist_ok=True)

    title_string = (
        f'Uni-RCM Metrics for class {args.class_name} '
        f'(h={args.hidden_dim}, L={args.num_blocks}, {args.epochs_no}ep, bs={args.train_batch_size}'
        f'{f", vq_beta_2d={args.vq_beta_2d}, vq_beta_3d={args.vq_beta_3d}" if orq is not None else ""})'
    )
    header_string = 'AUPRO@30% & AUPRO@10% & AUPRO@5% & AUPRO@1% & P-AUROC & I-AUROC'
    results_string = (
        f'{au_pros[0]:.3f} & {au_pros[1]:.3f} & {au_pros[2]:.3f} & '
        f'{au_pros[3]:.3f} & {pixel_rocauc:.3f} & {image_rocauc:.3f}'
    )

    with open(result_file_name, "w") as f:
        f.write(title_string + '\n' + header_string + '\n' + results_string)

    print(title_string)
    print("AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% | P-AUROC | I-AUROC")
    print(
        f'  {au_pros[0]:.3f}   |   {au_pros[1]:.3f}   |   {au_pros[2]:.3f}  |   '
        f'{au_pros[3]:.3f}  |   {pixel_rocauc:.3f} |   {image_rocauc:.3f}'
    )

    return {
        'class_name': args.class_name,
        'au_pros': au_pros,
        'pixel_rocauc': pixel_rocauc,
        'image_rocauc': image_rocauc,
    }


def _save_qualitatives(rgb, depth, gt, cos_2d, cos_3d, cos_comb,
                       rgb_path, args):
    defect_class_str = rgb_path[0].split('/')[-3]
    image_name_str = rgb_path[0].split('/')[-1]

    save_path = os.path.join(
        args.qualitative_folder,
        f'{args.class_name}_uni_rcm_h{args.hidden_dim}_L{args.num_blocks}'
        f'_{args.epochs_no}ep_{args.train_batch_size}bs',
        defect_class_str,
    )
    os.makedirs(save_path, exist_ok=True)

    fig, axs = plt.subplots(2, 3, figsize=(10, 7))

    denormalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    rgb_vis = denormalize(rgb)

    axs[0, 0].imshow(rgb_vis.squeeze().permute(1, 2, 0).cpu().detach().numpy())
    axs[0, 0].set_title('RGB')
    axs[0, 1].imshow(gt.squeeze().cpu().detach().numpy())
    axs[0, 1].set_title('Ground-truth')
    axs[0, 2].imshow(depth.squeeze().permute(1, 2, 0).mean(axis=-1).cpu().detach().numpy())
    axs[0, 2].set_title('Depth')

    axs[1, 0].imshow(cos_3d.cpu().detach().numpy(), cmap=plt.cm.jet)
    axs[1, 0].set_title('3D Mapping Error')
    axs[1, 1].imshow(cos_2d.cpu().detach().numpy(), cmap=plt.cm.jet)
    axs[1, 1].set_title('2D Mapping Error')

    cos_comb_vis = cos_comb if cos_comb.dim() == 2 else cos_comb.reshape(224, 224)
    axs[1, 2].imshow(cos_comb_vis.cpu().detach().numpy(), cmap=plt.cm.jet)
    axs[1, 2].set_title('Combined Score')

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, image_name_str), dpi=256)

    if args.visualize_plot:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uni-RCM Inference')

    parser.add_argument('--dataset_path', default='your_dataset_path', type=str)
    parser.add_argument('--class_name', default='bagel', type=str)

    parser.add_argument('--checkpoint_folder', default='./checkpoints/checkpoints_UniRCM', type=str)
    parser.add_argument('--checkpoint_class_name', default='combine', type=str,
                        help='Training class name used in checkpoint folder/file name')

    parser.add_argument('--epochs_no', default=200, type=int)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--num_blocks', default=3, type=int)


    parser.add_argument('--orq_checkpoint',
                        default='./checkpoints/checkpoints_UniRCM_ORQ/combine/UniRCM_ORQ_combine_Krgb4096_Kxyz1024_L4_iter20.pth',
                        type=str,help='Optional ORQ checkpoint from uni_rcm_extract_orq.py')
    parser.add_argument('--num_embeddings_rgb', default=4096, type=int,
                        help='Fallback RGB codebook size if metadata is missing in checkpoint')
    parser.add_argument('--num_embeddings_xyz', default=1024, type=int,
                        help='Fallback XYZ codebook size if metadata is missing in checkpoint')
    parser.add_argument('--vq_layers', default=4, type=int,
                        help='Fallback VQ layers if metadata is missing in checkpoint')
    parser.add_argument('--vq_beta_2d', default=0.05, type=float,
                        help='Weight for ORQ anomaly signal in 2D')
    parser.add_argument('--vq_beta_3d', default=0.10, type=float,
                        help='Weight for ORQ anomaly signal in 3D')

    parser.add_argument('--qualitative_folder', default='./results/qualitatives_uni_rcm', type=str)
    parser.add_argument('--quantitative_folder', default='./results/quantitatives_uni_rcm', type=str)

    parser.add_argument('--visualize_plot', default=False, action='store_true')
    parser.add_argument('--produce_qualitatives', default=False, action='store_true')

    args = parser.parse_args()
    infer_UniRCM(args)
