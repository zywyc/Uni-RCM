import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dataset import get_data_loader
from models.features import MultimodalFeatures
from models.quantizer import build_quantizer

RGB_DIM = 768
XYZ_DIM = 1152


def set_seeds(seed=115):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def extract_train_features(feature_extractor, train_loader, device, batch_size):
    all_rgb_patches = []
    all_xyz_patches = []

    with torch.no_grad():
        for (rgb, pc, _), _ in tqdm(train_loader, desc='Extracting features'):
            rgb = rgb.to(device)
            pc = pc.to(device)

            if batch_size == 1:
                rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)
            else:
                rgb_patches = []
                xyz_patches = []
                for idx in range(rgb.shape[0]):
                    rgb_single, xyz_single = feature_extractor.get_features_maps(
                        rgb[idx].unsqueeze(0), pc[idx].unsqueeze(0)
                    )
                    rgb_patches.append(rgb_single)
                    xyz_patches.append(xyz_single)
                rgb_patch = torch.stack(rgb_patches, dim=0)
                xyz_patch = torch.stack(xyz_patches, dim=0)

            xyz_mask = xyz_patch.sum(dim=-1) == 0
            valid_mask = ~xyz_mask

            all_rgb_patches.append(rgb_patch[valid_mask].cpu())
            all_xyz_patches.append(xyz_patch[valid_mask].cpu())

    rgb_features = torch.cat(all_rgb_patches, dim=0)
    xyz_features = torch.cat(all_xyz_patches, dim=0)
    return rgb_features, xyz_features


def extract_orq(args):
    set_seeds()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = get_data_loader(
        'train',
        class_name=args.class_name,
        img_size=224,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        shuffle=False,
    )

    feature_extractor = MultimodalFeatures().to(device)
    for param in feature_extractor.parameters():
        param.requires_grad = False

    rgb_orq = build_quantizer(num_embeddings=args.num_embeddings_rgb, embedding_dim=RGB_DIM, vq_layers=args.vq_layers,).to(device)
    xyz_orq = build_quantizer(num_embeddings=args.num_embeddings_xyz, embedding_dim=XYZ_DIM, vq_layers=args.vq_layers,).to(device)

    print('[ORQ] Extracting frozen features from training split...')
    all_rgb_features, all_xyz_features = extract_train_features(feature_extractor, train_loader, device, args.batch_size)
    print(f'[ORQ] Valid RGB patches: {all_rgb_features.shape[0]}')
    print(f'[ORQ] Valid XYZ patches: {all_xyz_features.shape[0]}')

    print(f'[ORQ] Running K-Means for XYZ ORQ '
          f'(K={args.num_embeddings_xyz}, layers={args.vq_layers}, iters={args.orq_epochs})...')
    xyz_orq.train_kmeans(all_xyz_features, n_iter=args.orq_epochs)

    print(
        f'[ORQ] Running K-Means for RGB ORQ '
        f'(K={args.num_embeddings_rgb}, layers={args.vq_layers}, iters={args.orq_epochs})...')
    rgb_orq.train_kmeans(all_rgb_features, n_iter=args.orq_epochs)

    save_dir = os.path.join(args.checkpoint_savepath, args.class_name)
    os.makedirs(save_dir, exist_ok=True)

    suffix = f'_L{args.vq_layers}' if args.vq_layers > 1 else ''
    checkpoint_name = (
        f'UniRCM_ORQ_{args.class_name}_Krgb{args.num_embeddings_rgb}_Kxyz{args.num_embeddings_xyz}{suffix}'
        f'_iter{args.orq_epochs}.pth'
    )
    checkpoint_path = os.path.join(save_dir, checkpoint_name)

    checkpoint = {
        'class_name': args.class_name,
        'dataset_path': args.dataset_path,
        'num_embeddings_rgb': args.num_embeddings_rgb,
        'num_embeddings_xyz': args.num_embeddings_xyz,
        'vq_layers': args.vq_layers,
        'orq_epochs': args.orq_epochs,
        'rgb_dim': RGB_DIM,
        'xyz_dim': XYZ_DIM,
        'rgb_codebook': rgb_orq.state_dict(),
        'xyz_codebook': xyz_orq.state_dict(),
        'rgb_utilization': rgb_orq.get_utilization(),
        'xyz_utilization': xyz_orq.get_utilization(),
        'num_rgb_features': int(all_rgb_features.shape[0]),
        'num_xyz_features': int(all_xyz_features.shape[0]),
    }
    torch.save(checkpoint, checkpoint_path)

    print(f'[ORQ] Saved checkpoint: {checkpoint_path}')
    print(f'[ORQ] RGB utilization: {checkpoint["rgb_utilization"]:.3f}')
    print(f'[ORQ] XYZ utilization: {checkpoint["xyz_utilization"]:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract standalone RGB/XYZ ORQ codebooks for Uni-RCM inference.'
    )
    parser.add_argument(
        '--dataset_path',
        default='your_dataset_path',
        type=str,
        help='Training dataset root used to build ORQ.',
    )
    parser.add_argument(
        '--checkpoint_savepath',
        default='./checkpoints/checkpoints_UniRCM_ORQ',
        type=str,
        help='Directory where the ORQ checkpoint will be written.',
    )
    parser.add_argument(
        '--class_name',
        default='combine',
        type=str,
        help='Category name or combine for one-for-all ORQ extraction.',
    )
    
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for feature extraction.')
    parser.add_argument('--num_embeddings_rgb', default=4096, type=int, help='RGB ORQ size K.')
    parser.add_argument('--num_embeddings_xyz', default=1024, type=int, help='XYZ ORQ size K.')
    parser.add_argument('--vq_layers', default=4, type=int, help='1 for VQ, >1 for residual VQ.')
    parser.add_argument('--orq_epochs', default=20, type=int, help='K-Means iterations for ORQ.')

    extract_orq(parser.parse_args())
