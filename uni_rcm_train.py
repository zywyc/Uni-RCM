import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import swanlab as swl
import numpy as np

from tqdm import tqdm, trange

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.rcm_nets import UniRCM_Net, NUM_PATCHES


RGB_DIM = 768
XYZ_DIM = 1152


class UniRCM_Model(nn.Module):
    def __init__(self, rgb_dim=RGB_DIM, xyz_dim=XYZ_DIM,
                 num_patches=NUM_PATCHES, hidden_dim=512, num_blocks=3):
        super().__init__()

        self.net_2d3d = UniRCM_Net(rgb_dim, xyz_dim, num_patches, hidden_dim, num_blocks)
        self.net_3d2d = UniRCM_Net(xyz_dim, rgb_dim, num_patches, hidden_dim, num_blocks)

    def forward(self, rgb_patch, xyz_patch, xyz_mask=None):
        xyz_pred = self.net_2d3d(rgb_patch)
        rgb_pred = self.net_3d2d(xyz_patch)

        if xyz_mask is None:
            xyz_mask = (xyz_patch.sum(dim=-1) == 0)
        valid = ~xyz_mask

        cos_fn = nn.CosineSimilarity(dim=-1, eps=1e-06)
        cos_xyz = 1.0 - cos_fn(xyz_pred[valid], xyz_patch[valid]).mean()
        cos_rgb = 1.0 - cos_fn(rgb_pred[valid], rgb_patch[valid]).mean()

        mse_xyz = F.mse_loss(xyz_pred[valid], xyz_patch[valid])
        mse_rgb = F.mse_loss(rgb_pred[valid], rgb_patch[valid])

        losses = {
            'cos_loss': cos_xyz + cos_rgb,
            'mse_loss': mse_xyz + mse_rgb,
            'cos_xyz': cos_xyz,
            'cos_rgb': cos_rgb,
        }
        preds = {'xyz_pred': xyz_pred, 'rgb_pred': rgb_pred}
        return losses, preds


def set_seeds(sid=115):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def train_UniRCM(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = (
        f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'
        f'_h{args.hidden_dim}_L{args.num_blocks}'
    )

    swl.login(api_key="your_api_key")  # Replace with your actual API key
    swl.init(
        project='Uni-RCM',
        experiment_name=model_name,
        config=vars(args),
        #mode = 'disabled',
    )

    train_loader = get_data_loader(
        "train", class_name=args.class_name, img_size=224,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size, shuffle=True,
    )

    feature_extractor = MultimodalFeatures().to(device)
    for p in feature_extractor.parameters():
        p.requires_grad = False

    model = UniRCM_Model(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    directory = os.path.join(args.checkpoint_savepath, args.class_name)
    os.makedirs(directory, exist_ok=True)


    for epoch in trange(args.epochs_no, desc='Training'):
        model.train()
        epoch_losses = []
        epoch_cos_sim_xyz = []
        epoch_cos_sim_rgb = []

        for (rgb, pc, _), _ in tqdm(train_loader, leave=False):
            rgb, pc = rgb.to(device), pc.to(device)

            with torch.no_grad():
                if args.batch_size == 1:
                    rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)
                    rgb_patch = rgb_patch.unsqueeze(0)
                    xyz_patch = xyz_patch.unsqueeze(0)
                else:
                    rps, xps = [], []
                    for i in range(rgb.shape[0]):
                        rp, xp = feature_extractor.get_features_maps(
                            rgb[i].unsqueeze(0), pc[i].unsqueeze(0))
                        rps.append(rp)
                        xps.append(xp)
                    rgb_patch = torch.stack(rps)
                    xyz_patch = torch.stack(xps)

            losses, _ = model(rgb_patch, xyz_patch)

            loss = losses['cos_loss'] + args.mse_weight * losses['mse_loss']
            batch_cos_sim_xyz = 1.0 - losses['cos_xyz'].item()
            batch_cos_sim_rgb = 1.0 - losses['cos_rgb'].item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            swl.log({
                "train/loss": loss.item(),
                "train/cos_loss": losses['cos_loss'].item(),
                "train/mse_loss": losses['mse_loss'].item(),
                "train/cos_sim_xyz": batch_cos_sim_xyz,
                "train/cos_sim_rgb": batch_cos_sim_rgb,
            })

            epoch_losses.append(loss.item())
            epoch_cos_sim_xyz.append(batch_cos_sim_xyz)
            epoch_cos_sim_rgb.append(batch_cos_sim_rgb)

        avg_loss = np.mean(epoch_losses)

        swl.log({
            "epoch/avg_loss": avg_loss,
            "epoch/cos_sim_xyz": np.mean(epoch_cos_sim_xyz),
            "epoch/cos_sim_rgb": np.mean(epoch_cos_sim_rgb),
        })

        swl.log({"global/loss": avg_loss})

        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(directory, f'UniRCM_{model_name}.pth')
            torch.save(model.state_dict(), ckpt_path)

    ckpt_path = os.path.join(directory, f'UniRCM_{model_name}.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint: {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uni-RCM Training')
    parser.add_argument('--dataset_path',
                        default='your_dataset_path', type=str)
    parser.add_argument('--checkpoint_savepath',
                        default='./checkpoints/checkpoints_UniRCM', type=str)
    parser.add_argument('--class_name', default='combine', type=str)
    parser.add_argument('--epochs_no', default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--num_blocks', default=3, type=int)
    parser.add_argument('--mse_weight', default=0.01, type=float)

    args = parser.parse_args()
    train_UniRCM(args)
