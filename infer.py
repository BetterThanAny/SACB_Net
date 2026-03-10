"""
Inference script for SACB-Net.

Loads a trained checkpoint and evaluates on the test/validation set.
Computes Dice scores and Jacobian determinant statistics.

Usage:
    python infer.py --checkpoint experiments/xxx.pth.tar --dataset ixi
    python infer.py --checkpoint experiments/xxx.pth.tar --dataset lpba --save_results
    python infer.py --checkpoint experiments/xxx.pth.tar --dataset abd --data_dir /path/to/data
"""

import os
import argparse
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from model import SACB_Net
from dataset import datasets, trans


def get_args():
    parser = argparse.ArgumentParser(description='SACB-Net Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth.tar)')
    parser.add_argument('--dataset', type=str, default='ixi',
                        choices=['ixi', 'lpba', 'abd'],
                        help='Dataset to evaluate on (default: ixi)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Base data directory (overrides base_dir env var)')
    parser.add_argument('--k', type=int, default=7,
                        help='num_k parameter for SACB_Net (default: 7)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--save_dir', type=str, default='results/',
                        help='Directory to save inference results (default: results/)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save warped images and flow fields as .npz')
    return parser.parse_args()


def compute_jacobian_stats(flow):
    """Compute Jacobian determinant statistics from a flow field.

    Args:
        flow: numpy array of shape (3, D, H, W)

    Returns:
        dict with Jacobian determinant statistics
    """
    jac_det = utils.jacobian_determinant_vxm(flow)
    num_neg = np.sum(jac_det <= 0)
    total = np.prod(jac_det.shape)
    return {
        'percent_neg_jac': num_neg / total * 100,
        'num_neg': int(num_neg),
        'total': int(total),
        'mean_jac': float(np.mean(jac_det)),
        'std_jac': float(np.std(jac_det)),
    }


def main():
    args = get_args()

    # Set GPU
    torch.cuda.set_device(args.gpu)
    print(f'Using GPU: {torch.cuda.get_device_name(args.gpu)}')

    # Set data directory
    if args.data_dir:
        os.environ['base_dir'] = args.data_dir
    elif 'base_dir' not in os.environ:
        os.environ['base_dir'] = '/bask/projects/d/duanj-ai-imaging/xxc/dataset_all'

    # Dataset config
    if args.dataset == 'ixi':
        atlas_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/atlas.pkl')
        val_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/Val/')
        val_composed = transforms.Compose([
            trans.Seg_norm(),
            trans.NumpyType((np.float32, np.int16))
        ])
        val_set = datasets.IXIBrainInferDataset(
            glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
        dice_score = utils.dice_val_VOI
        img_size = (160, 192, 224)
    elif args.dataset == 'lpba':
        val_dir = os.path.join(os.getenv('base_dir'), 'LPBA_data_2/Val/')
        val_composed = transforms.Compose([
            trans.Seg_norm2(),
            trans.NumpyType((np.float32, np.int16))
        ])
        val_set = datasets.LPBABrainInferDatasetS2S(
            sorted(glob.glob(val_dir + '*.pkl')), transforms=val_composed)
        dice_score = utils.dice_LPBA
        img_size = (160, 192, 160)
    elif args.dataset == 'abd':
        val_dir = os.path.join(os.getenv('base_dir'), 'AbdomenCTCT/Val/')
        val_composed = transforms.Compose([
            trans.NumpyType((np.float32, np.int16))
        ])
        val_set = datasets.LPBABrainInferDatasetS2S(
            sorted(glob.glob(val_dir + '*.pkl')), transforms=val_composed)
        dice_score = utils.dice_abdo
        img_size = (192, 160, 224)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Initialize model
    model = SACB_Net(inshape=img_size, num_k=args.k)
    model.set_k(args.k)
    model.cuda()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{args.gpu}')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f'Checkpoint loaded: {args.checkpoint}')

    # Spatial transformer for warping segmentations (nearest interpolation)
    reg_model = utils.SpatialTransformer(size=img_size, mode='nearest').cuda()

    # Create save directory
    if args.save_results:
        os.makedirs(args.save_dir, exist_ok=True)

    # Inference
    dice_scores = []
    jac_stats_list = []

    print(f'\nEvaluating on {args.dataset} dataset ({len(val_set)} pairs)...\n')

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            data = [t.cuda() for t in data]
            x = data[0]      # moving image
            y = data[1]      # fixed image
            x_seg = data[2]  # moving segmentation
            y_seg = data[3]  # fixed segmentation

            # Forward pass
            x_warped, flow = model(x, y)

            # Warp segmentation with nearest interpolation
            def_out = reg_model(x_seg.float(), flow)

            # Compute Dice
            dsc = dice_score(def_out.long(), y_seg.long())
            dice_scores.append(dsc)

            # Compute Jacobian determinant stats
            flow_np = flow.detach().cpu().numpy()[0]
            jac_info = compute_jacobian_stats(flow_np)
            jac_stats_list.append(jac_info)

            print(f'[{idx + 1:3d}/{len(val_loader)}] '
                  f'Dice: {dsc:.4f}  |  '
                  f'%|J|<=0: {jac_info["percent_neg_jac"]:.3f}%')

            # Save results
            if args.save_results:
                np.savez_compressed(
                    os.path.join(args.save_dir, f'pair_{idx:04d}.npz'),
                    moving=x.cpu().numpy()[0, 0],
                    fixed=y.cpu().numpy()[0, 0],
                    warped=x_warped.cpu().numpy()[0, 0],
                    flow=flow_np,
                    moving_seg=x_seg.cpu().numpy()[0, 0],
                    fixed_seg=y_seg.cpu().numpy()[0, 0],
                    warped_seg=def_out.cpu().numpy()[0, 0],
                )

    # Summary
    dice_arr = np.array(dice_scores)
    jac_neg_arr = np.array([s['percent_neg_jac'] for s in jac_stats_list])

    print('\n' + '=' * 60)
    print(f'Dataset:    {args.dataset}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Num pairs:  {len(dice_scores)}')
    print('-' * 60)
    print(f'Dice:       {dice_arr.mean():.4f} +/- {dice_arr.std():.4f}')
    print(f'%|J|<=0:    {jac_neg_arr.mean():.4f} +/- {jac_neg_arr.std():.4f}')
    print('=' * 60)

    if args.save_results:
        print(f'\nResults saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
