import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import scipy.io
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from inr import models
from utils import *

from torch.utils.tensorboard import SummaryWriter


def main(args):
    args.datetime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    # fix seed
    set_seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # configure(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)
    save_yaml(args.log_dir, vars(args), 'config.yml')
    print_info(args)

    X, R = load_data(args.data, ['X', 'R'])
    h, w, c = X.shape
    factor = 4 # downsampling factor
    X_torch = torch.from_numpy(X)
    R_torch = torch.from_numpy(R)
    #####MSI and HSI################
    MSI = hwc2chw(hsi2msi(X_torch, R_torch))  # G=XR
    SNRm = args.snr
    [msi, sigmam] = add_noise(MSI, SNRm)
     # Simulate LR-HSI by filtering, downsampling (factor = 4) and adding noise
     # Get PSF for the HSI, gaussian = False -->Starg-Mutargh filter
    psf = get_hs_psf(N=5, n_channels=c, gaussian=True, sigma=1.)
    x = hwc2chw(X_torch).to(device, dtype=torch.float)
    hsi_LR = Ax(x, psf, ratio=factor)  # y=Hx
    SNRh = args.snr
    [hsi, sigmah] = add_noise(hsi_LR, SNRh)
    cond = 1e-3

    print(f'hsi, shape: {hsi.shape}, range: {hsi.min(), hsi.max()}')
    print(f'msi, shape: {msi.shape}, range: {msi.min(), msi.max()}')


    # build INR model, you may choose different INR w/ different params

    # args.hidden_features = 512
    # args.spatial_hidden_layers = 5
    spatial_params = {'nonlin': args.inr, 'in_features': 2, 'out_features': args.k,
                      'hidden_features': args.spatial_hidden_features,
                      'hidden_layers': args.spatial_hidden_layers, 'outermost_linear': True}

    if args.pos_enc:
        spatial_params['pos_encode'] = args.pos_enc

    spatial_model = models.get_INR(**spatial_params)

    print(f'spatial model:\n', spatial_model)

    temporal_params = {'nonlin': args.inr, 'in_features': 1, 'out_features': args.k,
                       'hidden_features': args.temporal_hidden_features,
                       'hidden_layers': args.temporal_hidden_layers, 'outermost_linear': True}

    if args.pos_enc:
        temporal_params['pos_encode'] = args.pos_enc

    temporal_model = models.get_INR(**temporal_params)
    print(f'temporal model:\n', temporal_model)

    spatial_coord = get_coordinate(h, w)  # 2D
    temporal_coord = get_coordinate(c)  # 1D

    if args.cuda:
        R_torch = R_torch.cuda()
        spatial_model = spatial_model.cuda()
        temporal_model = temporal_model.cuda()
        spatial_coord = spatial_coord.cuda()
        temporal_coord = temporal_coord.cuda()
    R_torch = R_torch.to(torch.float32)
    # https://pytorch.org/docs/stable/optim.html
    # Per-parameter options
    optimizer = torch.optim.Adam([{'params': spatial_model.parameters()},
                                  {'params': temporal_model.parameters()}],
                                 lr=args.lr, weight_decay=args.wd)

    # training
    # these two models do not contain sth like BN layers, eval and train model are almost the same
    spatial_model.train()
    temporal_model.train()
    t0 = time.time()
    best_psnr = -np.Inf
    for i in range(1, args.num_iters + 1):
        spatial = spatial_model(spatial_coord)  # [wh,k]
        temporal = temporal_model(temporal_coord)  # [c,k]

        output = spatial @ temporal.T  # [wh, c]

        # back to [0, 1]
        output = output / 2 + 0.5
        output = output.reshape(h, w, c)
        output = output.permute(1, 0, 2)
        """
        output: noise / 
        """
        # Network
        # output = network(output) #
        HP =Ax(hwc2chw(output),psf,factor)
        MP =hwc2chw(hsi2msi(output, R_torch))
        loss = ((HP -hsi) ** 2).mean() + args.lam * ((MP - msi) ** 2).mean()
        loss += tv_loss(spatial.T.view(1, args.k, h, w), args.alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.lr_decay:
            adjust_learning_rate(optimizer, args.lr, i, args.num_iters)

        writer.add_scalar('loss', loss.item(), i)

        if i % args.eval_freq == 0:

            # print(f'iter: [{i}/{args.num_iters}], loss: {loss.item()}')

            # save_checkpoint({'spatial_model': spatial_model.state_dict(),
            #                  'temporal_model': temporal_model.state_dict()}, False, args.log_dir,
            #                 f'checkpoint_iter_{i}.pth.tar')

            output = output.detach().cpu().numpy()
            
            #if args.clip:
            output = np.clip(output, 0., 1.)
            mse = ((output - X) ** 2).mean()
            psnr = peak_signal_noise_ratio(X, output, data_range=1.)
            ssim = structural_similarity(X, output, data_range=1., channel_axis=-1)
            ergas = compute_ergas(X, output, 4)
            sam = compute_sam(X, output)

            psnr_mean, psnr_std = average_psnr(X, output, data_range=1.)
            ssim_mean, ssim_std = average_ssim(X, output, data_range=1.)

            print(f'iter: [{i}/{args.num_iters}], '
                  f'range: [{output.min()}, {output.max()}], '
                  f'loss: {loss.item()}, '
                  f'mse: {mse.item()}, '
                  f'PSNR: {psnr}, '
                  f'SSIM: {ssim}, '
                  f'ERGAS: {ergas}, '
                  f'SAM: {sam}, '
                  f'AVG PSNR: {psnr_mean} ± {psnr_std}, '
                  f'AVG SSIM: {ssim_mean} ± {ssim_std}')
            # log
            writer.add_scalar('mse', mse.item(), i)
            writer.add_scalar('psnr', psnr, i)
            writer.add_scalar('ssim', ssim, i)
            writer.add_scalar('ergas', ergas, i)
            writer.add_scalar('sam', sam, i)

            writer.add_scalar('psnr_mean', psnr_mean, i)
            writer.add_scalar('psnr_std', psnr_std, i)

            writer.add_scalar('ssim_mean', ssim_mean, i)
            writer.add_scalar('ssim_std', ssim_std, i)

            if psnr_mean > best_psnr:
                best_psnr = psnr_mean
                np.save(os.path.join(args.log_dir, 'best_psnr.npy'), best_psnr)
                np.save(os.path.join(args.log_dir, 'best_psnr_output.npy'), output)

    print('time_cost: ', time.time() - t0)

    # save model last
    # save_checkpoint({'spatial_model': spatial_model.state_dict(),
    #                  'temporal_model': temporal_model.state_dict()}, False, args.log_dir,
    #                 f'checkpoint_last.pth.tar')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoising')

    parser.add_argument('--data', type=str,
                        help='data file path')

    parser.add_argument('--snr', type=int, help='SNR parameter for adding noise')

    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                        help='learning rate (default: 3e-5)')

    parser.add_argument('--lr_decay', action='store_true',
                        help='cosine learning rate decay')

    parser.add_argument('--wd', type=float, default=0., metavar='WD',
                        help='weight decay (default: 0.0)')

    parser.add_argument('--num_iters', type=int, default=10000, metavar='NUM_ITERS',
                        help='training iterations (default: 1000)')

    parser.add_argument('--clip', action='store_true',
                        help='clip data range in [0., 1.]')

    # parser.add_argument('--save_denoised', action='store_true',
    #                     help='save denoised results')

    parser.add_argument('--cuda', action='store_true',
                        help='use cuda')

    parser.add_argument('--log_dir', default='./runs', type=str, metavar='PATH',
                        help='where checkpoints and logs to be saved (default: ./runs)')
    parser.add_argument('--eval_freq', default=100, type=int, metavar='N',
                        help='evaluation frequency (default: 100)')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (defualt: 42)')

    parser.add_argument('--inr', type=str, help='inr model')
    parser.add_argument('--pos_enc', action='store_true',
                        help='position encoding for ReLU')

    parser.add_argument('--spatial_hidden_features', type=int, default=512, help='hidden_features (default: 512)')
    parser.add_argument('--spatial_hidden_layers', type=int, default=5, help='hidden layers for spatial model')
    parser.add_argument('--temporal_hidden_features', type=int, default=128, help='hidden_features (default: 12)')
    parser.add_argument('--temporal_hidden_layers', type=int, default=6, help='hidden layers for temporal model')

    # parser.add_argument('--non_negative', action='store_true',
    #                     help='force spatial to be non-negative')

    parser.add_argument('--k', type=int, help='num of rank')
    parser.add_argument('--lam', type=float, help='lambda hyper-parameter')
    parser.add_argument('--alpha', type=float, help='alpha hyper-parameter')
    args = parser.parse_args()

    main(args)
