import os
import random
import shutil
import sys

import PIL
import math
import numpy as np
import scipy.io
import torch
import torchvision
import yaml
from PIL import Image
from torch.fft import fft2 as fft2
from torch.fft import ifft2 as ifft2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forward_difference_3(U):
    # Calculate U along the x direction finite difference
    Duy = U[:, 1:, :] - U[:, :-1, :]
    Duy = torch.cat((Duy, U[:, 0:1, :] - U[:, -1:, :]), dim=1) 

    # Calculate U along the y direction finite difference
    Dux = U[1:, :, :] - U[:-1, :, :]
    Dux = torch.cat((Dux, U[0:1, :, :] - U[-1:, :, :]), dim=0) 

    # Calculate U along the z direction finite difference
    Duz = U[:, :, 1:] - U[:, :, :-1]
    Duz = torch.cat((Duz, U[:, :, 0:1] - U[:, :, -1:]), dim=2) 

    return Dux, Duy, Duz
def average_psnr(X_true, X_pred, data_range=1.):
    assert X_true.ndim == X_pred.ndim == 3
    # channel last
    psnr_list = []
    for i in range(X_true.shape[-1]):
        psnr_list.append(peak_signal_noise_ratio(X_true[:, :, i], X_pred[:, :, i], data_range=data_range))
    psnr_list = np.asarray(psnr_list)
    return psnr_list.mean(), psnr_list.std()


def average_ssim(X_true, X_pred, data_range=1.):
    assert X_true.ndim == X_pred.ndim == 3
    # channel last
    ssim_list = []
    for i in range(X_true.shape[-1]):
        # ssim = structural_similarity(X, output, data_range=1., channel_axis=-1)
        ssim_list.append(structural_similarity(X_true[:, :, i], X_pred[:, :, i], data_range=data_range))
    ssim_list = np.asarray(ssim_list)
    return ssim_list.mean(), ssim_list.std()


def tv_loss(x, mul_factor):  # anisotropic
    '''Calculates TV loss for an image `x`.'''

    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :h_x - 1, :])).sum()
    w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :w_x - 1])).sum()
    return mul_factor * 2 * (h_tv / count_h + w_tv / count_w) / batch_size



def tv_loss_tem(x, mul_factor):  # anisotropic
    '''Calculates TV loss for an image `x`.'''

    def _tensor_size(t):
        return t.size()[0] * t.size()[1] 
    h_x = x.size()[0]
    w_x = x.size()[1]
    count_h = _tensor_size(x[1:, :])
    h_tv = torch.abs((x[1:, :] - x[:h_x-1, :])).sum()
    return mul_factor * 2 * (h_tv / count_h) 


def tv_loss_isotropic(x, beta=0.5):  # unstable
    '''Calculates TV loss for an image `x`.

    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)

    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))


def compute_sam(x_true, x_pred):
    assert x_true.ndim == 3 and x_pred.ndim == 3 and x_true.shape == x_pred.shape
    c = x_true.shape[-1]
    x_true = x_true.reshape(-1, c)
    x_pred = x_pred.reshape(-1, c)
    sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1) + 1e-7)
    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()
    return mSAM
def compute_sam_torch_tmp(x_true, x_pred):
    w,c=x_true.size()
    sam = torch.sum(x_true * x_pred, dim=1) / (torch.norm(x_true, p=2, dim=1) * torch.norm(x_pred, p=2, dim=1) + 1e-7)
    sam = torch.acos(sam) * 180 / np.pi
    mSAM = sam.mean()
    return mSAM.item() 


def compute_sam_torch(x_true, x_pred):
    sam = torch.sum(x_true * x_pred, dim=2) / (torch.norm(x_true, p=2, dim=2) * torch.norm(x_pred, p=2, dim=2) + 1e-7)
    [h,w,c]=x_true.size()
    sam=h*w*c-sam
    mSAM = sam.mean()
    return mSAM.item() 

def compute_ergas(x_true, x_pred, scale_factor):
    assert x_true.ndim == 3 and x_pred.ndim == 3 and x_true.shape == x_pred.shape
    c = x_true.shape[-1]
    err = x_true - x_pred
    ERGAS = 0
    for i in range(c):
        ERGAS = ERGAS + np.mean(err[:, :, i] ** 2 / np.mean(x_true[:, :, i]) ** 2)
    ERGAS = (100 / scale_factor) * np.sqrt((1 / c) * ERGAS)
    return ERGAS


def adjust_learning_rate(optimizer, lr, epoch, epochs):
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.5 * lr * (1 + math.cos(math.pi * epoch / epochs))


def get_coordinate(*length):
    dim = len(length)
    if dim == 2:
        h, w = length
        # max_len = max(h, w)
        max_len = max(h, w)  # n
        grids = torch.linspace(-1, 1, steps=max_len)  # [n]
        # stack([n, n], [n, n]) -> [n, n, 2]
        coords = torch.stack(torch.meshgrid(grids, grids), dim=-1)

        if h >= w:
            minor = int((h - w) / 2)
            coords = coords[:, minor:w + minor]
        else:
            minor = int((w - h) / 2)
            coords = coords[minor:h + minor, :]

        coords = coords.reshape(-1, dim)

    elif dim == 1:
        coords = torch.linspace(-1, 1, steps=length[0]).view(-1, 1)
    else:
        raise NotImplementedError(f"{dim}-D coordinates are not supported!")
    return coords


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_path = os.path.join(save_dir, filename)
    torch.save(state, file_path)
    if is_best:
        best_file_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(file_path, best_file_path)


def save_images(data: np.ndarray, idx: list, prefix: str, save_dir: str):
    assert data.ndim == 3

    for i in idx:
        img = Image.fromarray(data[:, :, i])
        # only support grayscale image
        img = img.convert('L')
        img.save(os.path.join(save_dir, f'{prefix}_{i}.png'))


def load_data(data_path, keys):
    mat = scipy.io.loadmat(data_path)

    data = []
    for k in keys:
        data.append(mat[k])
        print(f'{k}.shape: {mat[k].shape}, type: {type(mat[k])}')
    return data


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_info(args):
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))


def save_yaml(dir, args, save_name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, save_name), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

def hwc2chw(x): 
    return(x.permute(2,0,1))
def chw2hwc(x):
    return(x.permute(1,2,0))
def get_filter():
    h = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
    return torch.from_numpy(h/np.sum(h))
def gaussian_filter(N=15, sigma=2.0):
    n = (N - 1) / 2.0
    y, x = np.ogrid[-n:n + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return torch.from_numpy(h)

def add_noise(x,SNRdb=40.):
    '''Add noise SNRdb to clean image x (CHW)'''
    x=x.to(device)
    SNR = 10**(SNRdb/10.)
    sigma = torch.sqrt(torch.mean(x**2)/SNR)
    sigma = sigma.to(device)
    xnoise = x + sigma*torch.randn(x.shape).to(device)
    return xnoise,sigma

def Ax(img,psf,ratio):
    '''Filtering and downsampling by A
    input: image (c x m x n); psf: PSF (); ratio: downsampling ratio'''
    c,m,n = img.shape
    x = img.to(device)
    h = expand_shift_psf(psf,c,m,n).to(device)
    img = torch.real(ifft2(fft2(x)*fft2(h))) #filtering
    imgd = img[:,::ratio,::ratio] #downsampling
    return imgd

def get_hs_psf(N=7,n_channels=93, gaussian=True,sigma=2.):
    '''Assume all HS bands have the same psf
    which is Gaussian with sigma=1.'''
    psf = torch.zeros(n_channels,N,N)
    if gaussian:
        for i in range(n_channels):
            psf[i,:,:]=gaussian_filter(N=N,sigma=sigma)
    else:
        for i in range(n_channels):
            psf[i,:,:]=get_filter()
    return psf

def hsi2msi(X,R):
    """Convert HSI X to MSI M by multiplying with spectral response R: M=X*R
    X: (HxWxNh), R(NhxNm)"""
    [r,c,b]=X.size()
    x=im2mat(X)
    if R.shape[0]!=b:
        R=torch.transpose(R,0,1)
    xout = torch.mm(x,R)
    return mat2im(xout,r)

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def im2mat(X):
    """X(r,c,b)-->X(r*c,b)"""
    mat=reshape_fortran(X,(X.shape[0]*X.shape[1],X.shape[2]))
    return mat

def mat2im(X,r):
    """X(r*c,b)-->X(r*c,b)"""
    c=int(X.shape[0]/r)
    b=X.shape[1]
    return reshape_fortran(X,(r,c,b))


def expand_shift_psf(psf,c,m,n):
    '''pad psf to (mxn) with zeros, move the psf to the pad-center and shift--> reduce the boundary effect after convolution
    input: PSF (small size)
    output: PSF expand and shift to (c x m x n)'''
    midm=m//2
    midn=n//2
    if len(psf.shape)<3:
        midx=psf.shape[0]//2
        midy=psf.shape[1]//2
        y = torch.zeros(c, m, n)
    else:
        midx = psf.shape[1] // 2
        midy = psf.shape[2] // 2
        y = torch.zeros(psf.shape[0], m, n)
    y[:, midm - midx :midm + midx+1, midn - midy:midn + midy+1] = psf
    return torch.fft.fftshift(y,dim=[1,2])
