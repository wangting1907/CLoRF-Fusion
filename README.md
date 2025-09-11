Hyperspectral and multispectral image fusion with arbitrary resolution through self-supervised representations
=====
Implementation
------
Setup
-----
First, prepare your dataset as:

<img width="294" height="50" alt="e72c578c6c42957775f348d01fb67f3c" src="https://github.com/user-attachments/assets/a50720b9-2985-46f3-81f1-bf19583344cd" />

The matlab data include high-resolution hyperpsectral data and spectral response function.

Second, Simulate LR-HSI and HR-MSI, and add nosie:

``` 
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
```

Configure parameters in train_tv_spatial.sh

Run demo
-----
run CLoRF:
```
sh train_tv_spatial.sh > train_pavia_university.out &
```
Citation
-----
If you use our method or our code in your research, please kindly cite it:

```
@article{wang2025hyperspectral,
  title={Hyperspectral and Multispectral Image Fusion with Arbitrary Resolution Through Self-Supervised Representations: T. Wang et al.},
  author={Wang, Ting and Yan, Zipei and Li, Jizhou and Zhao, Xile and Wang, Chao and Ng, Michael},
  journal={International Journal of Computer Vision},
  pages={1--21},
  year={2025},
  publisher={Springer}
}
```
