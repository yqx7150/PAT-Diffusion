import scipy.io as io
import os
from skimage.measure import compare_psnr,compare_ssim
import cv2
import matplotlib.pyplot as plt

for iii in range(25):

    # aaa_path = str(iii)+'-000000.png.mat'
    # aaa = io.loadmat(os.path.join('/home/lqg/桌面/sde-hank-cut-aloha-new-patch/results',aaa_path))['data']

    # bbb_path = 'ori'+str(iii)+'-000000.png.mat'
    # bbb = io.loadmat(os.path.join('/home/lqg/桌面/sde-hank-cut-aloha-new-patch/results',bbb_path))['data']

    aaa = io.loadmat(os.path.join('/home/lqg/桌面/sde-hank-cut-aloha-new-patch/results/Rec','000007.png.mat'))['data']
    bbb = cv2.imread('/home/lqg/桌面/sde-hank-cut-aloha-new-patch/lzdata/test_bedroom_256_ori/000007.png')
    bbb = bbb / 255.
    
    print(aaa.shape,aaa.dtype,aaa.max(),aaa.min())
    print(bbb.shape,bbb.dtype,bbb.max(),bbb.min())
  

    psnr = compare_psnr(255.* aaa, 255. * bbb, data_range=255)
    ssim = compare_ssim(aaa, bbb, data_range=1,multichannel=True)
    print(' PSNR:', psnr,' SSIM:', ssim)
