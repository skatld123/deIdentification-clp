import os
import cv2
import math
import numpy as np
import fnmatch

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

def psnr(img1, img2):
    image1 = img1.astype(np.float64)
    image2 = img2.astype(np.float64)
    mse = np.mean((image1 - image2)**2)
    if mse == 0:
        return 0, 0
    else:
        return 10 * math.log10(255.0 / math.sqrt(mse)), mse

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def read_file_real_A_B(dir):
    patterns_A_B = ["*real_B.*", "*fake_B.*"]
    imgs_A_B = []
    img_list = os.listdir(dir)
    for pattern_a_b in patterns_A_B:
        for filename in fnmatch.filter(img_list, pattern_a_b):
            imgs_A_B.append(filename)
    return imgs_A_B

def calculate_psnr_ssim(list, dir):
    list_A = []
    list_B = []
    for element in list:
        if "_real_B.png" in element:
            list_A.append(element)
        elif "_fake_B.png" in element:
            list_B.append(element)

    total_psnr = 0
    total_ssim = 0
    total_mse = 0
    for real_a in list_A:
        real_a_name = real_a[:-11]
        for real_b in list_B:
            real_b_name = real_b[:-11]
            if real_a_name == real_b_name:
                img_real_a = np.fromfile(os.path.join(dir, real_a), np.uint8)
                img_real_a = cv2.imdecode(img_real_a, cv2.IMREAD_COLOR)
                img_real_b = np.fromfile(os.path.join(dir, real_b), np.uint8)
                img_real_b = cv2.imdecode(img_real_b, cv2.IMREAD_COLOR)

                psnr_score, mse_score = psnr(img_real_a, img_real_b)
                total_mse += mse_score
                total_psnr += psnr_score
                total_ssim += calculate_ssim(img_real_a, img_real_b)
    assert len(list_A) == len(list_B)

    return total_psnr/len(list_A), total_ssim/len(list_A), total_mse/len(list_A)

dir_path = '/root/deid-lp-GAN/results/license-plate_cyclegan_vanilla_v2/test_latest/images'
a_b_list = read_file_real_A_B(os.path.join(current_dir, dir_path))
avg_psnr, avg_ssim, avg_mse = calculate_psnr_ssim(a_b_list, os.path.join(current_dir, dir_path))
print(f'average PSNR : {avg_psnr}, average SSIM : {avg_ssim}, average MSE : {avg_mse}')