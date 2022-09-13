import os
import cv2
import math
import numpy as np
from skimage.measure import compare_ssim as ssim
from scipy.ndimage import convolve
import random

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse)) 
    return psnr 

def calculate_accuracy():
    print('Kernel Loss (Weighted L1 + Discrminator Losses')
    fold_A = 'D:/Somani/Dataset/256p/test_B/'
    f_name = []
    images = 0
    for z in range(0,20):
        images += 1
        f_name.append(random.randint(0, 826))
    #print(f_name)
    print(images)

    for i in range(0,4):
        total_psnr = 0
        total = 0
        if i == 0:
            print('DiceBCE')
            fold_B = 'D:/Somani/Model/HD/results/K-3-DiceBCE-50/test_latest/images'
        elif i == 1:
            print('Focal')
            fold_B = 'D:/Somani/Model/HD/results/K-3-focal-50/test_latest/images'
        elif i == 2:
            print('IoU')
            fold_B = 'D:/Somani/Model/HD/results/K-3-iou-50/test_latest/images'
        elif i == 3:
            print('MSE')
            fold_B = 'D:/Somani/Model/HD/results/K-3-lsgan-50/test_latest/images'
        img_list = os.listdir(fold_A)
        num_imgs = len(img_list)
        #x = num_imgs / 2 
        #print(num_imgs)

        #for n in range(1,num_imgs,2):
        
        for n in f_name:
            name_A = img_list[n]
            path_A = os.path.join(fold_A, name_A)
            name_B = name_A
            name_B = name_B.replace('.png', '_synthesized_image.jpg')
            path_B = os.path.join(fold_B, name_B)
            #print(path_A)
            #print(path_B)
            #print("*************")
            weights = [   [5.,35.,55.,35.,5.],
                          [35.,125.,181.,125.,25.],
                          [55.,181.,255.,181.,55.],
                          [35.,125.,181.,125.,25.],
                          [5.,35.,55.,35.,5.]]
            #weights = weights.view(1,1,5,5).repeat(3,1,1,1)
            imgA = cv2.imread(path_A, 1)
            #img1 = convolve(imgA, weights)
            #img1 = ((img1 - np.min(img1))/(np.max(img1) - np.min(img1))) * 255
            #img1 = cv2.resize(img1,(256,256))
            #print(imgA.shape)
            imgB = cv2.imread(path_B, 1)
            #img2 = convolve(imgB, weights)
            #img2 = ((img2 - np.min(img2))/(np.max(img2) - np.min(img2))) * 255
            #print(img2.shape) 

            s = ssim(imgA,imgB, multichannel=True)
            p = PSNR(imgA, imgB)
            total_psnr += p
            total = total + s
        print("SSIM: ",total/images)
        print("PSNR: ",total_psnr / images )

calculate_accuracy()