import os
import numpy as np
import cv2
import argparse
import random

import matplotlib.pyplot as plt
from scipy.stats import itemfreq

fold_A = "D:/Somani/D/PNG/test_A/"
fold_B = "D:/Somani/D/PNG/test_B/"
fold_A1_out = "D:/Somani/D/PNG/Crop/A1/"
fold_B1_out = "D:/Somani/D/PNG/Crop/B1/"
fold_A2_out = "D:/Somani/D/PNG/Crop/A2/"
fold_B2_out = "D:/Somani/D/PNG/Crop/B2/"

splits = os.listdir(fold_A)

for sp in splits:
    img_fold_A = os.path.join(fold_A, sp)
    img_fold_B = os.path.join(fold_B, sp)
    img_list = os.listdir(img_fold_A)

    num_imgs = len(img_list)
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_A1_out = os.path.join(fold_A1_out, sp)
    img_fold_B1_out = os.path.join(fold_B1_out, sp)
    img_fold_A2_out = os.path.join(fold_A2_out, sp)
    img_fold_B2_out = os.path.join(fold_B2_out, sp)

    
    if not os.path.isdir(img_fold_A1_out):
        os.makedirs(img_fold_A1_out)
    if not os.path.isdir(img_fold_B1_out):
        os.makedirs(img_fold_B1_out) 
    
    if not os.path.isdir(img_fold_A2_out):
        os.makedirs(img_fold_A2_out)
    if not os.path.isdir(img_fold_B2_out):
        os.makedirs(img_fold_B2_out) 

    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        # if args.use_AB:
        #     name_B = name_A.replace('_A.', '_B.')
        # else:
        name_B = name_A
                
        path_B = os.path.join(img_fold_B, name_B)
        
        min_value, std_value = 0 , 0
        k = 1
        ## k = 1 => 68%
        ## k = 2 => 95%
        ## k = 3 => 99.7%

        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_A1_out = name_A
            name_B1_out = name_B
            
            name_A2_out = name_A
            name_B2_out = name_B

            # if args.use_AB:
            #     name_AB = name_AB.replace('_A.', '.')  # remove _A
            path_A1_out = os.path.join(img_fold_A1_out, name_A)
            path_B1_out = os.path.join(img_fold_B1_out, name_B)
            path_A2_out = os.path.join(img_fold_A2_out, name_A)
            path_B2_out = os.path.join(img_fold_B2_out, name_B)

            im_A = cv2.imread(path_A, 0)
            im_B = cv2.imread(path_B, 0)
            #print(im_B.shape)
            
            mean_value, std_value = im_B.mean(), np.std(im_B)
            min_value = mean_value + k * std_value
            print("Mean Value:{} & std_value:{}",mean_value,std_value)
           
            for l in range(25): 
                li, perc = 0,0                    # 256p data 25 parts of 1024^2.
                i = random.randint(0, 1024-256)
                j = random.randint(0, 1024-256)
                x = im_A[i:i+256,j:j+256]
                y = im_B[i:i+256,j:j+256]
                temp = y
                li = np.where(temp >= min_value)
                perc = len(li[0]) / (256 * 256)
                print(len(li[0]))

                if (perc >= 0.14): 
                  path_x = "{}_{}.png". format(path_A1_out[:-4],l)
                  cv2.imwrite(path_x, x)
                  path_y = "{}_{}.png". format(path_B1_out[:-4],l)
                  cv2.imwrite(path_y, y)

                elif (perc >= 0.11 and perc <0.14):
                  path_x = "{}_{}.png". format(path_A2_out[:-4],l)
                  cv2.imwrite(path_x, x)
                  path_y = "{}_{}.png". format(path_B2_out[:-4],l)
                  cv2.imwrite(path_y, y)

                else:
                  continue