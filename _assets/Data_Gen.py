import os 
import numpy as np 
import cv2 
import argparse 
import random 
 
parser = argparse.ArgumentParser('create image pairs') 
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges') 
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg') 
parser.add_argument('--fold_A_out', dest='fold_A_out', help='output directory of A', type=str, default='../dataset/test_A') 
parser.add_argument('--fold_B_out', dest='fold_B_out', help='output directory of B', type=str, default='../dataset/test_B') 
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000) 
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true') 
args = parser.parse_args() 
 
for arg in vars(args): 
    print('[%s] = ' % arg, getattr(args, arg)) 
 
splits = os.listdir(args.fold_A) 
 
for sp in splits: 
    img_fold_A = os.path.join(args.fold_A, sp) 
    img_fold_B = os.path.join(args.fold_B, sp) 
    img_list = os.listdir(img_fold_A) 
    if args.use_AB: 
        img_list = [img_path for img_path in img_list if '_A.' in img_path] 
 
    num_imgs = min(args.num_imgs, len(img_list)) 
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list))) 
    img_fold_A_out = os.path.join(args.fold_A_out, sp) 
    img_fold_B_out = os.path.join(args.fold_B_out, sp) 
    if not os.path.isdir(img_fold_A_out): 
        os.makedirs(img_fold_A_out) 
    print('split = %s, number of images = %d' % (sp, num_imgs)) 
    if not os.path.isdir(img_fold_B_out): 
        os.makedirs(img_fold_B_out) 
    print('split = %s, number of images = %d' % (sp, num_imgs)) 
     
    for n in range(num_imgs): 
        name_A = img_list[n] 
        path_A = os.path.join(img_fold_A, name_A) 
        if args.use_AB: 
            name_B = name_A.replace('_A.', '_B.') 
        else: 
            name_B = name_A 
        path_B = os.path.join(img_fold_B, name_B) 
        if os.path.isfile(path_A) and os.path.isfile(path_B): 
            name_A_out = name_A 
            name_B_out = name_B 
 
            if args.use_AB: 
                name_AB = name_AB.replace('_A.', '.')  # remove _A 
            path_A_out = os.path.join(img_fold_A_out, name_A_out) 
            path_B_out = os.path.join(img_fold_B_out, name_B_out) 
            im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR 
            #print(im_A.shape) 
            im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR 
            #print(im_B.shape) 
 
             
            for l in range(16):                     # 256p data 16 random crop of 1024^2. 
                i = random.randint(0, 1024-256) 
                j = random.randint(0, 1024-256) 
                x = im_A[i:i+256,j:j+256] 
                y = im_B[i:i+256,j:j+256] 
                #im_xy= np.concatenate([x, y], 1) 
                path_x = "{}_{}.png". format(path_A_out[:-4],l) 
                path_y = "{}_{}.png". format(path_B_out[:-4],l) 
                cv2.imwrite(path_x, x) 
                cv2.imwrite(path_y, y) 
