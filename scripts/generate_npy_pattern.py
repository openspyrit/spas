# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:25:25 2025

@author: admin
"""

from spas.generate import walsh_patterns
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

Np = 2
DMD_minor_size = 768
pattern_dim = "1D"
scan_mode = 'Walsh'
pattern_prefix = scan_mode + '_' + str(Np) + 'x' + str(Np)
pattern_source = '../Patterns/' + pattern_dim + '/' + pattern_prefix
pattern_source_temp = '../Patterns/temp/' + pattern_dim + '/' + pattern_prefix

if os.path.isdir(pattern_source_temp) == False:
            os.mkdir(pattern_source_temp)
            
walsh_patterns(N = Np, save_data = True, path = pattern_source_temp + '/', N_DMD = DMD_minor_size, pattern_dim = pattern_dim)

if os.path.isdir(pattern_source) == False:
            os.mkdir(pattern_source)
            
for im_num in range(Np*2):
# im_num = -1
    image_path = pattern_source_temp + '/' + pattern_prefix + '_' + str(im_num) + '.png'
    image_path_out = pattern_source + '/' + pattern_prefix + '_' + str(im_num) + '.npy'
    
    
    im = Image.open(image_path)
    
    im = Image.open(image_path)
    im_HD = np.array(im,dtype=np.uint8)
    im_new = im_HD[:, 128:1024-128]
    im_Np = cv2.resize(im_new, (Np,Np), interpolation = cv2.INTER_NEAREST)
    im_vec = im_Np.ravel()
    
    np.save(image_path_out, im_vec)
    
# plt.figure()
# plt.imshow(im_new)
# plt.colorbar()