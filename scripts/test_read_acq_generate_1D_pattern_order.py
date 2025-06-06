# -*- coding: utf-8 -*-
"""
Created on Tue May 27 11:24:57 2025

@author: admin
"""

import numpy as np
from matplotlib import pyplot as plt
# import matplotlib.image as mpimg




for i in range(128):#da.shape[3]):
    data_path = '../../data/2025-06-03_test/obj_cat_source_white_LED_Walsh_im_64x64_ti_8ms_zoom_x1/np_data_' + str(i) + '.npz'
    d = np.load(data_path)
    da = d['arr_0']
    if i <= 5 or i >= 120:
        img16 = da
        img8 = (img16/256).astype('uint8')
        plt.figure()
        plt.imshow(img8)
        plt.title(i)


import pickle
for i in range(256):#da.shape[3]):
    data_path = '../../data/2025-06-03_test/obj_cat8_source_white_LED_Walsh_im_128x128_ti_10ms_zoom_x1/spectral_' + str(i) + '.pkl'
    with open(data_path, "rb") as fp:
        da = pickle.load(fp)
        
    if i <= 5 or (i >= 120 and i < 128) or i > 250:
        img16 = da
        img8 = (img16/256).astype('uint8')
        plt.figure()
        plt.imshow(img8)
        plt.title(i)
        plt.colorbar()

# # generate pattern order
# Np                       = 64 
# pattern_dim              = '1D'
# scan_mode                = 'Walsh'  #'Walsh_inv' #'Raster_inv' #'Raster' #

# pattern_order_source = 'C:/openspyrit/spas/stats/' + pattern_dim + '/pattern_order_' + scan_mode + '_' + str(Np) + 'x' + str(Np) + '.npz'

# pattern_order = np.load(pattern_order_source)
# data = pattern_order['pattern_order']



# Np                       = 16 
# pattern_dim              = '1D'
# pattern_order_source = 'C:/openspyrit/spas/stats/' + pattern_dim + '/pattern_order_' + scan_mode + '_' + str(Np) + 'x' + str(Np) + '.npz'

# li = np.arange(2*Np)
# pattern_order = np.array(li, dtype=np.uint16)

# np.savez(pattern_order_source[:len(pattern_order_source)-4], pattern_order = pattern_order, pos_neg = True)