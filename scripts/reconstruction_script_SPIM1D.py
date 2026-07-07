# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:32:03 2026

@author: equipe-onli
"""

import numpy as np


#%% read raw_data
filename = 'E:/openspyrit/data/2026-07-06_calib_light_sheet/obj_ball-fluo-cuve_source_laser-473nm_Lc_600.0nm_Gr_1_Walsh_sparse_im_4x256_ti_500ms_zoom_x1/raw_data/spectral_Ny_7.3225737mm_Gr_1_Lc_600.0nm_NA_0.npz'
raw_data_file = np.load(filename)
raw_data3 = raw_data_file['arr_0']
raw_data_file.close()
raw_data = np.empty((raw_data3.shape[0], raw_data3.shape[1], raw_data3.shape[2], 1, 1, 1))
raw_data[:,:,:, 0,0,0] = raw_data3


