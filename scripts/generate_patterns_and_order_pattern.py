# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:49:52 2022

@author: mahieu
"""

###############################################################################
# prog to generate patterns and its order
# Covariance matrix is required to generate Walsh patterns
###############################################################################

from spas.generate import walsh_patterns #, raster_patterns
from spas.generate import generate_hadamard_order
import numpy as np
import os
import time

################################## INPUT ######################################
Np_tab = [8]#[32, 64, 128]    # Number of pixels in one dimension of the image (image: NpxNp)
scan_mode_tab = ['Walsh']#, 'Raster']
zoom = []     # leave empty to execute all the possible zoom, otherwise specify one or more values of zoom
DMD_minor_size = 768    # minor size of the DMD
spas_path = 'C:/openspyrit/spas/'
######################## CREATE ALL THE POSSIBLE ZOOM #########################
stop = 0
if len(zoom) == 0:
    if max(Np_tab) > DMD_minor_size:
        print('Error, The image size cannot be higher than the minimum size of the DMD')
        print('max(Np) <= DMD_minor_size')
        print('program stop')
        stop = 1
    else:
        zoom_vector = [1, 2, 3, 4]
        z = 3
        while 1:    
            z = z*2
            zoom_vector.append(z)    
            if z >= DMD_minor_size:
                break
else:
    zoom_vector = zoom
################################## BEGIN ######################################
t0 = time.time()
if stop == 0:
    for scan_mode in scan_mode_tab:
        for Np in Np_tab:
            max_zoom = DMD_minor_size//Np
            zoom_tab = zoom_vector[:zoom_vector.index(DMD_minor_size//Np)+1]        
            for zoom in zoom_tab:
                t00 = time.time()
                print('zoom = ' + str(zoom) + ' || Np = ' + str(Np) + ' || scan mode : ' + scan_mode)
                ############################# PATH ################################                
                pattern_order_source = spas_path + 'stats/pattern_order_' + scan_mode + '_' + str(Np) + 'x' + str(Np) + '.npz'
                pattern_source       = spas_path + 'Patterns_test/Zoom_x' + str(zoom) + '/' + scan_mode + '_' + str(Np) + 'x' + str(Np)
                pattern_prefix       = scan_mode + '_' + str(Np) + 'x' + str(Np)
                ########################## CREATE PATH ############################
                if os.path.isdir(pattern_source) == False:
                    if os.path.isdir(spas_path + 'Patterns_test/Zoom_x' + str(zoom)) == False:
                        os.mkdir(spas_path + 'Patterns_test/Zoom_x' + str(zoom))
                        os.mkdir(pattern_source)
                    else:
                        os.mkdir(pattern_source)                    
                ##################### generate patterns #######################
                if scan_mode == 'Walsh':                
                    walsh_patterns(N = Np, save_data = True, path = pattern_source + '/', N_DMD = DMD_minor_size//zoom)
                elif scan_mode == 'Raster':
                    raster_patterns(N=Np, save_data = True, path = pattern_source + '/', N_DMD = DMD_minor_size//zoom)   
                #################### delay of the loop ########################
                hours, rem = divmod(time.time()-t00, 3600)
                minutes, seconds = divmod(rem, 60)
                print("       delay : {:0>2}h{:0>2}m{:0>2}s".format(int(hours),int(minutes),int(seconds)))
                ###################### elapsed time ###########################
                hours, rem = divmod(time.time()-t0, 3600)
                minutes, seconds = divmod(rem, 60)
                print("elapsed time : {:0>2}h{:0>2}m{:0>2}s".format(int(hours),int(minutes),int(seconds)))
            ####################### generate pattern order ####################    
            if scan_mode == 'Walsh':               
                cov_path = spas_path + 'stats/Cov_' + str(Np) + 'x' + str(Np) + '.npy'
                generate_hadamard_order(N = Np, name = 'pattern_order_' + scan_mode + '_' + str(Np) + 'x' + str(Np), cov_path = cov_path, pos_neg = True)
            elif scan_mode == 'Raster':
                pattern_order=np.arange(0, Np**2, dtype=np.uint16)
                np.savez(pattern_order_source[:len(pattern_order_source)-4], pattern_order = pattern_order, pos_neg = False)
elapsed = time.time() - t0
print('FINISHED')
################################### END #######################################


# #%% generate inverted patterns and its order from existed files
# from PIL import Image, ImageOps
# from matplotlib import pyplot as plt
# import shutil
# import os

# ################################## INPUT ######################################
# Np = 32    # Number of pixels in one dimension of the image (image: NpxNp)
# scan_mode_orig = 'Walsh'    
# zoom = 1
# ############################# PATH ################################  
# scan_mode = scan_mode_orig + '_inv' 
# spas_path = 'C:/openspyrit/spas/'             
# pattern_order_source_orig = spas_path + 'stats/pattern_order_' + scan_mode_orig + '_' + str(Np) + 'x' + str(Np) + '.npz'
# pattern_source_orig       = spas_path + 'Patterns/Zoom_x' + str(zoom) + '/' + scan_mode_orig + '_' + str(Np) + 'x' + str(Np)
# pattern_prefix_orig       = scan_mode_orig + '_' + str(Np) + 'x' + str(Np)
# pattern_order_source = spas_path + 'stats/pattern_order_' + scan_mode + '_' + str(Np) + 'x' + str(Np) + '.npz'
# pattern_source       = spas_path + 'Patterns/Zoom_x' + str(zoom) + '/' + scan_mode + '_' + str(Np) + 'x' + str(Np)
# pattern_prefix       = scan_mode + '_' + str(Np) + 'x' + str(Np)
# ########################## CREATE PATH ############################

# if os.path.isdir(pattern_source) == False:
#     if os.path.isdir(spas_path + 'Patterns/Zoom_x' + str(zoom)) == False:
#         os.mkdir(spas_path + 'Patterns/Zoom_x' + str(zoom))
#         os.mkdir(pattern_source)
#     else:
#         os.mkdir(pattern_source) 
# ######################## load fig and inverted it #####################        
# if scan_mode_orig == 'Walsh':
#     fac = 2
# else:
#     fac = 1
    
# for ind in range(Np**2*fac):
#     im = Image.open(pattern_source_orig + '/' + pattern_prefix_orig + '_'+str(ind) + '.png')
    
#     # plt.figure()
#     # plt.imshow(im)
#     # plt.colorbar()
    
#     im_invert = ImageOps.invert(im)
    
#     # plt.figure()
#     # plt.imshow(im_invert)
#     # plt.colorbar()
    
#     im_invert.save(pattern_source + '/' + pattern_prefix + '_'+str(ind) + '.png', quality=95)

# ####################### copy pattern order ####################
# shutil.copyfile(pattern_order_source_orig, pattern_order_source)

















