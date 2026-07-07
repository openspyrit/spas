# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:00:04 2026

@author: equipe-onli
"""

from matplotlib import pyplot as plt
import numpy as np
import os

ny = '7.5000029'
ilc = '610.0'
arm = 'spatial'

file_name = arm + '_Ny_' + ny + 'mm_Gr_' + str(1) + '_Lc_' + ilc + 'nm_NA_' + str(0)
# path_name = all_path.raw_data_path + '/' + file_name + '.npz'
fold = '2026-05-29_calib_light_sheet'
cuve_nbr = 3
# path_name = '../../data/' + fold + '/obj_fluo_cuve' + str(cuve_nbr) + '_source_laser-473nm_Lc_610.0nm_Gr_1_Walsh_im_128x128_ti_20ms_zoom_x1/raw_data/spatial_Ny_7.5000029mm_Gr_1_Lc_610.0nm_NA_0.npz'
path_name = '../../data/' + fold + '/obj_fluo_cuve' + str(cuve_nbr) + '_source_laser-473nm_Lc_610.0nm_Gr_1_Walsh_sparse_im_4x256_ti_20ms_zoom_x1/raw_data/spatial_Ny_7.5000029mm_Gr_1_Lc_610.0nm_NA_0.npz'
raw_data_file = np.load(path_name)
raw_data = raw_data_file['arr_0']
raw_data_file.close()

spatial_acqui = np.empty((raw_data.shape), dtype = np.int16)
 
raw_data = np.rot90(raw_data, k=1, axes=(0,1))
spatial_acqui[:, :, :] = raw_data

spatial_acqui = np.squeeze(spatial_acqui)  
#%% plot spatial acqui
for i in range(spatial_acqui.shape[2]):
    if i < 10 or i == 33 or i == 63 or i == 127:
        plt.figure()
        plt.imshow(spatial_acqui[:, :, i])
        plt.title('pattern n°' + str(i))  

prof = np.empty((spatial_acqui.shape[1], spatial_acqui.shape[2]))    
for i in range(spatial_acqui.shape[2]):
    prof[:, i] = np.mean(spatial_acqui[int(spatial_acqui.shape[0]/2)-5:int(spatial_acqui.shape[0]/2)+5, :, i], axis=0)
    if i < 10 or i == 33 or i == 63 or i == 127:
        plt.figure()
        plt.plot(prof[:,i])
        plt.title('pattern n°' + str(i)) 

def binArray(data, axis, binstep, binsize, func=np.nanmean):
    """
    Binning on an array
    
    Parameters
    ----------
    data : TYPE
        data is your array.
    axis : TYPE
        axis is the axis you want to been.
    binstep : TYPE
        binstep is the number of points between each bin (allow overlapping bins).
    binsize : TYPE
        binsize is the size of each bin.
    func : TYPE, optional
        func is the function you want to apply to the bin (np.max for maxpooling, np.mean for an average ...). The default is np.nanmean.

    Returns
    -------
    data : TYPE
        The binning array.

    """
    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    data = data.transpose(argdims)
    data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data

prof2 = np.rot90(prof, 3)
prof3 = prof2#np.flip(prof2, axis=1)
prof4 = binArray(prof3, 1, 4, 4)

pos = prof4[0::2, :]
neg = prof4[1::2, :]

max1 = pos.max()
max2 = neg.max()
maxi = max(max1,max2)

pos = pos / maxi
neg = neg / maxi
# plt.figure()
# plt.plot(pos[:, 0])

save_path = '../../result/' + fold + '/' + str(cuve_nbr) + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
plt.figure()
plt.imshow(pos)
plt.title('pos')
plt.colorbar()
plt.savefig(save_path + 'pos')

plt.figure()
plt.imshow(neg)
plt.title('neg')
plt.colorbar()
plt.savefig(save_path + 'neg')

plt.figure()
plt.imshow(pos - neg)
plt.title('diff')
plt.colorbar()
plt.savefig(save_path + 'diff')

plt.figure()
plt.imshow(pos + neg)
plt.title('sum')
plt.colorbar()
plt.savefig(save_path + 'sum')















