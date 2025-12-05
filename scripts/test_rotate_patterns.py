# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 16:46:57 2025

@author: admin
"""

import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

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

pattern_order = np.linspace(0, 64)
pattern_dim              = '1D'
scan_mode                = 'Walsh' 
Np = 64
dmd_height = 768
dmd_width = 1024

pattern_source       = '../Patterns/' + pattern_dim + '/' + scan_mode + '_' + str(Np) + 'x' + str(Np)
pattern_prefix       = scan_mode + '_' + str(Np) + 'x' + str(Np)

path_base = Path(pattern_source)

# for index,pattern_name in enumerate(tqdm(pattern_order, unit=' patterns', total=len(pattern_order))):
#     # read numpy patterns
path = '..\\Patterns\\1D\\Walsh_64x64\\Walsh_64x64_10.npy' #path_base.joinpath(f'{pattern_prefix}_{pattern_name}.npy')
im = np.load(path) 
im_mat = np.reshape(im, [Np,Np])

patterns = np.zeros((dmd_height, dmd_width), dtype=np.uint8)
    
plt.figure()
plt.imshow(im_mat)

im_mat_rot = rotate(im_mat, angle=45)

plt.figure()
plt.imshow(im_mat_rot)

bin_fact = im_mat_rot.shape[0]/Np
bin_image = binArray(im_mat_rot, 0, bin_fact, bin_fact)

plt.figure()
plt.imshow(bin_image)

bin_image2 = binArray(bin_image, 1, bin_fact, bin_fact)

plt.figure()
plt.imshow(bin_image2)


