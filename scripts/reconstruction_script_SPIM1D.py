# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:32:03 2026

@author: equipe-onli
"""

import os
# os.chdir('/home/mahieu/openspyrit/spim/spas/scripts')
os.chdir('E:\\openspyrit\\spas\\scripts')
import numpy as np


#%% read raw_data
data_folder_name = '2026-07-06_calib_light_sheet'
data_name = 'obj_ball-fluo-cuve_source_laser-473nm_Lc_600.0nm_Gr_1_Walsh_sparse_im_4x256_ti_500ms_zoom_x1'

foldername = '../../data/' + data_folder_name + '/' + data_name + '/'
spectral_raw_data_name = 'raw_data/spectral_Ny_7.3225737mm_Gr_1_Lc_600.0nm_NA_0.npz'
raw_data_file = np.load(foldername + spectral_raw_data_name)
raw_data3 = raw_data_file['arr_0']
raw_data_file.close()
raw_data = np.empty((raw_data3.shape[0], raw_data3.shape[1], raw_data3.shape[2], 1, 1, 1))
raw_data[:,:,:, 0,0,0] = raw_data3
#%% read metadata
from spas.acquisition_SPIM1D import read_metadata, func_path

file_path = foldername + 'metadata.json'
dmd_params, spectrograph_params, cam_spat_params, cam_spec_params, acquisition_params = read_metadata(file_path)

all_path = func_path(data_folder_name, data_name, ask_overwrite = False)

#%% Specral reconstruction
from spas.reconstruction_SPIM1D import live_hadamard_reco, spatial_reco
from spas.visualization_SPIM1D import plot_acqui
from scipy import signal

new_raw = np.zeros(raw_data.shape)
for i in range(raw_data.shape[2]):
    new_raw[:,:,i,0,0,0] = signal.medfilt2d(raw_data[:,:,i,0,0,0], kernel_size=5)
    

    
had_reco_all = live_hadamard_reco(new_raw, acquisition_params)
# Spatial reconstruction
spatial_acqui = spatial_reco(acquisition_params, all_path)
# Plot Spatial and had reconstruction
plot_acqui(had_reco_all, spatial_acqui, acquisition_params, all_path)
#%% read experimental Hadamard matrix, the light sheet

from spyrit.misc.walsh_hadamard import walsh_matrix
import matplotlib.pyplot as plt
import torch
# Load Hadamard matrices
# (https://github.com/openspyrit/spyrit-examples/blob/master/2025_hLSFM/main_v3_recon_net_EGFP-DsRed_14_all_slices.ipynb)

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



all_path_bu = all_path
all_path_bu.raw_data_path = '../../data/2026-07-02_calib_light_sheet/obj_fluo_cuve_source_laser-473nm_Lc_580.0nm_Gr_1_Walsh_sparse_im_4x256_ti_25ms_zoom_x1/raw_data'
acquisition_params_bu = acquisition_params
acquisition_params_bu.Nz = 7.5001265
acquisition_params_bu.Lc[0] = (580.0, 1)
acquisition_params_bu.NAverages = 1

spatial_acqui = spatial_reco(acquisition_params_bu, all_path_bu)
prof = np.empty((spatial_acqui.shape[1], spatial_acqui.shape[2]))    
for i in range(spatial_acqui.shape[2]):
    # prof[:, i] = spatial_acqui[1024, :, i]
    prof[:, i] = np.mean(spatial_acqui[1024-1:1024+1, :, i], axis = 0)

prof2 = np.rot90(prof, 1)
prof3 = np.flip(prof2, axis = 0)
prof4 = binArray(prof3, 1, 2, 2)
pos = prof4[0::2, :]
neg = prof4[1::2, :]

H_exp = pos - neg

H_exp_rogn = H_exp[:, int(pos.shape[1]/4):int(pos.shape[1]/4 + pos.shape[1]/2)]
H_exp_bin = binArray(H_exp_rogn, axis = 1, binstep = 4, binsize = 4)

plt.figure()
plt.imshow(H_exp_bin)
plt.title('H_exp_bin')

# # Load positve data
# prep_pos = np.array(raw_data[:, :, 0::2, 0, 0, 0], dtype = np.int64)
# # Load negative data
# prep_neg =  np.array(raw_data[:, :, 1::2, 0, 0, 0], dtype = np.int64)

# Load positve data
prep_pos = np.array(new_raw[:, :, 0::2, 0, 0, 0], dtype = np.int64)
# Load negative data
prep_neg =  np.array(new_raw[:, :, 1::2, 0, 0, 0], dtype = np.int64)

# spectral dimension comes first
prep_pos = np.swapaxes(prep_pos, 0, 1)
prep_neg = np.swapaxes(prep_neg, 0, 1)

# prep_pos = np.rot90(prep_pos, 2)
# prep_neg = np.rot90(prep_neg, 2)

plt.figure()
plt.imshow(prep_pos[:,:,0])
plt.title('prep_pos')
plt.colorbar()

# param #2
y = prep_pos - prep_neg 

y = torch.from_numpy(y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

y = y.to(device)
M = H_exp_bin.shape[0]
N = H_exp_bin.shape[1]
y = y.reshape(-1,1,N,M)

plt.figure()
plt.imshow(y[:,0,:,0].cpu().numpy())
plt.title('y')
plt.colorbar()
#%% RECO
from spyrit.core.meas  import Linear
from spyrit.core.recon import PinvNet
from spyrit.misc.disp import add_colorbar, noaxis

linop = Linear(torch.from_numpy(H_exp_bin), meas_shape = (1,N), device=device)
recon = PinvNet(linop, store_H_pinv=True, device=device)

# init output
rec = np.zeros((N, N))
recs = []
lambdas = []

lambda_all = acquisition_params.wavelengths
# lambda_central_list = [560, 580, 600, 620, 640]
lambda_central_list = [spectrograph_params.position]
c_step = 20

for lambda_central in lambda_central_list:
    with torch.no_grad():
        c_central = np.argmin((lambda_all-lambda_central)**2) # Central channel
        m = y[c_central-c_step:c_central+c_step].sum(0, keepdim=True).to(device, torch.float32)
        print(f'reconstructing spectral bin from channels: {c_central-c_step}--{c_central+c_step}')
        rec_gpu = recon.reconstruct_pinv(m)
        rec = rec_gpu.cpu().detach().numpy().squeeze()
        rec = np.moveaxis(rec, 0, -1) # spectral channel is now the last axis
        rec = np.flip(rec,0)
        rec = np.fliplr(rec)
        rec = np.rot90(rec,-1)
        
        # Plot 
        fig, axs = plt.subplots(1, 1, figsize=(5,5))
        im = axs.imshow(rec) 
        axs.set_title(f'{lambda_central} nm') 
        add_colorbar(im, 'bottom') 
        noaxis(axs)
                
    recs.append(rec)
    lambdas.append(lambda_central)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    