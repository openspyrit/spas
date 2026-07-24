#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 09:08:24 2025
@author: mahieu

The main software to acquire in 1D an hyperspectral cube with the SPIM
"""


# packages
import os
os.chdir('E:\\openspyrit\\spas\\scripts')
import numpy as np
import time
import math
from spas.transfer_data_to_girder import transfer_data_SPIM1D
from spas.DMD_module import init_DMD, disconnect_DMD, change_patterns, setup_DMD, play_one_pattern
from spas.spectro_ShamrockAndor_module import init_spectrograph, disconnect_spectrograph, setup_spectrograph
from spas.cam_Andor_module import init_cam_spat, init_cam_spec, disconnect_cam, setup_cam, snapshot_cam, display_cam
from spas.PI_module import init_PI, disconnect_stage, read_position, move_to_middle, stage_adjustment, stage_parameters, go_to_zero
from spas.shutter_TSC001_module import ThorlabsShutter
from spas.flipping_mirror_MFF101_module import MFF
from spas.acquisition_SPIM1D import AcquisitionParameters, func_path, acquire, define_wavelengths_matrix, plot_spectrum
from spas.reconstruction_SPIM1D import live_hadamard_reco, spatial_reco
from spas.visualization_SPIM1D import plot_acqui

from matplotlib import pyplot as plt
#%% Initialize hardware
spectrograph = init_spectrograph(model = 'andor_shamrock')
DMD, DMD_initial_memory = init_DMD(dmd_lib_version = '4.3')
cam_spat = init_cam_spat(SN = 'VSC-10323')
cam_spec = init_cam_spec(SN = 'VSC-23585')
stage = init_PI(Model = 'C-884', SN = '0000000000', verbose = True)
shutter = ThorlabsShutter("85855593")
mirror = MFF(SN = '37010810')
#%% Move the PI stage to the middle
move_to_middle(stage.pidevice, stage.stage_tools)
position = read_position(stage.pidevice, stage.stage_tools, verbose = True)
# go_to_zero(stage.pidevice, stage.stage_tools)
#%% setup Spatial Camera
cam_spat_params = setup_cam(cam = cam_spat, 
                            expos_time  = 0.5,  # (s)
                            gain        = 1,        # 1 or 2                              
                            width       = 2048,     # max = 2048
                            height      = 2048,     # max = 2048
                            offsetX     = 1,        # min = 1
                            offsetY     = 1,        # min = 1
                            binningX    = 1,        # int < 2048
                            binningY    = 1,        # int < 2048
                            encodPix    = 12,       # 12 or 16 bit
                            snapshot    = True)    # if false => acquire video, if True => acquire an image 
#%% get a snapshot of the spatial camera
mirror.set_position('spatial', verbose = True)
shutter.open()
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spat_params, zoom = 1, pattern_to_display = 'gray_0', 
                              pattern_dim = '1D', scan_mode = 'Walsh_sparse', Np = 128, pattern_thickness = 16) # white, black or gray_ + pattern number
data = snapshot_cam(cam = cam_spat, tilt_image = True) # data_format accepted: 8 or 16 bits
DMD.Halt()
shutter.close()
#%% display spatial camera in continous mode
mirror.set_position('spatial', verbose = True)
shutter.open()
stage_adjustment(stage.pidevice)
time.sleep(1)
play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spat_params, pattern_to_display = 'gray_0', 
                 
                 pattern_dim = '1D', scan_mode = 'Walsh_sparse', Np = 128, pattern_thickness = 16) 
display_cam(cam = cam_spat, cam_params = cam_spat_params, display_max = True, display_profile = True)
DMD.Halt()
shutter.close()
position = read_position(stage.pidevice, stage.stage_tools, verbose = True)
#%% setup the Spectrograph
spectrograph_params = setup_spectrograph(spectrograph,
                                         grating_nbr =   1, print_select   = True,   # Arg:  1 
                                         position    = 600, print_position = True,   # the central wavelength of the grating
                                         slit_width  = 20000)                          # the width of the slit in (µm)
#%% setup Spectral Camera
cam_spec_params = setup_cam(cam = cam_spec, 
                            expos_time  = 0.5,        # (s)
                            gain        = 1,        # 1 or 2                              
                            width       = 2048,     # max = 2048
                            height      = 2048,     # max = 2048
                            offsetX     = 1,        # 1
                            offsetY     = 1,        # 1
                            binningX    = 8,        # int < 2048        
                            binningY    = 8,        # int < 2048
                            encodPix    = 12,       # 12 or 16 bit
                            snapshot    = False)    # if false => acquire video, if True => acquire an image   
#%% get a snapshot of the spectral camera
mirror.set_position('spectral', verbose = True)
shutter.open()
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spec_params, zoom = 1, pattern_to_display = 'gray_0', 
                              pattern_dim = '1D', scan_mode = 'Walsh_sparse', Np = 128, pattern_thickness = 4) 
data = snapshot_cam(cam = cam_spec, tilt_image = False) # data_format accepted: 8 or 16 bits
DMD.Halt()
plot_spectrum(data, cam_spec_params, spectrograph_params)
shutter.close()
#%% display spectral camera in continous mode
mirror.set_position('spectral', verbose = True)
shutter.open()
stage_adjustment(stage.pidevice)
time.sleep(1)
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spec_params, pattern_to_display = 'white', 
                              pattern_dim = '1D', scan_mode = 'Walsh_sparse', Np = 128, pattern_thickness = 4) 
display_cam(cam = cam_spec, cam_params = cam_spec_params, display_max = False, display_integral = True)
DMD.Halt()
shutter.close()
position = read_position(stage.pidevice, stage.stage_tools, verbose = True)
#%% setup acquisition
setup_version            = 'setup_v1.0'
collection_access        = 'public' #'private'#
Np                       = 128      # Number of pixels in one dimension of the image (image: NpxNp)
Nz                       = np.array([position[1]])# to move the stage to acquire the third spatial dimension
pattern_thickness        = 16
ti                       = cam_spec_params.exposure_time_μs / 1000 # Integration time of the spectral camera
NAverages                = 1        # Number of avegare (the acquisition is accumulated before moving the grating)
NRepetitions             = len(Nz)        # Number of repetitions (grating change after that, the acquisition is repeated)
Lc                       = [(spectrograph_params.position, spectrograph_params.grating.current_grating_nbr)]#, (630, 1), (600, 1), (630, 1), (660, 1), (690, 1), (720, 1)]#, (922, 1), (927, 1)] # [(0, 1), (780, 1), (785, 1), (795, 1), (805, 1), (810, 1)] #[(0, 1), (676, 1), (686, 1), (696, 1), (706, 1), (716, 1)] #[(0, 1), (557, 1), (567, 1), (577, 1), (587, 1), (597, 1)] #[(0, 1), (526, 1), (536, 1), (546, 1), (556, 1), (566, 1)] #[(0, 1), (416, 1), (426, 1), (436, 1), (446, 1), (456, 1)] ##, (832, 2), (852, 2), (872, 2), (892, 2), (912, 2), (932, 2), (952, 2), (972, 2), (992, 2)]## # a vector containig the central wavelength following by the grating number
array_to_move            = np.linspace(1, 10, 10, endpoint=True)
zoom                     = 1        # Numerical zoom applied in the DMD
xw_offset                = 128      # Default = 128
yh_offset                = 0        # Default = 0
pattern_compression      = 1
pattern_dim              = '1D'
scan_mode                = 'Walsh_sparse'  #'Walsh_inv' #'Raster_inv' #'Raster' #
source                   = 'lasers-532nm'#white_LED'#'No source'#'White_Zeiss_lamp'#'Thorlabs_White_halogen_lamp'#'HG-1_Oceanoptics'#No-light'#'Bioblock'#'Laser_405nm_1.2W_A_0.14'#'''#' + white LED might'#
object_name              = 'leg_n1-3_p1_y0.05'#'fluo_µsphere-gel-bin10' #
data_folder_name         = '2026-07-24_Drosophila'#'Patient-69_exvivo_LGG_BU'
data_name                = 'obj_' + object_name + '_source_' + source + '_Lc_' + str(Lc[0][0]) + 'nm_Gr_' + str(Lc[0][1]) + '_' + scan_mode + '_im_'+str(pattern_thickness)+'x'+str(Np)+'_ti_'+str(round(ti))+'ms_zoom_x'+str(zoom)

all_path = func_path(data_folder_name, data_name, ask_overwrite = False)
if 'mask_index' not in locals(): mask_index = [];  x_mask_coord = []; y_mask_coord = [] # execute "mask_index = []" to not apply the mask

if all_path.aborted == False:
    output_directory     = all_path.subfolder_path
    pattern_order_source = '../stats/' + pattern_dim + '/pattern_order_' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np) + '.npz'
    pattern_source       = '../Patterns/' + pattern_dim + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np)
    pattern_prefix       = scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np)
    experiment_name      = data_name
    light_source         = source,
    object               = object_name
    filter               = 'Notch filter at 473 and 532 nm' #+ OD=0.3',''No filter',#'linear colored filter',#'Orange filter (600nm)',#'Dichroic_420nm',#'HighPass_500nm + LowPass_750nm + Dichroic_560nm',#'BandPass filter 560nm Dl=10nm',#'None', # + , #'Nothing',#'Diffuser + HighPass_500nm + LowPass_750nm',##'Microsope objective x40',#'' linear colored filter + OD#0',#'Nothing',#
    description          = 'Sample: 5–6 Drosophila melanogaster legs with GFP and RFP fluoresence in muscles. Central wavelength of the spectrograph is ok, wavelength vector is not calibrated, warning! Optical setup: DMD -- 200mm --> Lens(f=200) -- 200mm --> objx4 -- 18mm --> sample' 
    
    acquisition_params = AcquisitionParameters(pattern_compression = pattern_compression, pattern_dimension_x = pattern_thickness, pattern_dimension_y = Np, 
                                               zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset, mask_index = mask_index, 
                                               x_mask_coord = x_mask_coord, y_mask_coord = y_mask_coord, output_directory = output_directory, 
                                               pattern_order_source = pattern_order_source, pattern_source = pattern_source, pattern_prefix = pattern_prefix, 
                                               experiment_name = experiment_name, light_source = light_source, object = object, filter = filter, 
                                               NAverages = NAverages, NRepetitions = NRepetitions, Lc = Lc, Nz = Nz, description = description)
    
    acquisition_params.wavelengths = define_wavelengths_matrix(cam_spec_params, Lc, display_figure = False, verbose = True)  
    acquisition_params.wavelengths = acquisition_params.wavelengths[0, :]
    acquisition_params.wavelengths = np.linspace(spectrograph_params.position - 50,
                                                 spectrograph_params.position + 50, 
                                                 int(cam_spec_params.width))
    
    stage_params = stage_parameters
    stage_params.array_to_move = array_to_move
                        
    try: 
        change_patterns(DMD = DMD, acquisition_params = acquisition_params, zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset, 
                        force_change = True) 
    except: 
        print('pass by exception')
        pass
                  
    DMD_params = setup_DMD(DMD = DMD, DMD_initial_memory = DMD_initial_memory, acquisition_params = acquisition_params, 
                           integration_time = ti, add_illumination_time = 30000) # 30000 si bin ?x? (je pense 4x4, mais possible 8x8), 1000000 pour la mesure de la calibration du feuillet
    

    if DMD_params.patterns != None:
        acq_time = Np*NRepetitions*NAverages*len(Lc)*(ti + DMD_params.add_illumination_time_us / 1000)
        print('Total expected acq time  : ' + str(int(acq_time/1000 // 60)) + ' min ' + 
              str(math.floor(acq_time/1000 % 60)) + ' s ' + str(round((acq_time / 1000 % 1) * 1000)) + ' ms')
else:
    print('setup aborted')
#%% Acquire
# time.sleep(20)
raw_data =  acquire(DMD                 = DMD,
                    DMD_params          = DMD_params,
                    cam_spat            = cam_spat,
                    cam_spat_params     = cam_spat_params,
                    cam_spec            = cam_spec,
                    cam_spec_params     = cam_spec_params,
                    spectrograph        = spectrograph,
                    spectrograph_params = spectrograph_params,
                    stage               = stage,
                    shutter             = shutter,
                    mirror              = mirror,
                    acquisition_params  = acquisition_params,
                    all_path            = all_path,
                    verbose             = False,
                    acquisition_arm     = 'spectral')
#%% Specral reconstruction
from scipy import signal
new_raw = np.zeros(raw_data.shape)
for i in range(raw_data.shape[2]):
    new_raw[:,:,i,0,0,0] = signal.medfilt2d(raw_data[:,:,i,0,0,0], kernel_size=3)
    
had_reco_all = live_hadamard_reco(new_raw, acquisition_params)
#%% Spatial reconstruction
spatial_acqui = spatial_reco(acquisition_params, all_path)
#%% Plot Spatial and had reconstruction
plot_acqui(had_reco_all, spatial_acqui, acquisition_params, all_path)#, med_filt = True)
#%% reco avec la matrix de Had experimentale, mon propre code
from spyrit.misc.walsh_hadamard import walsh_matrix
import matplotlib.pyplot as plt
import torch

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

# Load Hadamard matrices
# (https://github.com/openspyrit/spyrit-examples/blob/master/2025_hLSFM/main_v3_recon_net_EGFP-DsRed_14_all_slices.ipynb)
# H_exp = np.load(Path(data_folder + mat_folder) / f'motifs_Hadamard_{M}_{N}.npy')
# H_exp /= H_exp[0,16:500].mean()


all_path_bu = all_path
all_path_bu.raw_data_path = '../../data/2026-07-21_calib_light_sheet/obj_fluo-cuve3_source_laser-473nm_Lc_620.0nm_Gr_1_Walsh_sparse_im_4x128_ti_750ms_zoom_x1/raw_data'
acquisition_params_bu = acquisition_params
acquisition_params_bu.Nz = 7.5000029
acquisition_params_bu.Lc[0] = (620.0, 1)
acquisition_params_bu.NAverages = 1

# spatial_acqui = spatial_reco(acquisition_params_bu, all_path_bu)

spatial_acqui_arr = np.load(all_path_bu.raw_data_path + '/spatial_Ny_7.5000029mm_Gr_1_Lc_620.0nm_NA_0.npz')
spatial_acqui = spatial_acqui_arr['arr_0']

rogne1 = 620
rogne2 = 1320

# prof = np.empty((spatial_acqui.shape[1], spatial_acqui.shape[2]))    
prof = np.empty((rogne2 - rogne1, spatial_acqui.shape[2]))  

for i in range(spatial_acqui.shape[2]):
    prof[:, i] = np.squeeze(np.mean(spatial_acqui[rogne1:rogne2, 1024-10:1024+10, i], axis = 1))
    if 1 == 2:#i >= 2 and i <= 6:
        plt.figure()
        plt.plot(prof[:, i])
        plt.title('i = ' + str(i))

prof2 = np.rot90(prof, 1)
prof3 = np.flip(prof2, axis = 0)
prof4 = binArray(prof3, 1, 2.734, 2.734)
pos = prof4[0::2, :]
neg = prof4[1::2, :]

H_exp = pos - neg

# H_exp_path = '../../data/2026-07-21_calib_light_sheet/obj_fluo-cuve3_source_laser-473nm_Lc_620.0nm_Gr_1_Walsh_sparse_im_4x256_ti_750ms_zoom_x1/H_exp.npy'
# H_exp = np.load(H_exp_path)


# plt.figure()
# plt.imshow(H_exp)
# plt.title('H_exp')

H_exp_rogn = H_exp[:, 512:512+1024]
# H_exp_bin = binArray(H_exp_rogn, axis = 1, binstep = 4, binsize = 4)
H_exp_bin = H_exp

# plt.figure()
# plt.imshow(H_exp_bin)
# plt.title('H_exp_bin')

# Load positve data
prep_pos = np.array(raw_data[:, :, 0::2, 0, 0, 0], dtype = np.int64)
# Load negative data
prep_neg =  np.array(raw_data[:, :, 1::2, 0, 0, 0], dtype = np.int64)



# spectral dimension comes first
# prep_pos = np.moveaxis(prep_pos, -1, 0)
# prep_neg = np.moveaxis(prep_neg, -1, 0)
prep_pos = np.swapaxes(prep_pos, 0, 1)
prep_neg = np.swapaxes(prep_neg, 0, 1)

prep_pos

# plt.figure()
# plt.imshow(prep_pos[:,:,0])
# plt.title('prep_pos')
# plt.colorbar()

# plt.figure()
# plt.imshow(prep_neg[:,:,0])
# plt.title('prep_neg')
# plt.colorbar()

# param #2
y = prep_pos - prep_neg 

# plt.figure()
# plt.imshow(y[:,:,0])
# plt.title('y')
# plt.colorbar()


y = torch.from_numpy(y)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

y = y.to(device)
M = H_exp_bin.shape[0]
N = H_exp_bin.shape[1]
y = y.reshape(-1,1,N,M)

# plt.figure()
# plt.imshow(y[:,0,:,0])
# plt.title('y')
# plt.colorbar()

from spyrit.core.meas  import Linear
from spyrit.core.recon import PinvNet
from spyrit.misc.disp import add_colorbar, noaxis

linop = Linear(torch.from_numpy(H_exp_bin), meas_shape = (1,256), device=device) 
recon = PinvNet(linop, store_H_pinv=True, device=device)
#%% plot the reco by H_exp
save_fig = True
# init output
rec = np.zeros((N, N))
recs = []
lambdas = []

lambda_all = acquisition_params.wavelengths
# lambda_central_list = [560, 580, 600, 620, 640]
lambda_central_list = [spectrograph_params.position]
c_step = 10

for lambda_central in lambda_central_list:
    with torch.no_grad():
        c_central = np.argmin((lambda_all-lambda_central)**2) # Central channel
        m = y[c_central-c_step:c_central+c_step].sum(0, keepdim=True).to(device, torch.float32)
        print(f'reconstructing spectral bin from channels: {c_central-c_step}--{c_central+c_step}')
        rec_gpu = recon.reconstruct_pinv(m)
        rec = rec_gpu.cpu().detach().numpy().squeeze()
        rec = np.moveaxis(rec, 0, -1) # spectral channel is now the last axis
        # rec = np.flip(rec,0)
        rec = np.fliplr(rec)
        rec = np.rot90(rec,1)
        
        # Plot 
        fig, axs = plt.subplots(1, 1, figsize=(5,5))
        im = axs.imshow(rec) 
        axs.set_title(f'{lambda_central} nm') 
        add_colorbar(im, 'bottom') 
        noaxis(axs)
        if save_fig:
            plt.savefig(all_path.overview_path + '/spectral_GRAY_IMAGE_had_reco_by_H_exp_' + str(lambda_central) + 'nm.png', dpi=300, bbox_inches='tight')
                
    recs.append(rec)
    lambdas.append(lambda_central)
#%%
import collections
collections.Callable = collections.abc.Callable
fig_folder = './figure/'
from pathlib import Path
from spyrit.misc.disp import add_colorbar, noaxis
import matplotlib.pyplot as plt
from spyrit.misc.walsh_hadamard import walsh_matrix
import numpy as np
import torch

from spyrit.core.meas import LinearSplit
from spyrit.core.noise import Poisson
from spyrit.core.prep import Rerange, UnsplitRescale
from spyrit.core.nnet import Unet, Identity
from spyrit.core.train import load_net

from spyrit.core.recon import TikhoNet
from spyrit.core.nnet import Unet, Identity
from typing import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_rec = True
save_fig = True

M = cam_spec_params.width
N = Np

# Data Paths
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D'  # main directory
mat_folder = '/Reconstruction/Mat_rc/'  # prepped matrices
data_subfolder = 'data_2023_03_14/'  # raw data
prep_folder = '/Preprocess/'  # prepped raw data

H_exp = np.load(Path(data_folder + mat_folder) / f'motifs_Hadamard_{M}_{N}.npy')
H_exp /= H_exp[0,16:500].mean()


H_tar = walsh_matrix(Np)
H_tar = H_tar[:M]

# plot
f, axs = plt.subplots(2, 1)
axs[0].set_title('Target measurement patterns')
im = axs[0].imshow(H_tar, cmap='gray') 
add_colorbar(im, 'bottom')
axs[0].get_xaxis().set_visible(False)

axs[1].set_title('Experimental EGFP measurement patterns')
im = axs[1].imshow(H_exp, cmap='gray') 
add_colorbar(im, 'bottom')
axs[1].get_xaxis().set_visible(False)

H = torch.from_numpy(H_exp)
#%% débruitage
import numpy as np
from scipy import signal
import spyrit.misc.walsh_hadamard as wh
from matplotlib import pyplot as plt

new_raw = np.zeros((512,512,256))
for i in range(raw_data.shape[2]):
    new_raw[:,:,i] = signal.medfilt2d(raw_data[:,:,i,0,0,0], kernel_size=3)

plt.figure()
plt.imshow(new_raw[:,:,1])
    
M_sub = new_raw[:,:,0::2] - new_raw[:,:,1::2]
M_sub = M_sub.astype(float)
Npatterns = 256
temp_had_reco = wh.fwht(M_sub) / Npatterns
had_reco = np.swapaxes(temp_had_reco, 2, 1) 
s=np.sum(had_reco[:,:,100:400],axis=2)

plt.figure()
plt.imshow(s)

#%% transfer data to girder
transfer_data_SPIM1D(DMD_params, cam_spat_params, cam_spec_params, spectrograph_params, acquisition_params,
                    setup_version, data_folder_name, data_name, collection_access, upload_metadata = 1)
#%% Disconnect
disconnect_spectrograph(spectrograph, goto_zero = False)
disconnect_DMD(DMD)
disconnect_cam(cam_spat)
disconnect_cam(cam_spec)
disconnect_stage(stage)
shutter.disconnect()
mirror.disconnect()
#%% below, old prog
# #%% Neural Network setup (executed it just one time)
# network_param = ReconstructionParameters(
#     # Reconstruction network    
#     M = Np*Np,                  # Number of measurements
#     img_size = 128,             # Image size of the NN reconstruction
#     arch = 'dc-net',            # Main architecture
#     denoi = 'unet',             # Image domain denoiser (possibility to do not apply, put : None)
#     subs = 'rect',              # Subsampling scheme
    
#     # Training
#     data = 'imagenet',          # Training database
#     N0 = 10,                    # Intensity (max of ph./pixel)
    
#     # Optimisation (from train2.py)
#     num_epochs = 30,            # Number of training epochs
#     learning_rate = 0.001,      # Learning Rate
#     step_size = 10,             # Scheduler Step Size
#     gamma = 0.5,                # Scheduler Decrease Rate   
#     batch_size = 256,           # Size of the training batch
#     regularization = 1e-7       # Regularisation Parameter
#     )

# cov_folder = 'C:/openspyrit/stat/ILSVRC2012_v10102019/'
# cov_path = Path(cov_folder) / f'Cov_8_{network_param.img_size}x{network_param.img_size}.npy'
# model_folder = 'C:/openspyrit/models/'
# model, device = setup_reconstruction(cov_path, model_folder, network_param)
# #%% Neural Network Reconstruction
# plot_reco_with_NN(acquisition_parameters, spectral_data, model, device, network_param, all_path, cov_path)
# #%% Draw a ROI
# # Comment data_folder_name & data_name to draw a ROI in the current acquisition, else specify the acquisition name
# data_folder_name = '2025-01-16_myFirstAcq'
# data_name = 'obj_cat_source_white_LED_Walsh_im_64x64_ti_1ms_zoom_x1'
# mask_index, x_mask_coord, y_mask_coord = extract_ROI_coord(DMD_params, acquisition_parameters, all_path, 
#                                                            data_folder_name, data_name, GT, ti, Np)























