#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 09:08:24 2025
@author: mahieu

The main software to acquire in 1D an hyperspectral cube with the single pixel camera
"""


#%% packages
import time
import math
import os
os.chdir('C:\\openspyrit\\spas\\scripts')
from spas.DMD_module import init_DMD, disconnect_DMD, change_patterns, setup_DMD, play_one_pattern
from spas.spectro_SP_module import init_spectrograph, disconnect_spectrograph, setup_spectrograph
from spas.cam_Ximea_module import init_cam_spat, init_cam_spec, disconnect_cam, setup_cam, snapshot_cam, display_cam
from spas.acquisition_SPC1D import AcquisitionParameters, func_path, acquire, define_wavelengths_matrix, plot_spectrum
from spas.reconstruction_SPC1D import hadamard_reco
from spas.visualization_SCP1D import plot_reco_without_NN
from spas.transfer_data_to_girder import transfer_data_SPC1D
#%% Initialize hardware
DMD, DMD_initial_memory = init_DMD(dmd_lib_version = '4.2')
spectrograph = init_spectrograph(model = 'CM110')
cam_spat = init_cam_spat(SN = 'BRCID2503000')
cam_spec = init_cam_spec(SN = 'BRMID2503000')
#%% setup the Spectrograph
spectrograph_params = setup_spectrograph(spectrograph,
                                         grating_nbr =    2, print_select   = True,   # Arg:  1 (High Resoluton), 2 (Low Resoluton)
                                         position    =  550, print_position = True,   # the central wavelength of the grating
                                         unit        = 'nm', print_unit     = True,   # Arg: 'A', 'nm', 'µm'
                                         slit_width  = 300)                          # the width of the slit in (µm)
#%% setup Spatial Camera
cam_spat_params = setup_cam(cam = cam_spat, 
                            cameras_nbr = 2,        # number of camera 
                            expos_time  = 0.008,     # [0.001 - 1000] ms
                            frame_rate  = 4000,     # maximum is applied, depending of the exposure time 
                            gain        = 10,       # [0 - 18.07] dB
                            auto_wb     = True,     # auto white balance: [True or False]
                            gammaY      = 0.31,     # [0.3 - 1]                                
                            width       = 544,      #1280,# [32 - 1280]
                            height      = 400,      #864,# [4 - 864]
                            offsetX     = 320, #0,#
                            offsetY     = 200, #0,#
                            snapshot    = True)    # if false => acquire video, if True => acquire an image 
#%% get a snapshot of the spatial camera
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spat_params, pattern_to_display = 'black') # white, black or gray
data = snapshot_cam(cam = cam_spat, data_format = 8, tilt_image = False) # data_format accepted: 8 or 16 bits
DMD.Halt()
#%% display spatial camera in continous mode
play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spat_params, pattern_to_display = 'black') # white, black or gray
display_cam(cam = cam_spat)
DMD.Halt()
#%% setup Spectral Camera
cam_spec_params = setup_cam(cam = cam_spec, 
                            cameras_nbr = 2,
                            expos_time  = 1.5,         # [0.001 - 1000] ms
                            frame_rate  = 4000,      # maximum is applied, depending of the exposure time 
                            gain        = 12,        # [0 - 18.07] dB
                            gammaY      = 0.31,      # [0.3 - 1]                                   
                            width       = 768,#672,#1280, #832, #L 960, #1280,#,      # [32 - 1280]
                            height      = 480,       # [4 - 864]
                            offsetX     = 260,#0, #150,#$100, #3 100, #170,#160,
                            offsetY     = 150,#230,#75,#150,
                            binningX    = 1,         # [1, 2, 4, 8 & 16]
                            binningY    = 1,         # [1, 2, 4, 8 & 16]
                            snapshot    = False)     # if false => acquire video, if True => acquire an image    
#%% get a snapshot of the spectral camera
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spec_params, pattern_to_display = 'white') # white, black or gray
data = snapshot_cam(cam = cam_spec, data_format = 16, binX = 1, binY = 1, disp_bin_effect = False, tilt_image = False) # data_format accepted: 8 or 16 bits
DMD.Halt()
plot_spectrum(data, cam_spec_params, spectrograph_params)
#%% display spectral camera in continous mode
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spec_params, pattern_to_display = 'white') # white, black or gray
display_cam(cam = cam_spec, display_max = True)
DMD.Halt()
#%% setup acquisition
setup_version            = 'setup_v2.0'
collection_access        = 'public' #'private'#
Np                       = 128      # Number of pixels in one dimension of the image (image: NpxNp)
ti                       = cam_spec_params.exposure_time_μs / 1000        # Integration time of the spectral camera
NAverages                = 1 # Number of avegare (the acquisition is accumulated before moving the grating)
NRepetitions             = 1 # Number of repetitions (grating change after that, the acquisition is repeated)
Lc                       = [(550, 2)]#[(0, 1), (897, 1), (902, 1), (907, 1), (912, 1), (917, 1), (922, 1), (927, 1)] # [(0, 1), (780, 1), (785, 1), (795, 1), (805, 1), (810, 1)] #[(0, 1), (676, 1), (686, 1), (696, 1), (706, 1), (716, 1)] #[(0, 1), (557, 1), (567, 1), (577, 1), (587, 1), (597, 1)] #[(0, 1), (526, 1), (536, 1), (546, 1), (556, 1), (566, 1)] #[(0, 1), (416, 1), (426, 1), (436, 1), (446, 1), (456, 1)] ##, (832, 2), (852, 2), (872, 2), (892, 2), (912, 2), (932, 2), (952, 2), (972, 2), (992, 2)]## # a vector containig the central wavelength following by the grating number
zoom                     = 1        # Numerical zoom applied in the DMD
xw_offset                = 128#320#128#+192# - 130     # Default = 128
yh_offset                = 0#192#0#+192# - 50        # Default = 0
pattern_compression      = 1
pattern_dim              = '1D'
scan_mode                = 'Walsh'  #'Walsh_inv' #'Raster_inv' #'Raster' #
source                   = 'white_LED'#'HG-1_Oceanoptics'#'Thorlabs_White_halogen_lamp'#'White_Zeiss_lamp'#No-light'#'Bioblock'#'Laser_405nm_1.2W_A_0.14'#'''#' + white LED might'#
object_name              = 'Cat_SP_only' #'Ray-912' #'ray_405'#'nothing'   #'Arduino_box_position_1'#'biopsy-9-posterior-margin'#GP-without-sample'##-OP'#
data_folder_name         = '2025-07-25_decrease_size_cat'#'Patient-69_exvivo_LGG_BU'
data_name                = 'obj_' + object_name + '_source_' + source + '_' + scan_mode + '_im_'+str(Np)+'x'+str(Np)+'_ti_'+str(ti)+'ms_zoom_x'+str(zoom)

all_path = func_path(data_folder_name, data_name, ask_overwrite = False)
if 'mask_index' not in locals(): mask_index = [];  x_mask_coord = []; y_mask_coord = [] # execute "mask_index = []" to not apply the mask

if all_path.aborted == False:
    output_directory     = all_path.subfolder_path
    pattern_order_source = '../stats/' + pattern_dim + '/pattern_order_' + scan_mode + '_' + str(Np) + 'x' + str(Np) + '.npz'
    pattern_source       = '../Patterns/' + pattern_dim + '/' + scan_mode + '_' + str(Np) + 'x' + str(Np)
    pattern_prefix       = scan_mode + '_' + str(Np) + 'x' + str(Np)
    experiment_name      = data_name
    light_source         = source,
    object               = object_name
    filter               = 'Diffuser' #+ OD=0.3',''No filter',#'linear colored filter',#'Orange filter (600nm)',#'Dichroic_420nm',#'HighPass_500nm + LowPass_750nm + Dichroic_560nm',#'BandPass filter 560nm Dl=10nm',#'None', # + , #'Nothing',#'Diffuser + HighPass_500nm + LowPass_750nm',##'Microsope objective x40',#'' linear colored filter + OD#0',#'Nothing',#
    description          = 'for illumination: f=80mm + f=60mm, collection: spherical lens f=75mm only'
    
    acquisition_params = AcquisitionParameters(pattern_compression = pattern_compression, pattern_dimension_x = Np, pattern_dimension_y = Np, 
                                               zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset, mask_index = mask_index, 
                                               x_mask_coord = x_mask_coord, y_mask_coord = y_mask_coord, output_directory = output_directory, 
                                               pattern_order_source = pattern_order_source, pattern_source = pattern_source, pattern_prefix = pattern_prefix, 
                                               experiment_name = experiment_name, light_source = light_source, object = object, filter = filter, 
                                               NAverages = NAverages, NRepetitions = NRepetitions, Lc = Lc, description = description)
    
    import numpy as np
    # acquisition_params.wavelengths = np.linspace(450, 750, cam_spec_params.width)
    acquisition_params.wavelengths = define_wavelengths_matrix(cam_spec_params, Lc)
    acquisition_params.wavelengths = acquisition_params.wavelengths[0, :]
                        
    try: 
        change_patterns(DMD = DMD, acquisition_params = acquisition_params, zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset, 
                        force_change = True) 
    except: 
        pass
                  
    DMD_params = setup_DMD(DMD = DMD, DMD_initial_memory = DMD_initial_memory, acquisition_params = acquisition_params, integration_time = ti, add_illumination_time = 0) 

    if DMD_params.patterns != None:
        print('Total expected acq time  : ' + str(int(acquisition_params.pattern_amount*(ti+0.044)/1000 // 60)) + ' min ' + 
              str(math.floor(acquisition_params.pattern_amount*(ti+0.044)/1000 % 60)) + ' s ' + str(round((acquisition_params.pattern_amount*(ti+0.044) / 1000 % 1) * 1000)) + ' ms')
else:
    print('setup aborted')
#%% Acquire
# time.sleep(5)

acquire(DMD                 = DMD,
        DMD_params          = DMD_params,
        cam_spat            = cam_spat,
        cam_spat_params     = cam_spat_params,
        cam_spec            = cam_spec,
        cam_spec_params     = cam_spec_params,
        spectrograph        = spectrograph,
        spectrograph_params = spectrograph_params,
        acquisition_params  = acquisition_params,
        all_path            = all_path,
        verbose             = False)
#%% spectral data Reconstruction
# data_folder_name = '2025-06-27_lens_tuning'
# data_name = 'obj_Cat_fs-75mm_Lc-40mm_source_white_LED_Walsh_im_128x128_ti_1.25ms_zoom_x1'
had_reco = hadamard_reco(data_folder_name, data_name, mean_NA = False, mean_NR = False, save_spectral_data = False, 
                         save_spatial_data = False, bin_fact = cam_spec_params.height/Np/zoom, zoom = zoom)
#%% Plot
from matplotlib import pyplot as plt

if len(had_reco.shape) > 3: 
    NLc = had_reco.shape[3]
    for i in range(NLc):
        LLc = acquisition_params.Lc[i][0]    
        plot_reco_without_NN(acquisition_params, had_reco[:,:,:,i], all_path)
else:
    plot_reco_without_NN(acquisition_params, had_reco, all_path)    
    #%% transfer data to girder
    transfer_data_SPC1D(DMD_params, cam_spat_params, cam_spec_params, spectrograph_params, acquisition_params,
                        setup_version, data_folder_name, data_name, collection_access, upload_metadata = 1)
#%% spatial data Reconstruction
from matplotlib import pyplot as plt
import pickle
plot_fig = True
i = 0
data_path_folder = '../../data/' + data_folder_name + '/' + data_name + '/raw_data/'
data_file_list = os.listdir(data_path_folder)
for file in data_file_list:
    if file.startswith('spatial'):
        data_path = data_path_folder + file
        with open(data_path, "rb") as fp:
            da = pickle.load(fp)
        
        if plot_fig == True:    
            if i <= 5:
                img16 = da
                img8 = (img16/256).astype('uint8')
                plt.figure()
                plt.imshow(img8)
                plt.title(i)
                plt.colorbar()
        
        i = i + 1
        
# Np = 128
# spatial_data = np.empty((864, 1280, Np*2), dtype = np.uint16)
# plot_fig = True
# for i in range(256):#da.shape[3]):
#     # data_path = '../../data/2025-06-10_test/obj_cat2_source_white_LED_Walsh_im_128x128_ti_1.1ms_zoom_x1/raw_data/spectral_NR_1_Gr_2_L_550nm_NA_1_NS_' + str(i) + '.pkl'
#     data_path = 'C:/openspyrit/data/' + data_folder_name + '/' + data_name + '/raw_data/spatial_NR_0_Gr_2_Lc_550nm_NA_0_NS_' + str(i) + '.pkl'
#     with open(data_path, "rb") as fp:
#         da = pickle.load(fp)
        
#     # spatial_data[:, :, i] = da

#     if plot_fig == True:    
#         if i <= 5 or (i >= 120 and i < 128) or i > 250:
#             img16 = da
#             img8 = (img16/256).astype('uint8')
#             plt.figure()
#             plt.imshow(img8)
#             plt.title(i)
#             plt.colorbar()
#%% Disconnect
disconnect_DMD(DMD)
disconnect_spectrograph(spectrograph, goto_zero = False)
disconnect_cam(cam_spat)
disconnect_cam(cam_spec)

#%% below, old prog







#%% Neural Network setup (executed it just one time)
network_param = ReconstructionParameters(
    # Reconstruction network    
    M = Np*Np,                  # Number of measurements
    img_size = 128,             # Image size of the NN reconstruction
    arch = 'dc-net',            # Main architecture
    denoi = 'unet',             # Image domain denoiser (possibility to do not apply, put : None)
    subs = 'rect',              # Subsampling scheme
    
    # Training
    data = 'imagenet',          # Training database
    N0 = 10,                    # Intensity (max of ph./pixel)
    
    # Optimisation (from train2.py)
    num_epochs = 30,            # Number of training epochs
    learning_rate = 0.001,      # Learning Rate
    step_size = 10,             # Scheduler Step Size
    gamma = 0.5,                # Scheduler Decrease Rate   
    batch_size = 256,           # Size of the training batch
    regularization = 1e-7       # Regularisation Parameter
    )

cov_folder = 'C:/openspyrit/stat/ILSVRC2012_v10102019/'
cov_path = Path(cov_folder) / f'Cov_8_{network_param.img_size}x{network_param.img_size}.npy'
model_folder = 'C:/openspyrit/models/'
model, device = setup_reconstruction(cov_path, model_folder, network_param)
#%% Neural Network Reconstruction
plot_reco_with_NN(acquisition_parameters, spectral_data, model, device, network_param, all_path, cov_path)
#%% Draw a ROI
# Comment data_folder_name & data_name to draw a ROI in the current acquisition, else specify the acquisition name
data_folder_name = '2025-01-16_myFirstAcq'
data_name = 'obj_cat_source_white_LED_Walsh_im_64x64_ti_1ms_zoom_x1'
mask_index, x_mask_coord, y_mask_coord = extract_ROI_coord(DMD_params, acquisition_parameters, all_path, 
                                                           data_folder_name, data_name, GT, ti, Np)























