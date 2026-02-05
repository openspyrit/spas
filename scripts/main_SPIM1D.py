#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 09:08:24 2025
@author: mahieu

The main software to acquire in 1D an hyperspectral cube with the single pixel camera
"""


# packages
# import time
import math
import os
os.chdir('E:\\openspyrit\\spas\\scripts')
from spas.DMD_module import init_DMD, disconnect_DMD, change_patterns, setup_DMD, play_one_pattern
from spas.spectro_ShamrockAndor_module import init_spectrograph, disconnect_spectrograph, setup_spectrograph
from spas.cam_Andor_module import init_cam_spat, init_cam_spec, disconnect_cam, setup_cam, snapshot_cam, display_cam
from spas.PI_module import init_PI, disconnect_stage, read_position, move_to_middle, manual_adjustment_stage
from spas.acquisition_SPIM1D import AcquisitionParameters, func_path, acquire, define_wavelengths_matrix, plot_spectrum
# from spas.reconstruction_SPC1D import hadamard_reco
# from spas.visualization_SCP1D import plot_reco_without_NN
# from spas.transfer_data_to_girder import transfer_data_SPC1D
#%% Initialize hardware
DMD, DMD_initial_memory = init_DMD(dmd_lib_version = '4.3')
spectrograph = init_spectrograph(model = 'andor_shamrock')
cam_spat = init_cam_spat(SN = 'VSC-10323')
cam_spec = init_cam_spec(SN = 'VSC-23585')
pidevice, stage_tools = init_PI(Model = 'C-884', SN = '0000000000', verbose = True)
#%% Move the PI stage to the middle
move_to_middle(pidevice, stage_tools)
position = read_position(pidevice, stage_tools, verbose = True)
#%%
# here add manual displacment
# Emergency stop
stage_tools.stopall(pidevice)
#%% setup the Spectrograph
spectrograph_params = setup_spectrograph(spectrograph,
                                         grating_nbr =   1, print_select   = True,   # Arg:  1 
                                         position    = 600, print_position = True,   # the central wavelength of the grating
                                         slit_width  = 200)                          # the width of the slit in (µm)
#%% setup Spatial Camera
cam_spat_params = setup_cam(cam = cam_spat, 
                            expos_time  = 0.2,  # (s)
                            gain        = 1,        # 1 or 2                              
                            width       = 2048,     # max = 2048
                            height      = 2048,     # max = 2048
                            offsetX     = 1,        # 1
                            offsetY     = 1,        # 1
                            binningX    = 1,        # int < 2048
                            binningY    = 1,        # int < 2048
                            snapshot    = False)    # if false => acquire video, if True => acquire an image 
#%% get a snapshot of the spatial camera
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spat_params, zoom = 1, pattern_to_display = 'gray', pattern_dim = '1D',
                              scan_mode = 'Walsh', Np = 256, pattern_thickness = 16) # white, black or gray
data = snapshot_cam(cam = cam_spat, tilt_image = True) # data_format accepted: 8 or 16 bits
DMD.Halt()
#%% display spatial camera in continous mode
manual_adjustment_stage(pidevice)
play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spat_params, pattern_to_display = 'gray', pattern_dim = '1D', 
                              scan_mode = 'Walsh', Np = 256, pattern_thickness = 16) 
# display_cam(cam = cam_spat, binningX = 1, binningY = 1)
display_cam(cam = cam_spat, cam_params = cam_spat_params)
DMD.Halt()
#%% setup Spectral Camera
cam_spec_params = setup_cam(cam = cam_spec, 
                            expos_time  = 0.5,        # (s)
                            gain        = 2,        # 1 or 2                              
                            width       = 2048,     # max = 2048
                            height      = 2048,     # max = 2048
                            offsetX     = 1,        # 1
                            offsetY     = 1,        # 1
                            binningX    = 4,        # int < 2048        
                            binningY    = 4,        # int < 2048
                            snapshot    = False)    # if false => acquire video, if True => acquire an image   
#%% get a snapshot of the spectral camera
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spec_params, zoom = 1, pattern_to_display = 'white', pattern_dim = '1D', 
                              scan_mode = 'Walsh', Np = 256, pattern_thickness = 16) 
data = snapshot_cam(cam = cam_spec, tilt_image = False) # data_format accepted: 8 or 16 bits
DMD.Halt()
plot_spectrum(data, cam_spec_params, spectrograph_params)
#%% display spectral camera in continous mode
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spec_params, pattern_to_display = 'white', pattern_dim = '1D', 
                              scan_mode = 'Walsh', Np = 256, pattern_thickness = 16) 
display_cam(cam = cam_spec, cam_params = cam_spec_params)
DMD.Halt()
#%% setup acquisition
setup_version            = 'setup_v2.0'
collection_access        = 'public' #'private'#
Np                       = 256      # Number of pixels in one dimension of the image (image: NpxNp)
pattern_thickness        = 16
ti                       = cam_spec_params.exposure_time_μs / 1000 # Integration time of the spectral camera
NAverages                = 1        # Number of avegare (the acquisition is accumulated before moving the grating)
NRepetitions             = 1        # Number of repetitions (grating change after that, the acquisition is repeated)
Lc                       = [(spectrograph_params.position, spectrograph_params.grating.current_grating_nbr)]#, (570, 1), (600, 1), (630, 1), (660, 1), (690, 1), (720, 1)]#, (922, 1), (927, 1)] # [(0, 1), (780, 1), (785, 1), (795, 1), (805, 1), (810, 1)] #[(0, 1), (676, 1), (686, 1), (696, 1), (706, 1), (716, 1)] #[(0, 1), (557, 1), (567, 1), (577, 1), (587, 1), (597, 1)] #[(0, 1), (526, 1), (536, 1), (546, 1), (556, 1), (566, 1)] #[(0, 1), (416, 1), (426, 1), (436, 1), (446, 1), (456, 1)] ##, (832, 2), (852, 2), (872, 2), (892, 2), (912, 2), (932, 2), (952, 2), (972, 2), (992, 2)]## # a vector containig the central wavelength following by the grating number
zoom                     = 1        # Numerical zoom applied in the DMD
xw_offset                = 128      # Default = 128
yh_offset                = 0        # Default = 0
pattern_compression      = 1
pattern_dim              = '1D'
scan_mode                = 'Walsh'  #'Walsh_inv' #'Raster_inv' #'Raster' #
source                   = 'laser-532nm'#white_LED'#'No source'#'White_Zeiss_lamp'#'Thorlabs_White_halogen_lamp'#'HG-1_Oceanoptics'#No-light'#'Bioblock'#'Laser_405nm_1.2W_A_0.14'#'''#' + white LED might'#
object_name              = 'fluo_µsphere-gel-bin5' 
data_folder_name         = '2026-01-29_test_SPIM'#'Patient-69_exvivo_LGG_BU'
data_name                = 'obj_' + object_name + '_source_' + source + '_Lc_' + str(Lc[0][0]) + 'nm_Gr_' + str(Lc[0][1]) + '_' + scan_mode + '_im_'+str(Np)+'x'+str(Np)+'_ti_'+str(round(ti))+'ms_zoom_x'+str(zoom)

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
    filter               = 'Diffuser' #+ OD=0.3',''No filter',#'linear colored filter',#'Orange filter (600nm)',#'Dichroic_420nm',#'HighPass_500nm + LowPass_750nm + Dichroic_560nm',#'BandPass filter 560nm Dl=10nm',#'None', # + , #'Nothing',#'Diffuser + HighPass_500nm + LowPass_750nm',##'Microsope objective x40',#'' linear colored filter + OD#0',#'Nothing',#
    description          = 'with Notch filter.'
    
    acquisition_params = AcquisitionParameters(pattern_compression = pattern_compression, pattern_dimension_x = pattern_thickness, pattern_dimension_y = Np, 
                                               zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset, mask_index = mask_index, 
                                               x_mask_coord = x_mask_coord, y_mask_coord = y_mask_coord, output_directory = output_directory, 
                                               pattern_order_source = pattern_order_source, pattern_source = pattern_source, pattern_prefix = pattern_prefix, 
                                               experiment_name = experiment_name, light_source = light_source, object = object, filter = filter, 
                                               NAverages = NAverages, NRepetitions = NRepetitions, Lc = Lc, description = description)
    
    acquisition_params.wavelengths = define_wavelengths_matrix(cam_spec_params, Lc, display_figure = True, verbose = True)  
    acquisition_params.wavelengths = acquisition_params.wavelengths[0, :]
                        
    try: 
        change_patterns(DMD = DMD, acquisition_params = acquisition_params, zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset, 
                        force_change = False) 
    except: 
        print('pass by exception')
        pass
                  
    DMD_params = setup_DMD(DMD = DMD, DMD_initial_memory = DMD_initial_memory, acquisition_params = acquisition_params, 
                           integration_time = ti, add_illumination_time = 30000) 
    

    if DMD_params.patterns != None:
        print('Total expected acq time  : ' + str(int(acquisition_params.pattern_amount*(ti + 30)/1000 // 60)) + ' min ' + 
              str(math.floor(acquisition_params.pattern_amount*(ti + 30)/1000 % 60)) + ' s ' + 
              str(round((acquisition_params.pattern_amount*(ti + 30) / 1000 % 1) * 1000)) + ' ms')
else:
    print('setup aborted')
#%% Acquire
# time.sleep(20)
raw_data = acquire(DMD                 = DMD,
                   DMD_params          = DMD_params,
                   cam_spat            = cam_spat,
                   cam_spat_params     = cam_spat_params,
                   cam_spec            = cam_spec,
                   cam_spec_params     = cam_spec_params,
                   spectrograph        = spectrograph,
                   spectrograph_params = spectrograph_params,
                   acquisition_params  = acquisition_params,
                   all_path            = all_path,
                   verbose             = False,
                   acquisition_arm     = 'spectral')
#%% spectral data Reconstruction
import spyrit.misc.walsh_hadamard as wh
import numpy as np
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
#%%
M_sub = raw_data[:, :, 0::2] - raw_data[:, :, 1::2]
bin_fact = 4
bin_image = binArray(M_sub, 0, bin_fact, bin_fact)
had_reco = wh.fwht(bin_image) / Np
had_reco = np.swapaxes(had_reco, 2, 1) 
#%% plot
plt.figure()
plt.imshow(had_reco.sum(axis=2))
plt.colorbar()
plt.title('sum of wavelength')

plt.figure()
plt.plot(acquisition_params.wavelengths, np.sum(np.sum(had_reco, axis=1), axis=0))
plt.xlabel('wavelength (nm)')
plt.grid()
plt.title('spectrum')
#%% plot raw data
for i in range(0,4,1):
    plt.figure()
    plt.imshow(raw_data[:,:,i])
    plt.colorbar()
    plt.title('i = ' + str(i))

profile_pattern = np.mean(np.mean(raw_data, axis = 1), axis = 0)

plt.figure()
plt.plot(profile_pattern, 'o')
plt.title('pattern integration')

vec = []
plt.figure()
for i in range(4):
    plt.plot(np.mean(raw_data[:, :, i], axis = 0))
    vec.append(i)

plt.legend(vec)
plt.grid()
plt.title('spectrum of pattern n°:')
plt.show()
#%% plot proflies
for i in range(100, 110):
    plt.figure()
    plt.imshow(raw_data[:,:,i])
    plt.title('i = ' + str(i))
#%% from tikhonow
from spyrit.core.inverse import Tikhonov as tik
M_sub = raw_data[:, :, 0::2] - raw_data[:, :, 1::2]
had_reco = tik(M_sub, sigma = 0.1)# / Np

#%%
# data_folder_name = '2025-09-02_test_save-mpz'
# data_name = 'obj_cat_pickle18.06dB_source_white_LED_Gr_2_Walsh_im_64x64_ti_1.0ms_zoom_x1'
had_reco, spectral_data = hadamard_reco(data_folder_name, data_name, mean_NA = False, mean_NR = False, save_spectral_data = True, 
                         save_spatial_data = False, bin_fact = cam_spec_params.height/Np/1.3, zoom = zoom)   
# cam_spec_params.height/Np/.73
import numpy as np
# had_reco = np.delete(had_reco, (0), axis=0)
had_reco[-1,:,:] = 0
# #%% 
# from matplotlib import pyplot as plt
# i_init = 40
# i = i_init
# while True:
#     plt.figure()
#     plt.imshow(spectral_data[:,:,i])
#     plt.colorbar()
#     plt.title('i = ' + str(i))
#     i = i + 1
#     if i > 20 + i_init:
#         break
#%% Plot
if len(had_reco.shape) > 3: 
    NLc = had_reco.shape[3]
    for i in range(2):#NLc):
        LLc = acquisition_params.Lc[i][0]    
        plot_reco_without_NN(acquisition_params, had_reco[:,:,:,i], all_path)
else:
    plot_reco_without_NN(acquisition_params, had_reco, all_path)    
    #%% transfer data to girder
    transfer_data_SPC1D(DMD_params, cam_spat_params, cam_spec_params, spectrograph_params, acquisition_params,
                        setup_version, data_folder_name, data_name, collection_access, upload_metadata = 1)
#%% spatial data Reconstruction
from matplotlib import pyplot as plt
import numpy as np

fi = 'spatial'#'spectral'#
plot_fig = True
i = 0
data_path_folder = '../../data/' + data_folder_name + '/' + data_name + '/raw_data/'
data_file_list = os.listdir(data_path_folder)
for file in data_file_list:
    if file.startswith(fi):
        data_path = data_path_folder + file
        # data_path = 'test.npz'
        file = np.load(data_path)    
        da = file['arr_0']
        
        if plot_fig == True:    
            if i <= 5:
                img16 = da
                if fi == 'spatial' and cam_spat.get_imgdataformat() == 'XI_RGB48':
                    img8 = (img16/256).astype('uint8')
                    img8 = np.flip(np.flip(img8, axis = 1), axis = 0)
                elif fi == 'spectral':
                    img8 = da
                else:
                    img8 = np.flip(np.flip(da, axis = 1), axis = 0)
                # img8 = da
                plt.figure()
                plt.imshow(img8)
                plt.title(i)
                plt.colorbar()
        
        i = i + 1
#%% Disconnect
disconnect_DMD(DMD)
disconnect_spectrograph(spectrograph, goto_zero = False)
disconnect_cam(cam_spat)
disconnect_cam(cam_spec)
disconnect_stage(pidevice, stage_tools)
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























