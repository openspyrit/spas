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
from spas.transfer_data_to_girder import transfer_data_SPIM1D
import time
import math
from spas.DMD_module import init_DMD, disconnect_DMD, change_patterns, setup_DMD, play_one_pattern
from spas.spectro_ShamrockAndor_module import init_spectrograph, disconnect_spectrograph, setup_spectrograph
from spas.cam_Andor_module import init_cam_spat, init_cam_spec, disconnect_cam, setup_cam, snapshot_cam, display_cam
from spas.PI_module import init_PI, disconnect_stage, read_position, move_to_middle, stage_adjustment, stage_parameters, go_to_zero
from spas.shutter_TSC001_module import ThorlabsShutter
from spas.flipping_mirror_MFF101_module import MFF
from spas.acquisition_SPIM1D import AcquisitionParameters, func_path, acquire, define_wavelengths_matrix, plot_spectrum
from spas.reconstruction_SPIM1D import live_hadamard_reco, spatial_reco
from spas.visualization_SPIM1D import plot_acqui
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
                            expos_time  = 0.25,  # (s)
                            gain        = 1,        # 1 or 2                              
                            width       = 2048,     # max = 2048
                            height      = 2048,     # max = 2048
                            offsetX     = 1,        # 1
                            offsetY     = 1,        # 1
                            binningX    = 1,        # int < 2048
                            binningY    = 1,        # int < 2048
                            encodPix    = 12,       # 12 or 16 bit
                            snapshot    = True)    # if false => acquire video, if True => acquire an image 
#%% get a snapshot of the spatial camera
mirror.set_position('spatial', verbose = True)
shutter.open()
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spat_params, zoom = 1, pattern_to_display = 'white', pattern_dim = '1D',
                              scan_mode = 'Walsh', Np = 256, pattern_thickness = 4) # white, black or gray_ + pattern number
data = snapshot_cam(cam = cam_spat, tilt_image = True) # data_format accepted: 8 or 16 bits
DMD.Halt()
shutter.close()
#%% display spatial camera in continous mode
mirror.set_position('spatial', verbose = True)
shutter.open()
stage_adjustment(stage.pidevice)
time.sleep(1)
play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spat_params, pattern_to_display = 'gray_63', pattern_dim = '1D', 
                 scan_mode = 'Walsh', Np = 256, pattern_thickness = 4) #gray_33
display_cam(cam = cam_spat, cam_params = cam_spat_params, display_profile = True)
DMD.Halt()
shutter.close()
position = read_position(stage.pidevice, stage.stage_tools, verbose = True)
#%% setup the Spectrograph
spectrograph_params = setup_spectrograph(spectrograph,
                                         grating_nbr =   1, print_select   = True,   # Arg:  1 
                                         position    = 595, print_position = True,   # the central wavelength of the grating
                                         slit_width  = 20000)                          # the width of the slit in (µm)
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
                            encodPix    = 12,       # 12 or 16 bit
                            snapshot    = False)    # if false => acquire video, if True => acquire an image   
#%% get a snapshot of the spectral camera
mirror.set_position('spectral', verbose = True)
shutter.open()
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spec_params, zoom = 1, pattern_to_display = 'white', pattern_dim = '1D', 
                              scan_mode = 'Walsh', Np = 256, pattern_thickness = 4) 
data = snapshot_cam(cam = cam_spec, tilt_image = False) # data_format accepted: 8 or 16 bits
DMD.Halt()
plot_spectrum(data, cam_spec_params, spectrograph_params)
shutter.close()
#%% display spectral camera in continous mode
mirror.set_position('spectral', verbose = True)
shutter.open()
# stage_adjustment(stage.pidevice)
# time.sleep(1)
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spec_params, pattern_to_display = 'white', pattern_dim = '1D', 
                              scan_mode = 'Walsh', Np = 256, pattern_thickness = 4) 
display_cam(cam = cam_spec, cam_params = cam_spec_params, display_max = False, display_integral = True)
DMD.Halt()
shutter.close()
#%% setup acquisition
setup_version            = 'setup_v1.0'
collection_access        = 'public' #'private'#
Np                       = 256      # Number of pixels in one dimension of the image (image: NpxNp)
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
scan_mode                = 'Walsh'  #'Walsh_inv' #'Raster_inv' #'Raster' #
source                   = 'laser-473nm'#white_LED'#'No source'#'White_Zeiss_lamp'#'Thorlabs_White_halogen_lamp'#'HG-1_Oceanoptics'#No-light'#'Bioblock'#'Laser_405nm_1.2W_A_0.14'#'''#' + white LED might'#
object_name              = 'fluo_µsphere-gel-bin10' #'fluo_cuve'#
data_folder_name         = '2026-04-17_test'#'Patient-69_exvivo_LGG_BU'
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
    description          = 'CL=285mm. with Notch filter.'
    
    acquisition_params = AcquisitionParameters(pattern_compression = pattern_compression, pattern_dimension_x = pattern_thickness, pattern_dimension_y = Np, 
                                               zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset, mask_index = mask_index, 
                                               x_mask_coord = x_mask_coord, y_mask_coord = y_mask_coord, output_directory = output_directory, 
                                               pattern_order_source = pattern_order_source, pattern_source = pattern_source, pattern_prefix = pattern_prefix, 
                                               experiment_name = experiment_name, light_source = light_source, object = object, filter = filter, 
                                               NAverages = NAverages, NRepetitions = NRepetitions, Lc = Lc, Nz = Nz, description = description)
    
    acquisition_params.wavelengths = define_wavelengths_matrix(cam_spec_params, Lc, display_figure = True, verbose = True)  
    acquisition_params.wavelengths = acquisition_params.wavelengths[0, :]
    acquisition_params.wavelengths = np.linspace(spectrograph_params.position - 50,
                                                 spectrograph_params.position + 50, 
                                                 int(cam_spec_params.width))
    
    stage_params = stage_parameters
    stage_params.array_to_move = array_to_move
                        
    try: 
        change_patterns(DMD = DMD, acquisition_params = acquisition_params, zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset, 
                        force_change = False) 
    except: 
        print('pass by exception')
        pass
                  
    DMD_params = setup_DMD(DMD = DMD, DMD_initial_memory = DMD_initial_memory, acquisition_params = acquisition_params, 
                           integration_time = ti, add_illumination_time = 30000) 
    

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
#%% Reconstruction
from scipy import signal
new_raw = np.zeros(raw_data.shape)
for i in range(raw_data.shape[2]):
    new_raw[:,:,i,0,0,0] = signal.medfilt2d(raw_data[:,:,i,0,0,0], kernel_size=3)
    
had_reco_all = live_hadamard_reco(new_raw, acquisition_params)
spatial_acqui = spatial_reco(acquisition_params, all_path)
#%% Plot
plot_acqui(had_reco_all, spatial_acqui, acquisition_params, all_path)
#%% plot raw data
from matplotlib import pyplot as plt

for i in range(3):
    plt.figure()
    plt.imshow(np.squeeze(raw_data[:, :, i, 0, 0, 0]))
    plt.colorbar()
    plt.title('pattern n° ' + str(i))

plt.figure()
plt.imshow(np.squeeze(raw_data[:, 0, :, 0, 0, 0]))
plt.title('y = 0')

plt.figure()
plt.imshow(np.squeeze(raw_data[0, :, :, 0, 0, 0]))
plt.title('x = 0')

plt.figure()
plt.plot(np.squeeze(np.mean(np.mean(raw_data, axis=1),axis=0)))
plt.title('mean for each pattern')
plt.xlabel('pattern number')
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
disconnect_spectrograph(spectrograph, goto_zero = True)
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























