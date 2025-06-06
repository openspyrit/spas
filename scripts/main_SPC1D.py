#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 09:08:24 2025
@author: mahieu

The main software to acquire in 1D an hyperspectral cube with the single pixel camera
"""


#%% new packages
from spas.DMD_module import init_DMD, disconnect_DMD, change_patterns, setup_DMD, play_one_pattern
from spas.spectro_SP_module import init_spectrograph, disconnect_spectrograph, setup_spectrograph
from spas.cam_Ximea_module import init_cam_spat, init_cam_spec, disconnect_cam, setup_cam, snapshot_cam, display_cam
from spas.acquisition_SPC1D import AcquisitionParameters, func_path
import os
os.chdir('C:\\openspyrit\\spas\\scripts')
# from matplotlib import pyplot as plt
#%% old packages
# from spas.reconstruction import reconstruction_hadamard
# from spas.reconstruction_nn import ReconstructionParameters, setup_reconstruction
# from spas.visualization import snapshotVisu, plot_reco_without_NN, plot_reco_with_NN, extract_ROI_coord
# from spas.transfer_data_to_girder import transfer_data_2arms
# import spyrit.misc.walsh_hadamard as wh
# import time
# from pathlib import Path
#%% Initialize hardware
DMD, DMD_initial_memory = init_DMD(dmd_lib_version = '4.2')
spectrograph = init_spectrograph(model = 'CM110')
cam_spat = init_cam_spat(SN = 'BRCID2503000')
cam_spec = init_cam_spec(SN = 'BRMID2503000')
#%% setup the Spectrograph
Spectrograph_Parameters = setup_spectrograph(spectrograph,
                                             grating_nbr =    2, print_select   = True,   # Arg:  1 (High Resoluton), 2 (Low Resoluton)
                                             position    =  550, print_position = True,
                                             unit        = 'nm', print_unit     = True)    # Arg: 'A', 'nm', 'µm'
#%% setup Spatial Camera
cam_spat_Parameters = setup_cam(cam = cam_spat, 
                                cameras_nbr = 2,        # number of camera 
                                expos_time  = 0.025,     # [0.001 - 1000] ms
                                frame_rate  = 4000,     # maximum is applied, depending of the exposure time 
                                gain        = 0,       # [0 - 18.07] dB
                                auto_wb     = True,     # auto white balance: [True or False]
                                gammaY      = 0.31,        # [0.3 - 1]                                
                                width       = 768,     # [32 - 1280]
                                height      = 576,      # [4 - 864]
                                offsetX     = 286,
                                offsetY     = 235)
#%% get a snapshot of the spatial camera
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spat_Parameters, pattern_to_display = 'black') # white, black or gray
data = snapshot_cam(cam = cam_spat, data_format = 8) # data_format accepted: 8 or 16 bits
DMD.Halt()
#%% display spatial camera in continous mode
play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spat_Parameters, pattern_to_display = 'black') # white, black or gray
display_cam(cam = cam_spat)
DMD.Halt()
#%% setup Spectral Camera
cam_spec_Parameters = setup_cam(cam = cam_spec, 
                                cameras_nbr = 2,
                                expos_time  = 2,       # [0.001 - 1000] ms
                                frame_rate  = 4000,      # maximum is applied, depending of the exposure time 
                                gain        = 18,        # [0 - 18.07] dB
                                gammaY      = 0.31,       # [0.3 - 1]                                
                                width       = 1280,      # [32 - 1280]
                                height      = 864,       # [4 - 864]
                                offsetX     = 0,
                                offsetY     = 0,
                                binningX    = 1,         # [1, 2, 4, 8 & 16]
                                binningY    = 1)         # [1, 2, 4, 8 & 16]
#%% get a snapshot of the spectral camera
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spec_Parameters, pattern_to_display = 'white') # white, black or gray
data = snapshot_cam(cam = cam_spec, data_format = 16, binX = 4, binY = 4, disp_bin_effect = True) # data_format accepted: 8 or 16 bits
DMD.Halt()
#%% display spectral camera in continous mode
DMD_params = play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_spec_Parameters, pattern_to_display = 'white') # white, black or gray
display_cam(cam = cam_spec)
DMD.Halt()
#%% setup acquisition
setup_version            = 'setup_v2.0'
collection_access        = 'public' #'private'#
Np                       = 128      # Number of pixels in one dimension of the image (image: NpxNp)
ti                       = cam_spec_Parameters.exposure_time_μs / 1000        # Integration time of the spectrometer   
zoom                     = 1        # Numerical zoom applied in the DMD
xw_offset                = 128      # Default = 128
yh_offset                = 0        # Default = 0
pattern_compression      = 1
pattern_dim              = '1D'
scan_mode                = 'Walsh'  #'Walsh_inv' #'Raster_inv' #'Raster' #
source                   = 'white_LED'#White_Zeiss_lamp'#No-light'#'Bioblock'#'Thorlabs_White_halogen_lamp'#'Laser_405nm_1.2W_A_0.14'#'''#' + white LED might'#'HgAr multilines Source (HG-1 Oceanoptics)'
object_name              = 'cat'   #'Arduino_box_position_1'#'biopsy-9-posterior-margin'#GP-without-sample'##-OP'#
data_folder_name         = '2025-06-06_test'#'Patient-69_exvivo_LGG_BU'
data_name                = 'obj_' + object_name + '_source_' + source + '_' + scan_mode + '_im_'+str(Np)+'x'+str(Np)+'_ti_'+str(ti)+'ms_zoom_x'+str(zoom)

all_path = func_path(data_folder_name, data_name, ask_overwrite = True)
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
    description          = 'V2 SPAS in construction'
    
    acquisition_parameters = AcquisitionParameters(pattern_compression = pattern_compression, pattern_dimension_x = Np, pattern_dimension_y = Np, 
                                                   zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset, mask_index = mask_index, 
                                                   x_mask_coord = x_mask_coord, y_mask_coord = y_mask_coord, output_directory = output_directory, 
                                                   pattern_order_source = pattern_order_source, pattern_source = pattern_source, pattern_prefix = pattern_prefix, 
                                                   experiment_name = experiment_name, light_source = light_source, object = object, filter = filter, description = description)
                            
    try: 
        change_patterns(DMD = DMD, acquisition_params = acquisition_parameters, zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset, force_change = False) 
    except: 
        pass
          
        
    DMD_params = setup_DMD(DMD = DMD, DMD_initial_memory = DMD_initial_memory, acquisition_params = acquisition_parameters, integration_time = ti, add_illumination_time = 0) 
    
    # spectrometer_params, DMD_params, camPar = setup_acqui(spectrometer = spectrometer, DMD = DMD, camPar = camPar, DMD_initial_memory = DMD_initial_memory, 
    #                                                       metadata = metadata, acquisition_params = acquisition_parameters, DMD_output_synch_pulse_delay = 0, 
    #                                                       integration_time = ti)

    if DMD_params.patterns != None:
        print('Total expected acq time  : ' + str(int(acquisition_parameters.pattern_amount*(ti+0.356)/1000 // 60)) + ' min ' + 
              str(round(acquisition_parameters.pattern_amount*(ti+0.356)/1000 % 60)) + ' s')
else:
    print('setup aborted')

#%% Acquire

import time
import threading
from ximea import xiapi
import numpy as np
import math
import pickle
# import PIL.Image

def runCam_thread(cam, Spectrograph_Parameters, acquisition_parameters,  Np, start_chrono): 
    """Acquire video with the IDS monochrome camera in a thread

    Parameters:
    ----------
    camPar (CAM):
        Metadata object of the IDS monochrome camera
    start_chrono : int
        to save a delay for each acquisition frame of the video.

    Returns:
    -------
    None.
    """

    img = xiapi.Image()
    arm = cam.arm
    NR = 1
    NA = 1

    file_name = arm + '_NR_' + str(NR) + '_Gr_' + str(Spectrograph_Parameters.grating.current_grating_nbr) + '_L_' + str(Spectrograph_Parameters.position) + 'nm_NA_' + str(NA) + '_NS_'
    ####################### start data acquisition ############################
    print('Starting ' + arm + ' data acquisition...\n')
    cam.start_acquisition()
    
    print_time = False
    print_inc = False
    time_stmp_0 = (img.tsSec) + ((img.tsUSec)/1000000)
    
    
    i = 0
    while True: 
        counter = time.time() - start_chrono
        if i >= 2*Np:
            print('iteration reach : ' + str(2*Np))
            print('thread breaked by iteration')
            print('saving acqui ...')
            # cam.stop_acquisition()
            break
        elif counter > math.ceil(acquisition_parameters.pattern_amount*(ti+0.356)/1000) + 4:
            print('delay > ' + str(math.ceil(acquisition_parameters.pattern_amount*(ti+0.356)/1000) + 4) + 's')
            print('thread breaked by time')
            print('saving acqui ...')
            
            break        
        else:
            ############## get data and pass them from cameras to img #################
            if print_time == True:
                t1 = time.time()
            cam.get_image(img)
            if print_time == True:
                print('t1 = ' + str(time.time() - t1))
            ################### get image data as numpy array #########################
            if print_time == True:
                t2 = time.time()
            data_np = img.get_image_data_numpy()#(invert_rgb_order = True)  
            if print_time == True:
                print('t2 = ' + str(time.time() - t2))
            if print_time == True:
                t3 = time.time()
            
            with open(all_path.raw_data_path + '/' + file_name + str(i) + '.pkl', 'wb') as outp:
                pickle.dump(data_np, outp, pickle.HIGHEST_PROTOCOL)
            # image = PIL.Image.fromarray(data_np, 'L')
            # time_stmp = (img.tsSec) + ((img.tsUSec)/1000000)
            # image.save(all_path.raw_data_path + '/' + file_name + str(i) + '.BMP')
            # print('Time Stamp_0:{}s' .format(time_stmp_0))
            # print('Time Stamp:{}s' .format(time_stmp))
            # print('i: ' + str(i) + '-->  Time Stamp: ' + str(time_stmp - time_stmp_0))
            
            if print_time == True:
                print('t3 = ' + str(time.time() - t3))
            
            i = i + 1

        if print_inc == True:
            if i % 10 == 0 or i >= 250:
                print('i = ' + str(i))


start_chrono = time.time()
x = threading.Thread(target = runCam_thread, args=(cam_spat, Spectrograph_Parameters, acquisition_parameters, Np, start_chrono))
x.start()

x1 = threading.Thread(target = runCam_thread, args=(cam_spec, Spectrograph_Parameters, acquisition_parameters, Np, start_chrono))
x1.start()

time.sleep(0.9)

DMD.Run(loop=False)

first_pass = True
while(True):
    if first_pass == True:
        time.sleep(math.ceil(acquisition_parameters.pattern_amount*(ti+0.356)/1000))
        first_pass = False
        
    time.sleep(0.1)
    counter = time.time() - start_chrono
    if counter > math.ceil(acquisition_parameters.pattern_amount*(ti+0.356)/1000) + 5:
        print('delay > ' + str(math.ceil(acquisition_parameters.pattern_amount*(ti+0.356)/1000 + 5)) + 's')
        print('main loop stopped')
        break
        
DMD.Halt()

print('\n----------- COUNTERS SPATIAL CAM -----------') # reading counters
cam_spat.set_counter_selector('XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES')
print('Transport skipped frames: ', cam_spat.get_counter_value())
cam_spat.set_counter_selector('XI_CNT_SEL_API_SKIPPED_FRAMES')
print('API skipped frames: ', cam_spat.get_counter_value())
cam_spat.set_counter_selector('XI_CNT_SEL_TRANSPORT_TRANSFERRED_FRAMES')
print('Transferred frames: ', str(cam_spat.get_counter_value()) + ' / ' + str(2*Np))

print('\n----------- COUNTERS SPECTRAL CAM -----------') # reading counters
cam_spec.set_counter_selector('XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES')
print('Transport skipped frames: ', cam_spec.get_counter_value())
cam_spec.set_counter_selector('XI_CNT_SEL_API_SKIPPED_FRAMES')
print('API skipped frames: ', cam_spec.get_counter_value())
cam_spec.set_counter_selector('XI_CNT_SEL_TRANSPORT_TRANSFERRED_FRAMES')
print('Transferred frames: ', str(cam_spec.get_counter_value()) + ' / ' + str(2*Np))

cam_spat.stop_acquisition()
cam_spec.stop_acquisition()

# spectral_data = acquire(
#     ava                 = spectrometer,
#     DMD                 = DMD,
#     camPar              = camPar,
#     metadata            = metadata,
#     spectrometer_params = spectrometer_params,
#     DMD_params          = DMD_params,
#     acquisition_params  = acquisition_parameters,
#     repetitions         = 1,
#     reconstruct         = False)
#%% Hadamard Reconstruction

#%% Disconnect
disconnect_DMD(DMD)
disconnect_spectrograph(spectrograph, goto_zero = False)
disconnect_cam(cam_spat)
disconnect_cam(cam_spec)
#%% below, old prog









#%% Setup acquisition and send pattern to the DMD
setup_version            = 'setup_v1.3.1'
collection_access        = 'public' #'private'#
Np                       = 64       # Number of pixels in one dimension of the image (image: NpxNp)
ti                       = 1        # Integration time of the spectrometer   
zoom                     = 1        # Numerical zoom applied in the DMD
xw_offset                = 128      # Default = 128
yh_offset                = 0      # Default = 0
pattern_compression      = 1
scan_mode                = 'Walsh'  #'Walsh_inv' #'Raster_inv' #'Raster' #
source                   = 'white_LED'#White_Zeiss_lamp'#No-light'#'Bioblock'#'Thorlabs_White_halogen_lamp'#'Laser_405nm_1.2W_A_0.14'#'''#' + white LED might'#'HgAr multilines Source (HG-1 Oceanoptics)'
object_name              = 'cat_roi_fh'#'Arduino_box_position_1'#'biopsy-9-posterior-margin'#GP-without-sample'##-OP'#
data_folder_name         = '2025-01-16_myFirstAcq2'#'Patient-69_exvivo_LGG_BU'
data_name                = 'obj_' + object_name + '_source_' + source + '_' + scan_mode + '_im_'+str(Np)+'x'+str(Np)+'_ti_'+str(ti)+'ms_zoom_x'+str(zoom)

camPar.acq_mode          = 'snapshot'# 'video'   #
camPar.vidFormat         = 'avi'     #'bin'#
camPar.insert_patterns   = 0         # 0: no insertion / 1: insert white patterns for the camera / In the case of snapshot, put 0 to avoid bad reco
camPar.gate_period       = 16        # a multiple of the integration time of the spectro, between [2 - 16] (2: insert one white pattern between each pattern)
camPar.black_pattern_num = 1         # insert the picture number (in the pattern_source folder) of the pattern you want to insert
all_path = func_path(data_folder_name, data_name, ask_overwrite = True)
if 'mask_index' not in locals(): mask_index = [];  x_mask_coord = []; y_mask_coord = [] # execute "mask_index = []" to not apply the mask

if all_path.aborted == False:
    metadata = MetaData(
        output_directory     = all_path.subfolder_path,
        pattern_order_source = 'C:/openspyrit/spas/stats/pattern_order_' + scan_mode + '_' + str(Np) + 'x' + str(Np) + '.npz',
        pattern_source       = 'C:/openspyrit/spas/Patterns/' + scan_mode + '_' + str(Np) + 'x' + str(Np),
        pattern_prefix       = scan_mode + '_' + str(Np) + 'x' + str(Np),
        experiment_name      = data_name,
        light_source         = source,
        object               = object_name,
        filter               = 'Diffuser', #+ OD=0.3',''No filter',#'linear colored filter',#'Orange filter (600nm)',#'Dichroic_420nm',#'HighPass_500nm + LowPass_750nm + Dichroic_560nm',#'BandPass filter 560nm Dl=10nm',#'None', # + , #'Nothing',#'Diffuser + HighPass_500nm + LowPass_750nm',##'Microsope objective x40',#'' linear colored filter + OD#0',#'Nothing',#
        description          = 'test with pinehole to have a point source'
        # description          = 'two positions of the lens 80mm, P1:12cm (zoom=0.5), P2:22cm (zoom=1.5) from the DMD. Dichroic plate (T:>420nm, R:<420nm), HighPass_500nm in front of the cam, GP: Glass Plate, OP: other position, OA: out of anapath',
                        )    
    try: change_patterns(DMD = DMD, acquisition_params = acquisition_parameters, zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset,
                         force_change = False)
    except: pass
          
    acquisition_parameters = AcquisitionParameters(pattern_compression = pattern_compression, pattern_dimension_x = Np, pattern_dimension_y = Np, 
                                                   zoom = zoom, xw_offset = xw_offset, yh_offset = yh_offset, mask_index = mask_index, 
                                                   x_mask_coord = x_mask_coord, y_mask_coord = y_mask_coord)
        
    spectrometer_params, DMD_params, camPar = setup_2arms(spectrometer = spectrometer, DMD = DMD, camPar = camPar, DMD_initial_memory = DMD_initial_memory, 
                                                          metadata = metadata, acquisition_params = acquisition_parameters, DMD_output_synch_pulse_delay = 0, 
                                                          integration_time = ti)

    if DMD_params.patterns != None:
        print('Total expected acq time  : ' + str(int(acquisition_parameters.pattern_amount*(ti+0.356)/1000 // 60)) + ' min ' + 
              str(round(acquisition_parameters.pattern_amount*(ti+0.356)/1000 % 60)) + ' s')
else:
    print('setup aborted')
#%% Acquire
# time.sleep(0)
if camPar.acq_mode == 'video':
    spectral_data = acquire_2arms(
        ava                 = spectrometer,
        DMD                 = DMD,
        camPar              = camPar,
        metadata            = metadata,
        spectrometer_params = spectrometer_params,
        DMD_params          = DMD_params,
        acquisition_params  = acquisition_parameters,
        repetitions         = 1,
        reconstruct         = False)
elif camPar.acq_mode == 'snapshot':
    snapshot(camPar, all_path.pathIDSsnapshot, all_path.pathIDSsnapshot_overview)
    spectral_data = acquire(
        ava                 = spectrometer,
        DMD                 = DMD,
        metadata            = metadata,
        spectrometer_params = spectrometer_params,
        DMD_params          = DMD_params,
        acquisition_params  = acquisition_parameters,
        repetitions         = 1,
        reconstruct         = False)
    
    save_metadata_2arms(metadata, DMD_params, spectrometer_params, camPar, acquisition_parameters)
#%% Hadamard Reconstruction
Q = wh.walsh2_matrix(Np)
GT = reconstruction_hadamard(acquisition_parameters, 'walsh', Q, spectral_data, Np)
plot_reco_without_NN(acquisition_parameters, GT, all_path)
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
#%% transfer data to girder
transfer_data_2arms(metadata, acquisition_parameters, spectrometer_params, DMD_params, camPar,
                    setup_version, data_folder_name, data_name, collection_access, upload_metadata = 1)
#%% Draw a ROI
# Comment data_folder_name & data_name to draw a ROI in the current acquisition, else specify the acquisition name
data_folder_name = '2025-01-16_myFirstAcq'
data_name = 'obj_cat_source_white_LED_Walsh_im_64x64_ti_1ms_zoom_x1'
mask_index, x_mask_coord, y_mask_coord = extract_ROI_coord(DMD_params, acquisition_parameters, all_path, 
                                                           data_folder_name, data_name, GT, ti, Np)
#%% Disconnect
disconnect_2arms(spectrometer, DMD, camPar)
#%% Disconnect
disconnect_DMD






















