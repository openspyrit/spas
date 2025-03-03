"""
Example of an acquisition followed by a reconstruction using 100 % of the
Hadamard patterns and then, a reconstruction using 1/4 of the patterns 
(subsampled) with a DenoiCompNet model and a noise model. Reconstructions are 
performed after the acquisition and not in "real-time".
"""

from spas.acquisition_SPC2D import init_2arms, setup_cam, AcquisitionParameters, setup_2arms, setup, acquire, acquire_2arms, snapshot, disconnect_2arms, captureVid, displaySpectro, setup_tuneSpectro, change_patterns
from spas.metadata import MetaData, func_path, save_metadata_2arms
from spas.reconstruction import reconstruction_hadamard
from spas.reconstruction_nn import ReconstructionParameters, setup_reconstruction, reorder_subsample
from spas.noise import load_noise
from spas.visualization import snapshotVisu, displayVid, plot_reco_without_NN, plot_reco_with_NN, extract_ROI_coord
from spas.transfer_data_to_girder import transfer_data_2arms
import spyrit.misc.walsh_hadamard as wh
from spas import reconstruct
import time
from pathlib import Path
import numpy as np
#%% Initialize hardware
spectrometer, DMD, DMD_initial_memory, camPar = init_2arms(dmd_lib_version = '4.2') # possible version : '4.1', '4.2' or '4.3'
#%% Define the AOI of the camera
# Warning, not all values are allowed for Width and Height (max: 2076x3088 | ex: 768x544)
camPar.rectAOI.s32X.value      = 1100#  // X
camPar.rectAOI.s32Y.value      = 640#   // Y
camPar.rectAOI.s32Width.value  = 768#1544#3088#1544 # 3000#3088#         // Width must be multiple of 8
camPar.rectAOI.s32Height.value = 544#1038#2076#1730#2000## 1038#  2076   // Height   

camPar = captureVid(camPar)
#%% Set Camera Parameters 
# It is advice to execute this cell twice to take into account the parameter changement
camPar = setup_cam(camPar, 
    pixelClock   = 474,   # Allowed values : [118, 237, 474] (MHz)
    fps          = 220,   # FrameRate boundary : [1 - No value(depend of the AOI size)]
    Gain         = 0,     # Gain boundary : [0 100]
    gain_boost   = 'OFF', # set1"ON"to activate gain boost, "OFF" to deactivate
    nGamma       = 1,     # Gamma boundary : [1 - 2.2]
    ExposureTime = .65,# Exposure Time (ms) boudary : [0.013 - 56.221] 
    black_level  = 4)     # lack Level boundary : [0 255]

snapshotVisu(camPar)
#%% Display video in continuous mode for optical tuning
displayVid(camPar)
#%% Display the spectrum in continuous mode for optical tuning
pattern_to_display = 'white' #'gray'#'black', 
ti   = 1 # Integration time of the spectrometer  
zoom = 1 # Numerical zoom applied in the DMD

metadata, spectrometer_params, DMD_params, acquisition_parameters = setup_tuneSpectro(spectrometer = spectrometer, DMD = DMD, DMD_initial_memory = DMD_initial_memory,
                                                                                      pattern_to_display = pattern_to_display, ti = ti, zoom = zoom, xw_offset = 128, yh_offset = 0)
displaySpectro(ava = spectrometer, DMD = DMD, metadata = metadata, spectrometer_params = spectrometer_params, DMD_params = DMD_params, acquisition_params = acquisition_parameters)
#%% Setup acquisition and send pattern to the DMD
setup_version            = 'setup_v1.3.1'
collection_access        = 'public' #'private'#
Np                       = 64       # Number of pixels in one dimension of the image (image: NpxNp)
ti                       = 1        # Integration time of the spectrometer   
zoom                     = 2        # Numerical zoom applied in the DMD
xw_offset                = 428      # Default = 128
yh_offset                = 26      # Default = 0
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






















