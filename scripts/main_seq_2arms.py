"""
Example of an acquisition followed by a reconstruction using 100 % of the
Hadamard patterns and then, a reconstruction using 1/4 of the patterns 
(subsampled) with a DenoiCompNet model and a noise model. Reconstructions are 
performed after the acquisition and not in "real-time".
"""

# from spas import *
from spas.acquisition import init_2arms, setup_cam, AcquisitionParameters, setup_2arms, acquire, acquire_2arms, snapshot, disconnect_2arms, captureVid
from spas.metadata import MetaData, func_path
from spas.reconstruction import reconstruction_hadamard
from spas.reconstruction_nn import ReconstructionParameters, setup_reconstruction, reorder_subsample
from spas.noise import load_noise
from spas.visualization import snapshotVisu, displayVid, plot_reco_without_NN, plot_reco_with_NN 
from spas.transfer_data_to_girder import transfer_data_2arms
import spyrit.misc.walsh_hadamard as wh
from spas import reconstruct
import time
#%% Initialize hardware
spectrometer, DMD, DMD_initial_memory, camPar = init_2arms()
#%% Define the AOI 
# Warning, not all values are allowed for Width and Height (max: 2076x3088 | ex: 768x544)
camPar.rectAOI.s32X.value      = 1100#0 #0#  // X
camPar.rectAOI.s32Y.value      = 765#0#0#800#   // Y
camPar.rectAOI.s32Width.value  = 768# 1544 #3000#3088#3088#         // Width must be multiple of 8
camPar.rectAOI.s32Height.value = 544 #1730#2000## 1038#  2076   // Height   

camPar = captureVid(camPar)
#%% Set Camera Parameters 
# It is advice to execute this cell twice to take into account the parameter changement
camPar = setup_cam(camPar, 
    pixelClock   = 474,   # Allowed values : [118, 237, 474] (MHz)
    fps          = 220,   # FrameRate boundary : [1 - No value(depend of the AOI size)]
    Gain         = 0,     # Gain boundary : [0 100]
    gain_boost   = 'OFF', # set "ON" to activate gain boost, "OFF" to deactivate
    nGamma       = 1,     # Gamma boundary : [1 - 2.2]
    ExposureTime = 0.04,# Exposure Time (ms) boudary : [0.013 - 56.221] 
    black_level  = 5)     # Black Level boundary : [0 255]

snapshotVisu(camPar)
#%% Display video in continous mode for optical tuning
displayVid(camPar)
#%% Setup acquisition and send pattern to the DMD
setup_version            = 'setup_v1.3.1'
Np                       = 64       # Number of pixels in one dimension of the image (image: NpxNp)
zoom                     = 1       # Numerical zoom applied in the DMD
ti                       = 1        # Integration time of the spectrometer   
scan_mode                = 'Walsh'  #'Raster' #'Walsh_inv' #'Raster_inv' #
data_folder_name         = '2023-05-12_test_ALP4'
data_name                = 'cat_' + scan_mode + '_im_'+str(Np)+'x'+str(Np)+'_ti_'+str(ti)+'ms_zoom_x'+str(zoom)

camPar.acq_mode          = 'snapshot'#'video'   #
camPar.vidFormat         = 'avi'     #'bin'#
camPar.insert_patterns   = 0         # 0: no insertion / 1: insert white patterns for the camera
camPar.gate_period       = 1        # a multiple of the integration time of the spectro, between [2 - 16] (2: insert one white pattern between each pattern)
camPar.black_pattern_num = 1         # insert the picture number (in the pattern_source folder) of the pattern you want to insert
all_path = func_path(data_folder_name, data_name)

metadata = MetaData(
    output_directory     = all_path.subfolder_path,
    pattern_order_source = 'C:/openspyrit/spas/stats/pattern_order_' + scan_mode + '_' + str(Np) + 'x' + str(Np) + '.npz',
    pattern_source       = 'C:/openspyrit/spas/Patterns/Zoom_x' + str(zoom) + '/' + scan_mode + '_' + str(Np) + 'x' + str(Np),
    pattern_prefix       = scan_mode + '_' + str(Np) + 'x' + str(Np),

    experiment_name = data_name,
    light_source    = 'White LED light',#'Zeiss KL2500 white lamp',#'LED Laser 385nm + optical fiber 600µm, P = 30 mW',#'the sun',#'IKEA lamp 10W LED1734G10',#or BlueLaser 50 mW',#' (74) + 'Bioblock power: II',#'HgAr multilines Source (HG-1 Oceanoptics)',#'Nothing',#
    object          = 'Cat',#two little tubes containing PpIX at 634 and 620 state',#'Apple',#',#'USAF',#'Nothing''color checker'
    filter          = 'None', #'BandPass filter 560nm Dl=10nm',#'HighPass_500nm + LowPass_750nm',# + optical density = 0.1', #'Nothing',#'Diffuser + HighPass_500nm + LowPass_750nm',##'Microsope objective x40',#'' linear colored filter + OD#0',#'Nothing',#
    description     = 'test after changing metadata.py and acquisition.py to be used on Linux and MacOs plateform. We would like to be sure that SPAS for Windows is ok')
    
acquisition_parameters = AcquisitionParameters(
    pattern_compression = 1.0,
    pattern_dimension_x = Np,
    pattern_dimension_y = Np)
    
spectrometer_params, DMD_params, camPar = setup_2arms(
    spectrometer       = spectrometer, 
    DMD                = DMD,
    camPar             = camPar,
    DMD_initial_memory = DMD_initial_memory,
    metadata           = metadata, 
    acquisition_params = acquisition_parameters,
    DMD_output_synch_pulse_delay = 42,
    integration_time   = ti)    
#%% Acquire
time.sleep(0)
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
#%% Hadamard Reconstruction
Q = wh.walsh2_matrix(Np)
GT = reconstruction_hadamard(acquisition_parameters.patterns, 'walsh', Q, spectral_data, Np)
plot_reco_without_NN(acquisition_parameters, GT, Q, all_path)
#%% Neural Network Reconstruction
t0 = time.time()
network_param = ReconstructionParameters(
    # Reconstruction network    
    M = 64*64,          # Number of measurements
    img_size = 128,     # Image size
    arch = 'dc-net',    # Main architecture
    denoi = 'unet',     # Image domain denoiser
    subs = 'rect',      # Subsampling scheme
    
    # Training
    data = 'imagenet',  # Training database
    N0 = 10,            # Intensity (max of ph./pixel)
    
    # Optimisation (from train2.py)
    num_epochs = 30,       # Number of training epochs
    learning_rate = 0.001, # Learning Rate
    step_size = 10,        # Scheduler Step Size
    gamma = 0.5,           # Scheduler Decrease Rate   
    batch_size = 256,      # Size of the training batch
    regularization = 1e-7 # Regularisation Parameter
    )
        
cov_path = 'C:/openspyrit/stat/ILSVRC2012_v10102019/Cov_8_128x128.npy'
model_folder = 'C:/openspyrit/models/'
model, device = setup_reconstruction(cov_path, model_folder, network_param)
meas = reorder_subsample(spectral_data.T, acquisition_parameters, network_param) # Reorder and subsample
reco = reconstruct(model, device, meas) # Reconstruction
plot_reco_with_NN(acquisition_parameters, reco, all_path)
print('elapsed time = ' + str(round(time.time()-t0)) + ' s')
#%% transfer data to girder
transfer_data_2arms(metadata, acquisition_parameters, spectrometer_params, DMD_params, camPar,
                    setup_version, data_folder_name, data_name, upload_metadata = 1)
#%% Disconnect
disconnect_2arms(spectrometer, DMD, camPar)

