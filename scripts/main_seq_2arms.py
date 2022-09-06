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
from spas.reconstruction_nn import ReconstructionParameters, setup_reconstruction
from spas.noise import load_noise
from spas.visualization import snapshotVisu, displayVid, plot_reco_without_NN, plot_reco_with_NN 
from spas.transfer_data_to_girder import transfer_data_2arms
import spyrit.misc.walsh_hadamard as wh
#%% Initialize hardware
spectrometer, DMD, DMD_initial_memory, camPar = init_2arms()
#%% Define the AOI 
# Warning, not all values are allowed for Width and Height (max: 2076x3088 | ex: 768x544)
camPar.rectAOI.s32X.value      = 880#  // X
camPar.rectAOI.s32Y.value      = 590#   // Y
camPar.rectAOI.s32Width.value  = 768#1544 # 3088      // Width must be multiple of 8
camPar.rectAOI.s32Height.value = 544#1038# 2076#     // Height   

camPar = captureVid(camPar)
#%% Set Camera Parameters 
# It is advice to execute this cell twice to take into account the parameter changement
camPar = setup_cam(camPar, 
    pixelClock   = 474,   # Allowed values : [118, 237, 474] (MHz)
    fps          = 220,   # FrameRate boundary : [1 - No value(depend of the AOI size)]
    Gain         = 0,     # Gain boundary : [0 100]
    gain_boost   = 'OFF', # set "ON" to activate gain boost, "OFF" to deactivate
    nGamma       = 1,     # Gamma boundary : [1 - 2.2]
    ExposureTime = 0.09,   # Exposure Time (ms) boudary : [0.013 - 56.221] 
    black_level  = 0)     # Black Level boundary : [0 255]

snapshotVisu(camPar)
#%% Display video in continous mode for optical tuning
displayVid(camPar)
#%% Setup acquisition and send pattern to the DMD
setup_version            = 'setup_v1.3'
data_folder_name         = '2022-09-02_optical_tuning'
data_name                = 'DMD_12cm_f75mm_6.5cm_f50mm_6.5cm_f30mm_1.5cm_x20_0mm_diam1.5mm'

camPar.acq_mode          = 'snapshot'#'video'   #
camPar.vidFormat         = 'avi'     #'bin'#
camPar.insert_patterns   = 0         # 0: no insertion / 1: insert white patterns for the camera
camPar.gate_period       = 16        # a multiple of the integration time of the spectro, between [1 - 16]
camPar.black_pattern_num = 1         # insert the picture number (in the pattern_source folder) of the pattern you want to insert

all_path = func_path(data_folder_name, data_name)

metadata = MetaData(
    output_directory     = all_path.subfolder_path,
    pattern_order_source = 'C:/openspyrit/spas/stats/pattern_order.npz',#'../communication/raster.txt',#
    pattern_source       = 'C:/openspyrit/spas/Patterns/PosNeg/DMD_Walsh_64x64',#'../Patterns/RasterScan_64x64',#
    pattern_prefix       = 'Walsh_64x64',#'RasterScan_64x64_1',#

    experiment_name = data_name,
    light_source    = 'White LED light',#'HgAr multilines Source (HG-1 Oceanoptics)',#'Nothing',#
    object          = 'The cat',#'Nothing',#'USAF',#'Star Sector',#'Nothing'
    filter          = 'Diffuser + HighPass_500nm + LowPass_750nm',#' linear colored filter + OD#0',#'Nothing',#
    description     = 'Test')
    
acquisition_parameters = AcquisitionParameters(
    pattern_compression = 1.0,
    pattern_dimension_x = 64,
    pattern_dimension_y = 64)
    
spectrometer_params, DMD_params, camPar = setup_2arms(
    spectrometer       = spectrometer, 
    DMD                = DMD,
    camPar             = camPar,
    DMD_initial_memory = DMD_initial_memory,
    metadata           = metadata, 
    acquisition_params = acquisition_parameters,
    DMD_output_synch_pulse_delay = 42,
    integration_time   = 1)    
#%% Setup reconstruction
network_params = ReconstructionParameters(
    img_size       = 64,
    CR             = 1024,
    denoise        = True,
    epochs         = 40,
    learning_rate  = 1e-3,
    step_size      = 20,
    gamma          = 0.2,
    batch_size     = 256,
    regularization = 1e-7,
    N0             = 50.0,
    sig            = 0.0,
    arch_name      = 'c0mp')
        
cov_path   = 'C:/openspyrit/spas/stats/new-nicolas/Cov_64x64.npy'
mean_path  = 'C:/openspyrit/spas/stats/new-nicolas/Average_64x64.npy'
model_root = 'C:/openspyrit/spas/models/new-nicolas/'
H          = wh.walsh2_matrix(64)/64
        
model, device = setup_reconstruction(cov_path, mean_path, H, model_root, network_params)
noise = load_noise('C:/openspyrit/spas/noise-calibration/fit_model2.npz')

reconstruction_params = {
    'model'  : model,
    'device' : device,
    'batches': 1,
    'noise'  : noise}
#%% Acquire
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
#%% Reconstruction without NN
Q = wh.walsh2_matrix(64)
GT = reconstruction_hadamard(acquisition_parameters.patterns, 'walsh', Q, spectral_data)
plot_reco_without_NN(acquisition_parameters, GT, Q, all_path.had_reco_path, 
                     all_path.fig_had_reco_path)
#%% Reconstruct with NN
plot_reco_with_NN(acquisition_parameters, network_params, spectral_data, noise, 
                  model, device)
#%% transfer data to girder
transfer_data_2arms(metadata, acquisition_parameters, spectrometer_params, DMD_params, camPar,
                    setup_version, data_folder_name, data_name)
#%% Disconnect
disconnect_2arms(spectrometer, DMD, camPar)