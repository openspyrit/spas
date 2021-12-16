"""
Example of an acquisition followed by a reconstruction using 100 % of the
Hadamard patterns and then, a reconstruction using 1/4 of the patterns 
(subsampled) with a DenoiCompNet model and a noise model. Reconstructions are 
performed after the acquisition and not in "real-time".
"""

from spas import *
import os
import numpy as np
import spyrit.misc.walsh_hadamard as wh
import spas.transfer_data_to_girder as transf
from spas import plot_spec_to_rgb_image as plt_rgb
from matplotlib import pyplot as plt
    
#%% Init
spectrometer, DMD, DMD_initial_memory = init()
    
#%% Setup acquisition and send pattern to the DMD
setup_version = 'setup_v1.2'
data_folder_name = '2021-12-15_test_3ieme'
data_name = 'tomato_slice'

if not os.path.exists('../data/' + data_folder_name):
    os.makedirs('../data/' + data_folder_name)

subfolder_path = '../data/' + data_folder_name + '/' + data_name    
overview_path = subfolder_path + '/overview'
if not os.path.exists(overview_path):
    os.makedirs(overview_path)

data_path = subfolder_path + '/' + data_name
had_reco_path = data_path + '_had_reco.npz'    
nn_reco_path = data_path + '_nn_reco.npz'     
fig_had_reco_path = overview_path + '/' + 'HAD_RECO_' + data_name   
fig_nn_reco_path = overview_path + '/' + 'NN_RECO_' + data_name

metadata = MetaData(
    output_directory=subfolder_path,
    pattern_order_source='../stats/pattern_order.npz',#'../communication/raster.txt',#
    pattern_source='../Patterns/PosNeg/DMD_Walsh_64x64',#'../Patterns/RasterScan_64x64',#
    pattern_prefix='Walsh_64x64',#'RasterScan_64x64_1',#
    experiment_name=data_name,
    light_source='White LED light',#'Nothing',#'HgAr multilines Source (HG-1 Oceanoptics)',#
    object='nothing',#'Nothing',#'USAF',#'Star Sector',#'Nothing'
    filter='Diffuser',#' linear colored filter + OD#0',#'Nothing',#'OD#0',#
    description='system description : DMD_120mm_f50_10mm_MOx20_0mm_redFiber')
    
acquisition_parameters = AcquisitionParameters(
    pattern_compression=1.0,
    pattern_dimension_x=64,
    pattern_dimension_y=64)
    
spectrometer_params, DMD_params = setup(
    spectrometer=spectrometer, 
    DMD=DMD,
    DMD_initial_memory=DMD_initial_memory,
    metadata=metadata, 
    acquisition_params=acquisition_parameters,
    integration_time=1.0,)
    
#%% Setup reconstruction

network_params = ReconstructionParameters(
    img_size=64,
    CR=1024,
    denoise=True,
    epochs=40,
    learning_rate=1e-3,
    step_size=20,
    gamma=0.2,
    batch_size=256,
    regularization=1e-7,
    N0=50.0,
    sig=0.0,
    arch_name='c0mp',)
        
cov_path = '../stats/new-nicolas/Cov_64x64.npy'
mean_path = '../stats/new-nicolas/Average_64x64.npy'
model_root = '../models/new-nicolas/'
H = wh.walsh2_matrix(64)/64
        
model, device = setup_reconstruction(cov_path, mean_path, H, model_root, network_params)
noise = load_noise('../noise-calibration/fit_model2.npz')

reconstruction_params = {
    'model': model,
    'device': device,
    'batches': 1,
    'noise': noise,
}

#%% Acquire
spectral_data = acquire(
    ava=spectrometer,
    DMD=DMD,
    metadata=metadata,
    spectrometer_params=spectrometer_params,
    DMD_params=DMD_params,
    acquisition_params=acquisition_parameters,
    repetitions=1,
    reconstruct=False)

#%% Reconstruction without NN
Q = wh.walsh2_matrix(64)

GT = reconstruction_hadamard(acquisition_parameters.patterns, 'walsh', Q, spectral_data)

F_bin, wavelengths_bin, bin_width = spectral_binning(GT.T, acquisition_parameters.wavelengths, 530, 730, 8)
F_bin_rot = np.rot90(F_bin, axes=(1,2))
F_bin_flip = F_bin_rot[:,::-1,:]


F_bin_1px, wavelengths_bin, bin_width = spectral_slicing(GT.T, acquisition_parameters.wavelengths, 530, 730, 8)


plot_color(F_bin_flip, wavelengths_bin)
plot_color(F_bin_1px, wavelengths_bin)


#%% RGB view
image_arr = plt_rgb.plot_spec_to_rgb_image(GT, acquisition_parameters.wavelengths)
plt.imshow(image_arr)

#%% Reconstruct with NN
F_bin, wavelengths_bin, bin_width, noise_bin = spectral_binning(spectral_data.T, acquisition_parameters.wavelengths, 530, 730, 8, noise)
recon = reconstruct(model, device, F_bin[:,0:8192//4], 1, noise_bin)           
plot_color(recon, wavelengths_bin)
plt.show()


plt.imshow(np.sum(recon, axis=0))
plt.title('NN reco, sum of all wavelengths')
plt.show()

F_bin, wavelengths_bin, bin_width, noise_bin = spectral_slicing(spectral_data.T, acquisition_parameters.wavelengths, 514, 751, 8, noise)
recon2 = reconstruct(model, device, F_bin[:,0:8192//4], 4, noise_bin)
plot_color(recon2, wavelengths_bin)

#%% transfer data to girder
transf.transfer_data(metadata, acquisition_parameters, spectrometer_params, DMD_params, 
                               setup_version, data_folder_name, data_name)

#%% Disconnect
disconnect(spectrometer, DMD)