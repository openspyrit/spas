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
    
import ctypes as ct
import ALP4
#%% Init
spectrometer, DMD, DMD_initial_memory = init()
#%% Setup acquisition and send pattern to the DMD
setup_version = 'setup_v1.3'
data_folder_name = '2022-06-17_test_oldProg'
data_name = 'without_cam'

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
    pattern_order_source = 'C:/openspyrit/spas/stats/pattern_order.npz',#'../communication/raster.txt',#
    pattern_source = 'C:/openspyrit/spas/Patterns/PosNeg/DMD_Walsh_64x64',#'../Patterns/RasterScan_64x64',#'C:/spas/Programs/Python/Patterns/PosNeg_x2/DMD_Walsh_64x64',#
    pattern_prefix = 'Walsh_64x64',#'RasterScan_64x64_1',#
    experiment_name=data_name,
    light_source = 'White LED light',#'HgAr multilines Source (HG-1 Oceanoptics)',#'Nothing',#
    object = 'Cat',#'SeimensStar',#'Nothing',#'USAF',#'Star Sector',#'Nothing'
    filter = 'Diffuserr + LongPass at 500nm + ShortPass at 750nm',#' linear colored filter + OD#0',#'Nothing',#'OD#0',#
    description='test 27.5°, f75mm')
    
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
        
cov_path = 'C:/openspyrit/spas/stats/new-nicolas/Cov_64x64.npy'
mean_path = 'C:/openspyrit/spas/stats/new-nicolas/Average_64x64.npy'
model_root = 'C:/openspyrit/spas/models/new-nicolas/'
H = wh.walsh2_matrix(64)/64
        
model, device = setup_reconstruction(cov_path, mean_path, H, model_root, network_params)
noise = load_noise('C:/openspyrit/spas/noise-calibration/fit_model2.npz')

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
# GT = reconstruction_raster(spectral_data)/64/64
# had_reco_path = data_path + '_raster_reco.npz'
GT = np.rot90(GT, 1)
GT = np.rot90(GT, 1)
# GT = np.rot90(GT, 1)
# GT = np.fliplr(GT)

# GT = GT[:,:,::-1]

if not os.path.exists(had_reco_path):
    np.savez_compressed(had_reco_path, GT)
    
F_bin, wavelengths_bin, bin_width = spectral_binning(GT.T, acquisition_parameters.wavelengths, 530, 730, 8)
F_bin_rot = np.rot90(F_bin, axes=(1,2))
F_bin_flip = F_bin_rot[:,::-1,:]
F_bin_1px, wavelengths_bin, bin_width = spectral_slicing(GT.T, acquisition_parameters.wavelengths, 530, 730, 8)

#plt.figure()
plot_color(F_bin_flip, wavelengths_bin)
plt.savefig(fig_had_reco_path + '_spatial_view_sum_wavelength_binning.png')
plt.show()

#plt.figure()
plot_color(F_bin_1px, wavelengths_bin)
plt.savefig(fig_had_reco_path + '_spatial_view_single_slide_by_wavelength.png')
plt.show()

#plt.figure()
plt.imshow(np.sum(GT[:,:,193:877], axis=2))#[:,:,193:877] #(540-625 nm)
plt.title('Sum of all wavelengths')
plt.savefig(fig_had_reco_path + '_spatial_view_sum_of_wavelengths.png')
plt.show()

########### RGB view
print('Beging RGB convertion ...')
image_arr = plt_rgb.plot_spec_to_rgb_image(GT, acquisition_parameters.wavelengths)
print('RGB convertion finished')
plt.figure()
plt.imshow(image_arr, extent=[0, 10.5, 0, 10.5])
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.savefig(fig_had_reco_path + '_RGB_view.png')
plt.show()

########### spectral view
GT50 = GT[16:48,16:48,:]
GT25 = GT[24:40,24:40,:]
plt.figure()
plt.plot(acquisition_parameters.wavelengths, np.mean(np.mean(GT25,axis=1),axis=0))
plt.plot(acquisition_parameters.wavelengths, np.mean(np.mean(GT50,axis=1),axis=0))
plt.plot(acquisition_parameters.wavelengths, np.mean(np.mean(GT,axis=1),axis=0))
plt.grid()
plt.title("% of region from the center of the image")
plt.legend(['25%', '50%', '100%'])
plt.xlabel(r'$\lambda$ (nm)')
plt.savefig(fig_had_reco_path + '_spectral_view.png')
plt.show()

# sp1 = GT[36,36,:]
# sp2  = GT[34,35,:]
# sp3  = GT[35,36,:]
# sp4  = GT[36,37,:]

# plt.figure()
# plt.plot(acquisition_parameters.wavelengths, sp1)
# plt.plot(acquisition_parameters.wavelengths, sp2)
# plt.plot(acquisition_parameters.wavelengths, sp3)
# plt.plot(acquisition_parameters.wavelengths, sp4)
# plt.legend(['1', '2', '3','4'])
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