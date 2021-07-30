"""
Example of an acquisition of 1/4 of the Hadamard patterns and then performs a 
reconstruction using 1/4 of the patterns  a DenoiCompNet model and a noise model
in "real-time", using multiprocessing.
"""
import spyrit.misc.walsh_hadamard as wh
from spas import *

if __name__ == '__main__':

#%% Init
    spectrometer, DMD, DMD_initial_memory = init()
        
    #%% Setup
    metadata = MetaData(
        output_directory='../data/.../',
        pattern_order_source='../communication/...',
        pattern_source='../Patterns/.../',
        pattern_prefix='Hadamard_64x64',
        experiment_name='...',
        light_source='...',
        object='...',
        filter='...',
        description='...')
        
    acquisition_parameters = AcquisitionParameters(
        pattern_compression=0.25,
        pattern_dimension_x=64,
        pattern_dimension_y=64)
    
    spectrometer_params, DMD_params = setup(
        spectrometer=spectrometer, 
        DMD=DMD,
        DMD_initial_memory=DMD_initial_memory,
        metadata=metadata, 
        acquisition_params=acquisition_parameters,
        integration_time=1.0)
    
    network_params = {
        'img_size': 64,
        'CR': 1024,
        'net_arch': 0,
        'denoise': True,
        'epochs': 20,
        'learning_rate': 1e-3,
        'step_size': 10,
        'gamma': 0.5,
        'batch_size': 256,
        'regularization': 1e-7,
        'N0': 2500,
        'sig': 0.5        
        }
        
    cov_path = './...'
    mean_path = './...'
    H = wh.walsh2_matrix(64)/64
    model_root = './...'
        
    model, device = setup_reconstruction(cov_path, mean_path, H, model_root, network_params)
    noise = load_noise('../noise-calibration/fit_model.npz')
    
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
        repetitions=3,
        verbose=True,
        reconstruct=True,
        reconstruction_params=reconstruction_params)

    #%% Disconnect
    disconnect(spectrometer, DMD)