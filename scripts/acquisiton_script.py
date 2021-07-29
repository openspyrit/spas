from singlepixel import *

#%% Init
spectrometer, DMD, DMD_initial_memory = init() # just once, two consecutively returns an error
    
#%% Setup
metadata = MetaData(
    output_directory='../meas/',
    pattern_order_source='../stats_download/Cov_64x64.npy', # covariance matrix or 
    pattern_source='../Walsh_64x64/',
    pattern_prefix='Walsh_64x64',
    experiment_name='my_first_measurement',
    light_source='white_lamp',
    object='no_object',
    filter='no_filter',
    description='my_first_description')
    
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
    integration_time=1.0)

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

#%% Disconnect
disconnect(spectrometer, DMD)