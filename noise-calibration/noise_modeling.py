# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""
Modeling noise as a sum of gaussian process and a poisson process.
"""

if __name__ == '__main__':

    import numpy as np
    from matplotlib import pyplot as plt

    from singlepixel import *
        
    #%% Init
    spectrometer, DMD, DMD_initial_memory = init()
    spectral_data_black = []
    spectral_data_white = []    
    
    #%% Setup and acquire black patterns
    times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    for integration_time in times:
        
        metadata = MetaData(
            output_directory='../data/20-03-2021-noise_model5/',
            pattern_order_source='../communication/10k-black.txt',
            pattern_source='../Patterns/PosNeg/DMD_Hadamard_64x64/Hadamard_64x64_',
            experiment_name='/black-acq-' + str(integration_time),
            light_source='None',
            object='None',
            filter='None',
            description='Acquisition with black patterns')
            
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
            integration_time=integration_time,
            pos_neg=False)
        
        print(f'Acquiring integration_time = {integration_time} ms')
        spectral_data = acquire(
            ava=spectrometer,
            DMD=DMD,
            metadata=metadata,
            spectrometer_params=spectrometer_params,
            DMD_params=DMD_params,
            acquisition_params=acquisition_parameters,
            repetitions=1)
    
        spectral_data_black.append(spectral_data)
    
    #%% Disconnect
    disconnect(spectrometer, DMD)

    #%% Analysis
    spectral_data_black = np.asarray(spectral_data_black, dtype=np.float32)
    mean_black = np.mean(spectral_data_black, axis=1)
    var_black = np.var(spectral_data_black, axis=1)

    #%% Init
    spectrometer, DMD, DMD_initial_memory = init()
    
    #%% Setup and acquire white patterns
    times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    for integration_time in times:
        
        metadata = MetaData(
            output_directory='../data/20-03-2021-noise_model5/',
            pattern_order_source='../communication/10k-white.txt',
            pattern_source='../Patterns/PosNeg/DMD_Hadamard_64x64/Hadamard_64x64_',
            experiment_name='/white-acq-' + str(integration_time),
            light_source='None',
            object='None',
            filter='None',
            description='Acquisition with white patterns')
            
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
            integration_time=integration_time,
            pos_neg=False)
            
        print(f'Acquiring integration_time = {integration_time} ms')
        spectral_data = acquire(
            ava=spectrometer,
            DMD=DMD,
            metadata=metadata,
            spectrometer_params=spectrometer_params,
            DMD_params=DMD_params,
            acquisition_params=acquisition_parameters,
            repetitions=1)
    
        spectral_data_white.append(spectral_data)
        
    #%% Disconnect
    disconnect(spectrometer, DMD)
    
    #%% Save times
    np.savetxt('../data/20-03-2021-noise_model5/times.txt',times)
    
    #%% Analysis
    
    spectral_data_white = np.asarray(spectral_data_white, dtype=np.float32)
    mean_white = np.mean(spectral_data_white, axis=1)
    var_white = np.var(spectral_data_white, axis=1)
    wavelengths = acquisition_parameters.wavelengths
    
    #%% Saving data
    
    np.savez_compressed('../data/20-03-2021-noise_model5/raw_data.npz', spectral_data_black=spectral_data_black,
                        spectral_data_white=spectral_data_white)

    #%% Plot measured variables as a function of lambda
    
    plt.figure()
    plt.plot(wavelengths, mean_black[0,:], 'b')
    plt.plot(wavelengths, mean_white[0,:], 'r')
    plt.xlabel('Wavelenght (nm)')
    plt.ylabel('Mean')
    plt.title(f'Mean for integration_time = {times[0]} ms')
    plt.legend(['$µ_{black}$','$µ_{white}$'])
    
    plt.figure()
    plt.plot(wavelengths, var_black[0,:], 'b')
    plt.plot(wavelengths, var_white[0,:], 'r')
    plt.xlabel('Wavelenght (nm)')
    plt.ylabel('Var')
    plt.title(f'Variance for integration_time = {times[0]} ms')
    plt.legend(['$\sigma_{black}^{2}$','$\sigma_{white}^{2}$'])
    
    #%% Plot measured variables as a function of time
    
    plt.figure()
    plt.plot(times, mean_black[:,1024], label='$µ_{black}$')
    plt.plot(times, mean_white[:,1024], label='$µ_{white}$')
    plt.xlabel('Integration time (ms)')
    plt.ylabel('$µ$')
    plt.title(f'Mean for $\lambda$ = {wavelengths[2000]:.2f} nm')
    plt.legend()
    
    plt.figure()
    plt.plot(times, var_black[:,1024], label='$\sigma_{black}^{2}$')
    plt.plot(times, var_white[:,1024], label='$\sigma_{white}^{2}$')
    plt.xlabel('Integration time (ms)')
    plt.ylabel('$\sigma^{2}$')
    plt.title(f'Variance for $\lambda$ = {wavelengths[2000]:.2f} nm')
    plt.legend()
    
    #%% Poisson noise
    
    mean_poisson = mean_white - mean_black
    var_poisson = var_white - var_black
    
    plt.figure()
    plt.plot(wavelengths, mean_poisson[0,:])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('$µ_{poisson}$')
    plt.title(f'Mean for integration_time = {times[0]} ms')
    
    plt.figure()
    plt.plot(wavelengths, var_poisson[0,:])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('$\sigma^{2}_{poisson}$')
    plt.title(f'Variance for integration_time = {times[0]} ms')
    
    plt.figure()
    plt.plot(times, mean_poisson[:,1024])
    plt.xlabel('Integration time (ms)')
    plt.ylabel('$µ_{poisson}$')
    plt.title(f'Mean for $\lambda$ = {wavelengths[0]:.2f} nm')
    
    plt.figure()
    plt.plot(times, var_poisson[:,1024])
    plt.xlabel('Integration time (ms)')
    plt.ylabel('$\sigma^{2}_{poisson}$')
    plt.title(f'Variance for $\lambda$ = {wavelengths[0]:.2f} nm')
    
    
    #%% Linear 
    
    # Fitting data for each wavelength
    coeff1 = np.zeros((mean_poisson.shape[1]))
    for index in range(mean_poisson.shape[1]):
        x = mean_poisson[:,index]
        A = np.stack([x,np.zeros(mean_poisson.shape[0])]).T
        y = var_poisson[:,index]
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        coeff1[index] = m
    
    np.savetxt('../data/20-03-2021-noise_model5/fit_model.txt', coeff1)
    #%% Plot linear fit
    
    lambda_index = 0
    plt.figure()
    plt.plot(mean_poisson[:,lambda_index], var_poisson[:,lambda_index], 'o', label='Original data', markersize=10)
    plt.plot(mean_poisson[:,lambda_index], coeff1[lambda_index]*mean_poisson[:,lambda_index] + 0, 'r', label='Fitted line')
    plt.xlabel('$µ_{poisson}$')
    plt.ylabel('$\sigma^{2}_{poisson}$')
    plt.title(f'Poisson variance as a function of poisson mean ($\lambda$={wavelengths[lambda_index]:.2f} nm)')
    plt.legend()
        
    #%% Plot k²
    
    plt.figure()
    plt.plot(wavelengths, coeff1)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('k²')