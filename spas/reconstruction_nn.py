# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""Functions for image reconstruction using Neural Networks.

Implements functions for loading, reconstructing and plotting using Neural 
Networks developed with spyrit. Allows the possibility of "real-time" 
reconstructions and plots using multiprocessing features.
"""

from dataclasses import InitVar, dataclass, field
from enum import IntEnum
from time import perf_counter_ns, sleep
from typing import Tuple, Union
from multiprocessing import Queue
from pathlib import Path

import torch
import numpy as np
from matplotlib import pyplot as plt
from spyrit.learning.model_Had_DCAN import compNet, noiCompNet, DenoiCompNet 
from spyrit.learning.nets import load_net

from spas.noise import noiseClass

class netType(IntEnum):
    """Possible model architectures.
    """
    c0mp = 0
    comp = 1
    pinv = 2
    free = 3


@dataclass
class ReconstructionParameters:
    """Reconstruction parameters for loading a reconstruction model.
    """
    img_size: int
    CR: int
    denoise: bool
    epochs: int
    learning_rate: float
    step_size: int
    gamma: float
    batch_size: int
    regularization: float
    N0: float
    sig: float

    arch_name: InitVar[str]

    _net_arch: int = field(init=False)


    def __post_init__(self, arch_name):
        self.arch_name = arch_name


    @property
    def arch_name(self):
        return netType(self._net_arch).name


    @arch_name.setter
    def arch_name(self, arch_name):
        self._net_arch = int(netType[arch_name])
    

def setup_reconstruction(cov_path: str, mean_path: str, H: np.ndarray, 
    model_root: str, network_params: ReconstructionParameters
    ) -> Tuple[Union[compNet, noiCompNet, DenoiCompNet], str]:
    """Loads a neural network for reconstruction.

    Args:
        cov_path (str): 
            Path to the covariance matrix.
        mean_path (str): 
            Path to the mean matrix.
        H (nd.array): 
            Hadamard matrix with patterns.
        model_root (str): 
            Folder containing trained models for reconstruction.
        network_params (ReconstructionParameters): 
            Parameters used to load the model.

    Returns:
        Tuple[Union[compNet, noiCompNet, DenoiCompNet], str]:
            model (compNet, noiCompNet, DenoiCompNet):
                Loaded model.
            device (str):
                Device to which the model was loaded.
    """

    net_type = ['c0mp', 'comp', 'pinv', 'free']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    Cov_had = np.load(cov_path) / network_params.img_size**2
    Mean_had = np.load(mean_path) / network_params.img_size

    suffix = '_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(
           network_params.img_size, network_params.CR, 
           network_params.epochs, network_params.learning_rate,
           network_params.step_size, network_params.gamma,
           network_params.batch_size, network_params.regularization)

    recon_type = ""
    if network_params.N0 == 0:
        train_type = ''
    else:
        train_type = '_N0_{}_sig_{}'.format(network_params.N0,
                                            network_params.sig)
        if network_params.denoise == True:
            recon_type+="_Denoi"

        # Training parameters
        arch = network_params.arch_name

        suffix = 'NET_' + arch + train_type + recon_type + suffix

        title = Path(model_root) / suffix

    if network_params.N0 == 0:
        model = compNet(
            network_params.img_size, 
            network_params.CR, 
            Mean_had, 
            Cov_had, 
            network_params._net_arch,
            H)
    
    elif network_params.denoise == False:
        model = noiCompNet(
            network_params.img_size, 
            network_params.CR, 
            Mean_had, 
            Cov_had, 
            network_params._net_arch,
            network_params.N0,
            network_params.sig, 
            H)

    elif network_params.denoise == True:
        model = DenoiCompNet(
            network_params.img_size, 
            network_params.CR, 
            Mean_had, 
            Cov_had, 
            network_params._net_arch,
            network_params.N0,
            network_params.sig, 
            H,
            None)

    torch.cuda.empty_cache()

    load_net(title, model, device)
    model = model.to(device)

    return model, device


def reconstruct(model: Union[compNet, noiCompNet, DenoiCompNet],
    device: str, spectral_data: np.ndarray, batches : int, 
    noise_model: noiseClass = None, is_process: bool = False) -> np.ndarray:
    """Reconstructs images from experimental data.

    Using a loaded model, reconstructs images in batches. It can be used in
    'real-time' using multiprocessing to allow reconstruction and acquisition at
    the same time.

    Args:
        model (Union[compNet, noiCompNet, DenoiCompNet]): 
            Pre-loaded model for reconstruction.
        device (str):
            Device to which the model was loaded. Used to make sure the spectral
            data is loaded to the same device as the model.
        spectral_data (np.ndarray):
            Spectral data acquired, must have the dimensions
            (spectral dimension x patterns).
        batches (int):
            Number of batches for reconstruction in case the device does not
            have enough memory to reconstruct all data at a single time. E.g.
            when reconstructing data distributed along many wavelengths, there
            may be too many data.
        noise_model (noiseClass, optional): 
            Loaded noise model in case the reconstruction should use a denoising
            method. Defaults to None.
        is_process (bool, optional): 
            If True, reconstruction is performed in 'real-time' mode, using
            multiprocessing for reconstruction, thus allowing reconstruction and
            acquisition simultaneously. Defaults to False.

    Returns:
        np.ndarray: Reconstructed images following the dimensions:
        spectral dimension x image size x image size.
    """

    # Implemented only the case for Denoi reconstruction
    
    proportion = spectral_data.shape[0]//batches # Amount of wavelengths per batch
    
    recon = np.zeros((spectral_data.shape[0], 64, 64))

    start = perf_counter_ns()

    with torch.no_grad():
        for batch in range(batches):
                
            lambda_indeces = range(proportion * batch, proportion * (batch+1))

            info = (f'batch {batch},'
                f'reconstructed wavelength range: {lambda_indeces}')
            
            if is_process:
                info = '#Recon process:' + info   

            print(info)
        
            C = noise_model.mu[lambda_indeces]
            s = noise_model.sigma[lambda_indeces]
            K = noise_model.K[lambda_indeces]
            
            n = len(C)
                
            C = torch.from_numpy(C).float().to(device).reshape(n,1,1)
            s = torch.from_numpy(s).float().to(device).reshape(n,1,1)
            K = torch.from_numpy(K).float().to(device).reshape(n,1,1)
        
            CR = spectral_data.shape[1]
            
            torch_img = torch.from_numpy(spectral_data[lambda_indeces,:])
            torch_img = torch_img.float()
            torch_img = torch.reshape(torch_img, (len(lambda_indeces), 1, CR)) # batches, channels, patterns
            torch_img = torch_img.to(device)
        
            torch_recon = model.forward_reconstruct_expe(torch_img,
                len(lambda_indeces), 1, model.n, model.n, C, s, K)
                
            recon[lambda_indeces,:,:] = torch_recon.cpu().detach().numpy().squeeze()
    
    end = perf_counter_ns()

    time_info = f'Reconstruction time: {(end-start)/1e+6} ms'

    if is_process:
        time_info = '#Recon process:' + time_info

    print(time_info)

    return recon


def reconstruct_process(model: Union[compNet, noiCompNet, DenoiCompNet],
    device: str, queue_to_recon: Queue, queue_reconstructed: Queue, 
    batches: int, noise_model: noiseClass, sleep_time: float = 0.3) -> None:
    """Performs reconstruction in real-time.

    Args:
        model (Union[compNet, noiCompNet, DenoiCompNet]): 
            Pre-loaded model for reconstruction.
        device (str):
            Device to which the model was loaded. Used to make sure the spectral
            data is loaded to the same device as the model.
        queue_to_recon (Queue):
            Multiprocessing queue containing spectral data acquired for
            reconstruction.
        queue_reconstructed (Queue):
            Multiprocessing queue containing reconstructed data for plotting.
            Used by a plotting process to show data in 'real-time'.
        batches (int):
            Number of batches for reconstruction in case the device does not
            have enough memory to reconstruct all data at a single time. E.g.
            when reconstructing data distributed along many wavelengths, there
            may be too many data.
        noise_model (noiseClass):
            Loaded noise model in case the reconstruction should use a denoising
            method. Defaults to None.
        sleep_time (float, optional):
            Estimated time between two reconstructions. If no data is received
            for reconstruction, the process may sleep for this given time in
            seconds. Defaults to 0.3.
    """

    while True:
        if not queue_to_recon.empty():

            message = queue_to_recon.get()

            if isinstance(message, str):
                if message == 'kill':
                    break

            else:
                spectral_data = message
                print('#Recon process: Got data')
                recon_data = reconstruct(model, device, spectral_data, batches,
                                    noise_model, True)
                queue_reconstructed.put(recon_data)

        else:
            sleep(sleep_time)

    # Removing any data left in queue
    while not queue_to_recon.empty():
        queue_to_recon.get_nowait()
    
    queue_reconstructed.put('kill') # Sends a message to stop plotting
    print('#Recon process: Ended reconstruction')


def plot_recon(queue: Queue, sleep_time: float = 0.3) -> None:
    """Plots reconstructed data in 'real-time'.

    Args:
        queue (Queue):
            Multiprocessing queue containing reconstructed data for plotting.
        sleep_time (float, optional): 
            Estimated time between two reconstructions. If no reconstruction is
            received, the plot main loop enters sleep mode. A plot is showed 
            during sleep_time in seconds. Defaults to 0.3.
    """


    print('#Plot process: Entered plot process')
    plt.ion()

    fig = plt.figure()

    while True:
        if not queue.empty():
            message = queue.get()

            if isinstance(message, str):
                if message == 'kill':
                    break
            else:
                recon_data = message
                print('#Plot process: Got plot')                
                plt.clf()
                plt.imshow(np.sum(recon_data, axis=0), cmap='gray')
                plt.title('Sum of wavelengths')
                plt.draw()
                plt.pause(sleep_time)

        else:
            sleep(sleep_time)

    plt.close('all')

    # Removing any data left in queue
    while not queue.empty():
        queue.get_nowait()

    print('#Plot process: Ended plot')