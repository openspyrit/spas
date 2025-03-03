# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""Acquisition utility functions.

    Acquisition module is a generic module that call function in different setup (SPC2D_1arm, SPC2D_2arms, SCP1D and SPIM)
    
"""

import warnings
from time import sleep, perf_counter_ns
from typing import NamedTuple, Tuple, List, Optional
from collections import namedtuple
from pathlib import Path
from multiprocessing import Process, Queue
import shutil    
import math

import numpy as np
from PIL import Image
##### DLL for the DMD
try:
    from ALP4 import ALP4, ALP_FIRSTFRAME, ALP_LASTFRAME
    from ALP4 import ALP_AVAIL_MEMORY, ALP_DEV_DYN_SYNCH_OUT1_GATE, tAlpDynSynchOutGate
    # print('ALP4 is ok in Acquisition file')
except:
    class ALP4:
        pass
##### DLL for the spectrometer Avantes 
try:
    from msl.equipment import EquipmentRecord, ConnectionRecord, Backend
    from msl.equipment.resources.avantes import MeasureCallback, Avantes
except:
    pass
    
from tqdm import tqdm
from spas.metadata_SPC2D import DMDParameters, MetaData, AcquisitionParameters
from spas.metadata_SPC2D import SpectrometerParameters, save_metadata, CAM, save_metadata_2arms
from spas.reconstruction_nn import reconstruct_process, plot_recon, ReconstructionParameters

# DLL for the IDS CAMERA
try:
    from pyueye import ueye, ueye_tools
except:
    print('ueye DLL not installed')

from matplotlib import pyplot as plt
from IPython import get_ipython
import ctypes as ct
import logging
import time
import threading


def _init_spectrometer() -> Avantes:
    """Initialize and connect to an Avantes Spectrometer.
    
    Returns:
        Avantes: Avantes spectrometer.
    """

    dll_path = Path(__file__).parent.parent.joinpath(
        'lib/avaspec3/avaspecx64.dll')

    record = EquipmentRecord(
    manufacturer='Avantes',
    model='AvaSpec-UCLS2048BCL-EVO-RS',  # update for your device
    serial='2011126U1',  # update for your device
    connection=ConnectionRecord(
        address=f'SDK::{dll_path}',
        backend=Backend.MSL))

    # Initialize Avantes SDK and establish the connection to the spectrometer
    ava = record.connect()
    print('Spectrometer connected')

    return ava


def _init_DMD(dmd_lib_version: str = '4.2') -> Tuple[ALP4, int]:
    """Initialize a DMD and clean its allocated memory from a previous use.
    
    Args:
        dmd_lib_version [str]: the version of the DMD library

    Returns:
        Tuple[ALP4, int]: Tuple containing initialized DMD object and DMD
        initial available memory.
    """

    # Initializing DMD
    stop_init = False
    if dmd_lib_version == '4.1':
        print('dmd lib version = ' + dmd_lib_version + ' not installed, please, install it at the location : "openspyrit/spas/alpV41"')
        stop_init = True
    elif dmd_lib_version == '4.2':
        dll_path = Path(__file__).parent.parent.joinpath('lib/alpV42').__str__()
        DMD = ALP4(version='4.2',libDir=dll_path)
    elif dmd_lib_version == '4.3':
        dll_path = Path(__file__).parent.parent.joinpath('lib/alpV43').__str__()
        DMD = ALP4(version='4.3',libDir=dll_path)
    else:
        print('unknown version of dmd library')
        stop_init = True
        
    if stop_init == False:
        DMD.Initialize(DeviceNum=None)
    
        #print(f'DMD initial available memory: {DMD.DevInquire(ALP_AVAIL_MEMORY)}')
        print('DMD connected')
    
        return DMD, DMD.DevInquire(ALP_AVAIL_MEMORY)
    else:
        print('DMD initialisation aborted')


def init(dmd_lib_version: str = '4.2') -> Tuple[Avantes, ALP4, int]:
    """Call functions to initialize spectrometer and DMD.
    
    Args:
        dmd_lib_version [str]: the version of the DMD library

    Returns:
        Tuple[Avantes, ALP4, int]: Tuple containing equipments and DMD initial
        available memory:
            Avantes:
                Connected spectrometer object.
            ALP4:
                Connected DMD object.
            DMD_initial_memory (int):
                Initial memory available in DMD after initialization.            
    """
    
    DMD, DMD_initial_memory = _init_DMD(dmd_lib_version)
    return _init_spectrometer(), DMD, DMD_initial_memory


def init_2arms(dmd_lib_version: str = '4.2') -> Tuple[Avantes, ALP4, int]:
    """Call functions to initialize spectrometer and DMD.
    
    Args:
        dmd_lib_version [str]: the version of the DMD library

    Returns:
        Tuple[Avantes, ALP4, int]: Tuple containing equipments and DMD initial
        available memory:
            Avantes:
                Connected spectrometer object.
            ALP4:
                Connected DMD object.
            DMD_initial_memory (int):
                Initial memory available in DMD after initialization.            
    """
    
    DMD, DMD_initial_memory = _init_DMD(dmd_lib_version)
    camPar = _init_CAM()
    return _init_spectrometer(), DMD, DMD_initial_memory, camPar


def _calculate_timings(integration_time: float = 1, 
                      integration_delay: int = 0, 
                      add_illumination_time: int = 300, 
                      synch_pulse_delay: int = 0, 
                      dark_phase_time: int = 44,
                      ) -> Tuple[int, int, int]:
    """Calculate spectrometer and DMD dependant timings.

    Args:
        integration_time (float): 
            Spectrometer exposure time during one scan in miliseconds. 
            Default is 1 ms.
        integration_delay (int):
            Parameter used to start the integration time not immediately after 
            the measurement request (or on an external hardware trigger), but 
            after a specified delay. Unit is based on internal FPGA clock cycle.
            Default is 0 us.
        add_illumination_time (int): 
            Extra time in microseconds to account for the spectrometer's 
            "dead time". Default is 365 us.
        synch_pulse_delay (int): 
            Time in microseconds between start of the frame synch output pulse 
            and the start of the pattern display (in master mode). Default is
            0 us.
        dark_phase_time (int):
            Time in microseconds taken by the DMD mirrors to completely tilt. 
            Minimum time for XGA type DMD is 44 us. Default is 44 us.

    Returns:
        [Tuple]: DMD timings which depend on spectrometer's parameters.
            synch_pulse_width: Duration of DMD's frame synch output pulse. Units
            in microseconds.
            illumination_time: Duration of the display of one pattern in a DMD
            sequence. Units in microseconds.
            picture_time: Time between the start of two consecutive pictures 
            (i.e. this parameter defines the image display rate). Units in
            microseconds.
    """

    illumination_time = (integration_delay/1000 + integration_time*1000 + 
        add_illumination_time)
    picture_time = illumination_time + dark_phase_time
    synch_pulse_width = round(illumination_time/2 + synch_pulse_delay)
    illumination_time = round(illumination_time)
    picture_time = round(picture_time)

    return synch_pulse_width, illumination_time, picture_time


def _setup_spectrometer(ava: Avantes, 
                        integration_time: float, 
                        integration_delay: int,
                        start_pixel: int,
                        stop_pixel: int,
                       ) -> Tuple[SpectrometerParameters, List[float]]:
    """Sets configurations in the spectrometer.
    
    Set all necessary configurations in the spectrometer preparing it for a
    measurement. Creates SpectrometerData containing its metadata. Gets the
    correct wavelengths depending on the selected pixels to be used.

    Args:
        ava (Avantes):
            Avantes spectrometer.
        integration_time (float):
            Spectrometer exposure time during one scan in miliseconds. 
        integration_delay (int):
            Parameter used to start the integration time not immediately after 
            the measurement request (or on an external hardware trigger), but 
            after a specified delay. Unit is based on internal FPGA clock cycle.
        start_pixel (int):
            Initial pixel data received from spectrometer.
        stop_pixel (int, optional):
            Last pixel data received from spectrometer. If None, then its value
            will be determined from the amount of available pixels in the
            spectrometer.
    Returns:
        Tuple[SpectrometerParameters, List[float]]: Metadata and wavelengths.
            spectrometer_params (SpectrometerParameters): 
                Spectrometer metadata.
            wavelengths (List): 
                List of float corresponding to the wavelengths associated with
                spectrometer's start and stop pixels. 
    """

    spectrometer_detector = ava.SensType(
        ava.get_parameter().m_Detector.m_SensorType).name

    # Get the number of pixels that the spectrometer has
    initial_available_pixels = ava.get_num_pixels()
    
    # print(f'\nThe spectrometer has {initial_available_pixels} pixels')

    # Enable the 16-bit AD converter for High-Resolution
    ava.use_high_res_adc(True)

    # Creating configuration block
    measconfig = ava.MeasConfigType()
        
    measconfig.m_StartPixel = start_pixel

    if stop_pixel is None:
        measconfig.m_StopPixel = initial_available_pixels - 1 
    else:
        measconfig.m_StopPixel = stop_pixel

    measconfig.m_IntegrationTime = integration_time
    measconfig.m_IntegrationDelay = integration_delay
    measconfig.m_NrAverages = 1 
    
    dark_correction = ava.DarkCorrectionType()
    dark_correction.m_Enable = 0
    dark_correction.m_ForgetPercentage = 100
    measconfig.m_CorDynDark = dark_correction
   
    smoothing = ava.SmoothingType()
    smoothing.m_SmoothPix = 0
    smoothing.m_SmoothModel = 0
    measconfig.m_Smoothing = smoothing

    measconfig.m_SaturationDetection = 1 
    
    trigger = ava.TriggerType()
    trigger.m_Mode = 2
    trigger.m_Source = 0
    trigger.m_SourceType = 0
    measconfig.m_Trigger = trigger
    
    control_settings = ava.ControlSettingsType()
    control_settings.m_StrobeControl = 0
    control_settings.m_LaserDelay = 0
    control_settings.m_LaserWidth = 0
    control_settings.LaserWaveLength = 0.00
    control_settings.m_StoreToRam = 0
    measconfig.m_Control = control_settings

    ava.prepare_measure(measconfig)

    spectrometer_params = SpectrometerParameters(
        high_resolution=True,
        initial_available_pixels=initial_available_pixels,
        detector=spectrometer_detector,
        configs=measconfig,
        version_info=ava.get_version_info())

    # Get the wavelength corresponding to each pixel
    wavelengths = ava.get_lambda()[
        spectrometer_params.start_pixel:spectrometer_params.stop_pixel+1]

    return spectrometer_params, np.asarray(wavelengths)


def _setup_DMD(DMD: ALP4, 
              add_illumination_time: int,
              initial_memory: int
              ) -> DMDParameters:
    """Create DMD metadata.

    Creates basic DMD metadata, but leaves most of its fields empty to be set
    later. Sets up the initial free memory present in the DMD.
    This function's name is used to create cohesion between spectrometer and DMD
    related functions.

    Args:
        DMD (ALP4):
            Connected DMD object.
        add_illumination_time (int):
            Extra time in microseconds to account for the spectrometer's 
            "dead time".
        initial_memory (int):
            Initial memory available in DMD after initialization.

    Returns:
        DMDParameters:
            DMD metadata object.
    """

    return DMDParameters(
        add_illumination_time_us=add_illumination_time,
        initial_memory=initial_memory,
        DMD=DMD)


def _sequence_limits(DMD: ALP4, 
                     pattern_compression: int, 
                     sequence_lenght: int,
                     pos_neg: bool = True) -> int:
    """Set sequence limits based on a sequence already uploaded to DMD.

    Args:
        DMD (ALP4): 
            Connected DMD object.
        pattern_compression (int):
            Percentage of total available patterns to be present in an
            acquisition sequence.
        sequence_lenght (int):
            Amount of patterns present in DMD memory.
        pos_neg (bool):
            Boolean indicating if sequence is formed by positive and negative
            patterns. Default is True.

    Returns:
        frames (int): 
            Amount of patterns to be used from a sequence based on the pattern
            compression.
    """

    # Choosing beggining of the sequence
    # DMD.SeqControl(ALP_BITNUM, 1)
    DMD.SeqControl(ALP_FIRSTFRAME, 0)

    # Choosing the end of the sequence
    if (round(pattern_compression * sequence_lenght) % 2 == 0) or not (pos_neg):
        frames = round(pattern_compression * sequence_lenght)
    else:
        frames = round(pattern_compression * sequence_lenght) + 1

    DMD.SeqControl(ALP_LASTFRAME, frames - 1)

    return frames


def _update_sequence(DMD: ALP4,
                     DMD_params: DMDParameters,
                     acquisition_params: AcquisitionParameters,
                     pattern_source: str,
                     pattern_prefix: str,
                     pattern_order: List[int],
                     bitplanes: int = 1):
    """Send new complete pattern sequence to DMD.

    Args:
        DMD (ALP4):
            Connected DMD object.
        DMD_params (DMDParameters):
            DMD metadata object to be updated with pattern related data and with
            memory available after patterns are sent to DMD.
        acquisition_params (AcquisitionParameters):
            Acquisition related metadata object. User must partially fill up
            with pattern_compression, pattern_dimension_x, pattern_dimension_y,
            zoom, x and y offest of patterns displayed on the DMD.
        pattern_source (str):
            Pattern source folder.
        pattern_preffix (str):
            Prefix used in pattern naming.
        pattern_order (List[int]):
            List of the pattern indices in a certain order for upload to DMD.
        bitplanes (int, optional): 
            Pattern bitplanes. Defaults to 1.
    """
    
    import cv2
    
    path_base = Path(pattern_source)

    seqId = DMD.SeqAlloc(nbImg=len(pattern_order), 
                             bitDepth=bitplanes)
    
    zoom = acquisition_params.zoom
    x_offset = acquisition_params.xw_offset
    y_offset = acquisition_params.yh_offset
    Np = acquisition_params.pattern_dimension_x
           
    dmd_height = DMD_params.display_height
    dmd_width = DMD_params.display_width
    len_im = int(dmd_height / zoom)  
  
    t = perf_counter_ns()

    # for adaptative patterns into a ROI
    apply_mask = False
    mask_index = acquisition_params.mask_index
        
    if len(mask_index) > 0:
        apply_mask = True
        Npx = acquisition_params.pattern_dimension_x
        Npy = acquisition_params.pattern_dimension_y
        mask_element_nbr = len(mask_index)
        x_mask_coord = acquisition_params.x_mask_coord
        y_mask_coord = acquisition_params.y_mask_coord 
        x_mask_length = x_mask_coord[1] - x_mask_coord[0]
        y_mask_length = y_mask_coord[1] - y_mask_coord[0]
    
    first_pass = True
    for index,pattern_name in enumerate(tqdm(pattern_order, unit=' patterns', total=len(pattern_order))):
        # read numpy patterns
        path = path_base.joinpath(f'{pattern_prefix}_{pattern_name}.npy')
        im = np.load(path) 
        
        patterns = np.zeros((dmd_height, dmd_width), dtype=np.uint8)
        
        if apply_mask == True: # for adaptative patterns into a ROI 
            pat_mask_all = np.zeros(y_mask_length*x_mask_length) # initialize a vector of lenght = size of the cropped mask           
            pat_mask_all[mask_index] = im[:mask_element_nbr] #pat_re_vec[:mask_element_nbr] # put the pattern into the vector    
            pat_mask_all_mat = np.reshape(pat_mask_all, [y_mask_length, x_mask_length]) # reshape the vector into a matrix of the 2d cropped mask
            # resize the matrix to the DMD size
            pat_mask_all_mat_DMD = cv2.resize(pat_mask_all_mat, (int(dmd_height*x_mask_length/(Npx*zoom)), int(dmd_height*y_mask_length/(Npy*zoom))), interpolation = cv2.INTER_NEAREST)
            
            if first_pass == True:
                first_pass = False
                len_im3 = pat_mask_all_mat_DMD.shape
                    
            patterns[y_offset:y_offset+len_im3[0], x_offset:x_offset+len_im3[1]] = pat_mask_all_mat_DMD 
        else: # send the entire square pattern without the mask
            im_mat = np.reshape(im, [Np,Np])
            im_HD = cv2.resize(im_mat, (int(dmd_height/zoom), int(dmd_height/zoom)), interpolation = cv2.INTER_NEAREST)
            
            if first_pass == True:
                len_im = im_HD.shape
                first_pass = False
                
            patterns[y_offset:y_offset+len_im[0], x_offset:x_offset+len_im[1]] = im_HD  
        
        # if pattern_name == 800:
        #     plt.figure()
        #     # plt.imshow(pat_c_re)
        #     # plt.imshow(pat_mask_all_mat)
        #     # plt.imshow(pat_mask_all_mat_DMD)
        #     plt.imshow(np.rot90(patterns,2))
        #     plt.colorbar()
        #     plt.title('pattern n°' + str(pattern_name))
        
        patterns = patterns.ravel()
        
        DMD.SeqPut(
            imgData=patterns.copy(),
            PicOffset=index, 
            PicLoad=1)
    
    print(f'\nTime for sending all patterns: '
          f'{(perf_counter_ns() - t)/1e+9} s')


def _setup_patterns(DMD: ALP4, 
                    metadata: MetaData, 
                    DMD_params: DMDParameters, 
                    acquisition_params: AcquisitionParameters,
                    cov_path: str = None, 
                    pattern_to_display: str = 'white', 
                    loop: bool = False) -> None:
    """Read and send patterns to DMD.

    Reads patterns from a file and sends a percentage of them to the DMD,
    considering positve and negative Hadamard patterns, which should be even in
    number. 
    Prints time taken to read all patterns and send the requested ones
    to DMD. 
    Updates available memory in DMD metadata object (DMD_params).

    Args:
        DMD (ALP4):
            Connected DMD object.
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file 
            outputs. Must be created and filled up by the user.
        DMD_params (DMDParameters):
            DMD metadata object to be updated with pattern related data and with
            memory available after patterns are sent to DMD.
        acquisition_params (AcquisitionParameters):
            Acquisition related metadata object. User must partially fill up
            with pattern_compression, pattern_dimension_x, pattern_dimension_y,
            zoom, x and y offest of patterns displayed on the DMD.
        loop (bool):
            is to projet in loop, one or few patterns continously (see AlpProjStartCont
            in the doc for more detail). Default is False
    """
    
    file = np.load(Path(metadata.pattern_order_source))
    pattern_order = file['pattern_order']               
    pos_neg = file['pos_neg']
    
    if loop == True:
        pos_neg = False
        if pattern_to_display == 'white':
            pattern_order = np.array(pattern_order[0:1], dtype=np.int16)
        elif pattern_to_display == 'black':
            pattern_order = np.array(pattern_order[1:2], dtype=np.int16)
        elif pattern_to_display == 'gray':
            index = int(np.where(pattern_order == 1953)[0])
            print(index)
            pattern_order = np.array(pattern_order[index:index+1], dtype=np.int16)
        
    bitplanes = 1
    DMD_params.bitplanes = bitplanes

    if (DMD_params.initial_memory - DMD.DevInquire(ALP_AVAIL_MEMORY) == 
        len(pattern_order)) and loop == False:
        print('Reusing patterns from previous acquisition')
        acquisition_params.pattern_amount = _sequence_limits(
            DMD, 
            acquisition_params.pattern_compression,
            len(pattern_order),
            pos_neg=pos_neg)

    else:
        if (DMD.Seqs):
            DMD.FreeSeq()

        _update_sequence(DMD, DMD_params, acquisition_params, metadata.pattern_source, metadata.pattern_prefix, 
                         pattern_order, bitplanes)
        print(f'DMD available memory after sequence allocation: '
        f'{DMD.DevInquire(ALP_AVAIL_MEMORY)}')
        acquisition_params.pattern_amount = _sequence_limits(
            DMD, 
            acquisition_params.pattern_compression, 
            len(pattern_order),
            pos_neg=pos_neg)        
        
    acquisition_params.patterns = (
        pattern_order[0:acquisition_params.pattern_amount])
    
    # Confirm memory allocated in DMD
    DMD_params.update_memory(DMD.DevInquire(ALP_AVAIL_MEMORY))


def _setup_patterns_2arms(DMD: ALP4, 
                          metadata: MetaData, 
                          DMD_params: DMDParameters, 
                          acquisition_params: AcquisitionParameters, 
                          camPar: CAM,
                          cov_path: str = None) -> None:
    """Read and send patterns to DMD.

    Reads patterns from a file and sends a percentage of them to the DMD,
    considering positve and negative Hadamard patterns, which should be even in
    number. 
    Prints time taken to read all patterns and send the requested ones
    to DMD. 
    Updates available memory in DMD metadata object (DMD_params).

    Args:
        DMD (ALP4):
            Connected DMD object.
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file 
            outputs. Must be created and filled up by the user.
        DMD_params (DMDParameters):
            DMD metadata object to be updated with pattern related data and with
            memory available after patterns are sent to DMD.
        acquisition_params (AcquisitionParameters):
            Acquisition related metadata object. User must partially fill up
            with pattern_compression, pattern_dimension_x, pattern_dimension_y,
            zoom, x and y offest of patterns displayed on the DMD.
        camPar (CAM):
            Metadata object of the IDS monochrome camera 
        cov_path (str): 
            Path to the covariance matrix used for reconstruction.
            It must be a .npy (numpy) or .pt (pytorch) file. It is converted to 
            a torch tensor for reconstruction.
            
    """

    file = np.load(Path(metadata.pattern_order_source))
    pattern_order = file['pattern_order']
    pattern_order = pattern_order.astype('int32')
    
    # copy the black pattern image (png) to the number = -1
    # black_pattern_dest_path = Path( metadata.pattern_source + '/' + metadata.pattern_prefix + '_' + '-1.png' )
    black_pattern_dest_path = Path( metadata.pattern_source + '/' + metadata.pattern_prefix + '_' + '-1.npy' )
                                  
    if black_pattern_dest_path.is_file() == False:
        # black_pattern_orig_path = Path( metadata.pattern_source + '/' + metadata.pattern_prefix + '_' + 
        #                               str(camPar.black_pattern_num) + '.png' )
        black_pattern_orig_path = Path( metadata.pattern_source + '/' + metadata.pattern_prefix + '_' + 
                                      str(camPar.black_pattern_num) + '.npy' )
        shutil.copyfile(black_pattern_orig_path, black_pattern_dest_path)                              
        
        
    # add white patterns for the camera
    if camPar.insert_patterns == 1:
        inc = 0
        while True:
            try:
                pattern_order[inc]  # except error from the end of array to stop the loop
                if (inc % camPar.gate_period) == 0:#16) == 0:
                    pattern_order = np.insert(pattern_order, inc, -1) # double white pattern is required if integration time is shorter than 3.85 ms
                    if camPar.int_time_spect < 3.85:
                        pattern_order = np.insert(pattern_order, inc+1, -1)
                        if camPar.int_time_spect < 1.65:
                            pattern_order = np.insert(pattern_order, inc+2, -1)
                            if camPar.int_time_spect < 1:
                                pattern_order = np.insert(pattern_order, inc+1, -1)
                                if camPar.int_time_spect <= 0.6:
                                    pattern_order = np.insert(pattern_order, inc+1, -1)
                inc = inc + 1
            except:
                # print('while loop finished')
                break
       
        # if camPar.int_time_spect < 1.75: # add one pattern at the beginning of the sequence when the integration time of the spectrometer is shorter than 1.75 ms
        #     print('no interleaving')
        #     #pattern_order = np.insert(pattern_order, 0, -1) 
        #     # pattern_order = np.insert(pattern_order, 0, -1)     
        if (len(pattern_order)%2) != 0: # Add one pattern at the end of the sequence if the pattern number is even
            pattern_order = np.insert(pattern_order, len(pattern_order), -1)
            print('pattern order is odd => a black image is automaticly insert, need to be deleted in the case for tuning the spectrometer')
               
    pos_neg = file['pos_neg']

    bitplanes = 1
    DMD_params.bitplanes = bitplanes

    if (DMD_params.initial_memory - DMD.DevInquire(ALP_AVAIL_MEMORY) == 
        len(pattern_order)):
        print('Reusing patterns from previous acquisition')
        acquisition_params.pattern_amount = _sequence_limits(
            DMD, 
            acquisition_params.pattern_compression,
            len(pattern_order),
            pos_neg=pos_neg)

    else:
        if (DMD.Seqs):
            DMD.FreeSeq()

        _update_sequence(DMD, DMD_params, acquisition_params, metadata.pattern_source, metadata.pattern_prefix, 
                         pattern_order, bitplanes)
        print(f'DMD available memory after sequence allocation: '
        f'{DMD.DevInquire(ALP_AVAIL_MEMORY)}')
        acquisition_params.pattern_amount = _sequence_limits(
            DMD, 
            acquisition_params.pattern_compression, 
            len(pattern_order),
            pos_neg=pos_neg)        

    acquisition_params.patterns = (
        pattern_order[0:acquisition_params.pattern_amount])
    
    acquisition_params.patterns_wp = acquisition_params.patterns
    
    # Confirm memory allocated in DMD
    DMD_params.update_memory(DMD.DevInquire(ALP_AVAIL_MEMORY))

def _setup_timings(DMD: ALP4, 
                   DMD_params: DMDParameters, 
                   picture_time: int, 
                   illumination_time: int, 
                   synch_pulse_delay: int, 
                   synch_pulse_width: int, 
                   trigger_in_delay: int,
                   add_illumination_time: int) -> None:
    """Setup pattern sequence timings in DMD.

    Send previously user-defined plus calculated timings to DMD.
    Updates DMD metadata with sequence and timing related data.
    This function has no default values for timings and lets the burden of
    setting them to the setup function.

    Args:
        DMD (ALP4):
            Connected DMD object.
        DMD_params (DMDParameters):
            DMD metadata object to be updated with pattern related data and with
            memory available after patterns are sent to DMD.
        picture_time (int):
            Time between the start of two consecutive pictures (i.e. this 
            parameter defines the image display rate). Units in microseconds.
        illumination_time (int):
            Duration of the display of one pattern in a DMD sequence. 
            Units in microseconds.
        synch_pulse_delay (int):
            Time in microseconds between start of the frame synch output pulse 
            and the start of the pattern display (in master mode).
        synch_pulse_width (int):
            Duration of DMD's frame synch output pulse. Units in microseconds.
        trigger_in_delay (int):
            Time in microseconds between the incoming trigger edge and the start
            of the pattern display on DMD (slave mode).
        add_illumination_time (int):
            Extra time in microseconds to account for the spectrometer's 
            "dead time".
    """

    DMD.SetTiming(illuminationTime=illumination_time,
                  pictureTime=picture_time,
                  synchDelay=synch_pulse_delay,
                  synchPulseWidth=synch_pulse_width,
                  triggerInDelay=trigger_in_delay)

    DMD_params.update_sequence_parameters(add_illumination_time, DMD=DMD)


def setup(spectrometer: Avantes,
          DMD: ALP4,
          DMD_initial_memory: int, 
          metadata: MetaData,
          acquisition_params: AcquisitionParameters,
          start_pixel: int = 0,
          stop_pixel: Optional[int] = None,
          integration_time: float = 1, 
          integration_delay: int = 0,
          DMD_output_synch_pulse_delay: int = 0, 
          add_illumination_time: int = 356,
          dark_phase_time: int = 44,
          DMD_trigger_in_delay: int = 0,
          pattern_to_display: str = 'white',
          loop: bool = False
          ) -> Tuple[SpectrometerParameters, DMDParameters]:
    """Setup everything needed to start an acquisition.

    Sets all parameters for DMD, spectrometer, DMD patterns and DMD timings.
    Must be called before every acquisition.

    Args:
        spectrometer (Avantes):
            Connected spectrometer (Avantes object).
        DMD (ALP4):
            Connected DMD.
        DMD_initial_memory (int):
            Initial memory available in DMD after initialization.
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file 
            outputs. Must be created and filled up by the user.
        acquisition_params (AcquisitionParameters):
            Acquisition related metadata object. User must partially fill up
            with pattern_compression, pattern_dimension_x, pattern_dimension_y.
        start_pixel (int):
            Initial pixel data received from spectrometer. Default is 0.
        stop_pixel (int, optional):
            Last pixel data received from spectrometer. Default is None if it
            should be determined from the amount of available pixels in the
            spectrometer.
        integration_time (float):
            Spectrometer exposure time during one scan in miliseconds. Default
            is 1 ms.
        integration_delay (int):
            Parameter used to start the integration time not immediately after 
            the measurement request (or on an external hardware trigger), but 
            after a specified delay. Unit is based on internal FPGA clock cycle.
            Default is 0 us.
        DMD_output_synch_pulse_delay (int):
            Time in microseconds between start of the frame synch output pulse 
            and the start of the pattern display (in master mode). Default is
            0 us.
        add_illumination_time (int):
            Extra time in microseconds to account for the spectrometer's 
            "dead time". Default is 365 us.
        dark_phase_time (int):
            Time in microseconds taken by the DMD mirrors to completely tilt. 
            Minimum time for XGA type DMD is 44 us. Default is 44 us.
        DMD_trigger_in_delay (int):
            Time in microseconds between the incoming trigger edge and the start
            of the pattern display on DMD (slave mode). Default is 0 us.
        pattern_to_display (string):
            display one pattern on the DMD to tune the spectrometer. Default is white 
            pattern
        loop (bool):
            is to projet in loop, one or few patterns continuously (see AlpProjStartCont
            in the doc for more detail). Default is False
    Raises:
        ValueError: Sum of dark phase and additional illumination time is lower
        than 400 us.

    Returns:
        Tuple[SpectrometerParameters, DMDParameters, List]: Tuple containing DMD
        and spectrometer relate metadata, as well as wavelengths.
            spectrometer_params (SpectrometerParameters):
                Spectrometer metadata object with spectrometer configurations.
            DMD_params (DMDParameters):
                DMD metadata object with DMD configurations.
    """

    if loop == False:
        path = Path(metadata.output_directory)
        if not path.exists():
            path.mkdir()

    if dark_phase_time + add_illumination_time < 350:
        raise ValueError(f'Sum of dark phase and additional illumination time '
                         f'is {dark_phase_time + add_illumination_time}.'
                         f' Must be greater than 350 µs.')

    elif dark_phase_time + add_illumination_time < 400:
        warnings.warn(f'Sum of dark phase and additional illumination time '
                      f'is {dark_phase_time + add_illumination_time}.'
                      f' It is recomended to choose at least 400 µs.')

    synch_pulse_width, illumination_time, picture_time = _calculate_timings(
        integration_time, 
        integration_delay, 
        add_illumination_time, 
        DMD_output_synch_pulse_delay, 
        dark_phase_time)

    spectrometer_params, wavelenghts = _setup_spectrometer(
        spectrometer, 
        integration_time, 
        integration_delay,
        start_pixel,
        stop_pixel)

    acquisition_params.wavelengths = np.asarray(wavelenghts, dtype=np.float64)

    DMD_params = _setup_DMD(DMD, add_illumination_time, DMD_initial_memory)

    _setup_patterns(DMD=DMD, metadata=metadata, DMD_params=DMD_params, 
                    acquisition_params=acquisition_params, loop=loop,
                    pattern_to_display=pattern_to_display)
    
    _setup_timings(DMD, DMD_params, picture_time, illumination_time, 
                   DMD_output_synch_pulse_delay, synch_pulse_width, 
                   DMD_trigger_in_delay, add_illumination_time)

    return spectrometer_params, DMD_params


def change_patterns(DMD: ALP4, 
                    acquisition_params: AcquisitionParameters, 
                    zoom: int = 1, 
                    xw_offset: int = 0, 
                    yh_offset: int = 0,
                    force_change: bool = False
                    ):
    """
    Delete patterns in the memory of the DMD in the case where the zoom or (x,y) offset change
    
    DMD (ALP4):
        Connected DMD.
    acquisition_params (AcquisitionParameters):
        Acquisition related metadata object. User must partially fill up
        with pattern_compression, pattern_dimension_x, pattern_dimension_y,
        zoom, x and y offest of patterns displayed on the DMD.    
    zoom (int):
        digital zoom. Deafult is x1. 
    xw_offset (int):
        offset int he width direction of the patterns for zoom > 1. 
        Default is 0. 
    yh_offset (int):
        offset int he height direction of the patterns for zoom > 1. 
        Default is 0.
    force_change (bool):
        to force the changement of the pattern sequence. Default is False.
    """
    
    if acquisition_params.zoom != zoom or acquisition_params.xw_offset != xw_offset or acquisition_params.yh_offset != yh_offset or force_change == True:
        if (DMD.Seqs):
            DMD.FreeSeq()

    
def setup_2arms(spectrometer: Avantes,
          DMD: ALP4,
          camPar: CAM,
          DMD_initial_memory: int, 
          metadata: MetaData,
          acquisition_params: AcquisitionParameters,
          start_pixel: int = 0,
          stop_pixel: Optional[int] = None,
          integration_time: float = 1, 
          integration_delay: int = 0,
          DMD_output_synch_pulse_delay: int = 0, 
          add_illumination_time: int = 356,
          dark_phase_time: int = 44,
          DMD_trigger_in_delay: int = 0          
          ) -> Tuple[SpectrometerParameters, DMDParameters]:
    """Setup everything needed to start an acquisition.

    Sets all parameters for DMD, spectrometer, DMD patterns and DMD timings.
    Must be called before every acquisition.

    Args:
        spectrometer (Avantes):
            Connected spectrometer (Avantes object).
        DMD (ALP4):
            Connected DMD.
        camPar (CAM):
            Metadata object of the IDS monochrome camera 
        DMD_initial_memory (int):
            Initial memory available in DMD after initialization.
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file 
            outputs. Must be created and filled up by the user.
        acquisition_params (AcquisitionParameters):
            Acquisition related metadata object. User must partially fill up
            with pattern_compression, pattern_dimension_x, pattern_dimension_y,
            zoom, x and y offest of patterns displayed on the DMD.
        start_pixel (int):
            Initial pixel data received from spectrometer. Default is 0.
        stop_pixel (int, optional):
            Last pixel data received from spectrometer. Default is None if it
            should be determined from the amount of available pixels in the
            spectrometer.
        integration_time (float):
            Spectrometer exposure time during one scan in miliseconds. Default
            is 1 ms.
        integration_delay (int):
            Parameter used to start the integration time not immediately after 
            the measurement request (or on an external hardware trigger), but 
            after a specified delay. Unit is based on internal FPGA clock cycle.
            Default is 0 us.
        DMD_output_synch_pulse_delay (int):
            Time in microseconds between start of the frame synch output pulse 
            and the start of the pattern display (in master mode). Default is
            0 us.
        add_illumination_time (int):
            Extra time in microseconds to account for the spectrometer's 
            "dead time". Default is 365 us.
        dark_phase_time (int):
            Time in microseconds taken by the DMD mirrors to completely tilt. 
            Minimum time for XGA type DMD is 44 us. Default is 44 us.
        DMD_trigger_in_delay (int):
            Time in microseconds between the incoming trigger edge and the start
            of the pattern display on DMD (slave mode). Default is 0 us.
    
    Raises:
        ValueError: Sum of dark phase and additional illumination time is lower
        than 400 us.

    Returns:
        Tuple[SpectrometerParameters, DMDParameters, List]: Tuple containing DMD
        and spectrometer relate metadata, as well as wavelengths.
            spectrometer_params (SpectrometerParameters):
                Spectrometer metadata object with spectrometer configurations.
            DMD_params (DMDParameters):
                DMD metadata object with DMD configurations.
    """

    path = Path(metadata.output_directory)
    if not path.exists():
        path.mkdir()
    
    if dark_phase_time + add_illumination_time < 350:
        raise ValueError(f'Sum of dark phase and additional illumination time '
                         f'is {dark_phase_time + add_illumination_time}.'
                         f' Must be greater than 350 µs.')

    elif dark_phase_time + add_illumination_time < 400:
        warnings.warn(f'Sum of dark phase and additional illumination time '
                      f'is {dark_phase_time + add_illumination_time}.'
                      f' It is recomended to choose at least 400 µs.')
    
    synch_pulse_width, illumination_time, picture_time = _calculate_timings(
        integration_time, 
        integration_delay, 
        add_illumination_time, 
        DMD_output_synch_pulse_delay, 
        dark_phase_time)

    spectrometer_params, wavelenghts = _setup_spectrometer(
        spectrometer, 
        integration_time, 
        integration_delay,
        start_pixel,
        stop_pixel)
    
    if camPar.gate_period > 16:
        gate_period = 16
        print('Warning, gate period is ' + str(camPar.gate_period) + ' >  than the max: 16.')
        print('Try to increase the FPS of the camera, or the integration time of the spectrometer.')
        print('Check the Pixel clock which must be = 474 MHz')
        print('Otherwise some frames will be lost.')
    elif camPar.gate_period <1:
        print('Warning, gate period is ' + str(camPar.gate_period) + ' <  than the min: 1.')
        gate_period = 1
    else:
        gate_period = camPar.gate_period
    
    camPar.gate_period = gate_period    
    Gate = tAlpDynSynchOutGate()
    Gate.byref[0] = ct.c_ubyte(gate_period)     # Period [1 to 16] (it is a multiple of the trig period which go to the spectro)
    Gate.byref[1] = ct.c_ubyte(1)   # Polarity => 0: active pulse is low, 1: high
    Gate.byref[2] = ct.c_ubyte(1)   # Gate1 ok to send TTL 
    Gate.byref[3] = ct.c_ubyte(0)   # Gate2 do not send TTL
    Gate.byref[4] = ct.c_ubyte(0)   # Gate3 do not send TTL
    DMD.DevControlEx(ALP_DEV_DYN_SYNCH_OUT1_GATE, Gate)
    camPar.gate_period = gate_period
    camPar.int_time_spect = integration_time

    acquisition_params.wavelengths = np.asarray(wavelenghts, dtype=np.float64)

    DMD_params = _setup_DMD(DMD, add_illumination_time, DMD_initial_memory)
    
    _setup_patterns_2arms(DMD=DMD, metadata=metadata, DMD_params=DMD_params, 
                    acquisition_params=acquisition_params, camPar=camPar)

    _setup_timings(DMD, DMD_params, picture_time, illumination_time, 
                   DMD_output_synch_pulse_delay, synch_pulse_width, 
                   DMD_trigger_in_delay, add_illumination_time)

    return spectrometer_params, DMD_params, camPar

 
def _calculate_elapsed_time(start_measurement_time: int, 
                          measurement_time: np.ndarray,
                          timestamps: List[int],
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate acquisition timings.

    Calculates elapsed time between each callback measurement taking into
    account the moment when the DMD started running a sequence.
    Calculates elapsed time between each spectrum acquired by the spectrometer
    based on the spectrometer's internal clock.

    Args:
        start_measurement_time (int): 
            Time in nanoseconds when DMD is set to start running a sequence.
        measurement_time (np.ndarray): 
            1D array with `int` type timings in nanoseconds when each callbacks
            starts. 
        timestamps (List[int]):
            1D array with measurement timestamps from spectrometer.
            Timestamps count ticks for the last pixel of the spectrum was
            received by the spectrometer microcontroller. Ticks are in 10 
            microsecond units since the spectrometer started.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple with measurement timings.
            measurement_time (np.ndarray): 
                1D array with `float` type elapsed times between each callback.
                Units in milliseconds. 
            timestamps (np.ndarray): 
                1D array with `float` type elapsed time between each measurement
                made by the spectrometer based on its internal clock. 
                Units in milliseconds.
    """

    measurement_time = np.concatenate(
        (start_measurement_time,measurement_time),axis=None)

    measurement_time = np.diff(measurement_time)/1e+6 # In ms
    timestamps = np.diff(timestamps)/100 # In ms

    return measurement_time, timestamps


def _save_acquisition(metadata: MetaData, 
                     DMD_params: DMDParameters, 
                     spectrometer_params: SpectrometerParameters, 
                     acquisition_parameters: AcquisitionParameters, 
                     spectral_data: np.ndarray) -> None:
    """Save all acquisition data and metadata.

    Args:
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file
            outputs.
        DMD_params (DMDParameters): 
            DMD metadata object with DMD configurations.
        spectrometer_params (SpectrometerParameters):
            Spectrometer metadata object with spectrometer configurations.
        acquisition_parameters (AcquisitionParameters):
            Acquisition related metadata object. 
        spectral_data (ndarray):
            1D array with `float` type spectrometer measurements. Array size
            depends on start and stop pixels previously set to the spectrometer.
    """

    # Saving collected data and timings
    path = Path(metadata.output_directory)
    path = path / f'{metadata.experiment_name}_spectraldata.npz'
    np.savez_compressed(path, spectral_data=spectral_data)

    # 'save_metadata' function is commented because the 'save_metadata_2arms' function is executed after the 'acquire' function in the "main_seq_2arms.py" prog
    # # Saving metadata
    # save_metadata(metadata, 
    #               DMD_params,
    #               spectrometer_params,
    #               acquisition_parameters)

def _save_acquisition_2arms(metadata: MetaData, 
                     DMD_params: DMDParameters, 
                     spectrometer_params: SpectrometerParameters, 
                     camPar: CAM,
                     acquisition_parameters: AcquisitionParameters, 
                     spectral_data: np.ndarray) -> None:
    """Save all acquisition data and metadata.

    Args:
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file
            outputs.
        DMD_params (DMDParameters): 
            DMD metadata object with DMD configurations.
        spectrometer_params (SpectrometerParameters):
            Spectrometer metadata object with spectrometer configurations.
        camPar (CAM):
            Metadata object of the IDS monochrome camera 
        acquisition_parameters (AcquisitionParameters):
            Acquisition related metadata object. 
        spectral_data (ndarray):
            1D array with `float` type spectrometer measurements. Array size
            depends on start and stop pixels previously set to the spectrometer.
    """

    # Saving collected data and timings
    path = Path(metadata.output_directory)
    path = path / f'{metadata.experiment_name}_spectraldata.npz'
    np.savez_compressed(path, spectral_data=spectral_data)

    # Saving metadata
    save_metadata_2arms(metadata, 
                  DMD_params,
                  spectrometer_params,
                  camPar,
                  acquisition_parameters)


def _acquire_raw(ava: Avantes, 
            DMD: ALP4,
            spectrometer_params: SpectrometerParameters, 
            DMD_params: DMDParameters, 
            acquisition_params: AcquisitionParameters,
            loop: bool = False
            ) -> NamedTuple:
    """Raw data acquisition.

    Setups a callback function to receive messages from spectrometer whenever a
    measurement is ready to be read. Reads a measurement via a callback.

    Args:
        ava (Avantes): 
            Connected spectrometer (Avantes object).
        DMD (ALP4): 
            Connected DMD.
        spectrometer_params (SpectrometerParameters): 
            Spectrometer metadata object with spectrometer configurations.
        DMD_params (DMDParameters):
            DMD metadata object with DMD configurations.
        acquisition_params (AcquisitionParameters): 
            Acquisition related metadata object.
        loop (bool):
            if True, projet continuously the pattern, see the AlpProjStartCont function
            if False, projet one time the seq of the patterns, see the AlpProjStart function (Default)

    Returns:
        NamedTuple: NamedTuple containig spectral data and measurement timings.
            spectral_data (ndarray):
                2D array of `float` of size (pattern_amount x pixel_amount)
                containing measurements received from the spectrometer for each
                pattern of a sequence.
            spectrum_index (int):
                Index of the last acquired spectrum. 
            timestamps (np.ndarray): 
                1D array with `float` type elapsed time between each measurement
                made by the spectrometer based on its internal clock. 
                Units in milliseconds.
            measurement_time (np.ndarray): 
                1D array with `float` type elapsed times between each callback.
                Units in milliseconds.
            start_measurement_time (float):
                Time when acquisition started.
            saturation_detected (bool):
                Boolean incating if saturation was detected during acquisition.
    """
    def register_callback(measurement_time, timestamps, 
                          spectral_data, ava):
        
        def measurement_callback(handle, info): # If we want to reconstruct during callback; can use it in here. Add function as parameter. 
            nonlocal spectrum_index
            nonlocal saturation_detected
            
            measurement_time[spectrum_index] = perf_counter_ns()

            if info.contents.value >= 0:                  
                timestamp,spectrum = ava.get_data()
                spectral_data[spectrum_index,:] = (
                    np.ctypeslib.as_array(spectrum[0:pixel_amount]))
                 
                if np.any(ava.get_saturated_pixels() > 0):
                    saturation_detected = True

                timestamps[spectrum_index] = np.ctypeslib.as_array(timestamp)
                
            else: # Set values to zero if an error occured
                spectral_data[spectrum_index,:] = 0
                timestamps[spectrum_index] = 0
            
            spectrum_index += 1
        
        return measurement_callback
    
    
    pixel_amount = (spectrometer_params.stop_pixel - 
                    spectrometer_params.start_pixel + 1)

    measurement_time = np.zeros((acquisition_params.pattern_amount))
    timestamps = np.zeros((acquisition_params.pattern_amount),dtype=np.uint32)
    spectral_data = np.zeros(
        (acquisition_params.pattern_amount,pixel_amount),dtype=np.float64)

    # Boolean to indicate if saturation was detected during acquisition
    saturation_detected = False 

    spectrum_index = 0 # Accessed as nonlocal variable inside the callback
    
    if loop == False:
        #spectro.register_callback(-2,acquisition_params.pattern_amount,pixel_amount)
        callback = register_callback(measurement_time, timestamps, 
                                     spectral_data, ava)
        measurement_callback = MeasureCallback(callback)
        ava.measure_callback(-2, measurement_callback)
    else:
        ava.measure(-1)
    
       
    DMD.Run(loop=loop) # if loop=False : Run the whole sequence only once, if loop=True : Run continuously one pattern 
    start_measurement_time = perf_counter_ns()
    
    if loop == False:
        while(True):
            if(spectrum_index >= acquisition_params.pattern_amount) and loop == False:
                break
            elif((perf_counter_ns() - start_measurement_time) / 1e+6 > 
                (2 * acquisition_params.pattern_amount * 
                DMD_params.picture_time_us / 1e+3)) and loop == False:
                print('Stopping measurement. One of the equipments may be blocked '
                'or disconnected.')
                break
            else:
                sleep(acquisition_params.pattern_amount *
                DMD_params.picture_time_us / 1e+6 / 10)
        DMD.Halt()
    else:                
        sleep(0.1)
        
        timestamp, spectrum = ava.get_data()
        spectral_data_1 = (np.ctypeslib.as_array(spectrum[0:pixel_amount]))
        
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.ion() # create GUI
        figure, ax = plt.subplots(figsize=(10, 8))
        line1, = ax.plot(acquisition_params.wavelengths, spectral_data_1)

        plt.title("Tune the Spectrometer", fontsize=20)
        plt.xlabel("Lambda (nm)")
        plt.ylabel("counts")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        printed = False
        while(True):            
            try:                          
                timestamp, spectrum = ava.get_data()
                spectral_data_1 = (np.ctypeslib.as_array(spectrum[0:pixel_amount]))

                line1.set_xdata(acquisition_params.wavelengths)
                line1.set_ydata(spectral_data_1) # updating data values

                figure.canvas.draw() # drawing updated values
                figure.canvas.flush_events() # flush prior plot
            
                if not printed:
                    print('Press "Ctrl + c" to exit')                       
                    if np.amax(spectral_data_1) >= 65535:
                        print('!!!!!!!!!! Saturation detected in the spectro !!!!!!!!!!')
                    printed = True
            
            except KeyboardInterrupt:
                if (DMD.Seqs):
                    DMD.Halt()
                    DMD.FreeSeq()
                plt.close()
                get_ipython().run_line_magic('matplotlib', 'inline')
                break
            
    ava.stop_measure()
    
    AcquisitionResult = namedtuple('AcquisitionResult', [
        'spectral_data', 
        'spectrum_index',
        'timestamps',
        'measurement_time',
        'start_measurement_time',
        'saturation_detected'])

    return AcquisitionResult(spectral_data, 
                             spectrum_index,
                             timestamps,
                             measurement_time,
                             start_measurement_time,
                             saturation_detected)


def acquire(ava: Avantes, 
            DMD: ALP4,
            metadata: MetaData, 
            spectrometer_params: SpectrometerParameters, 
            DMD_params: DMDParameters, 
            acquisition_params: AcquisitionParameters,
            repetitions: int = 1,
            verbose: bool = False,
            reconstruct: bool = False,
            reconstruction_params: ReconstructionParameters = None
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a complete acquisition.

    Performs single or multiple acquisitions using the same setup configurations
    previously chosen.
    Finnaly saves all acqusition related data and metadata.

    Args:
        ava (Avantes): 
            Connected spectrometer (Avantes object).
        DMD (ALP4): 
            Connected DMD.
        metadata (MetaData): 
            Metadata concerning the experiment, paths, file inputs and file 
            outputs. Must be created and filled up by the user.
        spectrometer_params (SpectrometerParameters): 
            Spectrometer metadata object with spectrometer configurations.
        DMD_params (DMDParameters):
            DMD metadata object with DMD configurations.
        acquisition_params (AcquisitionParameters): 
            Acquisition related metadata object.
        wavelengths (List[float]): 
            List of float corresponding to the wavelengths associated with
            spectrometer's start and stop pixels.
        repetitions (int):
            Number of times the acquisition will be repeated with the same
            configurations. Default is 1, a single acquisition.
        verbose (bool):
            Chooses if data concerning each acquisition should be printed to
            user. If False, only overall data regarding all repetitions is 
            printed. Default is False.
        reconstruct (bool): 
            If True, will perform reconstruction alongside acquisition using
            multiprocessing.
        reconstruction_params (ReconstructionParameters):
            Object containing parameters of the neural network to be loaded for
            reconstruction.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: Tuple containig spectral data and
        measurement timings.
            spectral_data (ndarray):
                2D array of `float` of size (pattern_amount x pixel_amount)
                containing measurements received from the spectrometer for each
                pattern of a sequence. 
            timestamps (np.ndarray): 
                1D array with `float` type elapsed time between each measurement
                made by the spectrometer based on its internal clock. 
                Units in milliseconds.
            measurement_time (np.ndarray): 
                1D array with `float` type elapsed times between each callback.
                Units in milliseconds.
    """

    loop = False # if true, is to projet continuously a unique pattern to tune the spectrometer

    if reconstruct == True:
        print('Creating reconstruction processes')

        # Creating a Queue for sending spectral data to reconstruction process
        queue_to_recon = Queue()

        # Creating a Queue for sending reconstructed images to plot
        queue_reconstructed = Queue()

        sleep_time = (acquisition_params.pattern_amount * 
                    DMD_params.picture_time_us/1e+6)

        # Creating reconstruction process
        recon_process = Process(target=reconstruct_process, 
                    args=(reconstruction_params.model,
                        reconstruction_params.device, 
                        queue_to_recon,
                        queue_reconstructed,
                        reconstruction_params.batches, 
                        reconstruction_params.noise,
                        sleep_time))

        # Creating plot process
        plot_process = Process(target=plot_recon, 
                        args=(queue_reconstructed, sleep_time))

        # Starting processes
        recon_process.start()
        plot_process.start()
        
    pixel_amount = (spectrometer_params.stop_pixel - 
                    spectrometer_params.start_pixel + 1)
    measurement_time = np.zeros(
        (acquisition_params.pattern_amount * repetitions))
    timestamps = np.zeros(
        ((acquisition_params.pattern_amount - 1) * repetitions), 
        dtype=np.float64)
    spectral_data = np.zeros(
        (acquisition_params.pattern_amount * repetitions,pixel_amount),
        dtype=np.float64)

    acquisition_params.acquired_spectra = 0
    print()

    for repetition in range(repetitions):
        if verbose:
            print(f"Acquisition {repetition}")

        AcquisitionResults = _acquire_raw(ava, DMD, spectrometer_params, 
            DMD_params, acquisition_params, loop)
    
        (data, spectrum_index, timestamp, time,
            start_measurement_time, saturation_detected) = AcquisitionResults

        print('Acquisition number : ' + str(repetition) + ' finished')

        if reconstruct == True:
            queue_to_recon.put(data.T)
            print('Data sent')

        time, timestamp = _calculate_elapsed_time(
            start_measurement_time, time, timestamp)

        begin = repetition * acquisition_params.pattern_amount
        end = (repetition + 1) * acquisition_params.pattern_amount
        spectral_data[begin:end] = data
        measurement_time[begin:end] = time

        begin = repetition * (acquisition_params.pattern_amount - 1)
        end = (repetition + 1) * (acquisition_params.pattern_amount - 1)
        timestamps[begin:end] = timestamp

        acquisition_params.acquired_spectra += spectrum_index

        acquisition_params.saturation_detected = saturation_detected
        
        if saturation_detected is True:
            print('!!!!!!!!!! Saturation detected in the spectro !!!!!!!!!!')
        # Print data for each repetition
        if (verbose):
            print('Spectra acquired: {}'.format(spectrum_index))
            print('Mean callback acquisition time: {} ms'.format(
               np.mean(time)))
            print('Total callback acquisition time: {} s'.format(
                np.sum(time)/1000))
            print('Mean spectrometer acquisition time: {} ms'.format(
                np.mean(timestamp)))
            print('Total spectrometer acquisition time: {} s'.format(
                np.sum(timestamp)/1000))

            # Print shape of acquisition matrix for one repetition    
            print(f'Partial acquisition matrix dimensions:'
                  f'{data.shape}')
            print()

    acquisition_params.update_timings(timestamps, measurement_time)
    # Real time between each spectrum acquisition by the spectrometer
    print('Complete acquisition done')
    print('Spectra acquired: {}'.format(acquisition_params.acquired_spectra))      
    print('Total acquisition time: {0:.2f} s'.format(acquisition_params.total_spectrometer_acquisition_time_s))
    
    _save_acquisition(metadata, DMD_params, spectrometer_params, 
                        acquisition_params, spectral_data)

    # Joining processes and closing queues
    if reconstruct == True:
        queue_to_recon.put('kill') # Sends a message to stop reconstruction
        recon_process.join()
        queue_to_recon.close()
        plot_process.join()
        queue_reconstructed.close()
        
    maxi = np.amax(spectral_data[0,:])
    print('------------------------------------------------')
    print('maximum in the spectrum = ' + str(maxi))
    print('------------------------------------------------')
    if maxi >= 65535:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!! warning, spectrum saturation !!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return spectral_data

def _acquire_raw_2arms(ava: Avantes, 
            DMD: ALP4,
            camPar: CAM,
            spectrometer_params: SpectrometerParameters, 
            DMD_params: DMDParameters, 
            acquisition_params: AcquisitionParameters,
            metadata,
            repetition,
            repetitions
            ) -> NamedTuple:
    """Raw data acquisition.

    Setups a callback function to receive messages from spectrometer whenever a
    measurement is ready to be read. Reads a measurement via a callback.

    Args:
        ava (Avantes): 
            Connected spectrometer (Avantes object).
        DMD (ALP4): 
            Connected DMD.
        camPar (CAM):
            Metadata object of the IDS monochrome camera 
        spectrometer_params (SpectrometerParameters): 
            Spectrometer metadata object with spectrometer configurations.
        DMD_params (DMDParameters):
            DMD metadata object with DMD configurations.
        acquisition_params (AcquisitionParameters): 
            Acquisition related metadata object.

    Returns:
        NamedTuple: NamedTuple containig spectral data and measurement timings.
            spectral_data (ndarray):
                2D array of `float` of size (pattern_amount x pixel_amount)
                containing measurements received from the spectrometer for each
                pattern of a sequence.
            spectrum_index (int):
                Index of the last acquired spectrum. 
            timestamps (np.ndarray): 
                1D array with `float` type elapsed time between each measurement
                made by the spectrometer based on its internal clock. 
                Units in milliseconds.
            measurement_time (np.ndarray): 
                1D array with `float` type elapsed times between each callback.
                Units in milliseconds.
            start_measurement_time (float):
                Time when acquisition started.
            saturation_detected (bool):
                Boolean incating if saturation was detected during acquisition.
    """
    # def for spectrometer acquisition
    def register_callback(measurement_time, timestamps, 
                          spectral_data, ava):
        
        def measurement_callback(handle, info): # If we want to reconstruct during callback; can use it in here. Add function as parameter. 
            nonlocal spectrum_index
            nonlocal saturation_detected

            measurement_time[spectrum_index] = perf_counter_ns()
            
            if info.contents.value >= 0:                  
                timestamp,spectrum = ava.get_data()
                spectral_data[spectrum_index,:] = (
                    np.ctypeslib.as_array(spectrum[0:pixel_amount]))
                 
                if np.any(ava.get_saturated_pixels() > 0):
                    saturation_detected = True

                timestamps[spectrum_index] = np.ctypeslib.as_array(timestamp)
                
            else: # Set values to zero if an error occured
                spectral_data[spectrum_index,:] = 0
                timestamps[spectrum_index] = 0
            
            spectrum_index += 1
        
        return measurement_callback
    
    # def for camera acquisition 
    if repetition == 0:
        camPar = stopCapt_DeallocMem(camPar)
        camPar.trigger_mode = 'hard'#'soft'#
        imageQueue(camPar)
        camPar = prepareCam(camPar, metadata)
        camPar.timeout = 1000   # time out in ms for the "is_WaitForNextImage" function
        start_chrono = time.time()
        x = threading.Thread(target = runCam_thread, args=(camPar, start_chrono))
        x.start()
    
    pixel_amount = (spectrometer_params.stop_pixel - 
                    spectrometer_params.start_pixel + 1)

    measurement_time = np.zeros((acquisition_params.pattern_amount))
    timestamps = np.zeros((acquisition_params.pattern_amount),dtype=np.uint32)
    spectral_data = np.zeros(
        (acquisition_params.pattern_amount,pixel_amount),dtype=np.float64)

    # Boolean to indicate if saturation was detected during acquisition
    saturation_detected = False 

    spectrum_index = 0 # Accessed as nonlocal variable inside the callback

    #spectro.register_callback(-2,acquisition_params.pattern_amount,pixel_amount)
    callback = register_callback(measurement_time, timestamps, 
                                 spectral_data, ava)
    measurement_callback = MeasureCallback(callback)
    ava.measure_callback(-2, measurement_callback)
    
    # time.sleep(0.5)
    # Run the whole sequence only once    
    DMD.Run(loop=False)
    start_measurement_time = perf_counter_ns()
    #sleep(13)
    
    while(True):
        if(spectrum_index >= acquisition_params.pattern_amount):
            break
        elif((perf_counter_ns() - start_measurement_time) / 1e+6 > 
            (2 * acquisition_params.pattern_amount * 
            DMD_params.picture_time_us / 1e+3)):
            print('Stopping measurement. One of the equipments may be blocked '
            'or disconnected.')
            break
        else:
            time.sleep(acquisition_params.pattern_amount *
            DMD_params.picture_time_us / 1e+6 / 10)

    ava.stop_measure()
    DMD.Halt()
    camPar.Exit = 2
    if repetition == repetitions-1:
        camPar = stopCam(camPar)
    #Yprint('MAIN :// camPar.camActivated = ' + str(camPar.camActivated))
    AcquisitionResult = namedtuple('AcquisitionResult', [
        'spectral_data', 
        'spectrum_index',
        'timestamps',
        'measurement_time',
        'start_measurement_time',
        'saturation_detected'])

    return AcquisitionResult(spectral_data, 
                             spectrum_index,
                             timestamps,
                             measurement_time,
                             start_measurement_time,
                             saturation_detected)


def acquire_2arms(ava: Avantes, 
            DMD: ALP4,
            camPar: CAM,
            metadata: MetaData, 
            spectrometer_params: SpectrometerParameters, 
            DMD_params: DMDParameters, 
            acquisition_params: AcquisitionParameters,
            repetitions: int = 1,
            verbose: bool = False,
            reconstruct: bool = False,
            reconstruction_params: ReconstructionParameters = None
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a complete acquisition.

    Performs single or multiple acquisitions using the same setup configurations
    previously chosen.
    Finnaly saves all acqusition related data and metadata.

    Args:
        ava (Avantes): 
            Connected spectrometer (Avantes object).
        DMD (ALP4): 
            Connected DMD.
        camPar (CAM):
            Metadata object of the IDS monochrome camera 
        metadata (MetaData): 
            Metadata concerning the experiment, paths, file inputs and file 
            outputs. Must be created and filled up by the user.
        spectrometer_params (SpectrometerParameters): 
            Spectrometer metadata object with spectrometer configurations.
        DMD_params (DMDParameters):
            DMD metadata object with DMD configurations.
        acquisition_params (AcquisitionParameters): 
            Acquisition related metadata object.
        wavelengths (List[float]): 
            List of float corresponding to the wavelengths associated with
            spectrometer's start and stop pixels.
        repetitions (int):
            Number of times the acquisition will be repeated with the same
            configurations. Default is 1, a single acquisition.
        verbose (bool):
            Chooses if data concerning each acquisition should be printed to
            user. If False, only overall data regarding all repetitions is 
            printed. Default is False.
        reconstruct (bool): 
            If True, will perform reconstruction alongside acquisition using
            multiprocessing.
        reconstruction_params (ReconstructionParameters):
            Object containing parameters of the neural network to be loaded for
            reconstruction.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: Tuple containig spectral data and
        measurement timings.
            spectral_data (ndarray):
                2D array of `float` of size (pattern_amount x pixel_amount)
                containing measurements received from the spectrometer for each
                pattern of a sequence. 
            timestamps (np.ndarray): 
                1D array with `float` type elapsed time between each measurement
                made by the spectrometer based on its internal clock. 
                Units in milliseconds.
            measurement_time (np.ndarray): 
                1D array with `float` type elapsed times between each callback.
                Units in milliseconds.
    """

    if reconstruct == True:
        print('Creating reconstruction processes')

        # Creating a Queue for sending spectral data to reconstruction process
        queue_to_recon = Queue()

        # Creating a Queue for sending reconstructed images to plot
        queue_reconstructed = Queue()

        sleep_time = (acquisition_params.pattern_amount * 
                    DMD_params.picture_time_us/1e+6)

        # Creating reconstruction process
        recon_process = Process(target=reconstruct_process, 
                    args=(reconstruction_params.model,
                        reconstruction_params.device, 
                        queue_to_recon,
                        queue_reconstructed,
                        reconstruction_params.batches, 
                        reconstruction_params.noise,
                        sleep_time))

        # Creating plot process
        plot_process = Process(target=plot_recon, 
                        args=(queue_reconstructed, sleep_time))

        # Starting processes
        recon_process.start()
        plot_process.start()
        
    pixel_amount = (spectrometer_params.stop_pixel - 
                    spectrometer_params.start_pixel + 1)
    measurement_time = np.zeros(
        (acquisition_params.pattern_amount * repetitions))
    timestamps = np.zeros(
        ((acquisition_params.pattern_amount - 1) * repetitions), 
        dtype=np.float64)
    spectral_data = np.zeros(
        (acquisition_params.pattern_amount * repetitions,pixel_amount),
        dtype=np.float64)

    acquisition_params.acquired_spectra = 0
    print()

    for repetition in range(repetitions):
        if verbose:
            print(f"Acquisition {repetition}")

        AcquisitionResults = _acquire_raw_2arms(ava, DMD, camPar, spectrometer_params, 
            DMD_params, acquisition_params, metadata, repetition, repetitions)
    
        (data, spectrum_index, timestamp, time,
            start_measurement_time, saturation_detected) = AcquisitionResults

        print('Acquisition number : ' + str(repetition) + ' finished')

        if reconstruct == True:
            queue_to_recon.put(data.T)
            print('Data sent')

        time, timestamp = _calculate_elapsed_time(
            start_measurement_time, time, timestamp)

        begin = repetition * acquisition_params.pattern_amount
        end = (repetition + 1) * acquisition_params.pattern_amount
        spectral_data[begin:end] = data
        measurement_time[begin:end] = time

        begin = repetition * (acquisition_params.pattern_amount - 1)
        end = (repetition + 1) * (acquisition_params.pattern_amount - 1)
        timestamps[begin:end] = timestamp

        acquisition_params.acquired_spectra += spectrum_index

        acquisition_params.saturation_detected = saturation_detected
    
        if saturation_detected is True:
            print('!!!!!!!!!! Saturation detected in the spectro !!!!!!!!!!')
        # Print data for each repetition
        if (verbose):
            print('Spectra acquired: {}'.format(spectrum_index))
            print('Mean callback acquisition time: {} ms'.format(
               np.mean(time)))
            print('Total callback acquisition time: {} s'.format(
                np.sum(time)/1000))
            print('Mean spectrometer acquisition time: {} ms'.format(
                np.mean(timestamp)))
            print('Total spectrometer acquisition time: {} s'.format(
                np.sum(timestamp)/1000))
    
            # Print shape of acquisition matrix for one repetition    
            print(f'Partial acquisition matrix dimensions:'
                  f'{data.shape}')
            print()
    
    acquisition_params.update_timings(timestamps, measurement_time)
    # Real time between each spectrum acquisition by the spectrometer
    print('Complete acquisition done')
    print('Spectra acquired: {}'.format(acquisition_params.acquired_spectra))      
    print('Total acquisition time: {0:.2f} s'.format(acquisition_params.total_spectrometer_acquisition_time_s))
    
    # delete acquisition with black pattern (white for the camera)
    if camPar.insert_patterns == 1:
        black_pattern_index = np.where(acquisition_params.patterns_wp == -1)
        # print('index of white patterns :')
        # print(black_pattern_index[0:38])
        if acquisition_params.patterns_wp.shape == acquisition_params.patterns.shape:
            acquisition_params.patterns = np.delete(acquisition_params.patterns, black_pattern_index)
        spectral_data = np.delete(spectral_data, black_pattern_index, axis = 0)
        acquisition_params.timestamps = np.delete(acquisition_params.timestamps, black_pattern_index[1:])
        acquisition_params.measurement_time = np.delete(acquisition_params.measurement_time, black_pattern_index)
        acquisition_params.acquired_spectra = len(acquisition_params.patterns)
    
    _save_acquisition_2arms(metadata, DMD_params, spectrometer_params, camPar, 
                        acquisition_params, spectral_data)

    # Joining processes and closing queues
    if reconstruct == True:
        queue_to_recon.put('kill') # Sends a message to stop reconstruction
        recon_process.join()
        queue_to_recon.close()
        plot_process.join()
        queue_reconstructed.close()
        
    maxi = np.amax(spectral_data[0,:])
    print('------------------------------------------------')
    print('maximum in the spectrum = ' + str(maxi))
    print('------------------------------------------------')
    if maxi >= 65535:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!! warning, spectrum saturation !!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return spectral_data


def setup_tuneSpectro(spectrometer, 
                      DMD, 
                      DMD_initial_memory,
                      pattern_to_display, 
                      ti : float = 1, 
                      zoom : int = 1,
                      xw_offset: int = 128,
                      yh_offset: int = 0,
                      mask_index : np.array = []):
    """ Setup the hadrware to tune the spectrometer in live. The goal is to find 
    the integration time of the spectrometer, noise is around 700 counts, 
    saturation is equal to 2**16=65535
    
    Args:
        spectrometer (Avantes):
            Connected spectrometer (Avantes object).
        DMD (ALP4):
            Connected DMD.
        DMD_initial_memory (int):
            Initial memory available in DMD after initialization.
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file 
            outputs. Must be created and filled up by the user.
        acquisition_params (AcquisitionParameters):
            Acquisition related metadata object. User must partially fill up
            with pattern_compression, pattern_dimension_x, pattern_dimension_y.
        pattern_to_display (string):
            display one pattern on the DMD to tune the spectrometer. Default is 
            white pattern
        ti (float):
            The integration time of the spectrometer during one scan in miliseconds. 
            Default is 1 ms.
        zoom (int):
            digital zoom on the DMD. Default is 1
        xw_offset (int):
            offset of the pattern in the DMD for zoom > 1 in the width (x) direction
        yh_offset (int):
            offset of the pattern in the DMD for zoom > 1 in the heihgt (y) direction   
        mask_index (Union[np.ndarray, str], optional):
            Array of `int` type corresponding to the index of the mask vector where
            the value is egal to 1
            
    return:
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file 
            outputs. Must be created and filled up by the user.
        spectrometer_params (SpectrometerParameters): 
            Spectrometer metadata object with spectrometer configurations.
        DMD_params (DMDParameters):
            DMD metadata object with DMD configurations.
    """
    
    data_folder_name = 'Tune'
    data_name = 'test'
    # all_path = func_path(data_folder_name, data_name)

    scan_mode   = 'Walsh'
    Np          = 16
    source      = ''
    object_name = ''

    metadata = MetaData(
        output_directory     = '',#all_path.subfolder_path,
        pattern_order_source = 'C:/openspyrit/spas/stats/pattern_order_' + scan_mode + '_' + str(Np) + 'x' + str(Np) + '.npz',
        pattern_source       = 'C:/openspyrit/spas/Patterns/' + scan_mode + '_' + str(Np) + 'x' + str(Np),
        pattern_prefix       = scan_mode + '_' + str(Np) + 'x' + str(Np),
        experiment_name      = data_name,
        light_source         = source,
        object               = object_name,
        filter               = '', 
        description          = ''
                        )
        
    acquisition_parameters = AcquisitionParameters(
        pattern_compression = 1,
        pattern_dimension_x = 16,
        pattern_dimension_y = 16,
        zoom                = zoom,
        xw_offset           = xw_offset,
        yh_offset           = yh_offset,
        mask_index          = []            )
    
    acquisition_parameters.pattern_amount = 1
        
    spectrometer_params, DMD_params = setup(
        spectrometer       = spectrometer, 
        DMD                = DMD,
        DMD_initial_memory = DMD_initial_memory,
        metadata           = metadata, 
        acquisition_params = acquisition_parameters,
        pattern_to_display = pattern_to_display,
        integration_time   = ti,           
        loop = True                         )
    
    return metadata, spectrometer_params, DMD_params, acquisition_parameters


def displaySpectro(ava: Avantes, 
            DMD: ALP4,
            metadata: MetaData, 
            spectrometer_params: SpectrometerParameters, 
            DMD_params: DMDParameters, 
            acquisition_params: AcquisitionParameters,
            reconstruction_params: ReconstructionParameters = None
            ):
    """Perform a continousely acquisition on the spectrometer for optical tuning.

    Send a pattern on the DMD to project light on the spectrometer. The goal is 
    to have a look on the amplitude of the spectrum to tune the illumination to
    avoid saturation (sat >= 65535) and noisy signal (amp <= 650).

    Args:
        ava (Avantes): 
            Connected spectrometer (Avantes object).
        DMD (ALP4): 
            Connected DMD.
        metadata (MetaData): 
            Metadata concerning the experiment, paths, file inputs and file 
            outputs. Must be created and filled up by the user.
        spectrometer_params (SpectrometerParameters): 
            Spectrometer metadata object with spectrometer configurations.
        DMD_params (DMDParameters):
            DMD metadata object with DMD configurations.
        acquisition_params (AcquisitionParameters): 
            Acquisition related metadata object.
        wavelengths (List[float]): 
            List of float corresponding to the wavelengths associated with
            spectrometer's start and stop pixels.
        reconstruction_params (ReconstructionParameters):
            Object containing parameters of the neural network to be loaded for
            reconstruction.
    """
    
    loop = True # is to project continuously a unique pattern to tune the spectrometer
    
    pixel_amount = (spectrometer_params.stop_pixel - 
                    spectrometer_params.start_pixel + 1)

    spectral_data = np.zeros(
        (acquisition_params.pattern_amount,pixel_amount),
        dtype=np.float64)

    acquisition_params.acquired_spectra = 0

    AcquisitionResults = _acquire_raw(ava, DMD, spectrometer_params, 
        DMD_params, acquisition_params, loop)

    (data, spectrum_index, timestamp, time,
        start_measurement_time, saturation_detected) = AcquisitionResults

    time, timestamp = _calculate_elapsed_time(
        start_measurement_time, time, timestamp)

    begin = acquisition_params.pattern_amount
    end = 2 * acquisition_params.pattern_amount
    spectral_data[begin:end] = data

    acquisition_params.acquired_spectra += spectrum_index

    acquisition_params.saturation_detected = saturation_detected


def check_ueye(func, *args, exp=0, raise_exc=True, txt=None):
    """Check for bad input value
    
    Args:
    ----------
    func : TYPE
        the ueye function.
    *args : TYPE
        the input value.
    exp : TYPE, optional
        DESCRIPTION. The default is 0.
    raise_exc : TYPE, optional
        DESCRIPTION. The default is True.
    txt : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    None.
    """
    
    ret = func(*args)
    if not txt:
        txt = "{}: Expected {} but ret={}!".format(str(func), exp, ret)
    if ret != exp:
        if raise_exc:
            raise RuntimeError(txt)
        else:
            logging.critical(txt)


def stopCapt_DeallocMem(camPar):
    """Stop capture and deallocate camera memory if need to change AOI
    
    Args:
    ----------
    camPar (CAM):
        Metadata object of the IDS monochrome camera 

    Returns:
    -------
    camPar (CAM):
        Metadata object of the IDS monochrome camera 
    """

    if camPar.camActivated == 1:        
        nRet = ueye.is_StopLiveVideo(camPar.hCam, ueye.IS_FORCE_VIDEO_STOP)
        if nRet == ueye.IS_SUCCESS:
            camPar.camActivated = 0
            print('video stop successful')
        else:
            print('problem to stop the video')
            
    if camPar.Memory == 1:        
        nRet = ueye.is_FreeImageMem(camPar.hCam, camPar.pcImageMemory, camPar.MemID)
        if nRet == ueye.IS_SUCCESS:
            camPar.Memory = 0
            print('deallocate memory successful')
        else:
            print('Problem to deallocate memory of the camera')
                
    return camPar


def stopCapt_DeallocMem_ExitCam(camPar):
    """Stop capture, deallocate camera memory if need to change AOI and disconnect the camera
    
    Args:
    ----------
    camPar (CAM):
        Metadata object of the IDS monochrome camera 

    Returns:
    -------
    camPar (CAM):
        Metadata object of the IDS monochrome camera 
    """
    if camPar.camActivated == 1:        
        nRet = ueye.is_StopLiveVideo(camPar.hCam, ueye.IS_FORCE_VIDEO_STOP)
        if nRet == ueye.IS_SUCCESS:
            camPar.camActivated = 0
            print('video stop successful')
        else:
            print('problem to stop the video')
            
    if camPar.Memory == 1:        
        nRet = ueye.is_FreeImageMem(camPar.hCam, camPar.pcImageMemory, camPar.MemID)
        if nRet == ueye.IS_SUCCESS:
            camPar.Memory = 0
            print('deallocate memory successful')
        else:
            print('Problem to deallocate memory of the camera')
    
    if camPar.Exit == 2:
        nRet = ueye.is_ExitCamera(camPar.hCam)
        if nRet == ueye.IS_SUCCESS:
            camPar.Exit = 0
            print('Camera disconnected')
        else:
            print('Problem to disconnect camera, need to restart spyder')
                
    return camPar


class ImageBuffer:
    """A class to allocate buffer in the camera memory
    """
    
    pcImageMemory = None
    MemID = None
    width = None
    height = None
    nbitsPerPixel = None


def imageQueue(camPar):
    """Create Imagequeue / Allocate 3 ore more buffers depending on the framerate / Initialize Image queue
    
    Args:
    ----------
    camPar (CAM):
        Metadata object of the IDS monochrome camera 

    Returns:
    -------
    None.

    """

    sleep(1)   # is required (delay of 1s was not optimized!!)
    buffers = []
    for y in range(10):
        buffers.append(ImageBuffer())

    for x in range(len(buffers)):
        buffers[x].nbitsPerPixel = camPar.nBitsPerPixel  # RAW8
        buffers[x].height = camPar.rectAOI.s32Height  # sensorinfo.nMaxHeight
        buffers[x].width = camPar.rectAOI.s32Width  # sensorinfo.nMaxWidth
        buffers[x].MemID = ueye.int(0)
        buffers[x].pcImageMemory = ueye.c_mem_p()
        check_ueye(ueye.is_AllocImageMem, camPar.hCam, buffers[x].width, buffers[x].height, buffers[x].nbitsPerPixel,
                   buffers[x].pcImageMemory, buffers[x].MemID)
        check_ueye(ueye.is_AddToSequence, camPar.hCam, buffers[x].pcImageMemory, buffers[x].MemID)

    check_ueye(ueye.is_InitImageQueue, camPar.hCam, ueye.c_int(0))
    if camPar.trigger_mode == 'soft':
        check_ueye(ueye.is_SetExternalTrigger, camPar.hCam, ueye.IS_SET_TRIGGER_SOFTWARE)
    elif camPar.trigger_mode == 'hard':
        check_ueye(ueye.is_SetExternalTrigger, camPar.hCam, ueye.IS_SET_TRIGGER_LO_HI)


def prepareCam(camPar, metadata):
    """Prepare the IDS monochrome camera before acquisition

    Args:
    ----------
    camPar (CAM):
        Metadata object of the IDS monochrome camera
    metadata (MetaData):
        Metadata concerning the experiment, paths, file inputs and file 
        outputs. Must be created and filled up by the user.

    Returns:
    -------
    camPar (CAM):
        Metadata object of the IDS monochrome camera

    """
    cam_path = metadata.output_directory + '\\' + metadata.experiment_name + '_video.' + camPar.vidFormat
    strFileName = ueye.c_char_p(cam_path.encode('utf-8'))
    
    if camPar.vidFormat == 'avi':         
        # print('Video format : AVI')
        camPar.avi = ueye.int()
        nRet = ueye_tools.isavi_InitAVI(camPar.avi, camPar.hCam)
        # print("isavi_InitAVI")
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("isavi_InitAVI ERROR")
        
        nRet = ueye_tools.isavi_SetImageSize(camPar.avi, camPar.m_nColorMode,  camPar.rectAOI.s32Width , camPar.rectAOI.s32Height, 0, 0, 0)
        nRet = ueye_tools.isavi_SetImageQuality(camPar.avi, 100)
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("isavi_SetImageQuality ERROR")
 
        nRet = ueye_tools.isavi_OpenAVI(camPar.avi, strFileName)
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("isavi_OpenAVI ERROR")
            print('Error code = ' + str(nRet))
            print('Certainly, it is a problem with the file name, Avoid special character like "µ" or try to redcue its size')
        
        nRet = ueye_tools.isavi_SetFrameRate(camPar.avi, camPar.fps)
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("isavi_SetFrameRate ERROR")
        nRet = ueye_tools.isavi_StartAVI(camPar.avi)
        # print("isavi_StartAVI")
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("isavi_StartAVI ERROR")

            
    elif camPar.vidFormat == 'bin':
        camPar.punFileID = ueye.c_uint()
        nRet = ueye_tools.israw_InitFile(camPar.punFileID, ueye_tools.IS_FILE_ACCESS_MODE_WRITE)
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("INIT RAW FILE ERROR")
            
        nRet = ueye_tools.israw_SetImageInfo(camPar.punFileID, camPar.rectAOI.s32Width, camPar.rectAOI.s32Height, camPar.nBitsPerPixel)
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("SET IMAGE INFO ERROR")
            
        if nRet == ueye.IS_SUCCESS: 
            # print('initFile ok')
            # print('SetImageInfo ok')
            nRet = ueye_tools.israw_OpenFile(camPar.punFileID, strFileName)
            # if nRet == ueye.IS_SUCCESS:
            #     # print('OpenFile success')
            
    # ---------------------------------------------------------
    # Activates the camera's live video mode (free run mode)
    # ---------------------------------------------------------
    nRet = ueye.is_CaptureVideo(camPar.hCam, ueye.IS_DONT_WAIT)

    if nRet != ueye.IS_SUCCESS:
        print("is_CaptureVideo ERROR")
    else:
        camPar.camActivated = 1
    
    return camPar
    
        
def runCam_thread(camPar, start_chrono): 
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
    
    imageinfo = ueye.UEYEIMAGEINFO()
    current_buffer = ueye.c_mem_p()
    current_id = ueye.int()
    # inc = 0
    entier_old = 0
    # time.sleep(0.01)
    while True: 
        nret = ueye.is_WaitForNextImage(camPar.hCam, camPar.timeout, current_buffer, current_id)
        if nret == ueye.IS_SUCCESS:
            check_ueye(ueye.is_GetImageInfo, camPar.hCam, current_id, imageinfo, ueye.sizeof(imageinfo))
            start_time = time.time()
            counter = start_time - start_chrono
            camPar.time_array.append(counter)
            if camPar.vidFormat == 'avi':
                nRet = ueye_tools.isavi_AddFrame(camPar.avi, current_buffer)  
            elif camPar.vidFormat == 'bin':   
                nRet = ueye_tools.israw_AddFrame(camPar.punFileID, current_buffer, imageinfo.u64TimestampDevice)
                                
            check_ueye(ueye.is_UnlockSeqBuf, camPar.hCam, current_id, current_buffer)
        else:
            print('Thread finished')
            break


def stopCam(camPar):
    """To stop the acquisition of the video

    Parameters
    ----------
    camPar (CAM):
        Metadata object of the IDS monochrome camera

    Returns
    -------
    camPar (CAM):
        Metadata object of the IDS monochrome camera
    """
    
    if camPar.vidFormat == 'avi':
        ueye_tools.isavi_StopAVI(camPar.hCam)
        ueye_tools.isavi_CloseAVI(camPar.hCam)
        ueye_tools.isavi_ExitAVI(camPar.hCam)
    elif camPar.vidFormat == 'bin':   
        ueye_tools.israw_CloseFile(camPar.punFileID)
        ueye_tools.israw_ExitFile(camPar.punFileID)
        camPar.punFileID = ueye.c_uint()

    return camPar


def disconnect(ava: Optional[Avantes]=None, 
               DMD: Optional[ALP4]=None):
    """Disconnect spectrometer and DMD.

    Disconnects equipments trying to stop a running pattern sequence (possibly 
    blocking correct functioning) and trying to free DMD memory to avoid errors
    in later acqusitions. 

    Args:
        ava (Avantes, optional): 
            Connected spectrometer (Avantes object). Defaults to None.
        DMD (ALP4, optional): 
            Connected DMD. Defaults to None.
    """

    if ava is not None:
        ava.disconnect()
        print('Spectro disconnected')
        
    if DMD is not None:
       
        # Stop the sequence display
        DMD.Halt()

        # Free the sequence from the onboard memory (if any is present)
        if (DMD.Seqs):
            DMD.FreeSeq()

        DMD.Free()
        print('DMD disconnected')


def disconnect_2arms(ava: Optional[Avantes]=None, 
                     DMD: Optional[ALP4]=None, 
                     camPar=None):
    """Disconnect spectrometer, DMD and the IDS monochrome camera.

    Disconnects equipments trying to stop a running pattern sequence (possibly 
    blocking correct functioning) and trying to free DMD memory to avoid errors
    in later acqusitions. 

    Args:
        ava (Avantes, optional): 
            Connected spectrometer (Avantes object). Defaults to None.
        DMD (ALP4, optional): 
            Connected DMD. Defaults to None.
       camPar (CAM):
           Metadata object of the IDS monochrome camera 
    """

    if ava is not None:
        ava.disconnect()
        print('Spectro disconnected')

    if DMD is not None:       
        # Stop the sequence display
        try:
            DMD.Halt()
            # Free the sequence from the onboard memory (if any is present)
            if (DMD.Seqs):
                DMD.FreeSeq()

            DMD.Free()  
            print('DMD disconnected')
            
        except:
            print('probelm to Halt the DMD')    

                
    if camPar.camActivated == 1:
        nRet = ueye.is_StopLiveVideo(camPar.hCam, ueye.IS_FORCE_VIDEO_STOP)
        if nRet == ueye.IS_SUCCESS:
           camPar.camActivated = 0 
        else:
            print('Problem to stop video, need to restart spyder')    
    
    if camPar.Memory == 1:
        nRet = ueye.is_FreeImageMem(camPar.hCam, camPar.pcImageMemory, camPar.MemID)
        if nRet == ueye.IS_SUCCESS:
            camPar.Memory = 0
        else:
            print('Problem to deallocate camera memory, need to restart spyder')
      
                
    if camPar.Exit == 1 or camPar.Exit == 2:
        nRet = ueye.is_ExitCamera(camPar.hCam)
        if nRet == ueye.IS_SUCCESS:
            camPar.Exit = 0
            print('Camera disconnected')
        else:
            print('Problem to disconnect camera, need to restart spyder')


def _init_CAM():
    """
    Initialize and connect to the IDS camera.
    
    Returns:
        CAM: a structure containing the parameters of the IDS camera
    """    
    camPar = CAM(hCam = ueye.HIDS(0),
                 sInfo = ueye.SENSORINFO(),
                 cInfo = ueye.CAMINFO(),
                 nBitsPerPixel = ueye.INT(8),
                 m_nColorMode = ueye.INT(),
                 bytes_per_pixel = int( ueye.INT(8)/ 8),
                 rectAOI = ueye.IS_RECT(),
                 pcImageMemory = ueye.c_mem_p(),
                 MemID = ueye.int(),
                 pitch = ueye.INT(),
                 fps = float(), 
                 gain = int(), 
                 gainBoost = str(),
                 gamma = float(), 
                 exposureTime = float(), 
                 blackLevel = int(),
                 camActivated = bool(),
                 pixelClock = ueye.uint(),
                 bandwidth = float(),
                 Memory = bool(),
                 Exit = int(),
                 vidFormat = str(),
                 gate_period = int(),
                 trigger_mode = str(),
                 avi = ueye.int(),
                 punFileID = ueye.c_uint(),
                 timeout = int(),
                 time_array = [],
                 int_time_spect = float(),
                 black_pattern_num = int(),
                 insert_patterns = bool(),
                 acq_mode = str(),
                 )          

    # # Camera Initialization ---
    ### Starts the driver and establishes the connection to the camera
    nRet = ueye.is_InitCamera(camPar.hCam, None)
    if nRet != ueye.IS_SUCCESS:
        print("is_InitCamera ERROR")

    ### Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
    nRet = ueye.is_GetCameraInfo(camPar.hCam, camPar.cInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetCameraInfo ERROR")

    ### You can query additional information about the sensor type used in the camera
    nRet = ueye.is_GetSensorInfo(camPar.hCam, camPar.sInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetSensorInfo ERROR")

    ### set camera parameters to default values
    nRet = ueye.is_ResetToDefault(camPar.hCam)
    if nRet != ueye.IS_SUCCESS:
        print("is_ResetToDefault ERROR")

    ### Set display mode to DIB
    nRet = ueye.is_SetDisplayMode(camPar.hCam, ueye.IS_SET_DM_DIB)

    ### Set the right color mode
    if int.from_bytes(camPar.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
        # setup the color depth to the current windows setting
        ueye.is_GetColorDepth(camPar.hCam, camPar.nBitsPerPixel, camPar.m_nColorMode)
        camPar.bytes_per_pixel = int(camPar.nBitsPerPixel / 8)
    elif int.from_bytes(camPar.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
        # for color camera models use RGB32 mode
        camPar.m_nColorMode = ueye.IS_CM_BGRA8_PACKED
        camPar.nBitsPerPixel = ueye.INT(32)
        camPar.bytes_per_pixel = int(camPar.nBitsPerPixel / 8)
    elif int.from_bytes(camPar.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
        # for color camera models use RGB32 mode
        camPar.m_nColorMode = ueye.IS_CM_MONO8
        camPar.nBitsPerPixel = ueye.INT(8)
        camPar.bytes_per_pixel = int(camPar.nBitsPerPixel / 8)
    else:
        # for monochrome camera models use Y8 mode
        camPar.m_nColorMode = ueye.IS_CM_MONO8
        camPar.nBitsPerPixel = ueye.INT(8)
        camPar.bytes_per_pixel = int(camPar.nBitsPerPixel / 8)
        # print("else")
    
    ### Get the AOI (Area Of Interest)
    sizeofrectAOI = ueye.c_uint(4*4)
    nRet = ueye.is_AOI(camPar.hCam, ueye.IS_AOI_IMAGE_GET_AOI, camPar.rectAOI, sizeofrectAOI)
    if nRet != ueye.IS_SUCCESS:
        print("AOI getting ERROR")
    
    camPar.camActivated = 0
    
    # Get current pixel clock
    getpixelclock = ueye.UINT(0)
    check_ueye(ueye.is_PixelClock, camPar.hCam, ueye.PIXELCLOCK_CMD.IS_PIXELCLOCK_CMD_GET, getpixelclock,
               ueye.sizeof(getpixelclock))
    
    camPar.pixelClock = getpixelclock
    # print('pixel clock = ' + str(getpixelclock) + ' MHz')
    
    # get the bandwidth (in MByte/s)
    camPar.bandwidth = ueye.is_GetUsedBandwidth(camPar.hCam)
    
    camPar.Exit = 1
    
    print('IDS camera connected')
    
    return camPar
    
    
def captureVid(camPar):
    """
    Allocate memory and begin video capture of the IDS camera

    Args:
        camPar : a structure containing the parameters of the IDS camera.

    Returns:
        camPar : a structure containing the parameters of the IDS camera.
    """
    camPar = stopCapt_DeallocMem_ExitCam(camPar)
    
    if camPar.Exit == 0:
        camPar = _init_CAM()
        camPar.Exit = 1
        
    
    ### Set the AOI
    sizeofrectAOI = ueye.c_uint(4*4)
    nRet = ueye.is_AOI(camPar.hCam, ueye.IS_AOI_IMAGE_SET_AOI, camPar.rectAOI, sizeofrectAOI)
    if nRet != ueye.IS_SUCCESS:
        print("AOI setting ERROR")

    width = camPar.rectAOI.s32Width
    height = camPar.rectAOI.s32Height

    ### Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
    nRet = ueye.is_AllocImageMem(camPar.hCam, width, height, camPar.nBitsPerPixel, camPar.pcImageMemory, camPar.MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_AllocImageMem ERROR")
    else:
        # Makes the specified image memory the active memory
        camPar.Memory = 1
        nRet = ueye.is_SetImageMem(camPar.hCam, camPar.pcImageMemory, camPar.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetImageMem ERROR")
        else:
            # Set the desired color mode
            nRet = ueye.is_SetColorMode(camPar.hCam, camPar.m_nColorMode)
    

    ### Activates the camera's live video mode (free run mode)
    nRet = ueye.is_CaptureVideo(camPar.hCam, ueye.IS_DONT_WAIT)
    if nRet != ueye.IS_SUCCESS:
        print("is_CaptureVideo ERROR")
        
        
    ### Enables the queue mode for existing image memory sequences
    nRet = ueye.is_InquireImageMem(camPar.hCam, camPar.pcImageMemory, camPar.MemID, width, height, camPar.nBitsPerPixel, camPar.pitch)
    if nRet != ueye.IS_SUCCESS:
        print("is_InquireImageMem ERROR")    

    camPar.camActivated = 1
    
    return camPar
 
def setup_cam(camPar, pixelClock, fps, Gain, gain_boost, nGamma, ExposureTime, black_level):
    """
    Set and read the camera parameters
    
    Args:
        pixelClock = [118, 237 or 474] (MHz)
        fps: fps boundary => [1 - No Value] sup limit depend of image size (216 fps for 768x544 pixels for example)
        Gain: Gain boundary => [0 100]
        gain_boost: 'ON' set "ON" to activate gain boost, "OFF" to deactivate
        nGamma: Gamma boundary => [1 - 2.2]
        ExposureTime: Exposure time (ms) boundarye => [0.032 - 56.221] 
        black_level: Black Level boundary => [0 255] 
    
    returns:
        CAM: a structure containing the parameters of the IDS camera
    """
    # It is necessary to execute twice this code to take account the parameter modification
    for i in range(2): 
        ############################### Set Pixel Clock ###############################
        ### Get range of pixel clock, result : range = [118 474] MHz (Inc = 0)
        getpixelclock = ueye.UINT(0)
        newpixelclock = ueye.UINT(0)
        newpixelclock.value = pixelClock
        PixelClockRange = (ueye.int * 3)()
    
        # Get pixel clock range
        nRet = ueye.is_PixelClock(camPar.hCam, ueye.IS_PIXELCLOCK_CMD_GET_RANGE, PixelClockRange, ueye.sizeof(PixelClockRange))
        if nRet == ueye.IS_SUCCESS:
            nPixelClockMin = PixelClockRange[0]
            nPixelClockMax = PixelClockRange[1]
            nPixelClockInc = PixelClockRange[2]
    
        # Set pixel clock
        check_ueye(ueye.is_PixelClock, camPar.hCam, ueye.PIXELCLOCK_CMD.IS_PIXELCLOCK_CMD_SET, newpixelclock,
                   ueye.sizeof(newpixelclock))
        # Get current pixel clock
        check_ueye(ueye.is_PixelClock, camPar.hCam, ueye.PIXELCLOCK_CMD.IS_PIXELCLOCK_CMD_GET, getpixelclock,
                   ueye.sizeof(getpixelclock))
        
        camPar.pixelClock = getpixelclock.value
        if i == 1:
            print('            pixel clock = ' + str(getpixelclock) + ' MHz')
        if getpixelclock == 118:
            if i == 1:
                print('Pixel clcok blocked to 118 MHz, it is necessary to unplug the camera if not desired')
        # get the bandwidth (in MByte/s)
        camPar.bandwidth = ueye.is_GetUsedBandwidth(camPar.hCam)
        if i == 1:
            print('              Bandwidth = ' + str(camPar.bandwidth) + ' MB/s')
        ############################### Set FrameRate #################################
        ### Read current FrameRate
        dblFPS_init = ueye.c_double()
        nRet = ueye.is_GetFramesPerSecond(camPar.hCam, dblFPS_init)
        if nRet != ueye.IS_SUCCESS:
            print("FrameRate getting ERROR")
        else:
            dblFPS_eff = dblFPS_init
            if i == 1:
                print('            current FPS = '+str(round(dblFPS_init.value*100)/100) + ' fps')
            if fps < 1:
                fps = 1
                if i == 1:
                    print('FPS exceed lower limit >= 1')
                
            dblFPS = ueye.c_double(fps)  
            if (dblFPS.value < dblFPS_init.value-0.01) | (dblFPS.value > dblFPS_init.value+0.01):
                newFPS = ueye.c_double()
                nRet = ueye.is_SetFrameRate(camPar.hCam, dblFPS, newFPS)
                time.sleep(1)
                if nRet != ueye.IS_SUCCESS:
                    print("FrameRate setting ERROR")
                else:
                    if i == 1:
                        print('                new FPS = '+str(round(newFPS.value*100)/100) + ' fps')
                    ### Read again the effective FPS / depend of the image size, 17.7 fps is not possible with the entire image size (ie 2076x3088)
                    dblFPS_eff = ueye.c_double() 
                    nRet = ueye.is_GetFramesPerSecond(camPar.hCam, dblFPS_eff)
                    if nRet != ueye.IS_SUCCESS:
                        print("FrameRate getting ERROR")
                    else:       
                        if i == 1:
                            print('          effective FPS = '+str(round(dblFPS_eff.value*100)/100) + ' fps')
        ############################### Set GAIN ######################################
        #### Maximum gain is depending of the sensor. Convertion gain code to gain to limit values from 0 to 100
        # gain_code = gain * slope + b
        gain_max_code = 1450
        gain_min_code = 100
        gain_max = 100
        gain_min = 0
        slope = (gain_max_code-gain_min_code)/(gain_max-gain_min)
        b = gain_min_code
        #### Read gain setting
        current_gain_code = ueye.c_int()
        current_gain_code = ueye.is_SetHWGainFactor(camPar.hCam, ueye.IS_GET_MASTER_GAIN_FACTOR, current_gain_code)
        current_gain = round((current_gain_code-b)/slope) 
        
        if i == 1:
            print('           current GAIN = '+str(current_gain))    
        gain_eff = current_gain
        
        ### Set new gain value
        gain = ueye.c_int(Gain)
        if gain.value != current_gain:
            if gain.value < 0:
                gain = ueye.c_int(0)
                if i == 1:
                    print('Gain exceed lower limit >= 0')
            elif gain.value > 100:
                gain = ueye.c_int(100)
                if i == 1:
                    print('Gain exceed upper limit <= 100')
            gain_code = ueye.c_int(round(slope*gain.value+b))
            
            ueye.is_SetHWGainFactor(camPar.hCam, ueye.IS_SET_MASTER_GAIN_FACTOR, gain_code)
            new_gain = round((gain_code-b)/slope)
            
            if i == 1:
                print('               new GAIN = '+str(new_gain))
            gain_eff = new_gain
        ############################### Set GAIN Boost ################################
        ### Read current state of the gain boost
        current_gain_boost_bool = ueye.is_SetGainBoost(camPar.hCam, ueye.IS_GET_GAINBOOST)
        if nRet != ueye.IS_SUCCESS:
            print("Gain boost ERROR")   
        if current_gain_boost_bool == 0:
            current_gain_boost = 'OFF'
        elif current_gain_boost_bool == 1:
            current_gain_boost = 'ON'
        
        if i == 1:
            print('current Gain boost mode = ' + current_gain_boost)
    
        ### Set the state of the gain boost
        if gain_boost != current_gain_boost: 
            if gain_boost == 'OFF':
                nRet = ueye.is_SetGainBoost (camPar.hCam, ueye.IS_SET_GAINBOOST_OFF)                
                print('         new Gain Boost : OFF')
                
            elif gain_boost == 'ON':
                nRet = ueye.is_SetGainBoost (camPar.hCam, ueye.IS_SET_GAINBOOST_ON)                
                print('         new Gain Boost : ON')
                    
            if nRet != ueye.IS_SUCCESS:
                print("Gain boost setting ERROR")          
        ############################### Set Gamma #####################################
        ### Check boundary of Gamma
        if nGamma > 2.2:
            nGamma = 2.2
            if i == 1:
                print('Gamma exceed upper limit <= 2.2')
        elif nGamma < 1:
            nGamma = 1
            if i == 1:
                print('Gamma exceed lower limit >= 1')
        ### Read current Gamma    
        c_nGamma_init = ueye.c_void_p() 
        sizeOfnGamma = ueye.c_uint(4)    
        nRet = ueye.is_Gamma(camPar.hCam, ueye.IS_GAMMA_CMD_GET, c_nGamma_init, sizeOfnGamma)
        if nRet != ueye.IS_SUCCESS:
            print("Gamma getting ERROR") 
        else:
            if i == 1:
                print('          current Gamma = ' + str(c_nGamma_init.value/100))
        ### Set Gamma
            c_nGamma = ueye.c_void_p(round(nGamma*100)) # need to multiply by 100 [100 - 220]
            if c_nGamma_init.value != c_nGamma.value:
                nRet = ueye.is_Gamma(camPar.hCam, ueye.IS_GAMMA_CMD_SET, c_nGamma, sizeOfnGamma)
                if nRet != ueye.IS_SUCCESS:
                    print("Gamma setting ERROR") 
                else:
                    if i == 1:
                        print('              new Gamma = '+str(c_nGamma.value/100))
        ############################### Set Exposure time #############################
        ### Read current Exposure Time
        getExposure = ueye.c_double()
        sizeOfpParam = ueye.c_uint(8)   
        nRet = ueye.is_Exposure(camPar.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, getExposure, sizeOfpParam)
        if nRet == ueye.IS_SUCCESS:
            getExposure.value = round(getExposure.value*1000)/1000
            
            if i == 1:
                print('  current Exposure Time = ' + str(getExposure.value) + ' ms')
        ### Get minimum Exposure Time
        minExposure = ueye.c_double()
        nRet = ueye.is_Exposure(camPar.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, minExposure, sizeOfpParam)
        ### Get maximum Exposure Time
        maxExposure = ueye.c_double()
        nRet = ueye.is_Exposure(camPar.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, maxExposure, sizeOfpParam)
        ### Get increment Exposure Time
        incExposure = ueye.c_double()
        nRet = ueye.is_Exposure(camPar.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, incExposure, sizeOfpParam)
        ### Set new Exposure Time
        setExposure = ueye.c_double(ExposureTime)
        if setExposure.value > maxExposure.value:
           setExposure.value = maxExposure.value 
           if i == 1:
               print('Exposure Time exceed upper limit <= ' + str(maxExposure.value))
        elif setExposure.value < minExposure.value:
           setExposure.value = minExposure.value
           if i == 1:
               print('Exposure Time exceed lower limit >= ' + str(minExposure.value))
    
        if (setExposure.value < getExposure.value-incExposure.value/2) | (setExposure.value > getExposure.value+incExposure.value/2):   
            nRet = ueye.is_Exposure(camPar.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, setExposure, sizeOfpParam)
            if nRet != ueye.IS_SUCCESS:
                print("Exposure Time ERROR")
            else:
                if i == 1:
                    print('      new Exposure Time = ' + str(round(setExposure.value*1000)/1000) + ' ms')
        ############################### Set Black Level ###############################
        current_black_level_c = ueye.c_uint()      
        sizeOfBlack_level = ueye.c_uint(4)    
        ### Read current Black Level
        nRet = ueye.is_Blacklevel(camPar.hCam, ueye.IS_BLACKLEVEL_CMD_GET_OFFSET, current_black_level_c, sizeOfBlack_level)
        if nRet != ueye.IS_SUCCESS:
            print("Black Level getting ERROR")
        else:
            if i == 1:
                print('    current Black Level = ' + str(current_black_level_c.value))
            
        ### Set Black Level 
        if black_level > 255:
            black_level = 255
            if i == 1:
                print('Black Level exceed upper limit <= 255')
        if black_level < 0:
            black_level = 0
            if i == 1:
                print('Black Level exceed lower limit >= 0')
            
        black_level_c = ueye.c_uint(black_level)
        if black_level != current_black_level_c.value  :            
            nRet = ueye.is_Blacklevel(camPar.hCam, ueye.IS_BLACKLEVEL_CMD_SET_OFFSET, black_level_c, sizeOfBlack_level)
            if nRet != ueye.IS_SUCCESS:
                print("Black Level setting ERROR")
            else:
                if i == 1:
                    print('        new Black Level = ' + str(black_level_c.value))
            
    
    camPar.fps = round(dblFPS_eff.value*100)/100
    camPar.gain = gain_eff
    camPar.gainBoost = gain_boost
    camPar.gamma = c_nGamma.value/100
    camPar.exposureTime = round(setExposure.value*1000)/1000
    camPar.blackLevel = black_level_c.value
    
    return camPar
    

def snapshot(camPar, pathIDSsnapshot, pathIDSsnapshot_overview):
    """
    Snapshot of the IDS camera
    
    Args:
        CAM: a structure containing the parameters of the IDS camera
    """
    array = ueye.get_data(camPar.pcImageMemory, camPar.rectAOI.s32Width, camPar.rectAOI.s32Height, camPar.nBitsPerPixel, camPar.pitch, copy=False)
    
    # ...reshape it in an numpy array...
    frame = np.reshape(array,(camPar.rectAOI.s32Height.value, camPar.rectAOI.s32Width.value))#, camPar.bytes_per_pixel))
    
    with pathIDSsnapshot.open('wb') as f: #('ab') as f: #(pathname, mode='w', encoding='utf-8') as f: #('ab') as f:
        np.save(f,frame)
    
    maxi = np.amax(frame)
    if maxi == 0:
        maxi = 1
    im = Image.fromarray(frame*math.floor(255/maxi))
    im.save(pathIDSsnapshot_overview)
    
    maxi = np.amax(frame)
    # print()
    # print('frame max = ' + str(maxi))
    # print('frame min = ' + str(np.amin(frame)))
    if maxi >= 255:
        print('Saturation detected')
        
    plt.figure
    plt.imshow(frame)#, cmap='gray', vmin=mini, vmax=maxi)  
    plt.colorbar(); 
    plt.show()
    
    