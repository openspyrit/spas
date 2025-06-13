# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""Acquisition utility functions.

    Acquisition module is a generic module that call function in different setup (SPC2D_1arm, SPC2D_2arms, SCP1D and SPIM)
    
"""

import warnings
from time import sleep, perf_counter_ns
from typing import NamedTuple, Tuple, List, Optional, Union
from collections import namedtuple
from pathlib import Path
from multiprocessing import Process, Queue
import shutil    
import math
import os

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
# ##### DLL for the spectrometer Avantes 
# try:
#     from msl.equipment import EquipmentRecord, ConnectionRecord, Backend
#     from msl.equipment.resources.avantes import MeasureCallback, Avantes
# except:
#     pass
    
from tqdm import tqdm
# from spas.metadata_SPC2D import DMDParameters, MetaData, AcquisitionParameters
# from spas.metadata_SPC2D import SpectrometerParameters, save_metadata, CAM, save_metadata_2arms
from spas.reconstruction_nn import reconstruct_process, plot_recon, ReconstructionParameters
#To be remove later
from spas.metadata_SPC2D import MetaData

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
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

# from spas.DMD_module import  DMDParameters
# import spas.DMD_module as DMD_mod
# init_DMD, calculate_timings, setup_DMD, setup_patterns, setup_timings, _sequence_limits, _update_sequence, disconnect_DMD,

@dataclass_json
@dataclass
class AcquisitionParameters:
    """Class containing acquisition specifications and timing results.

    This class is adapted to be reconstructed from a JSON file.

    Attributes:
        pattern_compression (float):
            Percentage of total available patterns to be present in an
            acquisition sequence.
        pattern_dimension_x (int):
            Length of reconstructed image that defines pattern length.
        pattern_dimension_y (int):
            Width of reconstructed image that defines pattern width.
        zoom (int):
            numerical zoom of the patterns
        xw_offset (int):
            offset of the pattern in the DMD for zoom > 1 in the width (x) direction
        yh_offset (int):
            offset of the pattern in the DMD for zoom > 1 in the heihgt (y) direction   
        mask_index (Union[np.ndarray, str], optional):
            Array of `int` type corresponding to the index of the mask vector where
            the value is egal to 1
        x_mask_coord (Union[np.ndarray, str], optional):
            coordinates of the mask in the x direction x[0] and x[1] are the first
            and last points respectively
        y_mask_coord (Union[np.ndarray, str], optional):
            coordinates of the mask in the y direction y[0] and y[1] are the first
            and last points respectively    
        pattern_amount (int, optional):
            Quantity of patterns sent to DMD for an acquisition. This value is
            calculated by an external function. Default in None.
        acquired_spectra (int, optional):
            Amount of spectra actually read from the spectrometer. This value is
            calculated by an external function. Default in None.
        mean_callback_acquisition_time_ms (float, optional):
            Mean time between 2 callback executions during an acquisition. This 
            value is calculated by an external function. Default in None.
        total_callback_acquisition_time_s (float, optional):
            Total time of callback executions during an acquisition. This value
            is calculated by an external function. Default in None.
        mean_spectrometer_acquisition_time_ms (float, optional):
            Mean time between 2 spectrometer measurements during an acquisition
            based on its own internal clock. This value is calculated by an
            external function. Default in None.
        total_spectrometer_acquisition_time_s (float, optional):
            Total time of spectrometer measurements during an acquisition
            based on its own internal clock. This value is calculated by an
            external function. Default in None.
        saturation_detected (bool, optional):
            Boolean incating if saturation was detected during acquisition.
            Default is None.
        patterns (Union[List[int],str], optional) = None
            List `int` or `str` containing all patterns sent to the DMD for an
            acquisition sequence. This value is set by an external function and
            its type can be modified by multiple functions during object
            creation, manipulation, when dumping to a JSON file or
            when reconstructing an AcquisitionParameters object from a JSON
            file. It is intended to be of type List[int] most of the execution
            List[int]time. Default is None.
        wavelengths (Union[np.ndarray, str], optional):
            Array of `float` type corresponding to the wavelengths associated
            with spectrometer's start and stop pixels.
        timestamps (Union[List[float], str], optional):
            List of `float` type elapsed time between each measurement
            made by the spectrometer based on its internal clock. Units in 
            milliseconds. Default is None.
        measurement_time (Union[List[float], str], optional):
            List of `float` type elapsed times between each callback. Units in
            milliseconds. Default is None.
        class_description (str):
            Class description used to improve redability when dumped to JSON
            file. Default is 'Acquisition parameters'.
    """

    pattern_compression: float
    pattern_dimension_x: int
    pattern_dimension_y: int
    zoom: Optional[int] = field(default=None) 
    xw_offset: Optional[int] = field(default=None) 
    yh_offset: Optional[int] = field(default=None) 
    mask_index: Optional[Union[np.ndarray, str]] = field(default=None, repr=False)
    x_mask_coord: Optional[Union[np.ndarray, str]] = field(default=None, repr=False)
    y_mask_coord: Optional[Union[np.ndarray, str]] = field(default=None, repr=False)
    
    pattern_amount: Optional[int] = 1
    acquired_spectra: Optional[int] = None

    mean_callback_acquisition_time_ms: Optional[float] = None
    total_callback_acquisition_time_s: Optional[float] = None
    mean_spectrometer_acquisition_time_ms: Optional[float] = None
    total_spectrometer_acquisition_time_s: Optional[float] = None

    saturation_detected: Optional[bool] = None

    patterns: Optional[Union[List[int], str]] = field(default=None, repr=False)
    patterns_wp: Optional[Union[List[int], str]] = field(default=None, repr=False)
    wavelengths: Optional[Union[np.ndarray, str]] = field(default=None, repr=False)
    timestamps: Optional[Union[List[float], str]] = field(default=None, repr=False)
    measurement_time: Optional[Union[List[float], str]] = field(default=None, repr=False)
    
    output_directory: Optional[str] = field(default=None)  
    pattern_order_source: Optional[str] = field(default=None)    
    pattern_source: Optional[str] = field(default=None)  
    pattern_prefix: Optional[str] = field(default=None)  
    experiment_name: Optional[str] = field(default=None)  
    light_source: Optional[str] = field(default=None)  
    object: Optional[str] = field(default=None)  
    filter: Optional[str] = field(default=None)  
    description: Optional[str] = field(default=None)  
    
    NRepetitions: Optional[int] = field(default=None) 
    NAverages: Optional[int] = field(default=None) 
    Lc: Optional[Union[List[int], str]] = field(default=None, repr=False)
    receive_last_trig:Optional[bool] = field(default=False, repr=False)
    
    class_description: str = 'Acquisition parameters'


    def undo_readable_pattern_order(self) -> None:
        """Changes the patterns attribute from `str` to `List` of `int`.

        When reconstructing an AcquisitionParameters object from a JSON file,
        this method turns the patterns, wavelengths, timestamps and 
        measurement_time attributes from a string to a list of integers
        containing the pattern indices used in that acquisition.
        """
        
        def to_float(str_arr):
            arr = []
            for s in str_arr:
                try:
                    num = float(s)
                    arr.append(num)
                except ValueError:
                    pass
            return arr
        
        self.patterns = self.patterns.strip('[').strip(']').split(', ')
        self.patterns = [int(s) for s in self.patterns if s.isdigit()]
        try:
            self.patterns_wp = self.patterns_wp.text.strip('[').strip(']').split(', ')
            self.patterns_wp = [int(s) for s in self.patterns_wp if s.isdigit()]
        except:
            print('patterns_wp has no attribute ''strip''')    

        if self.wavelengths:
            self.wavelengths = (
                self.wavelengths.strip('[').strip(']').split(', '))
            self.wavelengths = to_float(self.wavelengths)
            self.wavelengths = np.asarray(self.wavelengths)
        else:
            print('wavelenghts not present in metadata.'
            ' Reading data in legacy mode.')

        if self.timestamps:
            self.timestamps = self.timestamps.strip('[').strip(']').split(', ')
            self.timestamps = to_float(self.timestamps)
        else:
            print('timestamps not present in metadata.'
            ' Reading data in legacy mode.')

        if self.measurement_time:
            self.measurement_time = (
                self.measurement_time.strip('[').strip(']').split(', '))
            self.measurement_time = to_float(self.measurement_time)
        else:
            print('measurement_time not present in metadata.'
            ' Reading data in legacy mode.')

        if self.mask_index:
            self.mask_index = (
                self.mask_index.strip('[').strip(']').split(', '))
            self.mask_index = to_float(self.mask_index)
            self.mask_index = np.asarray(self.mask_index)
        else:
            print('mask_index not present in metadata.'
            ' Reading data in legacy mode.')
        
        if self.x_mask_coord:
            self.x_mask_coord = (
                self.x_mask_coord.strip('[').strip(']').split(', '))
            self.x_mask_coord = to_float(self.x_mask_coord)
            self.x_mask_coord = np.asarray(self.x_mask_coord)
        else:
            print('x_mask_coord not present in metadata.'
            ' Reading data in legacy mode.')
        
        if self.y_mask_coord:
            self.y_mask_coord = (
                self.y_mask_coord.strip('[').strip(']').split(', '))
            self.y_mask_coord = to_float(self.y_mask_coord)
            self.y_mask_coord = np.asarray(self.y_mask_coord)
        else:
            print('y_mask_coord not present in metadata.'
            ' Reading data in legacy mode.')
            
        if self.Lc:
            self.Lc = (
                self.Lc.strip('[').strip(']').split(', '))
            self.Lc = to_float(self.Lc)
            self.Lc = np.asarray(self.Lc)
        else:
            print('Lc not present in metadata.'
            ' Reading data in legacy mode.')
        
    @staticmethod
    def readable_pattern_order(acquisition_params_dict: dict) -> dict:
        """Turns list of patterns into a string.

        Turns the list of pattern attributes from an AcquisitionParameters 
        object (turned into a dictionary) into a string that will improve
        readability once all metadata is dumped into a JSON file.
        This function must be called before dumping.

        Args:
            acquisition_params_dict (dict): Dictionary obtained from converting 
            an AcquisitionParameters object.

        Returns:
            [dict]: Modified dictionary with acquisition parameters metadata.
        """

        def _hard_coded_conversion(data):
            s = '['
            for value in data:
                s += f'{value:.4f}, '
            s = s[:-2]
            s += ']'

            return s

        readable_dict = acquisition_params_dict
        readable_dict['patterns'] = str(readable_dict['patterns'])
        readable_dict['patterns_wp'] = str(readable_dict['patterns_wp'])
        
        readable_dict['wavelengths'] = _hard_coded_conversion(
            readable_dict['wavelengths'])
    
        readable_dict['timestamps'] = _hard_coded_conversion(
            readable_dict['timestamps'])

        readable_dict['measurement_time'] = _hard_coded_conversion(
            readable_dict['measurement_time'])
        
        readable_dict['mask_index'] = _hard_coded_conversion(
            readable_dict['mask_index'])
        
        readable_dict['x_mask_coord'] = _hard_coded_conversion(
            readable_dict['x_mask_coord'])
        
        readable_dict['y_mask_coord'] = _hard_coded_conversion(
            readable_dict['y_mask_coord'])
        
        readable_dict['Lc'] = _hard_coded_conversion(
            readable_dict['Lc'])

        return readable_dict


    def update_timings(self, timestamps: np.ndarray, 
                       measurement_time: np.ndarray):
        """Updates acquisition timings.

        Args:
            timestamps (ndarray): 
                Array of `float` type elapsed time between each measurement made
                by the spectrometer based on its internal clock. Units in 
                milliseconds.
            measurement_time (ndarray):
                Array of `float` type elapsed times between each callback. Units
                in milliseconds.
        """
        self.mean_callback_acquisition_time_ms = np.mean(measurement_time)
        self.total_callback_acquisition_time_s = np.sum(measurement_time) / 1000
        self.mean_spectrometer_acquisition_time_ms = np.mean(
            timestamps, dtype=float)
        self.total_spectrometer_acquisition_time_s = np.sum(timestamps) / 1000

        self.timestamps = timestamps
        self.measurement_time = measurement_time

 
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


# def setup_acqui(DMD: ALP4,
#           #camPar: CAM,
#           DMD_initial_memory: int, 
#           #metadata: MetaData,
#           acquisition_params: AcquisitionParameters,
#           start_pixel: int = 0,
#           stop_pixel: Optional[int] = None,
#           integration_time: float = 1, 
#           integration_delay: int = 0,
#           DMD_output_synch_pulse_delay: int = 0, 
#           add_illumination_time: int = 356,
#           dark_phase_time: int = 44,
#           DMD_trigger_in_delay: int = 0          
#           ):# -> Tuple[SpectrometerParameters, DMDParameters]:
#     """Setup everything needed to start an acquisition.

#     Sets all parameters for DMD, spectrometer, DMD patterns and DMD timings.
#     Must be called before every acquisition.

#     Args:
#         spectrometer (Avantes):
#             Connected spectrometer (Avantes object).
#         DMD (ALP4):
#             Connected DMD.
#         camPar (CAM):
#             Metadata object of the IDS monochrome camera 
#         DMD_initial_memory (int):
#             Initial memory available in DMD after initialization.
#         metadata (MetaData):
#             Metadata concerning the experiment, paths, file inputs and file 
#             outputs. Must be created and filled up by the user.
#         acquisition_params (AcquisitionParameters):
#             Acquisition related metadata object. User must partially fill up
#             with pattern_compression, pattern_dimension_x, pattern_dimension_y,
#             zoom, x and y offest of patterns displayed on the DMD.
#         start_pixel (int):
#             Initial pixel data received from spectrometer. Default is 0.
#         stop_pixel (int, optional):
#             Last pixel data received from spectrometer. Default is None if it
#             should be determined from the amount of available pixels in the
#             spectrometer.
#         integration_time (float):
#             Spectrometer exposure time during one scan in miliseconds. Default
#             is 1 ms.
#         integration_delay (int):
#             Parameter used to start the integration time not immediately after 
#             the measurement request (or on an external hardware trigger), but 
#             after a specified delay. Unit is based on internal FPGA clock cycle.
#             Default is 0 us.
#         DMD_output_synch_pulse_delay (int):
#             Time in microseconds between start of the frame synch output pulse 
#             and the start of the pattern display (in master mode). Default is
#             0 us.
#         add_illumination_time (int):
#             Extra time in microseconds to account for the spectrometer's 
#             "dead time". Default is 365 us.
#         dark_phase_time (int):
#             Time in microseconds taken by the DMD mirrors to completely tilt. 
#             Minimum time for XGA type DMD is 44 us. Default is 44 us.
#         DMD_trigger_in_delay (int):
#             Time in microseconds between the incoming trigger edge and the start
#             of the pattern display on DMD (slave mode). Default is 0 us.
    
#     Raises:
#         ValueError: Sum of dark phase and additional illumination time is lower
#         than 400 us.

#     Returns:
#         Tuple[SpectrometerParameters, DMDParameters, List]: Tuple containing DMD
#         and spectrometer relate metadata, as well as wavelengths.
#             spectrometer_params (SpectrometerParameters):
#                 Spectrometer metadata object with spectrometer configurations.
#             DMD_params (DMDParameters):
#                 DMD metadata object with DMD configurations.
#     """

#     path = Path(metadata.output_directory)
#     if not path.exists():
#         path.mkdir()
    
#     if dark_phase_time + add_illumination_time < 350:
#         raise ValueError(f'Sum of dark phase and additional illumination time '
#                          f'is {dark_phase_time + add_illumination_time}.'
#                          f' Must be greater than 350 µs.')

#     elif dark_phase_time + add_illumination_time < 400:
#         warnings.warn(f'Sum of dark phase and additional illumination time '
#                       f'is {dark_phase_time + add_illumination_time}.'
#                       f' It is recomended to choose at least 400 µs.')
    
#     synch_pulse_width, illumination_time, picture_time = _calculate_timings(
#         integration_time, 
#         integration_delay, 
#         add_illumination_time, 
#         DMD_output_synch_pulse_delay, 
#         dark_phase_time)

#     spectrometer_params, wavelenghts = _setup_spectrometer(
#         spectrometer, 
#         integration_time, 
#         integration_delay,
#         start_pixel,
#         stop_pixel)
    
#     if camPar.gate_period > 16:
#         gate_period = 16
#         print('Warning, gate period is ' + str(camPar.gate_period) + ' >  than the max: 16.')
#         print('Try to increase the FPS of the camera, or the integration time of the spectrometer.')
#         print('Check the Pixel clock which must be = 474 MHz')
#         print('Otherwise some frames will be lost.')
#     elif camPar.gate_period <1:
#         print('Warning, gate period is ' + str(camPar.gate_period) + ' <  than the min: 1.')
#         gate_period = 1
#     else:
#         gate_period = camPar.gate_period
    
#     camPar.gate_period = gate_period    
#     Gate = tAlpDynSynchOutGate()
#     Gate.byref[0] = ct.c_ubyte(gate_period)     # Period [1 to 16] (it is a multiple of the trig period which go to the spectro)
#     Gate.byref[1] = ct.c_ubyte(1)   # Polarity => 0: active pulse is low, 1: high
#     Gate.byref[2] = ct.c_ubyte(1)   # Gate1 ok to send TTL 
#     Gate.byref[3] = ct.c_ubyte(0)   # Gate2 do not send TTL
#     Gate.byref[4] = ct.c_ubyte(0)   # Gate3 do not send TTL
#     DMD.DevControlEx(ALP_DEV_DYN_SYNCH_OUT1_GATE, Gate)
#     camPar.gate_period = gate_period
#     camPar.int_time_spect = integration_time

#     acquisition_params.wavelengths = np.asarray(wavelenghts, dtype=np.float64)

#     DMD_params = _setup_DMD(DMD, add_illumination_time, DMD_initial_memory)
    
#     _setup_patterns_2arms(DMD=DMD, metadata=metadata, DMD_params=DMD_params, 
#                     acquisition_params=acquisition_params, camPar=camPar)

#     _setup_timings(DMD, DMD_params, picture_time, illumination_time, 
#                    DMD_output_synch_pulse_delay, synch_pulse_width, 
#                    DMD_trigger_in_delay, add_illumination_time)

#     return spectrometer_params, DMD_params, camPar

def _save_acquisition_2arms(# metadata: MetaData, 
                     DMD_params,#: DMD_mod.DMDParameters, 
                     # spectrometer_params: SpectrometerParameters, 
                     # camPar: CAM,
                     acquisition_parameters: AcquisitionParameters, 
                     spectral_data: np.ndarray) -> None:
    print('at this moment, do nothing')

# def _save_acquisition_2arms(metadata: MetaData, 
#                      DMD_params: DMDParameters, 
#                      spectrometer_params: SpectrometerParameters, 
#                      camPar: CAM,
#                      acquisition_parameters: AcquisitionParameters, 
#                      spectral_data: np.ndarray) -> None:
#     """Save all acquisition data and metadata.

#     Args:
#         metadata (MetaData):
#             Metadata concerning the experiment, paths, file inputs and file
#             outputs.
#         DMD_params (DMDParameters): 
#             DMD metadata object with DMD configurations.
#         spectrometer_params (SpectrometerParameters):
#             Spectrometer metadata object with spectrometer configurations.
#         camPar (CAM):
#             Metadata object of the IDS monochrome camera 
#         acquisition_parameters (AcquisitionParameters):
#             Acquisition related metadata object. 
#         spectral_data (ndarray):
#             1D array with `float` type spectrometer measurements. Array size
#             depends on start and stop pixels previously set to the spectrometer.
#     """

#     # Saving collected data and timings
#     path = Path(metadata.output_directory)
#     path = path / f'{metadata.experiment_name}_spectraldata.npz'
#     np.savez_compressed(path, spectral_data=spectral_data)

#     # Saving metadata
#     save_metadata_2arms(metadata, 
#                   DMD_params,
#                   spectrometer_params,
#                   camPar,
#                   acquisition_parameters)


def _acquire_raw_2arms(
            DMD: ALP4,
            # camPar: CAM,
            # spectrometer_params: SpectrometerParameters, 
            DMD_params,#: DMD_mod.DMDParameters, 
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
                # spectral_data[spectrum_index,:] = (
                #     np.ctypeslib.as_array(spectrum[0:pixel_amount]))
                 
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
        # camPar = stopCapt_DeallocMem(camPar)
        # camPar.trigger_mode = 'hard'#'soft'#
        # imageQueue(camPar)
        # camPar = prepareCam(camPar, metadata)
        # camPar.timeout = 1000   # time out in ms for the "is_WaitForNextImage" function
        start_chrono = time.time()
        # x = threading.Thread(target = runCam_thread, args=(camPar, start_chrono))
        # x.start()
    
    # pixel_amount = (spectrometer_params.stop_pixel - 
    #                 spectrometer_params.start_pixel + 1)

    measurement_time = np.zeros((acquisition_params.pattern_amount))
    timestamps = np.zeros((acquisition_params.pattern_amount),dtype=np.uint32)
    # spectral_data = np.zeros(
    #     (acquisition_params.pattern_amount,pixel_amount),dtype=np.float64)

    # Boolean to indicate if saturation was detected during acquisition
    saturation_detected = False 

    spectrum_index = 0 # Accessed as nonlocal variable inside the callback

    # #spectro.register_callback(-2,acquisition_params.pattern_amount,pixel_amount)
    # callback = register_callback(measurement_time, timestamps, 
    #                              spectral_data, ava)
    # measurement_callback = MeasureCallback(callback)
    # ava.measure_callback(-2, measurement_callback)
    
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

    # ava.stop_measure()
    DMD.Halt()
    # camPar.Exit = 2
    if repetition == repetitions-1:
        # camPar = stopCam(camPar)
        pass
    #Yprint('MAIN :// camPar.camActivated = ' + str(camPar.camActivated))
    AcquisitionResult = namedtuple('AcquisitionResult', [
        'spectral_data', 
        'spectrum_index',
        'timestamps',
        'measurement_time',
        'start_measurement_time',
        'saturation_detected'])

    return AcquisitionResult(#spectral_data, 
                             spectrum_index,
                             timestamps,
                             measurement_time,
                             start_measurement_time,
                             saturation_detected)

# def acquire(DMD: ALP4,
#             DMD_params: DMDParameters,
#             cam_spat: xiapi,
#             cam_spat_params: cam_Parameters,
#             cam_spec: xiapi,
#             cam_spec_params: cam_Parameters,
#             spectrograph        = spectrograph,
#             spectrograph_params = spectrograph_params,
#             acquisition_params  = acquisition_params):
#     """
    

#     Parameters
#     ----------
#     DMD : ALP4
#         DESCRIPTION.
#     DMD_params : DMDParameters
#         DESCRIPTION.
#     cam_spat : xiapi
#         DESCRIPTION.
#     cam_spat_params : cam_Parameters
#         DESCRIPTION.
#     cam_spec : xiapi
#         DESCRIPTION.
#     cam_spec_params : cam_Parameters
#         DESCRIPTION.
#     spectrograph : TYPE, optional
#         DESCRIPTION. The default is spectrograph.
#     spectrograph_params : TYPE, optional
#         DESCRIPTION. The default is spectrograph_params.
#     acquisition_params : TYPE, optional
#         DESCRIPTION. The default is acquisition_params.

#     Returns
#     -------
#     None.

#     """
    
#     pass


def acquire_2arms(
            DMD: ALP4,
            # camPar: CAM,
            metadata: MetaData, 
            DMD_params,#: DMD_mod.DMDParameters, 
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

    # if reconstruct == True:
    #     print('Creating reconstruction processes')

    #     # Creating a Queue for sending spectral data to reconstruction process
    #     queue_to_recon = Queue()

    #     # Creating a Queue for sending reconstructed images to plot
    #     queue_reconstructed = Queue()

    #     sleep_time = (acquisition_params.pattern_amount * 
    #                 DMD_params.picture_time_us/1e+6)

    #     # Creating reconstruction process
    #     recon_process = Process(target=reconstruct_process, 
    #                 args=(reconstruction_params.model,
    #                     reconstruction_params.device, 
    #                     queue_to_recon,
    #                     queue_reconstructed,
    #                     reconstruction_params.batches, 
    #                     reconstruction_params.noise,
    #                     sleep_time))

    #     # Creating plot process
    #     plot_process = Process(target=plot_recon, 
    #                     args=(queue_reconstructed, sleep_time))

    #     # Starting processes
    #     recon_process.start()
    #     plot_process.start()
        
    # pixel_amount = (spectrometer_params.stop_pixel - 
    #                 spectrometer_params.start_pixel + 1)
    measurement_time = np.zeros(
        (acquisition_params.pattern_amount * repetitions))
    timestamps = np.zeros(
        ((acquisition_params.pattern_amount - 1) * repetitions), 
        dtype=np.float64)
    # spectral_data = np.zeros(
    #     (acquisition_params.pattern_amount * repetitions,pixel_amount),
    #     dtype=np.float64)

    acquisition_params.acquired_spectra = 0
    print()

    for repetition in range(repetitions):
        if verbose:
            print(f"Acquisition {repetition}")

        AcquisitionResults = _acquire_raw_2arms(DMD, 
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
        # spectral_data[begin:end] = data
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
    
    # # delete acquisition with black pattern (white for the camera)
    # if camPar.insert_patterns == 1:
    #     black_pattern_index = np.where(acquisition_params.patterns_wp == -1)
    #     # print('index of white patterns :')
    #     # print(black_pattern_index[0:38])
    #     if acquisition_params.patterns_wp.shape == acquisition_params.patterns.shape:
    #         acquisition_params.patterns = np.delete(acquisition_params.patterns, black_pattern_index)
    #     spectral_data = np.delete(spectral_data, black_pattern_index, axis = 0)
    #     acquisition_params.timestamps = np.delete(acquisition_params.timestamps, black_pattern_index[1:])
    #     acquisition_params.measurement_time = np.delete(acquisition_params.measurement_time, black_pattern_index)
    #     acquisition_params.acquired_spectra = len(acquisition_params.patterns)
    
    _save_acquisition_2arms(#metadata, 
                            DMD_params, 
                            #spectrometer_params, camPar, 
                            acquisition_params, 
                            # spectral_data
                            )

    # Joining processes and closing queues
    if reconstruct == True:
        queue_to_recon.put('kill') # Sends a message to stop reconstruction
        recon_process.join()
        queue_to_recon.close()
        plot_process.join()
        queue_reconstructed.close()
        
    # maxi = np.amax(spectral_data[0,:])
    # print('------------------------------------------------')
    # print('maximum in the spectrum = ' + str(maxi))
    # print('------------------------------------------------')
    # if maxi >= 65535:
    #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #     print('!!!!! warning, spectrum saturation !!!!!!!!')
    #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # return spectral_data

class func_path:
    """
    A class that contain all the path to save the data
    
    Args:
        None
        
    Return:
        None
    """
    
    subfolder_path: str
    aborted: bool
    raw_data_path: str
    overview_path: str
    data_name: str
    data_path: str
    had_reco_path: str
    fig_had_reco_path: str
    nn_reco_path: str
    fig_nn_reco_path: str
    
    def __init__(self, data_folder_name, data_name, ask_overwrite=False):        
        if not os.path.exists('../../data/' + data_folder_name):
            os.makedirs('../../data/' + data_folder_name)
        
        self.subfolder_path = '../../data/' + data_folder_name + '/' + data_name
        if not os.path.exists(self.subfolder_path):
            os.makedirs(self.subfolder_path)
            aborted = False
        elif ask_overwrite == True:
            res = input('Acquisition already exists, overwrite it ?[y/n]')
            if res == 'n':
                aborted = True
            elif res == 'y':
                aborted = False
            else:
                print('')
                aborted = False
        else:
            aborted = False
                
        self.aborted = aborted
                   
        self.raw_data_path = self.subfolder_path + '/raw_data'
        if not os.path.exists(self.raw_data_path):
            os.makedirs(self.raw_data_path)

        self.overview_path = self.subfolder_path + '/overview'
        if not os.path.exists(self.overview_path):
            os.makedirs(self.overview_path)

        self.data_name = data_name
        self.data_path = self.subfolder_path + '/'# + data_name
        self.had_reco_path = self.data_path + 'had_reco.npz'         
        self.fig_had_reco_path = self.overview_path + '/'# + data_name   
        self.nn_reco_path = self.data_path + 'nn_reco.npz'
        self.fig_nn_reco_path = self.overview_path + '/'# + data_name 







    
    