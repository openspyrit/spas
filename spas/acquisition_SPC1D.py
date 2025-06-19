# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""Acquisition utility functions.

    Acquisition module is a generic module that call function in different setup (SPC2D_1arm, SPC2D_2arms, SCP1D and SPIM)
    
"""

# import warnings
# from time import sleep, perf_counter_ns
# from typing import NamedTuple 
from typing import Tuple, List, Optional, Union
# from collections import namedtuple
from pathlib import Path
# from multiprocessing import Process, Queue
# import shutil    
import math
import os
import json 

import numpy as np
# from PIL import Image
##### DLL for the DMD
try:
    from ALP4 import ALP4, ALP_FIRSTFRAME, ALP_LASTFRAME
    from ALP4 import ALP_AVAIL_MEMORY, ALP_DEV_DYN_SYNCH_OUT1_GATE, tAlpDynSynchOutGate
    # print('ALP4 is ok in Acquisition file')
except:
    class ALP4:
        pass
from ximea import xiapi
# ##### DLL for the spectrometer Avantes 
# try:
#     from msl.equipment import EquipmentRecord, ConnectionRecord, Backend
#     from msl.equipment.resources.avantes import MeasureCallback, Avantes
# except:
#     pass
    
# from tqdm import tqdm
# from spas.metadata_SPC2D import DMDParameters, MetaData, AcquisitionParameters
# from spas.metadata_SPC2D import SpectrometerParameters, save_metadata, CAM, save_metadata_2arms
from spas.reconstruction_nn import reconstruct_process, plot_recon, ReconstructionParameters
# from spas.metadata_SPC1D import save_metadata
#To be remove later
# from spas.metadata_SPC2D import MetaData

# # DLL for the IDS CAMERA
# try:
#     from pyueye import ueye, ueye_tools
# except:
#     print('ueye DLL not installed')

# from matplotlib import pyplot as plt
# from IPython import get_ipython
# import ctypes as ct
# import logging
import time
import threading
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import pickle
from progress.bar import Bar
from spas.spectro_SP_module import setup_spectrograph
from spas.cam_Ximea_module import counter_trigger

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
    Lc: Optional[Union[List[List[int]], str]] = field(default=None)
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
            LLc = [] * 2            
            a = (
                self.Lc.strip('[').strip(']').split(')('))
            for ia in range(len(a)):
                if ia % 2 == 0:
                    b = a[ia] + ')'
                else:
                    b = '(' + a[ia]   
                c = b.strip('(').strip(')').split(', ')
                d = to_float(c)
                LLc.append(d)
            
            # self.Lc = LLc
            LLLc = [] * 2
            for sublist in LLc:
                sublist[:] = map(int, sublist[:])
                LLLc.append(sublist)
            self.Lc = LLLc
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
        
        def _hard_coded_conversion_for_list(data):
            s = '['
            for index in range(len(data)):
                s += '('
                for value in data[:][index]:
                    s += f'{value:.0f}, '
                s = s[:-2]
                s += ')'
            s += ']'

            return s
        
        readable_dict = acquisition_params_dict
        readable_dict['patterns'] = str(readable_dict['patterns'])
        readable_dict['patterns_wp'] = str(readable_dict['patterns_wp'])
        
        readable_dict['wavelengths'] = _hard_coded_conversion(
            readable_dict['wavelengths'])
    
        # readable_dict['timestamps'] = _hard_coded_conversion(
        #     readable_dict['timestamps'])

        # readable_dict['measurement_time'] = _hard_coded_conversion(
        #     readable_dict['measurement_time'])
        
        # readable_dict['mask_index'] = _hard_coded_conversion(
        #     readable_dict['mask_index'])
        
        # readable_dict['x_mask_coord'] = _hard_coded_conversion(
        #     readable_dict['x_mask_coord'])
        
        # readable_dict['y_mask_coord'] = _hard_coded_conversion(
        #     readable_dict['y_mask_coord'])
        
        readable_dict['Lc'] = _hard_coded_conversion_for_list(
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
 

def read_metadata(file_path: str):
                                    # -> Tuple[DMDParameters,
                                    #        Spectrograph_Parameters,
                                    #        cam_Parameters,
                                    #        cam_Parameters,
                                    #        AcquisitionParameters]:
    """Reads metadata of a previous acquisition from JSON file.

    Args:
        file_path (str):
            Name of JSON file containing all metadata.

    Returns:
        Tuple[MetaData, AcquisitionParameters, SpectrometerParameters, 
        DMDParameters]:
            saved_metadata (MetaData):
                Metadata object read from JSON.
            saved_acquisition_params(AcquisitionParameters):
                AcquisitionParameters object read from JSON.
            saved_spectrometer_params(SpectrometerParameters):
                SpectrometerParameters object read from JSON.
            saved_dmd_params(DMDParameters):
                DMDParameters object read from JSON.
    """
    
    from spas.DMD_module import DMDParameters
    from spas.spectro_SP_module import Spectrograph_Parameters
    from spas.cam_Ximea_module import cam_Parameters
    from spas.acquisition_SPC1D import AcquisitionParameters
    
    
    file = open(file_path,'r')
    # data_folder_name = '2025-06-16_test'
    # data_name = 'obj_USAF3_source_white_LED_Walsh_im_128x128_ti_1.0ms_zoom_x1'
    # file = open(metadata_path,'r')
    data = json.load(file)
    file.close()
        
    for object in data:
        if object['class_description'] == 'DMD parameters':
            saved_dmd_params = DMDParameters.from_dict(object)
            
        if object['class_description'] == 'spectrograph SP parameters':
            saved_spectro_params = Spectrograph_Parameters.from_dict(object)
            # saved_spectro_params = Spectrograph_Parameters.undo_readable_class_spectro(object)
            # # saved_spectro_params.undo_readable_class_spectro(object)
            # break
        
        if object['class_description'] == 'spatial camera parameters':
            saved_cam_spat_params = cam_Parameters.from_dict(object)  
            
        if object['class_description'] == 'spectral camera parameters':
            saved_cam_spec_params = cam_Parameters.from_dict(object)  
            
        if object['class_description'] == 'Acquisition parameters':
            saved_acquisition_params = AcquisitionParameters.from_dict(object)
            saved_acquisition_params.undo_readable_pattern_order()
            

    return (saved_dmd_params, saved_spectro_params, 
            saved_cam_spat_params, saved_cam_spec_params, saved_acquisition_params)


def save_metadata(DMD_params,#: DMDParameters, 
                  spectrograph_params,#: Spectrograph_Parameters, 
                  cam_spat_params,#: cam_Parameters,
                  cam_spec_params,
                  acquisition_params: AcquisitionParameters) -> None:
    """Saves metadata to JSON file.

    Args:
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file
            outputs.
        DMD_params (DMDParameters):
            Class containing DMD configurations and status.
        spectrometer_params (SpectrometerParameters):
            Object containing spectrometer configurations.
        acquisition_parameters (AcquisitionParameters):
            Object containing acquisition specifications and timing results.
    """

    from spas.spectro_SP_module import Spectrograph_Parameters
    from spas.acquisition_SPC1D import AcquisitionParameters

    path = Path(acquisition_params.output_directory)
    with open(path / 'metadata.json', 'w', encoding='utf8') as output:

        output_params = [DMD_params.to_dict(), 
                         Spectrograph_Parameters.readable_class_spectro(spectrograph_params.to_dict()),
                         cam_spat_params.to_dict(),
                         cam_spec_params.to_dict(),
                         AcquisitionParameters.readable_pattern_order(acquisition_params.to_dict())]

        json.dump(output_params, output, ensure_ascii=False, indent=4)


def runCam_thread(cam, acquisition_params, DMD_params, all_path, NR: int = 1, iLc: int = 1, NA: int = 1, first_acqui: bool = True, verbose: bool = False): 
    """Acquire video with the Ximea camera in a thread

    Parameters:
    ----------
    cam (obj): 
        a object to drive the Ximea camera
    acquisition_params (class):
        the class of the acquisition parameters
    NR (int):
        the increment of the number of repetitions (default = 1)
    iLc (int):
        the increment of the central wavenlength and grating number (default = 1)
    NA (int):
        the increment of the number of averages (default = 1)
    first_acqui (bool):
        a boolean to start the video acquistion just at the first call. (default = True)
        
    Returns:
    -------
        None.
    """

    img = xiapi.Image()
    arm = cam.arm
    file_name = arm + '_NR_' + str(NR) + '_Gr_' + str(acquisition_params.Lc[iLc][1]) + '_Lc_' + str(acquisition_params.Lc[iLc][0]) + 'nm_NA_' + str(NA) + '_NS_'
    ####################### start data acquisition ############################
    if first_acqui:
        # print('Starting ' + arm + ' data acquisition...\n')
        cam.start_acquisition()
        
    start_chrono = time.time()
    i = 0
    # acquire snapshot
    if cam.snapshot:
        while True: 
            counter_time = time.time() - start_chrono
            if arm == "spatial":
                stop_it = 2
            elif arm == "spectral":
                stop_it = 1
            if i >= stop_it:
                acquisition_params.receive_last_trig = True
                if verbose:
                    print('\n iteration reach (' + arm + ') : ' + str(i) + ' in the thread \n')
                break
            elif counter_time > math.ceil(acquisition_params.pattern_amount * DMD_params.picture_time_us / 1e6) + 4:
                print('delay > ' + str(math.ceil(acquisition_params.pattern_amount * DMD_params.picture_time_us / 1e6) + 4) + 's in the thread \n')
                break        
            else:
                ############## get data and pass them from cameras to img #################
                cam.get_image(img)
                if i == stop_it - 1:
                    ################### get image data as numpy array #########################
                    data_np = img.get_image_data_numpy()#(invert_rgb_order = True)  
                    ################### write raw data in files #######################
                    with open(all_path.raw_data_path + '/' + file_name + str(i) + '.pkl', 'wb') as outp:
                        pickle.dump(data_np, outp, pickle.HIGHEST_PROTOCOL)      
                
                i = i + 1
                acquisition_params.receive_last_trig = False
    # acquire all the frames
    else:
        while True: 
            counter_time = time.time() - start_chrono
            if i >= acquisition_params.pattern_amount:
                acquisition_params.receive_last_trig = True
                if verbose:
                    print('\n iteration reach (' + arm + ') : ' + str(i) + ' in the thread \n')
                break
            elif counter_time > math.ceil(acquisition_params.pattern_amount * DMD_params.picture_time_us / 1e6) + 4:
                print('delay > ' + str(math.ceil(acquisition_params.pattern_amount * DMD_params.picture_time_us / 1e6) + 4) + 's in the thread \n')
                break        
            else:
                ############## get data and pass them from cameras to img #################
                cam.get_image(img)
                ################### get image data as numpy array #########################
                data_np = img.get_image_data_numpy()#(invert_rgb_order = True)  
                ################### write raw data in files #######################
                with open(all_path.raw_data_path + '/' + file_name + str(i) + '.pkl', 'wb') as outp:
                    pickle.dump(data_np, outp, pickle.HIGHEST_PROTOCOL)
    
                # # time_stmp = (img.tsSec) + ((img.tsUSec)/1000000)
                # print('i: ' + str(i) + '-->  Time Stamp: ' + str(time_stmp - time_stmp_0))           
                
                i = i + 1
                acquisition_params.receive_last_trig = False

def acquire(DMD: ALP4,
            DMD_params,
            cam_spat: xiapi,
            cam_spat_params,
            cam_spec: xiapi,
            cam_spec_params,
            spectrograph,
            spectrograph_params,
            acquisition_params,
            all_path,
            verbose):
    """
    

    Parameters
    ----------
    DMD : ALP4
        DESCRIPTION.
    DMD_params : DMDParameters
        DESCRIPTION.
    cam_spat : xiapi
        DESCRIPTION.
    cam_spat_params : cam_Parameters
        DESCRIPTION.
    cam_spec : xiapi
        DESCRIPTION.
    cam_spec_params : cam_Parameters
        DESCRIPTION.
    spectrograph : TYPE, optional
        DESCRIPTION. The default is spectrograph.
    spectrograph_params : TYPE, optional
        DESCRIPTION. The default is spectrograph_params.
    acquisition_params : TYPE, optional
        DESCRIPTION. The default is acquisition_params.

    Returns
    -------
    None.

    """
    
    first_acqui = True

    total_loop = acquisition_params.NRepetitions * acquisition_params.NAverages * len(acquisition_params.Lc)
    total_iter = acquisition_params.pattern_amount * total_loop

    bar = Bar('Processing', max = total_loop)
    verbose = False
    first_acqui = True
    boucle = 0
    for NR in range(acquisition_params.NRepetitions):#tqdm(range(acquisition_params.NRepetitions)):
        for iLc in range(len(acquisition_params.Lc)):#tqdm(range(len(acquisition_params.Lc))):
            setup_spectrograph(spectrograph,
                               grating_nbr =  acquisition_params.Lc[iLc][1], print_select   = False,
                               position    =  acquisition_params.Lc[iLc][0], print_position = False)
            for NA in range(acquisition_params.NAverages):#tqdm(range(acquisition_params.NAverages)):
                if verbose:
                    boucle = boucle + 1
                    print('-----------------------------------------')
                    print('loop = ' + str(boucle) + ' / ' + str(acquisition_params.NRepetitions * len(acquisition_params.Lc) * acquisition_params.NAverages))
                    
                    print('[NR = ' + str(NR + 1) + '/' + str(acquisition_params.NRepetitions) + ' --- Lc = ' + str(iLc + 1) + '/' + str(len(acquisition_params.Lc)) + ' --- NA = ' + str(NA + 1) + '/' + str(acquisition_params.NAverages) + ']')
                
                bar.next()
                
                x = threading.Thread(target = runCam_thread, args=(cam_spat, acquisition_params, DMD_params, all_path, NR, iLc, NA, first_acqui, verbose))
                x.start()

                x1 = threading.Thread(target = runCam_thread, args=(cam_spec, acquisition_params, DMD_params, all_path, NR, iLc, NA, first_acqui, verbose))
                x1.start()

                if first_acqui:
                    time.sleep(1)
                
                DMD.Run(loop=False)
                
                first_pass = True
                start_chrono = time.time()
                while(True):
                    if first_pass == True:
                        time.sleep(math.ceil(acquisition_params.pattern_amount * DMD_params.picture_time_us / 1e6))
                        first_pass = False
                        
                    time.sleep(0.1)
                    counter_time = time.time() - start_chrono
                    
                    if acquisition_params.receive_last_trig:
                        if verbose:
                            print('iteration reachs ' + str(acquisition_params.pattern_amount) + ' in the main loop \n')
                        break
                    elif counter_time > math.ceil(acquisition_params.pattern_amount * DMD_params.picture_time_us / 1e6) + 5:
                        print('delay > ' + str(math.ceil(acquisition_params.pattern_amount * DMD_params.picture_time_us / 1e6 + 5)) + 's in the main loop \n')
                        break
                        
                DMD.Halt()
                first_acqui = False

    print('\n----------- COUNTERS SPATIAL CAM -----------') # reading counters
    counter_trig = counter_trigger(cam_spat)
    print('Transport skipped frames: ', counter_trig[0])
    print('API skipped frames      : ', counter_trig[1])
    print('Transferred frames      : ', str(counter_trig[2]) + ' / ' + str(total_iter))

    print('\n----------- COUNTERS SPECTRAL CAM -----------') # reading counters
    counter_trig = counter_trigger(cam_spec)
    print('Transport skipped frames: ', counter_trig[0])
    print('API skipped frames      : ', counter_trig[1])
    print('Transferred frames      : ', str(counter_trig[2]) + ' / ' + str(total_iter))
    print('\n')
                
    time.sleep(1)
    cam_spat.stop_acquisition()
    cam_spec.stop_acquisition()

    bar.finish()
    
    save_metadata(DMD_params, spectrograph_params, cam_spat_params, cam_spec_params, acquisition_params)


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
        self.fig_had_reco_path = self.overview_path + '/spectral'   
        self.nn_reco_path = self.data_path + 'nn_reco.npz'
        self.fig_nn_reco_path = self.overview_path + '/spectral' 







    
    