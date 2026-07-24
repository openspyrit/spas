# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""Acquisition utility functions.

    Acquisition module is a generic module that call function in different setup (SPC2D_1arm, SPC2D_2arms, SCP1D and SPIM)
    
"""


from typing import Tuple, List, Optional, Union
from pathlib import Path   
import math
import os
import json 

import numpy as np
##### DLL for the DMD
try:
    from ALP4 import ALP4
except:
    class ALP4:
        pass

from pylablib.devices import Andor

import time
import threading

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from progress.bar import Bar
from spas.spectro_ShamrockAndor_module import setup_spectrograph
from scipy import interpolate
from spas.PI_module import move_an_axis

from matplotlib import pyplot as plt

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
    spec_timestamps: Optional[Union[List[float], str]] = field(default=None, repr=False)
    spat_timestamps: Optional[Union[List[float], str]] = field(default=None, repr=False)
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
    receive_last_trig_spat:Optional[bool] = field(default=False, repr=False)
    receive_last_trig_spec:Optional[bool] = field(default=False, repr=False)
    
    Nz: Optional[Union[np.ndarray, str]] = field(default=None, repr=True)
    
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

        if self.spat_timestamps:
            self.spat_timestamps = self.spat_timestamps.strip('[').strip(']').split(', ')
            self.spat_timestamps = to_float(self.spat_timestamps)
        else:
            print('spat_timestamps not present in metadata.'
            ' Reading data in legacy mode.')
        
        if self.spec_timestamps:
            self.spec_timestamps = self.spec_timestamps.strip('[').strip(']').split(', ')
            self.spec_timestamps = to_float(self.spec_timestamps)
        else:
            print('spec_timestamps not present in metadata.'
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
            
        if self.Nz:
            self.Nz = (
                self.Nz.strip('[').strip(']').split(', '))
            self.Nz = to_float(self.Nz)
            self.Nz = np.asarray(self.Nz)
        else:
            print('Nz not present in metadata.'
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
    
        readable_dict['spat_timestamps'] = _hard_coded_conversion(
            readable_dict['spat_timestamps'])
        
        readable_dict['spec_timestamps'] = _hard_coded_conversion(
            readable_dict['spec_timestamps'])

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
        
        readable_dict['Nz'] = _hard_coded_conversion(
            readable_dict['Nz'])

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
    from spas.cam_Andor_module import cam_Parameters
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
    from spas.acquisition_SPIM1D import AcquisitionParameters

    path = Path(acquisition_params.output_directory)
    with open(path / 'metadata.json', 'w', encoding='utf8') as output:

        output_params = [DMD_params.to_dict(), 
                         Spectrograph_Parameters.readable_class_spectro(spectrograph_params.to_dict()),
                         cam_spat_params.to_dict(),
                         cam_spec_params.to_dict(),
                         AcquisitionParameters.readable_pattern_order(acquisition_params.to_dict())]

        json.dump(output_params, output, ensure_ascii=False, indent=4)


import tkinter as tk

def popup():
    """
    A popup to tell that the spectral data were acquired, it is need to swhitch the miror to acquire the spatial data

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    result = {"choice": None}

    def on_ok():
        result["choice"] = "ok"
        window.destroy()

    def on_abort():
        result["choice"] = "abort"
        window.destroy()

    window = tk.Tk()
    window.title("Information")

    label = tk.Label(
        window,
        text="Spectral acquisition is done, please switch the mirror for the spatial camera.\nClick OK when it is done",
        padx=20,
        pady=20
    )
    label.pack()

    frame = tk.Frame(window)
    frame.pack(pady=10)

    abort_button = tk.Button(frame, text="Abort", command=on_abort, width=10)
    abort_button.pack(side="left", padx=10)

    ok_button = tk.Button(frame, text="OK", command=on_ok, width=10)
    ok_button.pack(side="right", padx=10)

    window.mainloop()
    return result["choice"]

   
def runCam_thread(cam, acquisition_params, DMD_params, all_path, NR: int = 0, iLc: int = 0, NA: int = 0, first_acqui: bool = True, verbose: bool = False): 
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
    # acquisition_params.receive_last_trig_spat = False
    exp_time = cam.get_attribute_value("ExposureTime")
    acquisition_params.receive_last_trig_spec = False
    arm = cam.arm
    file_name = arm + '_Ny_' + str(acquisition_params.Nz[NR]) + 'mm_Gr_' + str(acquisition_params.Lc[iLc][1]) + '_Lc_' + str(acquisition_params.Lc[iLc][0]) + 'nm_NA_' + str(NA)# + '_NS_'
    ####################### start data acquisition ############################
    cam.setup_acquisition(mode="sequence", nframes = acquisition_params.pattern_amount) 
    if first_acqui:
        print('Starting ' + arm + ' data acquisition...\n')
        cam.start_acquisition()
        
    start_chrono = time.time()
    i = 0
    # acquire snapshot
    if cam.snapshot:     
        data_np = np.zeros((cam.get_attribute_value("AOIHeight"), cam.get_attribute_value("AOIWidth"), 1), dtype=np.uint16)
        timestamps = np.zeros((1),dtype=np.float64)
        ############## wait for next frame and read it #################
        cam.wait_for_frame(timeout = exp_time + 5) # wait for the next available frame    
        data_np = cam.snap()
        
        outp = all_path.raw_data_path + '/' + file_name
        np.savez(outp, data_np, allow_pickle = False)
        
        if arm == "spatial":
            acquisition_params.receive_last_trig_spat = True
            acquisition_params.spat_timestamps = timestamps
        elif arm == "spectral":
            acquisition_params.receive_last_trig_spec = True
            acquisition_params.spec_timestamps = timestamps
            acquisition_params.pattern_amount = 1
            return data_np
        
    # acquire all the frames
    else:
        bar = Bar('Processing', max = acquisition_params.pattern_amount)
        timestamps = np.zeros((acquisition_params.pattern_amount),dtype=np.float64)        
        # data_np = np.empty((acquisition_params.pattern_amount, cam.get_attribute_value("AOIHeight"), cam.get_attribute_value("AOIWidth")), dtype=np.int16)  
        data_np = np.zeros((cam.get_attribute_value("AOIHeight"), cam.get_attribute_value("AOIWidth"), acquisition_params.pattern_amount), dtype=np.uint16)
        
        while True: 
            counter_time = time.time() - start_chrono
            if i >= acquisition_params.pattern_amount:
                # data_np = np.transpose(data_np, (1, 2, 0))
                outp = all_path.raw_data_path + '/' + file_name
                np.savez(outp, data_np, allow_pickle = False)
                
                if arm == 'spectral':
                    acquisition_params.receive_last_trig_spec = True
                    acquisition_params.spec_timestamps = timestamps
                elif arm == 'spatial':
                    acquisition_params.receive_last_trig_spat = True
                    acquisition_params.spat_timestamps = timestamps
                print('\n iteration reach (' + arm + ') : ' + str(i + 1) + ' in the thread \n')
                
                bar.finish()
                return data_np
                break
            elif counter_time > math.ceil(acquisition_params.pattern_amount * DMD_params.picture_time_us / 1e6) + 4:
                print('delay > ' + str(math.ceil(acquisition_params.pattern_amount * DMD_params.picture_time_us / 1e6) + 4) + 's in the thread \n')
                break        
            else:
                ############## wait for next frame and read it #################
                cam.wait_for_frame(timeout = exp_time + 2) # wait for the next available frame
                data = cam.read_newest_image()  
                ################### timestamp #################################
                timestamps[i] = cam.get_attribute_value("TimestampClock")       
                ################### get image data as numpy array #########################
                data_np[:, :, i] = data
       
                i = i + 1  
                
                bar.next()
                
                counter_time2 = time.time() - start_chrono
                if counter_time2 - counter_time > (cam.get_exposure() + 3):
                    print('problem with the trigger, delay longer than the exposure time')
                    break
    

class CamThread(threading.Thread):
    def __init__(self, *args):
        super().__init__()
        self.args = args
        self.data_np = None
        self.finished = threading.Event()

    def run(self):
        self.data_np = runCam_thread(*self.args)
        self.finished.set()


def acquire(DMD: ALP4,
            DMD_params,
            cam_spat: Andor,
            cam_spat_params,
            cam_spec: Andor,
            cam_spec_params,
            spectrograph,
            spectrograph_params,
            stage,
            shutter,
            mirror,
            acquisition_params,
            all_path,
            verbose,
            acquisition_arm):
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
    stage: class.
        DESCRIPTION. To control the PI stage for a 3D acquisition
    shutter: Class.
        DESCRIPTION. To control the shutter
    mirror: Class.
        DESCRIPTION. To control the flipping mirror MFF101
    acquisition_params : TYPE, optional
        DESCRIPTION. The default is acquisition_params.
        
    acquisition_arm: str.
        specify which arm acquisiton is used, spectral or spatial

    Returns
    -------
    None.

    """   

    total_loop = acquisition_params.NRepetitions * acquisition_params.NAverages * len(acquisition_params.Lc)
    # total_iter = acquisition_params.pattern_amount * total_loop
    first_acqui = True
    boucle = 0
    mirror.set_position(acquisition_arm, verbose = True)
    shutter.open()
    for iNR in range(acquisition_params.NRepetitions):#tqdm(range(acquisition_params.NRepetitions)):
        print('\n')
        move_an_axis(stage.pidevice, stage.stage_tools, axes = ['2'], array_to_move = [acquisition_params.Nz[iNR]], verbose = True)
        for iLc in range(len(acquisition_params.Lc)):#tqdm(range(len(acquisition_params.Lc))):
            setup_spectrograph(spectrograph,
                               grating_nbr =  acquisition_params.Lc[iLc][1], print_select   = False,
                               position    =  acquisition_params.Lc[iLc][0], print_position = True)
            for NA in range(acquisition_params.NAverages):#tqdm(range(acquisition_params.NAverages)):
                boucle = boucle + 1                
                print('Iteration : ' + str(boucle) + ' / ' + str(total_loop))
                acq_aborded = False
                if verbose:            
                    print('-----------------------------------------')
                    print('loop = ' + str(boucle) + ' / ' + str(acquisition_params.NRepetitions * len(acquisition_params.Lc) * acquisition_params.NAverages))                    
                    print('[NR = ' + str(iNR + 1) + '/' + str(acquisition_params.NRepetitions) + ' --- Lc = ' + str(iLc + 1) + '/' + str(len(acquisition_params.Lc)) + ' --- NA = ' + str(NA + 1) + '/' + str(acquisition_params.NAverages) + ']')
                                
                if acquisition_arm == 'spatial' and iLc == 0:
                    cam_thread = CamThread(cam_spat, acquisition_params, DMD_params, all_path, iNR, iLc, NA, first_acqui, verbose)
                    cam_thread.start()
                elif acquisition_arm == 'spectral':                      
                    cam_thread = CamThread(cam_spec, acquisition_params, DMD_params, all_path, iNR, iLc, NA, first_acqui, verbose)
                    cam_thread.start()
                else:
                    if iLc == 0:
                        print('Please, specify the acquisition arm: "spectral" or "spatial"')
                    else:
                        print('iLc > 0')
                    acq_aborded = True
                    print('!!!!! warning = acquisition aborted !!!!!')

                if acq_aborded == False:
                    if first_acqui:
                        time.sleep(2) # avant c'était 1.2, changement depuis acqui avec cam spat
                        begin_acqui = time.time()
                    
                    DMD.Run(loop=False)
                    
                    while not cam_thread.finished.is_set():
                        time.sleep(0.01)   # laisse Windows respirer
                    
                    raw_data = cam_thread.data_np
                    
                    DMD.Halt()
                    print('DMD stopped')
                    if first_acqui == True:
                        if raw_data is not None:
                            print('raw data is not None')
                            raw_data_arr = np.empty((raw_data.shape + (acquisition_params.NAverages, 
                                                                       len(acquisition_params.Lc), 
                                                                       acquisition_params.NRepetitions)), dtype=np.uint16)
                        else:
                            if acquisition_arm == 'spatial':
                                raw_data_arr = np.empty((cam_spat.get_attribute_value("AOIHeight"), 
                                                         cam_spat.get_attribute_value("AOIWidth"), 
                                                         acquisition_params.pattern_amount) + 
                                                        (acquisition_params.NAverages, 
                                                         len(acquisition_params.Lc), 
                                                         acquisition_params.NRepetitions), dtype=np.uint16)
                        first_acqui = False
                    
                    if acquisition_arm == 'spectral':
                        if cam_spec.snapshot:
                            raw_data_arr[:, :, NA, iLc, iNR] = raw_data
                        else:
                            raw_data_arr[:, :, :, NA, iLc, iNR] = raw_data  
                    # elif acquisition_arm == 'spatial':
                    #     if cam_spat.snapshot:
                    #         raw_data_arr[:, :, NA, iLc, iNR] = raw_data
                    #     else:
                    #         raw_data_arr[:, :, :, NA, iLc, iNR] = raw_data  
               
    acquisition_params.total_spectrometer_acquisition_time_s = time.time() - begin_acqui
    print('\n')
    print('\nTotal acquisition time = ' + str(round(acquisition_params.total_spectrometer_acquisition_time_s * 1000) / 1000) + ' s')
    
    shutter.close()
    print('shutter closed')

    if acquisition_arm == 'spatial':
        cam_spat.stop_acquisition()
        print('spatial cam stopped')
        try:
            a = acquisition_params.spat_timestamps[0]
            acquisition_params.spec_timestamps = []
        except:
            acquisition_params.spat_timestamps = np.empty(0)
            print('warning, timestamps for spatial camera is empty')
    elif acquisition_arm == 'spectral':   
        cam_spec.stop_acquisition()
        print('\n')
        print('spectral cam stopped')
        try:
            # a = acquisition_params.spec_timestamps[0]
            acquisition_params.spat_timestamps = []
        except:
            acquisition_params.spec_timestamps = np.empty(0)
            acquisition_params.spat_timestamps = []
            print('warning, timestamps for spatial camera set to zero')
            print('warning, timestamps for spectral camera is empty')

        
        # choice = popup()
        # print('choice = ' + choice)

        # if choice == "ok":
        print('----------------- other arm --------------------')
        if acquisition_arm == 'spatial':
            acquisition_arm = 'spectral'
        elif acquisition_arm == 'spectral':
            acquisition_arm = 'spatial'
        
        mirror.set_position(acquisition_arm, verbose = True)
        print(acquisition_arm + " acquisition is beginning")
        first_acqui = True
        shutter.open()           
        for iNR in range(acquisition_params.NRepetitions):#tqdm(range(acquisition_params.NRepetitions)):
            iLc = 0
            move_an_axis(stage.pidevice, stage.stage_tools, axes = ['2'], array_to_move = [acquisition_params.Nz[iNR]], verbose = True)
            
            if acquisition_arm == 'spatial':
                cam_thread = CamThread(cam_spat, acquisition_params, DMD_params, all_path, iNR, iLc, NA, first_acqui, verbose)
            elif acquisition_arm == 'spectral':                      
                cam_thread = CamThread(cam_spec, acquisition_params, DMD_params, all_path, iNR, iLc, NA, first_acqui, verbose)

            cam_thread.start()
            
            if acq_aborded == False:
                if first_acqui:
                    time.sleep(2) # avant c'était 1.2, changé pour faire l'acqui avec la cam spatiale
                    begin_acqui = time.time()
                    first_acqui = False
            
            DMD.Run(loop=False)
            
            while not cam_thread.finished.is_set():
                time.sleep(0.01)   # laisse Windows respirer

            DMD.Halt()
            cam_spat.stop_acquisition()
            print(acquisition_arm + ' cam stopped')
        
        shutter.close()
        # else:
        #     print("Spatial acquisition aborted")
        
        try:
            acquisition_params.spec_timestamps[0]
        except:
            acquisition_params.spec_timestamps = np.empty(0)
            print('warning, timestamps for spectral camera is empty')
        
        save_metadata(DMD_params, spectrograph_params, cam_spat_params, cam_spec_params, acquisition_params)
        return raw_data_arr 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def define_wavelengths_matrix(cam_spec_params, Lc: list = [], display_figure: bool = False, verbose: bool = False):
    """
    define the wavelength vector for each central wavelength of the spectrograph

    Parameters
    ----------
    cam_spec_params: class.
        the class containing the parameters of the spectral camera
    Lc : list, optional
        The list af the central wavelength of the spectrograph. The default is [].

    Returns
    -------
    A matrix fo the wavelength vector for each Lc.

    """
    
    if display_figure:
        from matplotlib import pyplot as plt
    
    NLc = len(Lc)
    for iLc in range(NLc):
        Lcc = Lc[iLc][0]
        Grating = Lc[iLc][1]
        
        if Grating == 2:
            Lc_array = np.array([365, 435, 546, 577, 696, 912])
            coeff_array = np.array([3.308, 3.311, 3.319, 3.324, 3.341, 3.365])
            ord_origine = np.array([-5.5, -9.56, -7.58, -3.09, -6.92, -3.89])
        elif Grating == 1:
            # Lc_array = np.array([365, 405, 436, 546, 696, 795])
            # coeff_array = np.array([14, 14.075, 14.3, 14.82, 15.75, 16.63])
            # ord_origine = np.array([0, 0, 0, 0, 0, 0])
            Lc_array = np.array([     365,   405,   436,   546,    696,   912])
            coeff_array = np.array([13.26, 14.18, 14.12, 14.94,  15.89, 18.36])
            ord_origine = np.array([ 3.94, -0.69, -6.5,  -3.53,  -9.63, -5.88])
    
        z = np.polyfit(Lc_array, coeff_array, 2)
        p = np.poly1d(z)
    
        # xnew = np.arange(Lc_array[0], Lc_array[-1])
        xnew = np.arange(200, 1000)
        if display_figure:
            plt.figure()
            plt.plot(Lc_array, coeff_array, '.', xnew, p(xnew), '-')
            plt.grid()
            plt.title('GR = ' + str(Grating))
            
        indx = find_nearest(xnew, Lcc)
        
        px_width = cam_spec_params.width
        px_offset = cam_spec_params.offsetX
        
        spam_lambda = px_width / p(xnew[indx]) # np.mean(p(xnew))
        
        L_offset = px_offset / p(xnew[indx]) # np.mean(p(xnew))
        L1 = Lcc - spam_lambda / 2 + L_offset
        L2 = Lcc + spam_lambda / 2 + L_offset
        
        if verbose:
            print('L_offset = ' + str(L_offset))
            print('spam_lambda = ' + str(spam_lambda))
            print('Lcc = ' + str(Lcc))
            print('L1 = ' + str(L1))
            print('L2 = ' + str(L2))
        
        xnew2 = np.linspace(L1, L2, px_width)
        if display_figure:
            plt.figure()
            plt.plot(Lc_array, coeff_array, '.', xnew2, p(xnew2), '-')
            plt.grid()
            plt.title('GR = ' + str(Grating))
        
        f = interpolate.interp1d(Lc_array, ord_origine, fill_value='extrapolate')    
        ynew = f(xnew2)
        
        w = np.empty([NLc, px_width])
        for i in range(px_width):
            # w[iLc, i] = (i - 640 - ynew[i]) / p(xnew2[i]) + Lcc + L_offset # cette ligne est pour la caméra Ximea (1280 px)
            w[iLc, i] = (i - int(px_width/2) - ynew[i]) / p(xnew2[i]) + Lcc + L_offset# c'est pour la Andor, attention, il faut vérifier que ce soit bien int(px_width/2) au lieu de 640 pour la Ximea
        
    return w


def plot_spectrum(data, cam_spec_params, spectrograph_params):
    """
    plot the spectrum of the spectral cam

    Parameters
    ----------
    data : np.array
        2d array of the spectral cam.
    cam_spec_params: class.
        A class containing the spectral cam parameters
    spectrograph_params: class.
        A class containing the spectrograph parameters

    Returns
    -------
    None.

    """
    from matplotlib import pyplot as plt
    
    wavelengths = define_wavelengths_matrix(cam_spec_params, [(spectrograph_params.position, spectrograph_params.grating.current_grating_nbr)])

    i, j = np.unravel_index(data.argmax(), data.shape)
    print('maximum found at the row : ' + str(i))
    
    data_m = data[i, :]
    plt.figure()
    plt.plot(wavelengths[0, :], data_m)
    plt.xlabel('Lambda (nm)')
    plt.ylabel('Intensity [0 - 1023]')
    plt.grid()
    

class func_path:
    """
    A class that contain all the path to save the data
    
    Args:
        None
        
    # Return:
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
    fig_spatial_path: str
    
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
        self.fig_spatial_path = self.overview_path + '/spatial_arm' 






    
    