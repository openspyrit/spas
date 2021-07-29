# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""Acquisition utility functions.

Typical usage example:

# Initialization
spectrometer, DMD, DMD_initial_memory = init()

# Acquisition
metadata = MetaData(
    output_directory=Path('./data/...'),
    pattern_order_source=Path('./communication/communication.txt'),
    pattern_source=Path('./Patterns/...),
    pattern_prefix='Hadamard_64x64'
    experiment_name='...',
    light_source='...',
    object='...',
    filter='...',
    description='...')

acquisition_parameters = AcquisitionParameters(
    pattern_compression=1.0,
    pattern_dimension_x=64,
    pattern_dimension_y=64)

spectrometer_params, DMD_params, wavelenghts = setup(
    spectrometer=spectrometer, 
    DMD=DMD,
    DMD_initial_memory=DMD_initial_memory,
    metadata=metadata, 
    acquisition_params=acquisition_parameters,
    integration_time=1.0,)

acquire(
    spectrometer,
    DMD, 
    metadata, 
    spectrometer_params, 
    DMD_params, 
    acquisition_parameters, 
    wavelenghts)

# Disconnect
disconnect(spectrometer, DMD)
"""

import warnings
from time import sleep, perf_counter_ns
from typing import NamedTuple, Tuple, List, Optional
from collections import namedtuple
from pathlib import Path
from multiprocessing import Process, Queue
    
import numpy as np
from PIL import Image
from msl.equipment import EquipmentRecord, ConnectionRecord, Backend
from msl.equipment.resources.avantes import MeasureCallback, Avantes
from ALP4 import ALP4, ALP_FIRSTFRAME, ALP_LASTFRAME
from ALP4 import ALP_AVAIL_MEMORY
from tqdm import tqdm

from .metadata import DMDParameters, MetaData, AcquisitionParameters
from .metadata import SpectrometerParameters, save_metadata
from .generate import generate_hadamard_order
from .reconstruction_nn import reconstruct_process, plot_recon, ReconstructionParameters


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
    print('Connected to spectrometer')

    return ava


def _init_DMD() -> Tuple[ALP4, int]:
    """Initialize a DMD and clean its allocated memory from a previous use.

    Returns:
        Tuple[ALP4, int]: Tuple containing initialized DMD object and DMD
        initial available memory.
    """

    # Initializing DMD

    dll_path = Path(__file__).parent.parent.joinpath('lib/alpV42').__str__()

    DMD = ALP4(version='4.2',libDir=dll_path)
    DMD.Initialize(DeviceNum=None)

    print(f'DMD initial available memory: {DMD.DevInquire(ALP_AVAIL_MEMORY)}')

    return DMD, DMD.DevInquire(ALP_AVAIL_MEMORY)


def init() -> Tuple[Avantes, ALP4, int]:
    """Call functions to initialize spectrometer and DMD.

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
    
    DMD, DMD_initial_memory = _init_DMD()
    return _init_spectrometer(), DMD, DMD_initial_memory


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
    
    print(f'\nThe spectrometer has {initial_available_pixels} pixels')

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


def _sequence_limits(DMD: ALP4, pattern_compression: int, 
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
    DMD.SeqControl(ALP_FIRSTFRAME, 0)

    # Choosing the end of the sequence
    if (round(pattern_compression * sequence_lenght) % 2 == 0) or not (pos_neg):
        frames = round(pattern_compression * sequence_lenght)
    else:
        frames = round(pattern_compression * sequence_lenght) + 1

    DMD.SeqControl(ALP_LASTFRAME, frames - 1)

    return frames


def _update_sequence(DMD: ALP4,
                     pattern_source: str,
                     pattern_prefix: str,
                     pattern_order: List[int],
                     bitplanes: int = 1):
    """Send new complete pattern sequence to DMD.

    Args:
        DMD (ALP4):
            Connected DMD object.
        pattern_source (str):
            Pattern source folder.
        pattern_preffix (str):
            Prefix used in pattern naming.
        pattern_order (List[int]):
            List of the pattern indices in a certain order for upload to DMD.
        bitplanes (int, optional): 
            Pattern bitplanes. Defaults to 1.
    """

    path_base = Path(pattern_source)

    seqId = DMD.SeqAlloc(nbImg=len(pattern_order), 
                             bitDepth=bitplanes)

    print(f'Pattern order size: {len(pattern_order)}')   
    t = perf_counter_ns()
    for index,pattern_name in enumerate(tqdm(pattern_order, unit=' patterns')):
        path = path_base.joinpath(f'{pattern_prefix}_{pattern_name}.png')
        image = Image.open(path)
        
        # Converting image to nparray and transforming it to a 1D vector (ravel)
        patterns = np.array(image,dtype=np.uint8).ravel()
        DMD.SeqPut(
            imgData=patterns.copy(),
            PicOffset=index, 
            PicLoad=1)

    print(f'\nTime for sending all patterns: '
          f'{(perf_counter_ns() - t)/1e+9} s')


def _setup_patterns(DMD: ALP4, metadata: MetaData, DMD_params: DMDParameters, 
                   acquisition_params: AcquisitionParameters,
                   cov_path: str = None, pos_neg: bool = True) -> None:
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
            outputs. Must be create and filled up by the user.
        DMD_params (DMDParameters):
            DMD metadata object to be updated with pattern related data and with
            memory available after patterns are sent to DMD.
        acquisition_params (AcquisitionParameters):
            Acquisition related metadata object. User must partially fill up
            with pattern_compression, pattern_dimension_x, pattern_dimension_y.
        pos_neg (bool):
            Boolean indicating if sequence is formed by positive and negative
            patterns. Default is True.
    """

    pattern_order_source = Path(metadata.pattern_order_source)

    if pattern_order_source.suffix == '.txt':

        # Pattern order is written directly in a text file
        file = open(pattern_order_source)
        pattern_order = file.readlines()[0].split(';')
        pattern_order.remove('')
        file.close()
        pattern_order = [int(pattern) for pattern in pattern_order]
        print(f'Found {len(pattern_order)} patterns')
    
    elif pattern_order_source.suffix == '.npy':

        # Pattern order needs to be calculated from covariance matrix
        pattern_order = generate_hadamard_order(64, pattern_order_source, pos_neg)

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

        _update_sequence(DMD, metadata.pattern_source, metadata.pattern_prefix, 
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


def _setup_timings(DMD: ALP4, DMD_params: DMDParameters, picture_time: int, 
                  illumination_time: int, synch_pulse_delay: int, 
                  synch_pulse_width: int, trigger_in_delay: int,
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
          pos_neg: bool = True,
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
            outputs. Must be create and filled up by the user.
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
        pos_neg (bool):
            Boolean indicating if sequence is formed by positive and negative
            patterns. Default is True.
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

    acquisition_params.wavelengths = np.asarray(wavelenghts, dtype=np.float32)

    DMD_params = _setup_DMD(DMD, add_illumination_time, DMD_initial_memory)

    _setup_patterns(DMD=DMD, metadata=metadata, DMD_params=DMD_params, 
                    acquisition_params=acquisition_params, pos_neg=pos_neg)
    _setup_timings(DMD, DMD_params, picture_time, illumination_time, 
                   DMD_output_synch_pulse_delay, synch_pulse_width, 
                   DMD_trigger_in_delay, add_illumination_time)

    return spectrometer_params, DMD_params

 
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

    # Saving metadata
    save_metadata(metadata, 
                  DMD_params,
                  spectrometer_params,
                  acquisition_parameters)


def _acquire_raw(ava: Avantes, 
            DMD: ALP4,
            spectrometer_params: SpectrometerParameters, 
            DMD_params: DMDParameters, 
            acquisition_params: AcquisitionParameters
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

    #spectro.register_callback(-2,acquisition_params.pattern_amount,pixel_amount)
    callback = register_callback(measurement_time, timestamps, 
                                 spectral_data, ava)
    measurement_callback = MeasureCallback(callback)
    ava.measure_callback(-2, measurement_callback)
    
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
            sleep(acquisition_params.pattern_amount *
            DMD_params.picture_time_us / 1e+6 / 10)

    ava.stop_measure()
    DMD.Halt()

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
    previosly chosen.
    Finnaly saves all acqusition related data and metadata.

    Args:
        ava (Avantes): 
            Connected spectrometer (Avantes object).
        DMD (ALP4): 
            Connected DMD.
        metadata (MetaData): 
            Metadata concerning the experiment, paths, file inputs and file 
            outputs. Must be create and filled up by the user.
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
        dtype=np.float32)
    spectral_data = np.zeros(
        (acquisition_params.pattern_amount * repetitions,pixel_amount),
        dtype=np.float64)

    acquisition_params.acquired_spectra = 0
    print()

    for repetition in range(repetitions):
        if verbose:
            print(f"Acquisition {repetition}")

        AcquisitionResults = _acquire_raw(ava, DMD, spectrometer_params, 
            DMD_params, acquisition_params)
    
        (data, spectrum_index, timestamp, time,
            start_measurement_time, saturation_detected) = AcquisitionResults

        print('Data acquired')

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

        # Print data for each repetition only if there are not too many repetitions
        if (verbose) and repetitions <= 10:
            if saturation_detected is True:
                print('Saturation detected!')
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
    print('Complete acquisition results:')
    print('Spectra acquired: {}'.format(
        acquisition_params.acquired_spectra))
    if acquisition_params.saturation_detected is True:
        print('Saturation detected!')
    print('Mean callback acquisition time: {} ms'.format(
        acquisition_params.mean_callback_acquisition_time_ms))
    print('Total callback acquisition time: {} s'.format(
        acquisition_params.total_callback_acquisition_time_s))
    print('Mean spectrometer acquisition time: {} ms'.format(
        acquisition_params.mean_spectrometer_acquisition_time_ms))
    print('Total spectrometer acquisition time: {} s'.format(
        acquisition_params.total_spectrometer_acquisition_time_s))
    print(f'Acquisition matrix dimension: {spectral_data.shape}')

    print(f'Saving data to {metadata.output_directory}')
    
    _save_acquisition(metadata, DMD_params, spectrometer_params, 
                        acquisition_params, spectral_data)

    # Joining processes and closing queues
    if reconstruct == True:
        queue_to_recon.put('kill') # Sends a message to stop reconstruction
        recon_process.join()
        queue_to_recon.close()
        plot_process.join()
        queue_reconstructed.close()

    return spectral_data


def disconnect(ava: Optional[Avantes]=None, DMD: Optional[ALP4]=None):
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

    if DMD is not None:
       
        # Stop the sequence display
        DMD.Halt()

        # Free the sequence from the onboard memory (if any is present)
        if (DMD.Seqs):
            DMD.FreeSeq()

        DMD.Free()