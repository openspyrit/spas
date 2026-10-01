#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 2026

@author: mahieu

Control of the Andor Shamrock spectrograph with the official Andor package
"pyAndorSpectrograph" (Andor SDK2, ATSpectrograph library).
The previous version, based on pylablib, is in spectro_ShamrockAndor_module_pylablib_pack.py

Install (in the conda env, from a writable copy of the folder):
    pip install "C:/Program Files/Andor SDK/Python/pyAndorSpectrograph"

Warning: Solis must be closed, otherwise the spectrograph is already in use.
"""

from pyAndorSpectrograph import ATSpectrograph

from dataclasses_json import dataclass_json
from dataclasses import dataclass, InitVar
from typing import Optional


# flipper mirrors (removable mirrors that select the input and the output port)
FLIPPERS = {'input': ATSpectrograph.INPUT_FLIPPER,
            'output': ATSpectrograph.OUTPUT_FLIPPER}

PORTS = {'direct': ATSpectrograph.DIRECT,
         'side': ATSpectrograph.SIDE}

PORT_NAMES = {value: key for key, value in PORTS.items()}

# slit index depending on the flipper and the port
SLITS = {('input', 'direct'):  ATSpectrograph.INPUT_DIRECT,
         ('input', 'side'):    ATSpectrograph.INPUT_SIDE,
         ('output', 'direct'): ATSpectrograph.OUTPUT_DIRECT,
         ('output', 'side'):   ATSpectrograph.OUTPUT_SIDE}


def check_return(spectrograph: ATSpectrograph, ret: int, func_name: str):
    """Raise an error if an ATSpectrograph function does not return SUCCESS.

    Args:
        spectrograph (ATSpectrograph):
            the object that control the spectrograph.
        ret (int):
            the code returned by the ATSpectrograph function.
        func_name (str):
            the name of the function, displayed in the error message.
    """
    if ret != ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
        (_, description) = spectrograph.GetFunctionReturnDescription(ret, 64)
        raise RuntimeError('Spectrograph: ' + func_name + ' failed, error ' + str(ret) + ' : ' + description)


def init_spectrograph(model: str = 'andor_shamrock', device: int = 0):
    """Initialize the communication with the spectrograph.

    Args:
        model (str):
            the model of the spectrograph. Only 'andor_shamrock' is accepted.
        device (int):
            the index of the spectrograph if several are connected. The default is 0.

    Returns:
        spectrograph (ATSpectrograph):
            An object containing all functions that control the spectrograph.
            The index of the spectrograph is stored in spectrograph.device
    """
    if model != 'andor_shamrock':
        print('Error, the model of the spectrograph must be : andor_shamrock. For another spectrograph, change the package that import the function init_spectrograph')
        return None

    spectrograph = ATSpectrograph()
    ret = spectrograph.Initialize("")
    if ret != ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
        (_, description) = spectrograph.GetFunctionReturnDescription(ret, 64)
        raise RuntimeError('Spectrograph: Initialize failed, error ' + str(ret) + ' : ' + description +
                           '. Check that the spectrograph is switched on and that Solis is closed.')

    (ret, nbr_devices) = spectrograph.GetNumberDevices()
    check_return(spectrograph, ret, 'GetNumberDevices')
    if device >= nbr_devices:
        spectrograph.Close()
        raise RuntimeError('Spectrograph: device ' + str(device) + ' not found, ' + str(nbr_devices) + ' spectrograph(s) detected')

    spectrograph.device = device
    (ret, serial_number) = spectrograph.GetSerialNumber(device, 64)
    check_return(spectrograph, ret, 'GetSerialNumber')
    print('Spectrograph ' + serial_number + ' connected')

    return spectrograph


def query_port(spectrograph: ATSpectrograph, flipper: str) -> Optional[str]:
    """Read the port selected by a flipper mirror.

    Args:
        flipper (str):
            'input' or 'output'.

    Returns:
        port (str):
            'direct' or 'side'. None if the flipper mirror is not present (only the direct port exists).
    """
    (ret, present) = spectrograph.IsFlipperMirrorPresent(spectrograph.device, FLIPPERS[flipper])
    check_return(spectrograph, ret, 'IsFlipperMirrorPresent')
    if not present:
        return None

    (ret, port) = spectrograph.GetFlipperMirror(spectrograph.device, FLIPPERS[flipper])
    check_return(spectrograph, ret, 'GetFlipperMirror')

    return PORT_NAMES[port]


def set_port(spectrograph: ATSpectrograph, flipper: str, port: str):
    """Move a flipper mirror to select the port.

    Args:
        flipper (str):
            'input' or 'output'.
        port (str):
            'direct' or 'side'.
    """
    if port not in PORTS:
        raise ValueError('port must be "direct" or "side", not "' + str(port) + '"')

    ret = spectrograph.SetFlipperMirror(spectrograph.device, FLIPPERS[flipper], PORTS[port])
    check_return(spectrograph, ret, 'SetFlipperMirror')


def query_slit_width(spectrograph: ATSpectrograph, flipper: str, port: str) -> Optional[float]:
    """Read the width (µm) of a motorized slit. Returns None if the slit is not motorized."""
    slit = SLITS[(flipper, port)]
    (ret, present) = spectrograph.IsSlitPresent(spectrograph.device, slit)
    check_return(spectrograph, ret, 'IsSlitPresent')
    if not present:
        return None

    (ret, width) = spectrograph.GetSlitWidth(spectrograph.device, slit)
    check_return(spectrograph, ret, 'GetSlitWidth')

    return round(width, 2)


@dataclass_json
@dataclass
class Grating:
    """Class containing the informations about the grating.

    Attributes:
        grooves:
            the number of grooves by mm
        blaze:
            the blaze wavelength (nm)
        current_grating_nbr:
            the grating currently used in the spectrograph
        number_of_grating:
            the number of available grating in the spectrograph
    """
    grooves: Optional[int] = None
    blaze: Optional[str] = None
    current_grating_nbr: Optional[int] = None
    number_of_grating: Optional[int] = None


def query_grating(spectrograph: ATSpectrograph) -> Grating:
    """Read the informations about the current grating."""
    device = spectrograph.device

    (ret, grating_nbr) = spectrograph.GetGrating(device)
    check_return(spectrograph, ret, 'GetGrating')

    (ret, number_of_grating) = spectrograph.GetNumberGratings(device)
    check_return(spectrograph, ret, 'GetNumberGratings')

    (ret, lines, blaze, home, offset) = spectrograph.GetGratingInfo(device, grating_nbr, 64)
    check_return(spectrograph, ret, 'GetGratingInfo')

    return Grating(grooves=round(lines),
                   blaze=blaze,
                   current_grating_nbr=grating_nbr,
                   number_of_grating=number_of_grating)


def query_position(spectrograph: ATSpectrograph) -> float:
    """Read the central wavelength (nm) of the grating."""
    (ret, wavelength) = spectrograph.GetWavelength(spectrograph.device)
    check_return(spectrograph, ret, 'GetWavelength')

    return round(wavelength, 2)


@dataclass_json
@dataclass
class Spectrograph_Parameters:
    """Class containing the Shamrock spectrograph parameters.

    Attributes:
        serial_number (str):
            the serial number of the spectrograph.
        position (float):
            the central wavelength of the grating (nm).
        grating (Grating):
            the informations about the current grating.
        input_port (str):
            the input port selected by the input flipper mirror: 'direct' or 'side'.
            None if no flipper mirror.
        output_port (str):
            the output port selected by the output flipper mirror: 'direct' or 'side'.
            None if no flipper mirror.
        slit_width (float):
            the width of the input slit (µm). Read from the spectrograph if the slit is motorized,
            otherwise given by the user in setup_spectrograph.
        slit_height (int):
            the height of the input slit (µm).
        resolution_th (float):
            the theoretical spectral resolution (nm).
        spectrograph (ATSpectrograph):
            An object containing all functions that control the spectrograph.
    """

    serial_number: Optional[str] = None
    position: Optional[float] = None
    grating: Optional[Grating] = None
    input_port: Optional[str] = None
    output_port: Optional[str] = None
    slit_width: Optional[float] = None
    slit_height: Optional[int] = 10000
    resolution_th: Optional[float] = None

    spectrograph: InitVar[ATSpectrograph] = None

    class_description: str = 'spectrograph Shamrock parameters'


    def __post_init__(self, spectrograph: Optional[ATSpectrograph] = None):
        """ Post initialization of attributes.

        Receives a spectrograph object and directly asks it for its configurations,
        then sets the Spectrograph_Parameters's attributes.
        During reconstruction from JSON, spectrograph is set to None and the function
        does nothing, letting initialization for the standard __init__ function.

        Args:
            spectrograph (ATSpectrograph, optional):
                Defaults to None.
        """
        if spectrograph is None:
            return

        (ret, self.serial_number) = spectrograph.GetSerialNumber(spectrograph.device, 64)
        check_return(spectrograph, ret, 'GetSerialNumber')

        self.position      = query_position(spectrograph)
        self.grating       = query_grating(spectrograph)
        self.input_port    = query_port(spectrograph, 'input')
        self.output_port   = query_port(spectrograph, 'output')
        # without flipper mirror, only the direct port exists
        self.slit_width    = query_slit_width(spectrograph, 'input', self.input_port or 'direct')
        self.resolution_th = 0.378 * 300 / self.grating.grooves


    @staticmethod
    def readable_class_spectro(spectro_params_dict: dict) -> dict:
        """Flattens the "grating" dictionary into "grating.xxx" keys to be readable in the metadata."""
        readable_spectro_dict = {}
        for key, value in spectro_params_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    readable_spectro_dict[key + '.' + sub_key] = sub_value
            else:
                readable_spectro_dict[key] = value

        return readable_spectro_dict


    @staticmethod
    def undo_readable_class_spectro(readable_spectro_dict: dict) -> dict:
        """Inverse of readable_class_spectro: gathers the "grating.xxx" keys into a "grating" dictionary."""
        spectro_params_dict = {}
        for key, value in readable_spectro_dict.items():
            if '.' in key:
                key, sub_key = key.split('.', 1)
                spectro_params_dict.setdefault(key, {})[sub_key] = value
            else:
                spectro_params_dict[key] = value

        return spectro_params_dict


def setup_spectrograph(spectrograph: ATSpectrograph,
                       grating_nbr: int = 1, print_select: bool = False,
                       position: float = 600, print_position: bool = False,
                       slit_width: Optional[float] = None,
                       input_port: Optional[str] = None, output_port: Optional[str] = None,
                       print_ports: bool = False):
    """ Setup the spectrograph to tune the cameras

    Parameters
    ----------
    spectrograph (ATSpectrograph):
        An object containing all functions that control the spectrograph.
    grating_nbr (int):
        the number of the selected grating. The default is 1.
    print_select (bool):
        a boolean to print or not the slected grating. The default is False.
    position (float). The default is 600:
        the position of the central wavelength of the grating (nm)
    print_position : bool, optional
        a boolean to print or not the position of the central wavenlength of the grating. The default is False.
    slit_width (float):
        the width of the input slit in µm. If the slit is motorized, the slit is moved,
        otherwise the value is only stored in the metadata. None (default) : nothing to do.
    input_port (str):
        'direct' or 'side', the input port selected by the input flipper mirror.
        None (default) : the port is not changed.
    output_port (str):
        'direct' or 'side', the output port selected by the output flipper mirror.
        None (default) : the port is not changed.
    print_ports (bool):
        a boolean to print or not the configuration of the input and output ports. The default is False.

    Returns
    -------
    Spectrograph_Parameters (obj)
        the metadata containing the spectrograph parameters.

    """
    device = spectrograph.device

    # input and output ports (flipper mirrors)
    for flipper, port in (('input', input_port), ('output', output_port)):
        current_port = query_port(spectrograph, flipper)
        if current_port is None:
            if port not in (None, 'direct'):
                print('No ' + flipper + ' flipper mirror, only the direct ' + flipper + ' port is available')
            elif print_ports:
                print(flipper + ' port : direct (no flipper mirror)')
        elif port is not None and port != current_port:
            set_port(spectrograph, flipper, port)
            if print_ports:
                print(flipper + ' port changed from ' + current_port + ' to ' + query_port(spectrograph, flipper))
        elif print_ports:
            print(flipper + ' port : ' + current_port)

    # grating
    current_grating = query_grating(spectrograph)
    if current_grating.current_grating_nbr == grating_nbr:
        if print_select:
            print(str(current_grating.grooves) + ' gr/mm and blaze wavelength = ' + current_grating.blaze + ' nm grating already selectionned. Nothing to do')
    else:
        ret = spectrograph.SetGrating(device, grating_nbr)
        check_return(spectrograph, ret, 'SetGrating')
        if print_select:
            current_grating = query_grating(spectrograph)
            print(str(current_grating.grooves) + ' gr/mm and blaze wavelength = ' + current_grating.blaze + ' nm grating selectionned.')

    # central wavelength
    current_position = query_position(spectrograph)
    if abs(current_position - position) < 0.01:
        if print_position:
            print('grating position already set to : ' + str(position) + ' nm. Nothing to do')
    else:
        ret = spectrograph.SetWavelength(device, position)
        check_return(spectrograph, ret, 'SetWavelength')
        if print_position:
            print('grating position set to : ' + str(query_position(spectrograph)) + ' nm.')

    # input slit
    current_input_port = query_port(spectrograph, 'input') or 'direct'
    if slit_width is not None and query_slit_width(spectrograph, 'input', current_input_port) is not None:
        ret = spectrograph.SetSlitWidth(device, SLITS[('input', current_input_port)], slit_width)
        check_return(spectrograph, ret, 'SetSlitWidth')

    spectrograph_params = Spectrograph_Parameters(spectrograph = spectrograph)
    if spectrograph_params.slit_width is None:
        # manual slit, the value given by the user is stored
        spectrograph_params.slit_width = slit_width
    elif print_ports:
        print('input slit width : ' + str(spectrograph_params.slit_width) + ' µm')

    return spectrograph_params


def disconnect_spectrograph(spectrograph: ATSpectrograph, goto_zero: bool = False):
    """ disconnect the spectrograph

    Parameters
    ----------
    spectrograph (ATSpectrograph):
        An object containing all functions that control the spectrograph.
    goto_zero (bool):
        a boolean to send the grating at the zero order. The default is False.

    Returns
    -------
    None.

    """
    if goto_zero:
        ret = spectrograph.GotoZeroOrder(spectrograph.device)
        check_return(spectrograph, ret, 'GotoZeroOrder')

    ret = spectrograph.Close()
    check_return(spectrograph, ret, 'Close')
    print('Spectrograph is disconnected')
