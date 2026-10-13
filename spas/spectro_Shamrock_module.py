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

Example:
    spectrograph = Shamrock()
    spectrograph.init(model = 'andor_shamrock', SN = 'SR-2070')
    spectrograph_params = spectrograph.setup(grating_nbr = 1, position = 600, input_port = 'side')
    spectrograph.disconnect(goto_zero = False)
"""

from pyAndorSpectrograph import ATSpectrograph

import ctypes
import time
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

# connection: number of retries before resetting the USB port, and delay (s) to let
# Windows enumerate the spectrograph again after the reset
CONNECT_RETRIES = 2
USB_RESET_DELAY = 5


def reset_ftdi_port(SN: Optional[str] = None) -> str:
    """Reset the USB port of the spectrograph (equivalent to unplug / plug the USB cable).

    The spectrograph is connected through an FTDI chip, controlled here with the FTDI
    library ftd2xx.dll (installed in System32 with the FTDI driver).

    Args:
        SN (str):
            the serial number of the spectrograph. The FTDI serial number starts with it
            (ex: 'SR-2070A' for SN = 'SR-2070'). If None, all FTDI devices whose
            description contains 'Shamrock' are reset.

    Returns:
        str: a message describing what has been done.
    """
    try:
        ftdi = ctypes.WinDLL('ftd2xx.dll')
    except OSError as error:
        return 'USB reset impossible, ftd2xx.dll not found (' + str(error) + ')'

    FT_OK = 0
    FT_OPEN_BY_SERIAL_NUMBER = 1

    nbr_devices = ctypes.c_ulong()
    if ftdi.FT_CreateDeviceInfoList(ctypes.byref(nbr_devices)) != FT_OK:
        return 'USB reset impossible, FT_CreateDeviceInfoList failed'

    reset_serials = []
    for index in range(nbr_devices.value):
        flags, dev_type, dev_id, loc_id = ctypes.c_ulong(), ctypes.c_ulong(), ctypes.c_ulong(), ctypes.c_ulong()
        serial = ctypes.create_string_buffer(16)
        description = ctypes.create_string_buffer(64)
        handle = ctypes.c_void_p()
        if ftdi.FT_GetDeviceInfoDetail(index, ctypes.byref(flags), ctypes.byref(dev_type), ctypes.byref(dev_id),
                                       ctypes.byref(loc_id), serial, description, ctypes.byref(handle)) != FT_OK:
            continue
        serial = serial.value.decode(errors = 'replace')
        description = description.value.decode(errors = 'replace')
        if (SN is not None and not serial.startswith(SN)) or (SN is None and 'shamrock' not in description.lower()):
            continue

        handle = ctypes.c_void_p()
        if ftdi.FT_OpenEx(serial.encode(), FT_OPEN_BY_SERIAL_NUMBER, ctypes.byref(handle)) != FT_OK:
            continue
        ftdi.FT_CyclePort(handle)
        ftdi.FT_Close(handle)
        reset_serials.append(serial)

    if not reset_serials:
        return 'USB reset impossible, no FTDI device found for SN = ' + str(SN)
    return 'USB port reset for FTDI device(s) ' + str(reset_serials)


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


class Shamrock:
    """Class that controls the Andor Shamrock spectrograph.

    Attributes:
        sdk (ATSpectrograph):
            the Andor object containing all functions of the spectrograph library.
            Can be used directly for functions not implemented here.
        device (int):
            the index of the spectrograph.
    """

    def __init__(self):
        self.sdk = None
        self.device = None


    def check_return(self, ret: int, func_name: str):
        """Raise an error if an ATSpectrograph function does not return SUCCESS.

        Args:
            ret (int):
                the code returned by the ATSpectrograph function.
            func_name (str):
                the name of the function, displayed in the error message.
        """
        if ret != ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
            (_, description) = self.sdk.GetFunctionReturnDescription(ret, 64)
            raise RuntimeError('Spectrograph: ' + func_name + ' failed, error ' + str(ret) + ' : ' + description)


    def init(self, model: str = 'andor_shamrock', SN: Optional[str] = None, device: int = 0):
        """Initialize the communication with the spectrograph.

        Args:
            model (str):
                the model of the spectrograph. Only 'andor_shamrock' is accepted.
            SN (str):
                the serial number of the spectrograph. If given, the spectrograph is searched
                by its serial number and device is not used. The default is None.
            device (int):
                the index of the spectrograph if several are connected and SN is None. The default is 0.
        """
        if model != 'andor_shamrock':
            print('Error, the model of the spectrograph must be : andor_shamrock. For another spectrograph, change the package that import the class Shamrock')
            return

        self.sdk = ATSpectrograph()

        # After the spectrograph is switched on, the first opening of its FTDI USB chip often
        # fails (Solis retries internally). So: retry, then reset the USB port and retry again.
        error = self._try_connect(SN, device)
        for attempt in range(1, CONNECT_RETRIES + 1):
            if error is None:
                break
            print('Spectrograph: ' + error + ' -> retry ' + str(attempt) + '/' + str(CONNECT_RETRIES))
            time.sleep(1)
            error = self._try_connect(SN, device)

        if error is not None:
            print('Spectrograph: ' + error + ' -> reset of the USB port of the spectrograph')
            reset_message = reset_ftdi_port(SN)
            print('Spectrograph: ' + reset_message)
            time.sleep(USB_RESET_DELAY)
            error = self._try_connect(SN, device)

        if error is not None:
            self.sdk = None
            raise RuntimeError('Spectrograph: ' + error + '. Check that the spectrograph is switched on, '
                               'that Solis is closed and that no other Python console is connected to it.')

        print('Spectrograph Andor Shamrock, SN = ' + self.get_serial_number() + ' connected')


    def _try_connect(self, SN: Optional[str], device: int) -> Optional[str]:
        """Initialize the ATSpectrograph library and search for the spectrograph.

        Args:
            SN (str):
                the serial number of the spectrograph, or None to use device.
            device (int):
                the index of the spectrograph, used if SN is None.

        Returns:
            Optional[str]: None if the spectrograph is connected (self.device is set),
            else the error message. In case of error, the library is closed.
        """
        ret = self.sdk.Initialize("")
        if ret != ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
            (_, description) = self.sdk.GetFunctionReturnDescription(ret, 64)
            self.sdk.Close()
            return 'Initialize failed, error ' + str(ret) + ' : ' + description

        (ret, nbr_devices) = self.sdk.GetNumberDevices()
        serial_numbers = []
        if ret == ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
            for index in range(nbr_devices):
                (ret_sn, serial_number) = self.sdk.GetSerialNumber(index, 64)
                serial_numbers.append(serial_number if ret_sn == ATSpectrograph.ATSPECTROGRAPH_SUCCESS else '?')

        if SN is not None:
            if SN in serial_numbers:
                self.device = serial_numbers.index(SN)
                return None
            self.sdk.Close()
            return 'SN = ' + SN + ' not found. Spectrograph(s) detected : ' + str(serial_numbers)

        if device < len(serial_numbers):
            self.device = device
            return None
        self.sdk.Close()
        return 'device ' + str(device) + ' not found. Spectrograph(s) detected : ' + str(serial_numbers)


    def get_serial_number(self) -> str:
        """Read the serial number of the spectrograph."""
        (ret, serial_number) = self.sdk.GetSerialNumber(self.device, 64)
        self.check_return(ret, 'GetSerialNumber')

        return serial_number


    def get_port(self, flipper: str) -> Optional[str]:
        """Read the port selected by a flipper mirror.

        Args:
            flipper (str):
                'input' or 'output'.

        Returns:
            port (str):
                'direct' or 'side'. None if the flipper mirror is not present (only the direct port exists).
        """
        (ret, present) = self.sdk.IsFlipperMirrorPresent(self.device, FLIPPERS[flipper])
        self.check_return(ret, 'IsFlipperMirrorPresent')
        if not present:
            return None

        (ret, port) = self.sdk.GetFlipperMirror(self.device, FLIPPERS[flipper])
        self.check_return(ret, 'GetFlipperMirror')

        return PORT_NAMES[port]


    def set_port(self, flipper: str, port: str):
        """Move a flipper mirror to select the port.

        Args:
            flipper (str):
                'input' or 'output'.
            port (str):
                'direct' or 'side'.
        """
        if port not in PORTS:
            raise ValueError('port must be "direct" or "side", not "' + str(port) + '"')

        ret = self.sdk.SetFlipperMirror(self.device, FLIPPERS[flipper], PORTS[port])
        self.check_return(ret, 'SetFlipperMirror')


    def _slit_index(self, flipper: str, port: Optional[str]) -> int:
        """Return the index of the slit. If port is None, the port currently selected is used."""
        if port is None:
            # without flipper mirror, only the direct port exists
            port = self.get_port(flipper) or 'direct'

        return SLITS[(flipper, port)]


    def get_slit_width(self, flipper: str = 'input', port: Optional[str] = None) -> Optional[float]:
        """Read the width (µm) of a motorized slit.

        Args:
            flipper (str):
                'input' (default) or 'output'.
            port (str):
                'direct' or 'side'. None (default): the port currently selected.

        Returns:
            width (float):
                the slit width (µm). None if the slit is not motorized.
        """
        slit = self._slit_index(flipper, port)
        (ret, present) = self.sdk.IsSlitPresent(self.device, slit)
        self.check_return(ret, 'IsSlitPresent')
        if not present:
            return None

        (ret, width) = self.sdk.GetSlitWidth(self.device, slit)
        self.check_return(ret, 'GetSlitWidth')

        return round(width, 2)


    def set_slit_width(self, width: float, flipper: str = 'input', port: Optional[str] = None):
        """Set the width (µm) of a motorized slit.

        Args:
            width (float):
                the slit width (µm).
            flipper (str):
                'input' (default) or 'output'.
            port (str):
                'direct' or 'side'. None (default): the port currently selected.
        """
        ret = self.sdk.SetSlitWidth(self.device, self._slit_index(flipper, port), width)
        self.check_return(ret, 'SetSlitWidth')


    def get_grating(self) -> Grating:
        """Read the informations about the current grating."""
        (ret, grating_nbr) = self.sdk.GetGrating(self.device)
        self.check_return(ret, 'GetGrating')

        (ret, number_of_grating) = self.sdk.GetNumberGratings(self.device)
        self.check_return(ret, 'GetNumberGratings')

        (ret, lines, blaze, home, offset) = self.sdk.GetGratingInfo(self.device, grating_nbr, 64)
        self.check_return(ret, 'GetGratingInfo')

        return Grating(grooves=round(lines),
                       blaze=blaze,
                       current_grating_nbr=grating_nbr,
                       number_of_grating=number_of_grating)


    def set_grating(self, grating_nbr: int):
        """Select the grating (the turret rotates)."""
        ret = self.sdk.SetGrating(self.device, grating_nbr)
        self.check_return(ret, 'SetGrating')


    def get_position(self) -> float:
        """Read the central wavelength (nm) of the grating."""
        (ret, wavelength) = self.sdk.GetWavelength(self.device)
        self.check_return(ret, 'GetWavelength')

        return round(wavelength, 2)


    def set_position(self, position: float):
        """Move the grating to the central wavelength position (nm)."""
        ret = self.sdk.SetWavelength(self.device, position)
        self.check_return(ret, 'SetWavelength')


    def setup(self,
              grating_nbr: int = 1, print_select: bool = False,
              position: float = 600, print_position: bool = False,
              slit_width: Optional[float] = None,
              input_port: Optional[str] = None, output_port: Optional[str] = None,
              print_ports: bool = False):
        """ Setup the spectrograph to tune the cameras

        Parameters
        ----------
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
        # input and output ports (flipper mirrors)
        for flipper, port in (('input', input_port), ('output', output_port)):
            current_port = self.get_port(flipper)
            if current_port is None:
                if port not in (None, 'direct'):
                    print('No ' + flipper + ' flipper mirror, only the direct ' + flipper + ' port is available')
                elif print_ports:
                    print(flipper + ' port : direct (no flipper mirror)')
            elif port is not None and port != current_port:
                self.set_port(flipper, port)
                if print_ports:
                    print(flipper + ' port changed from ' + current_port + ' to ' + self.get_port(flipper))
            elif print_ports:
                print(flipper + ' port : ' + current_port)

        # grating
        current_grating = self.get_grating()
        if current_grating.current_grating_nbr == grating_nbr:
            if print_select:
                print(str(current_grating.grooves) + ' gr/mm and blaze wavelength = ' + current_grating.blaze + ' nm grating already selectionned. Nothing to do')
        else:
            self.set_grating(grating_nbr)
            if print_select:
                current_grating = self.get_grating()
                print(str(current_grating.grooves) + ' gr/mm and blaze wavelength = ' + current_grating.blaze + ' nm grating selectionned.')

        # central wavelength
        current_position = self.get_position()
        if abs(current_position - position) < 0.01:
            if print_position:
                print('grating position already set to : ' + str(position) + ' nm. Nothing to do')
        else:
            self.set_position(position)
            if print_position:
                print('grating position set to : ' + str(self.get_position()) + ' nm.')

        # input slit
        if slit_width is not None and self.get_slit_width('input') is not None:
            self.set_slit_width(slit_width, 'input')

        spectrograph_params = Spectrograph_Parameters(spectrograph = self)
        if spectrograph_params.slit_width is None:
            # manual slit, the value given by the user is stored
            spectrograph_params.slit_width = slit_width
        elif print_ports:
            print('input slit width : ' + str(spectrograph_params.slit_width) + ' µm')

        return spectrograph_params


    def disconnect(self, goto_zero: bool = False):
        """ disconnect the spectrograph

        Parameters
        ----------
        goto_zero (bool):
            a boolean to send the grating at the zero order. The default is False.
        """
        if goto_zero:
            ret = self.sdk.GotoZeroOrder(self.device)
            self.check_return(ret, 'GotoZeroOrder')

        ret = self.sdk.Close()
        self.check_return(ret, 'Close')
        self.sdk = None
        self.device = None
        print('Spectrograph is disconnected')


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
            otherwise given by the user in Shamrock.setup.
        slit_height (int):
            the height of the input slit (µm).
        resolution_th (float):
            the theoretical spectral resolution (nm).
        spectrograph (Shamrock):
            the object that controls the spectrograph.
    """

    serial_number: Optional[str] = None
    position: Optional[float] = None
    grating: Optional[Grating] = None
    input_port: Optional[str] = None
    output_port: Optional[str] = None
    slit_width: Optional[float] = None
    slit_height: Optional[int] = 10000
    resolution_th: Optional[float] = None

    spectrograph: InitVar[Shamrock] = None

    class_description: str = 'spectrograph Shamrock parameters'


    def __post_init__(self, spectrograph: Optional[Shamrock] = None):
        """ Post initialization of attributes.

        Receives a spectrograph object and directly asks it for its configurations,
        then sets the Spectrograph_Parameters's attributes.
        During reconstruction from JSON, spectrograph is set to None and the function
        does nothing, letting initialization for the standard __init__ function.

        Args:
            spectrograph (Shamrock, optional):
                Defaults to None.
        """
        if spectrograph is None:
            return

        self.serial_number = spectrograph.get_serial_number()
        self.position      = spectrograph.get_position()
        self.grating       = spectrograph.get_grating()
        self.input_port    = spectrograph.get_port('input')
        self.output_port   = spectrograph.get_port('output')
        self.slit_width    = spectrograph.get_slit_width('input')
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
