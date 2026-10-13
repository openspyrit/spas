# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 2026

@author: mahieu

Laser combiner (6 lasers), driven through the creatis_pilot_combiner package.
"""

import sys
from dataclasses import dataclass, field
from typing import List, Optional
from dataclasses_json import dataclass_json

COMBINER_LIB_PATH = r"D:\materiel\gitLab\toolbox\Combiner\python_lib\src"
if COMBINER_LIB_PATH not in sys.path:
    sys.path.append(COMBINER_LIB_PATH)

try:
    from creatis_pilot_combiner import Combiner as CombinerDriver
except ImportError:   # package only needed to drive the lasers, not to read the metadata
    CombinerDriver = None


@dataclass_json
@dataclass
class Combiner_Parameters:
    """
    Class containing the laser combiner parameters, read from the lasers.
    lasers : one dict per connected laser, sorted by increasing wavelength :
        {'wavelength_nm', 'max_power_mW', 'power_mW', 'power_percent', 'state' ('ON' or 'OFF')}
    """
    class_description: str = 'laser combiner parameters'
    model: Optional[str] = None
    serial_number: Optional[str] = None
    lasers: List[dict] = field(default_factory = list)


class Combiner:
    """Class that controls the laser combiner. The lasers are detected automatically on the serial ports
    and called by their wavelength (nm): the closest connected wavelength within WAVELENGTH_TOLERANCE_NM is used.

    Example:
        combi = Combiner()
        combi.init(model = 'combiner', SN = '2')
        combi.laser_on(405)
        combi.set_power(405, 10, unit = '%')
        combi.laser_off(405)
        combi_params = combi.get_params()   # metadata : power and ON/OFF state of each laser
        combi.disconnect()
    """

    WAVELENGTH_TOLERANCE_NM = 5

    def __init__(self):

        self.device = None
        self.model = None
        self.serial_number = None
        self.lasers = {}   # {wavelength (nm): {'max_power': (mW), 'idn': combiner idn}}, sorted by increasing wavelength


    def init(self, model: str = 'combiner', SN: str = '2', verbose: bool = True):
        """
        Initialize the laser combiner and detect the connected lasers

        Parameters
        ----------
        model : str
            the model of the combiner. The default is 'combiner'.
        SN : str
            the identifier of the combiner. The default is '2'.
        verbose : bool
            print the connected lasers. The default is True.
            The detected lasers are stored in self.lasers.
        """
        if CombinerDriver is None:
            raise ImportError('creatis_pilot_combiner package not found in ' + COMBINER_LIB_PATH)
        self.device = CombinerDriver()
        self.device.init_lasers()
        self.model = model
        self.serial_number = str(SN)

        if self.device.n_laser == 0:
            print('Warning, combiner ' + model + ', id = ' + self.serial_number + ' : no laser detected')

        all_lasers = [laser for key in self.device.laser for laser in self.device.laser[key]]
        all_lasers.sort(key = lambda laser: laser.wavelength)
        self.lasers = {}
        for laser in all_lasers:
            self.lasers[laser.wavelength] = {'max_power': laser.max_power, 'idn': laser.idn}

        print('Combiner ' + model + ', id = ' + self.serial_number + ' connected')
        if verbose:
            self.print_lasers()

    def print_lasers(self):

        for nbr, (wavelength, laser) in enumerate(self.lasers.items(), start = 1):
            max_power = '?' if laser['max_power'] is None else f"{laser['max_power']:g}"
            print('Laser ' + str(nbr) + ': (' + f"{wavelength:g}" + ' nm, ' + max_power + ' mW) connected')

    def _idn(self, wavelength: float):
        """Return the combiner idn of the connected laser closest to wavelength (nm)."""
        available = list(self.lasers.keys())
        if available:
            closest = min(available, key = lambda w: abs(w - wavelength))
            if abs(closest - wavelength) <= self.WAVELENGTH_TOLERANCE_NM:
                return self.lasers[closest]['idn']
        raise ValueError('Laser ' + f"{wavelength:g}" + ' nm not connected, available wavelengths (nm) : ' +
                         str([f"{w:g}" for w in available]))

    def _laser(self, wavelength: float):
        """Return the laser object of the driver for the given wavelength (nm)."""
        key, index = self.device.get_laser_key_and_index(idn = self._idn(wavelength))
        return self.device.laser[key][index]

    def _state(self, laser):
        """Read the output state of a laser : TTL line if configured, else the laser itself."""
        ttl_port = getattr(laser, 'ttl_port', None)
        if ttl_port is not None and self.device.ni_board is not None:
            return self.device.ni_board.get_state(ttl_port)
        if hasattr(laser, 'get_emission_state'):   # Obis, Zouk
            return laser.get_emission_state()
        return laser.output                        # LuxX : last state sent

    def get_params(self, verbose: bool = False):
        """
        Read the power and the ON/OFF state of all the connected lasers (to be called just before
        the acquisition, as the powers and states change during the program)

        Parameters
        ----------
        verbose : bool
            print the parameters. The default is False.

        Returns
        -------
        combi_params : Combiner_Parameters
        """
        combi_params = Combiner_Parameters(model = self.model, serial_number = self.serial_number)
        for wavelength, laser_info in self.lasers.items():
            laser = self._laser(wavelength)
            try:
                laser.get_power_level()
            except Exception as e:
                print('Warning, power of the laser ' + f"{wavelength:g}" + ' nm not read (' + str(e) + '), last value kept')
            try:
                state = 'ON' if self._state(laser) else 'OFF'
            except Exception as e:
                print('Warning, state of the laser ' + f"{wavelength:g}" + ' nm not read (' + str(e) + ')')
                state = '?'
            combi_params.lasers.append({'wavelength_nm': wavelength,
                                        'max_power_mW': laser_info['max_power'],
                                        'power_mW': getattr(laser, 'power_mw', None),
                                        'power_percent': getattr(laser, 'power_pourcent', None),
                                        'state': state})
        if verbose:
            for laser in combi_params.lasers:
                power_mW = '?' if laser['power_mW'] is None else f"{laser['power_mW']:.4g}"
                power_percent = '?' if laser['power_percent'] is None else f"{laser['power_percent']:.3g}"
                print('Laser ' + f"{laser['wavelength_nm']:g}" + ' nm : ' + laser['state'] + ', ' + power_mW + ' mW (' + power_percent + ' %)')

        return combi_params

    def laser_on(self, wavelength: float):

        self.device.laser_on(idn = self._idn(wavelength))

    def laser_off(self, wavelength: float):

        self.device.laser_off(idn = self._idn(wavelength))

    def set_power(self, wavelength: float, power: float, unit: str = '%'):
        """
        Set the power of a laser

        Parameters
        ----------
        wavelength : float
            the wavelength of the laser (nm), e.g. 405.
        power : float
            the power.
        unit : str
            '%' or 'mW'. The default is '%'.
        """
        self.device.set_laser_power(power, unit = unit, idn = self._idn(wavelength))

    def disconnect(self):

        self.device.close()
        self.device = None
        self.lasers = {}
        print('Combiner disconnected')
