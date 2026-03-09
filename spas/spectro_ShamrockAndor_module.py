#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 09:14:44 2025

@author: mahieu
"""

# from spas.spectro_SP_lib import Spectrograph, grating
from spas.spectro_ShamrockAndor_lib import Spectrograph, grating
import pylablib as pll
from pylablib.devices import Andor
import math

from dataclasses_json import dataclass_json
from dataclasses import dataclass, InitVar
from typing import Optional


def init_spectrograph(model : str = 'andor_shamrock'):
    """Initialize the communication with the spectrograph.
    
    Args:
        model of the spectrograph [str]: the version of the DMD library

    Returns:
        spectrograph (obj): 
            An object containing the serial port communication object and all functions that control the spectrograph..
    """
    if model == 'andor_shamrock':
        spectrograph = Spectrograph()
        # pll.par["devices/dlls/" + model] = "F:/#Sources_logiciels/#OLD/Andor/Drivers/Shamrock64/ShamrockCIF.dll"
        pll.par["devices/dlls/" + model] = "E:/laurent/lib"

        spectro_list = Andor.list_shamrock_spectrographs()
        print('list of spectrograph : ' + spectro_list[0] + ' detected')
        spectrograph = Andor.ShamrockSpectrograph(idx=0)
        spec_is_open = spectrograph.is_opened()

        if spec_is_open == True:
            print('Spectrograph connected')
        
        return spectrograph
    else:
        print('Error, the model of the spectrograph must be : CM110. For another spectrograph, change the package that import the function init_spectrograph')


@dataclass_json
@dataclass
class Spectrograph_Parameters:
    """Class containing the spectrograph Spectral Products parameters. Further information into spectro_SP_lib.py.

    Attributes:
        spectrograph (obj):
            An object containing the serial port communication object and all functions that control the spectrograph.
    """

    # unit: Optional[str] = None
    position: Optional[int] = None
    grating: Optional[grating] = None
    slit_width: Optional[int] = None
    slit_height: Optional[int] = 4000
    resolution_th: Optional[float] = None
    # speed: Optional[int] = None
    # size: Optional[int] = None
        
    spectrograph: InitVar[Spectrograph] = None

    class_description: str = 'spectrograph SP parameters'


    def __post_init__(self, spectrograph: Optional[Spectrograph] = None):
        """ Post initialization of attributes.

        Receives a spectrograph object and directly asks it for its configurations, 
        then sets the SpectrographParameters's attributes.
        During reconstruction from JSON, spectrograph is set to None and the function
        does nothing, letting initialization for the standard __init__ function.

        Args:
            spectrograph (Spectrograph, optional): 
                Defaults to None.
        """
        if spectrograph is None:
            pass
        else:
            # self.unit            = Spectrograph.query_unit(spectrograph)
            self.position        = Spectrograph.query_position(spectrograph)
            self.grating         = Spectrograph.query_grating(spectrograph, grating)
            self.slit_width      = Spectrograph.slit_width
            self.resolution_th   = self.slit_width * 7.44 / self.grating.grooves
            # self.speed         = Spectrograph.query_speed(spectrograph, print_speed = False)
            # self.size          = Spectrograph.query_size(spectrograph, print_size = False)
    
    # def undo_readable_class_spectro(self):
    #     """Changes the time_array attribute from `str` to `List` of `int`."""
        
    #     print('icicicici')
        
    #     for item in self.keys():
    #         print(item)
    #         if item.find("grating"):
    #             print('grating found')
    #             sub_item = item[8:]
    #             self.item.sub_item = 'd'
    #         else:
    #             # sp = Spectrograph_Parameters.from_dict(item)
    #             # print(sp)
    #             pass
        
    #     # def to_float(str_arr):
    #     #     arr = []
    #     #     for s in str_arr:
    #     #         try:
    #     #             num = float(s)
    #     #             arr.append(num)
    #     #         except ValueError:
    #     #             pass
    #     #     return arr        
    
    
    @staticmethod
    def readable_class_spectro(spectro_params_dict: dict) -> dict:
        # pass
        """Turns class "grating into a readable dictionary
        """
        
        readable_spectro_dict = {}
        readable_spectro_dict_temp = spectro_params_dict
        for item in readable_spectro_dict_temp:
            stri = str(type(readable_spectro_dict_temp[item]))
            # print('----- item : ' + item)
            if item == 'grating':                  
                for sub_item in readable_spectro_dict_temp[item]().__dict__:
                    # print('---------- subitem = ' + sub_item)
                    readable_spectro_dict[item + '.' + sub_item] = getattr(readable_spectro_dict_temp[item], sub_item)#.value
            else:
                readable_spectro_dict[item] = readable_spectro_dict_temp[item]
                            
        return readable_spectro_dict
            

def setup_spectrograph(spectrograph: object,
                       grating_nbr: int = 1, print_select: bool = False,
                       position: int = 600,  print_position: bool = False,
                       slit_width: int = 200):
    """ Setup the spectrograph to tune the cameras
    Parameters
    ----------
    spectrograph (obj):
        The class containing the serial port communication object and all functions that control the spectrograph.
    grating_nbr (int):
        the number of the selected grating. The default is 1.
    print_select (bool):
        a boolean to print or not the slected grating. The default is False.
    position (int). The default is 600:
        the position of the central wavelength of the grating
    print_position : bool, optional
        a boolean to print or not the position of the central wavenlength of the grating. The default is False.
    slit_width (int):
        the width of the slit in µm.

    Returns
    -------
    SpectrographParameters (obj)
        the metadata containing the spectrograph parameters.

    """        
    current_grating  = Spectrograph.query_grating(spectrograph, grating)
    if current_grating.current_grating_nbr == grating_nbr: 
        if print_select:
            print(str(current_grating.grooves) + ' gr/mm and blaze wavelength = ' + str(current_grating.blaze) + ' nm grating already selectionned. Nothing to do')
    else:    
        Spectrograph.cmd_selectGrating(spectrograph, grating = current_grating, grating_nbr = grating_nbr, print_select = print_select)
        if print_select:
            current_grating  = Spectrograph.query_grating(spectrograph, grating)
            print(str(current_grating.grooves) + ' gr/mm and blaze wavelength = ' + str(current_grating.blaze) + ' nm grating selectionned.')
    
    current_position = Spectrograph.query_position(spectrograph)
    if current_position == position:
        if print_position:
            print('position already set to : ' + str(position) + ' nm. Nothing to do')
    else:
        Spectrograph.cmd_goto(spectrograph, position = position)
        if print_position:
            print('position set to : ' + str(Spectrograph.query_position(spectrograph)) + ' nm.')
    
    Spectrograph.slit_width = slit_width            
    
    return Spectrograph_Parameters(spectrograph = spectrograph)
    
    

def disconnect_spectrograph(spectrograph, goto_zero : bool = False):
    """ disconnect the spectrograph by releasing the RS232 communication port

    Parameters
    ----------
    spectrograph (obj):
        The class containing the serial port communication object and all functions that control the spectrograph.
    goto_zero (bool):
        a boolean to send the grating at home. The default is False.

    Returns
    -------
    None. Simply displays (in the "close_serial" function) whether the RS232 communication port has been successfully released.

    """
    if goto_zero == True:
        spectrograph.set_wavelength(0)
    
    spectrograph.close()
    # print('Spectrograph : ' + spectro_list[0] + ' closed')
    spec_is_open = spectrograph.is_opened()
    if spec_is_open == False:
        print('Spectrograph is disconnected')


# unit     = Spectrograph.query_unit(spectrograph, print_unit = True)
# position = Spectrograph.query_position(spectrograph, print_position = True)


# #%% Query functions
# def setup_spectro_SP():
# query_echo(serial_port)
# unit     = query_unit(serial_port, print_unit = True)
# position = query_position(serial_port, print_position = True)
# grating  = query_grating(serial_port, grating, print_grating_info = True)
# speed    = query_speed(serial_port, print_speed = True)
# size     = query_size(serial_port, print_size = True)
# #%% command functions
# # Below are examples of the command implemented. Pleass, Execute one line at a time
# cmd_unit(serial_port, unit = 'nm', print_unit = True)
# cmd_size(serial_port, size = 10, print_size = True)
# cmd_speed(serial_port, speed = 3000, print_speed = True)
# cmd_step(serial_port, print_position = True)
# cmd_selectGrating(serial_port, grating_nbr = 2, print_select = True)
# cmd_goto(serial_port, position = 600, unit = 'nm', print_position = True)
# cmd_scan(serial_port, start_position = 400, end_position = 800, unit = 'nm')
# cmd_reset(serial_port)
# #%% Close serial port. 
# close_serial(serial_port)
















