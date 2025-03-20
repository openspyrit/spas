#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 09:14:44 2025

@author: mahieu
"""
# from spas import spectro_SP_lib

# from spas.spectro_SP_lib import open_serial, close_serial, grating
# from spas.spectro_SP_lib import query_echo, query_unit, query_position, query_grating, query_speed, query_size
# from spas.spectro_SP_lib import cmd_unit, cmd_size, cmd_speed, cmd_step, cmd_goto, cmd_selectGrating, cmd_reset, cmd_scan

from spas.spectro_SP_lib import Spectro_SP, grating

from dataclasses_json import dataclass_json
from dataclasses import dataclass, InitVar
from typing import Optional


def init_spectro_SP():
    spectro_SP = Spectro_SP()
    spectro_SP.serial_port = 10#Spectro_SP.open_serial(comm_port = 'COM3')
    print('Spetrograph SP connected')
    
    return spectro_SP


@dataclass_json
@dataclass
class Spectro_SP_parameters():
    """Class containing the spectrograph Spectral Products configurations and status.

    Further information: spectro_SP_lib.py.

    Attributes:
        add_illumination_time_us (int):
    """
    
    def __init__(self, serial_port):
        self.serial_port = serial_port
        print('__init__')
    
    serial_port: object
    unit: Optional[str] = None
    position: Optional[int] = None
    grating: Optional[grating] = None
    speed: Optional[int] = None
    size: Optional[int] = None
    print('set to None')
        
    # spectro_SP: InitVar[Spectro_SP] = None

    class_description: str = 'spectrograph SP parameters'


    def __post_init__(self):
        """ Post initialization of attributes.

        Receives a DMD object and directly asks it for its configurations and
        status, then sets the majority of SpectrometerParameters's attributes.
        During reconstruction from JSON, DMD is set to None and the function
        does nothing, letting initialization for the standard __init__ function.

        Args:
            DMD (ALP4.ALP4, optional): 
                Connected DMD. Defaults to None.
        """
        # if spectro_SP == None:
        #     print('la')
        #     pass
        # else:
        #     pass
        
        # self.serial_port = object#Spectro_SP.query_unit(serial_port, print_unit = True)
        # serial_port = Spectro_SP.open_serial()
        print('sp= ' + str(self.serial_port))
        
        self.unit     = 'nm' # Spectro_SP.query_unit(Spectro_SP.serial_port, print_unit = True)
        self.position = 0 # Spectro_SP.query_position(Spectro_SP.serial_port, print_position = True)
        self.grating  = 1 # Spectro_SP.query_grating(Spectro_SP.serial_port, grating, print_grating_info = True)
        self.speed    = 3000 # Spectro_SP.query_speed(Spectro_SP.serial_port, print_speed = True)
        self.size     = 10 # Spectro_SP.query_size(Spectro_SP.serial_port, print_size = True)
        print('ici')
            

def setup_spectro_SP(Spectro_SP_params : Spectro_SP_parameters,
                    spectro_SP: object,
                    position: int = 600,
                    print_position: bool = True,
                    unit: str = 'nm',
                    print_unit: bool = True,                   
                    grating_nbr: int = 1,
                    print_select: bool = True,
                    speed: int = 3000,
                    print_speed: bool = True,
                    size: int = 10,
                    print_size: bool = True):
    
    Spectro_SP.cmd_goto(spectro_SP.serial_port, position = position, unit = 'nm', print_position = print_position)
    Spectro_SP.cmd_unit(spectro_SP.serial_port, unit = unit, print_unit = print_unit)
    Spectro_SP.cmd_size(spectro_SP.serial_port, size = size, print_size = print_size)
    Spectro_SP.cmd_speed(spectro_SP.serial_port, speed = speed, print_speed = print_speed)
    Spectro_SP.cmd_selectGrating(spectro_SP.serial_port, grating_nbr = grating_nbr, print_select = print_select)
    
    



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
















