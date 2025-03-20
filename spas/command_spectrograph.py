# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:52:13 2024

@author: mahieu
"""

# import sys
# # Set the path where the "CM110.py" module is installed in your computer.
# sys.path.append('../../spectrograph')

from spas.spectro_SP_lib import open_serial, close_serial, grating
from spas.spectro_SP_lib import query_echo, query_unit, query_position, query_grating, query_speed, query_size
from spas.spectro_SP_lib import cmd_unit, cmd_size, cmd_speed, cmd_step, cmd_goto, cmd_selectGrating, cmd_reset, cmd_scan

#%% Open serial port. 
serial_port = open_serial(comm_port = 'COM3')
#%% Query functions
query_echo(serial_port)
unit     = query_unit(serial_port, print_unit = True)
position = query_position(serial_port, print_position = True)
grating  = query_grating(serial_port, grating, print_grating_info = True)
speed    = query_speed(serial_port, print_speed = True)
size     = query_size(serial_port, print_size = True)
#%% command functions
# Below are examples of the command implemented. Pleass, Execute one line at a time
cmd_unit(serial_port, unit = 'nm', print_unit = True)
cmd_size(serial_port, size = 10, print_size = True)
cmd_speed(serial_port, speed = 3000, print_speed = True)
cmd_step(serial_port, print_position = True)
cmd_selectGrating(serial_port, grating_nbr = 2, print_select = True)
cmd_goto(serial_port, position = 600, unit = 'nm', print_position = True)
cmd_scan(serial_port, start_position = 400, end_position = 800, unit = 'nm')
cmd_reset(serial_port)
#%% Close serial port. 
close_serial(serial_port)
