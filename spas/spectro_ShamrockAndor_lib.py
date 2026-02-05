# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:55:45 2024

@author: mahieu
"""

import numpy as np
import math

#%% functions
class grating():
    """A class containing the informations about the grating.
    
    Attributes:
        grooves: 
            the number of grooves by mm
        blaze: 
            the blaze wavelength (nm)
        current_grating:
            the grating currently used in the spectrograph
        number_of_grating:
            the number of available grating in the spectrograph       
    """

    def __init__(self): 
        self.grooves = 0
        self.blaze = 0
        self.current_grating_nbr = 0
        self.number_of_grating = 0
        



class Spectrograph:
    """
    This class controls the spectrograph CM110 from Spectral Products.
    """   
    def query_position(self: object, 
                       print_position: bool = False) -> int:    
        """Read the position (in wavelength) of the grating inside the spectrograph.
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
            print_position (bool):
                a boolean to print or not (default) the result
                
        Returns:
            position (int):
                the position of the grating depending of the unit.
                (µm: micrometer, nm: nanometer, Å: Angström)
        """
        
        
        position = int(self.get_wavelength()*1e11) / 100
        
        if print_position:
            print("position = " + str(position) + ' nm')
            
        return position
      
        
    def query_grating(self: object,
                      grating: object,
                      print_grating_info: bool = False) -> grating:    
        """Read informations about the grating.
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
            grating (obj):
                the class containing the information of the grating ->
                (Grooves/mm, Blaze wavelength, current grating number, number of grating)
            print_grating_info (bool):
                a boolean to print or not (default) the grating informations
                
        Returns:
            grating (class):
                the characteristic of the selectd grating
        """
        
        grating_info = self.get_grating_info()
        
        # Query on the grooves number        
        grating.grooves = math.ceil(grating_info.lines)
        
        # Query on the blaze wavelength
        grating.blaze = grating_info.blaze_wavelength
        
        # Query on the current grating number
        grating.current_grating_nbr = self.get_grating()
        
        # Query on the total number of grating
        grating.number_of_grating = self.get_gratings_number()
        
        if print_grating_info == True:
            print("grooves/mm = " + str(grating.grooves))
            print("blaze wavelength = " + str(grating.blaze) + ' nm')
            print("current grating number = " + str(grating.current_grating_nbr))
            print("number of grating = " + str(grating.number_of_grating))
            
        return grating 
         
    
    def cmd_selectGrating(self: object,
                          grating: object = grating,
                          grating_nbr: int = 1,
                          print_select: bool = False):
        """Select the grating that will be used.
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
            grating (obj):
                the class containing the information of the grating ->
                (Grooves/mm, Blaze wavelength, current grating number, number of grating)
            grating_nbr (int = 1 default):
                To selecte the grating number. Valid values : 1 or 2
            print_select (bool):
                a boolean to print or not (default) the grating selected
                
        Returns:
            Nothing, just display a message if the command is not accepted or the new value if accepted
        """
        
        self.set_grating(grating_nbr)
        
        grating = self.query_grating(self, grating)  
        current_grating_nbr = grating.current_grating_nbr
        if current_grating_nbr == grating_nbr:
            if print_select:
                print('grating nrb set to : ' + str(grating.current_grating_nbr))
                
            current_position = self.query_position(self)   
            if current_position == 0:
                if print_select:
                    print('grating at home')
                            
               
    
    def cmd_goto(self: object, 
                 position: int = 0):
        """This command moves the grating to a selected position.
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
            position (int = 0 default):
                the position of the central wavelength of the grating
            unit (str): "µm", "nm" (default) or "A".
                the unit used in the GOTO, SCAN, SIZE, and CALIBRATE commands.
                
        Returns:
            Nothing, just display a message if the command is not accepted or the new position after the moving if accepted
        """
            
        self.set_wavelength(position * 1e-9)
        
    
#%% Example of how to use the functions
# serial_port = open_serial(comm_port = 'COM3')
# query_echo(serial_port)
# unit     = query_unit(serial_port, print_unit = True)
# position = query_position(serial_port, print_position = True)
# grating  = query_grating(serial_port, grating, print_grating_info = True)
# speed    = query_speed(serial_port, print_speed = True)
# size     = query_size(serial_port, print_size = True)

# cmd_unit(serial_port, unit = 'nm', print_unit = True)
# cmd_size(serial_port, size = 10, print_size = True)
# cmd_speed(serial_port, speed = 4000, print_speed = True)
# cmd_step(serial_port, print_position = True)
# cmd_selectGrating(serial_port, grating_nbr = 2, print_select = True)
# cmd_goto(serial_port, position = 550, unit = 'nm', print_position = True)
# cmd_scan(serial_port, start_position = 400, end_position = 800, unit = 'nm')
# cmd_reset(serial_port)
# close_serial(serial_port)





























