# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:55:45 2024

@author: mahieu
"""

import serial
import numpy as np
import time

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


class Spectro_SP:
    """
    This class controls the spectrograph CM110 from Spectral Products.
    """
    def query_echo(serial_port: object):    
        """The ECHO command is used to verify communications with the CM110.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
                
        Returns:
            Nothing, just print if communication is established or failed
        """
            
        send_cmd = bytes([27])
        try: 
            serial_port.write(send_cmd)    
            receive_serial = serial_port.readline()
            HiByte = receive_serial[0]
            
            if HiByte == 27:
                print("RS232 communication with the spectrograh established")
        except:
            print("Error: Attempting to use a port that is not open or used by another software !!")
    
    
    def open_serial(comm_port: str = 'COM3') -> object:
        """Open the serial port for RS232 communication with the spectrograph
        
        Args:
            comm_port (str):
                the communication port number that your computer assigns (Valid numbers: COM1-6) by opening the Device Manager / Ports(COM & LPT)
                
        Returns
            serial_port (obj)
                A object containing the serial port information
        """
        
        try:
            serial_port = serial.Serial(port = comm_port, baudrate = 9600, bytesize = 8, parity = 'N', stopbits = 1, timeout = 0.1, rtscts = True, dsrdtr = False, xonxoff = False)
            # query_echo(serial_port)
            
            return serial_port
        except:
            print('Error: Unable to open the port: ' + comm_port + '. Try the following possibilities:')
            print('       - Turn on the alimentation of the spectrograph')
            print('       - Connect the USB cable of the spectrograph')
            print('       - Check the COM port number by opening the Device Manager / Ports(COM & LPT)')
        
           
    def query_unit(serial_port: object, 
                   print_unit: bool = False) -> str:    
        """Read the unit used in the GOTO, SCAN, SIZE, and CALIBRATE commands of the current grating.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            print_unit (bool):
                a boolean to print or not (default) the result
                
        Returns:
            unit (str):
                the unit used in the GOTO, SCAN, SIZE, and CALIBRATE commands.
                (µm: micrometer, nm: nanometer, Å: Angström)
        """
            
        inc = 0
        while True:
            stop = False
            inc = inc + 1
            cmd = bytes([56])
            HiByte = bytes([14])
            send_cmd = cmd + HiByte       
            serial_port.readline() # used to flush the buffer
            serial_port.write(send_cmd)
                    
            receive_serial = serial_port.readline()
            HiByte = receive_serial[0]
            LoByte = receive_serial[1]
            unit_nbr = HiByte * 256 + LoByte
            if unit_nbr == 0:
                unit = 'µm'
            elif unit_nbr == 1:
                unit = 'nm'
            elif unit_nbr == 2:
                unit = 'A'
            else:
                print('problem to read the unit, value out of range')
                stop = True
                if inc == 1:
                    print('try a second time')
                else:
                    print('Error: unit reading failed !!')
            
            if stop == False or inc >= 2:
                break
    
        if stop == False:
            if print_unit == True:
                if unit == 'A':
                    unit_to_print = 'Å'
                else:
                    unit_to_print = unit
                print("Unit : " + unit_to_print)
                
            return unit
    
    
    def query_position(serial_port: object, 
                       print_position: bool = False) -> int:    
        """Read the position (in wavelength) of the grating inside the spectrograph.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            print_position (bool):
                a boolean to print or not (default) the result
                
        Returns:
            position (int):
                the position of the grating depending of the unit.
                (µm: micrometer, nm: nanometer, Å: Angström)
        """
        
        cmd = bytes([56])
        HiByte = bytes([0])
        send_cmd = cmd + HiByte
        serial_port.write(send_cmd)
        
        receive_serial = serial_port.readline()
        HiByte = receive_serial[0]
        LoByte = receive_serial[1]
        position = HiByte * 256 + LoByte
        
        if print_position == True:
            unit = query_unit(serial_port)
            if unit == 'A':
                unit_to_print = 'Å'
            else:
                unit_to_print = unit
            print("position = " + str(position) + ' ' + unit_to_print)
            
        return position
      
        
    def query_grating(serial_port: object,
                      grating: object,
                      print_grating_info: bool = False) -> grating:    
        """Read informations about the grating.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            grating (class):
                the class containing the information of the grating ->
                (Grooves/mm, Blaze wavelength, current grating number, number of grating)
            print_grating_info (bool):
                a boolean to print or not (default) the grating informations
                
        Returns:
            grating (class):
                the characteristic of the current grating
        """
    
        # Query on the grooves number
        cmd = bytes([56])
        HiByte = bytes([2])
        send_cmd = cmd + HiByte
        serial_port.write(send_cmd)
        
        receive_serial = serial_port.readline()
        HiByte = receive_serial[0]
        LoByte = receive_serial[1]
        grating.grooves = HiByte * 256 + LoByte
        
        # Query on the blaze wavelength
        cmd = bytes([56])
        HiByte = bytes([3])
        send_cmd = cmd + HiByte
        serial_port.write(send_cmd)
        
        receive_serial = serial_port.readline()
        HiByte = receive_serial[0]
        LoByte = receive_serial[1]
        grating.blaze = HiByte * 256 + LoByte
        
        # Query on the current grating number
        cmd = bytes([56])
        HiByte = bytes([4])
        send_cmd = cmd + HiByte
        serial_port.write(send_cmd)
        
        receive_serial = serial_port.readline()
        HiByte = receive_serial[0]
        LoByte = receive_serial[1]
        grating.current_grating_nbr = HiByte * 256 + LoByte
        
        # Query on the total number of grating
        cmd = bytes([56])
        HiByte = bytes([13])
        send_cmd = cmd + HiByte
        serial_port.write(send_cmd)
        
        receive_serial = serial_port.readline()
        HiByte = receive_serial[0]
        LoByte = receive_serial[1]
        grating.number_of_grating = HiByte * 256 + LoByte
        
        if print_grating_info == True:
            print("grooves/mm = " + str(grating.grooves))
            print("blaze wavelength = " + str(grating.blaze) + ' nm')
            print("current grating number = " + str(grating.current_grating_nbr))
            print("number of grating = " + str(grating.number_of_grating))
            
        return grating
    
    
    def query_speed(serial_port: object, 
                   print_speed: bool = False) -> int:    
        """Read the speed at which the monochromator may scan.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            print_speed (bool):
                a boolean to print or not (default) the result
                
        Returns:
            speed (int):
                the speed at which the monochromator may scan (Å/sec).
        """
        
        cmd = bytes([56])
        HiByte = bytes([5])
        send_cmd = cmd + HiByte
        serial_port.write(send_cmd)
        
        receive_serial = serial_port.readline()
        HiByte = receive_serial[0]
        LoByte = receive_serial[1]
        speed = HiByte * 256 + LoByte
    
        if print_speed == True:
            print("speed = " + str(speed) + ' Å/sec')
            
        return speed
    
    
    def query_size(serial_port: object, 
                       print_size: bool = False) -> int:    
        """Read the step size and the direction of the grating moving.
            If size is positive : rotation of the grating will increase the position in wavelength
            If size is negative : rotation of the grating will decrease the position in wavelength
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            print_size (bool):
                a boolean to print or not (default) the result
                
        Returns:
            size (int):
                the size of the step
        """
        
        cmd = bytes([56])
        HiByte = bytes([6])
        send_cmd = cmd + HiByte
        serial_port.write(send_cmd)
        
        receive_serial = serial_port.readline()
        HiByte = receive_serial[0]
        LoByte = receive_serial[1]
        size = HiByte * 256 + LoByte
        
        if size >= 128:
            size = 128 - size
        
        if print_size == True:
            unit = query_unit(serial_port)
            if unit == 'A':
                unit_to_print = 'Å'
            else:
                unit_to_print = unit
            print("step size  = " + str(size) + ' ' + unit_to_print)
            
        return size
    
    
    def cmd_unit(serial_port: object, 
                 unit: str = 'nm',
                 print_unit: bool = False):
        """This command allows the selection of units in the GOTO, SCAN, SIZE, and CALIBRATE commands of the current grating.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            unit (str): "µm", "nm" (default) or "A".
                the unit used in the GOTO, SCAN, SIZE, and CALIBRATE commands.
            print_unit (bool):
                a boolean to print or not (default) the result
                
        Returns:
            Nothing, just display a message if the command is not accepted or the new value if accepted
        """
    
        stop = False
        if unit == 'µm':
            unit_nbr = 0
        elif unit == 'nm':
            unit_nbr = 1
        elif unit == 'A':
            unit_nbr = 2
        else:
            print('Wrong input unit, please set to : µm, nm or A')
            print('command aborted')
            stop = True
        
        if stop == False:
            cmd = bytes([50])
            HiByte = bytes([unit_nbr])
            send_cmd = cmd + HiByte    
            serial_port.write(send_cmd) 
            
            ret = serial_port.readline()
            if len(ret) > 0:
                if ret[0] >= 128:
                    print('Command not accepted')    
                elif print_unit == True:
                    unit = query_unit(serial_port)
                    if unit == 'A':
                        unit_to_print = 'Å'
                    else:
                        unit_to_print = unit
                    print("Unit set to : " + unit_to_print)    
        
    
    def cmd_size(serial_port: object, 
                 size: int = 1,
                 print_size: bool = False):
        """This command determines the change in magnitude and the direction of the grating position after a STEP command.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            size (int = 1 default):
                the step size (in the preset unit) and the direction of the grating moving.
                To increase the position, set a value in the range[0:127].
                To decrease the position, set a value in the range[0:-127].
            print_unit (bool):
                a boolean to print or not (default) the result
                
        Returns:
            Nothing, just display a message if the command is not accepted or the new value if accepted
        """
        
        stop = False
        if abs(size) >= 128:
            print('value out of range. Valid range : [-127 ; 127]')
            print('command aborted')
            stop = True
        elif size < 0:
            new_size = abs(size) + 128
        else:
            new_size = size
        
        if stop == False:
            cmd = bytes([55])
            HiByte = bytes([int(new_size)])
            send_cmd = cmd + HiByte   
            serial_port.write(send_cmd) 
            
            ret = serial_port.readline()
            if len(ret) > 0:
                if ret[0] >= 128:
                    print('Command not accepted')    
                elif print_size == True:
                    unit = query_unit(serial_port)
                    if unit == 'A':
                        unit_to_print = 'Å'
                    else:
                        unit_to_print = unit
                        
                    current_size = query_size(serial_port)
                    if current_size >= 128:
                        new_current_size = 128 - current_size
                    else:
                        new_current_size = current_size
                    print("step size set to : " + str(new_current_size) + ' ' + unit_to_print)    
    
    
    def cmd_speed(serial_port: object, 
                 speed: int = 1000,
                 print_speed: bool = False):
        """Set the speed at which the grating may scan.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            speed (int = 1000 Å/s default):
                Values of speed are grating dependent. The function will find the nearest valid value depending of the grating.   
            print_speed (bool):
                A boolean to print or not (default) the result
                
        Returns:
            Nothing, just display a message if the command is not accepted or the new value if accepted
        """
        
        stop = False
        
        # possible valid values of the speed
        serial_port.readline()# used to flush the buffer
        current_grating  = query_grating(serial_port, grating)
        if current_grating.grooves == 3600:
            possible_speed = [333, 166, 83, 41, 20, 10, 5, 2, 1]
        elif current_grating.grooves == 2400:
            possible_speed = [500, 250, 125, 62, 31, 15, 7, 3, 1]
        elif current_grating.grooves == 1800:
            possible_speed = [666, 332, 166, 82, 40, 20, 10, 4, 2]
        elif current_grating.grooves == 1200:
            possible_speed = [1000, 500, 250, 125, 62, 31, 15, 7, 3, 1]
        elif current_grating.grooves == 600:
            possible_speed = [2000, 1000, 500, 250, 124, 62, 30, 14, 6, 2]
        elif current_grating.grooves == 300:
            possible_speed = [4000, 2000, 1000, 500, 248, 124, 60, 28, 12, 4]
        elif current_grating.grooves == 150:
            possible_speed = [8000, 4000, 2000, 1000, 496, 248, 120, 56, 24, 8]
        elif current_grating.grooves == 75:
            possible_speed = [16000, 8000, 4000, 2000, 992, 496, 240, 112, 48, 16]
        else:
            print('grating not referenced in the function "cmd_speed".')
            stop = True
        
        if stop == False:        
            try:
                possible_speed.index(speed)
            except: # find the closest valid speed if value is not valid
                print('desired speed does not match the possible values for the grating : ' + str(current_grating.grooves) + ' grooves/mm.')
                nearest_indx = np.argmin(np.abs(np.array(possible_speed) - speed))
                speed = possible_speed[nearest_indx]
                print('possible valid values : ' + str(possible_speed) + ' Å/s')
                print('The closest valid speed found is ' + str(speed) + ' Å/s')
                
            cmd = bytes([13])
            HiByte = bytes([int(np.floor(speed/256))])
            LoByte = bytes([int(speed%256)])
            send_cmd = cmd + HiByte + LoByte  
            serial_port.write(send_cmd) 
            
            ret = serial_port.readline()
            if len(ret) > 0:
                if ret[0] >= 128:
                    print('Command not accepted')                           
                elif print_speed == True:                    
                    current_speed = query_speed(serial_port)
                    print("speed set to : " + str(current_speed) + ' Å/s')      
    
    
    def cmd_step(serial_port: object,
                 print_position: bool = False):
        """Moove the grating by a preset amount defined by the SIZE command.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            print_position (bool):
                a boolean to print or not (default) the result
                
        Returns:
            Nothing, just display a message if the command is not accepted or the new position after the moving if accepted
        """
        
        cmd = bytes([54])
        send_cmd = cmd    
        serial_port.write(send_cmd) 
        
        ret = serial_port.readline()
        if len(ret) > 0:
            if ret[0] >= 128:
                print('Command not accepted')    
            elif print_position == True:
                position = query_position(serial_port)
                unit = query_unit(serial_port)
                if unit == 'A':
                    unit_to_print = 'Å'
                else:
                    unit_to_print = unit
                print("wavelength position = " + str(position) + ' ' + unit_to_print)    
         
    
    def cmd_selectGrating(serial_port: object,
                          grating_nbr: int = 1,
                 print_select: bool = False):
        """Select the grating that will be used.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            grating_nbr (int = 1 default):
                To selecte the grating number. Valid values : 1 or 2
            print_select (bool):
                a boolean to print or not (default) the grating selected
                
        Returns:
            Nothing, just display a message if the command is not accepted or the new value if accepted
        """
        
        serial_port.readline()# used to flush the buffer
        current_grating  = query_grating(serial_port, grating)
        if current_grating.current_grating_nbr == grating_nbr:
            print('grating already selectionned. Nothing to do')
        else:
            cmd = bytes([26])
            HiByte = bytes([grating_nbr])
            send_cmd = cmd + HiByte    
            serial_port.write(send_cmd) 
            print('grating change, please wait...')
            time.sleep(15)
            ret = serial_port.readline()
            if len(ret) > 0:
                if ret[0] >= 128:
                    print('Command not accepted')    
                elif print_select == True:            
                    print('selected grating:')
                    query_grating(serial_port, grating, print_grating_info = True)   
    
    
    def cmd_goto(serial_port: object, 
                 position: int = 0,
                 unit: str = 'nm',
                 print_position: bool = False):
        """This command moves the grating to a selected position.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            position (int = 0 default):
                the position (in wavelength)
            unit (str): "µm", "nm" (default) or "A".
                the unit used in the GOTO, SCAN, SIZE, and CALIBRATE commands.
            print_position (bool):
                a boolean to print or not (default) the position of the grating in wavelength
                
        Returns:
            Nothing, just display a message if the command is not accepted or the new position after the moving if accepted
        """
        
        stop = False
        
        serial_port.readline()# used to flush the buffer
        current_unit = query_unit(serial_port)
        if current_unit != unit:
            cmd_unit(serial_port, unit = unit)
        
        # find the factor to calculate the delay to move the grating because the unit of the speed is Å/s    
        if current_unit == "µm":
            fact_speed = 1/10000
        elif current_unit == "nm":
            fact_speed = 1/10
        elif current_unit == "A":
            fact_speed = 1
        
        current_position = query_position(serial_port)    
        
        if stop == False:
            # calculate the delay to move the grating
            speed = query_speed(serial_port)
            dist = abs(position - current_position)
            delay = np.ceil(dist *fact_speed / speed * 1000) / 1000
            if print_position == True:
                print('delay to move the grating is = ' + str(delay) + ' s')
            
            cmd = bytes([16])
            HiByte = bytes([int(np.floor(position/256))])
            LoByte = bytes([int(position%256)])
            send_cmd = cmd + HiByte + LoByte    
            serial_port.write(send_cmd)
            
            # waiting for the grating displacement
            time.sleep(delay)
            
            ret = serial_port.readline()
            if len(ret) > 0:
                if ret[0] >= 128:
                    print('Command not accepted') 
                elif print_position == True:
                    unit = query_unit(serial_port)
                    if unit == 'A':
                        unit_to_print = 'Å'
                    else:
                        unit_to_print = unit
                    pos = query_position(serial_port)
                    print("wavelength position set to : " + str(pos) + ' ' + unit_to_print)    
        
    
    def cmd_scan(serial_port: object,
                 start_position: int = 400,
                 end_position: int = 800,
                 unit: str = 'nm'):
        """This command moves the grating between a START position and an END position
        
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
            start_position (int=400 default):
                the start position of the scan
            end_position (int=800 default):
                the end position of the scan
            unit (str): "µm", "nm" (default) or "A".
                the unit used in the GOTO, SCAN, SIZE, and CALIBRATE commands.
        
        Returns:
            Nothing, just display a message if the command is not accepted
        """
    
        serial_port.readline()# used to flush the buffer
        current_unit = query_unit(serial_port)
        if current_unit != unit:
            cmd_unit(serial_port, unit = unit)
    
        cmd = bytes([12])
        start_HiByte = bytes([int(np.floor(start_position/256))])
        start_LoByte = bytes([int(start_position%256)])
        end_HiByte = bytes([int(np.floor(end_position/256))])
        end_LoByte = bytes([int(end_position%256)])
        send_cmd = cmd + start_HiByte + start_LoByte + end_HiByte + end_LoByte
        serial_port.write(send_cmd) 
    
        ret = serial_port.readline()
        if len(ret) > 0:
            if ret[0] >= 128:
                print('Command not accepted')    
    
    
    def cmd_reset(serial_port: object):
        """This command returns the grating to the home position.
    
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
                
        Returns:
            Nothing, just display a message if the command is accepted or not
        """
        
        cmd = bytes([255])
        HiByte = bytes([255])
        LoByte = bytes([255])
        send_cmd = cmd + HiByte + LoByte    
        serial_port.write(send_cmd) 
        
        ret = serial_port.readline()
        if len(ret) > 0:
            if ret[0] >= 128:
                print('Command not accepted')  
            else:
                print('grating goes back to home')
    
    
    def close_serial(serial_port):
        """Close the serial port
        
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
                
        Returns:
            Nothing, just display a message if the command is accepted or not
        """        
        
        serial_port.close()
        print("serial port closed")       
    
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





























