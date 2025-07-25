# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:55:45 2024

@author: mahieu
"""

import serial
import numpy as np

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
                print("RS232 communication established with the spectrograh")
                print('Spetrograph SP connected')
            else:
                print("Problem to communicate with the RS232 port")
        except:
            print("Error: Attempting to use a port that is not open or used by another software !!")
    
    
    def open_serial(self, comm_port: str = 'COM10') -> object:
        """Open the serial port for RS232 communication with the spectrograph. 
        
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
            comm_port (str):
                The communication port number that your computer assigns by opening the Device Manager / Ports(COM & LPT)
                
        Returns
            serial_port (obj)
                A object containing the serial port information
        """
        
        try:
            serial_port = serial.Serial(port = comm_port, baudrate = 9600, bytesize = 8, parity = 'N', stopbits = 1, timeout = 0.1, rtscts = True, dsrdtr = False, xonxoff = False)

            self.query_echo(serial_port)
            
            return serial_port
        except:
            print('Error: Unable to open the port: ' + comm_port + '. Try the following possibilities:')
            print('       - Turn on the alimentation of the spectrograph')
            print('       - Connect the USB cable of the spectrograph')
            print('       - Check the COM port number by opening the Device Manager / Ports(COM & LPT)')
        
           
    def query_unit(self: object, 
                   print_unit: bool = False) -> str:    
        """Read the unit used in the GOTO, SCAN, SIZE, and CALIBRATE commands of the current grating.
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
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
            # self.serial_port.flush() # used to flush the buffer
            self.serial_port.write(send_cmd)
                    
            receive_serial = self.serial_port.readline()
            if len(receive_serial) > 0:
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
            else:
                print('problem to read the unit, no return')
                stop = True
            
            if stop == False or inc >= 2:
                break
    
        if stop == False:
            if print_unit == True:
                if unit == 'A':
                    unit_to_print = 'Å'
                else:
                    unit_to_print = unit
                print("Current unit set to : " + unit_to_print)
                
            return unit
    
    
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
        
        cmd = bytes([56])
        HiByte = bytes([0])
        send_cmd = cmd + HiByte
        self.serial_port.write(send_cmd)

        receive_serial = self.serial_port.readline()
        if len(receive_serial) == 0:
            print('problem to read position')
            position = -1
        else:
            HiByte = receive_serial[0]
            LoByte = receive_serial[1]
            position = HiByte * 256 + LoByte
            
            if print_position == True:
                unit = self.query_unit(self)
                if unit == 'A':
                    unit_to_print = 'Å'
                else:
                    unit_to_print = unit
                print("position = " + str(position) + ' ' + unit_to_print)
            
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
    
        # Query on the grooves number
        cmd = bytes([56])
        HiByte = bytes([2])
        send_cmd = cmd + HiByte
        self.serial_port.write(send_cmd)
                
        ret  = self.serial_port.readline()
        if len(ret) > 0: 
            if ret[0] < 128:
                HiByte = ret[0]
                LoByte = ret[1]
                grating.grooves = HiByte * 256 + LoByte
            else:
                grating.grooves = 0
        else:
            grating.grooves = 0
        
        # Query on the blaze wavelength
        cmd = bytes([56])
        HiByte = bytes([3])
        send_cmd = cmd + HiByte
        self.serial_port.write(send_cmd)
        
        ret = self.serial_port.readline()
        if len(ret) > 0: 
            if ret[0] < 128:
                HiByte = ret[0]
                LoByte = ret[1]
                grating.blaze = HiByte * 256 + LoByte
            else:
                grating.blaze = 0
        else:
            grating.blaze = 0
        
        # Query on the current grating number
        cmd = bytes([56])
        HiByte = bytes([4])
        send_cmd = cmd + HiByte
        self.serial_port.write(send_cmd)
        
        ret = self.serial_port.readline()
        if len(ret) > 0: 
            if ret[0] < 128:
                HiByte = ret[0]
                LoByte = ret[1]
                grating.current_grating_nbr = HiByte * 256 + LoByte
            else:
                grating.current_grating_nbr = 0
        else:
            grating.current_grating_nbr = 0
        
        # Query on the total number of grating
        cmd = bytes([56])
        HiByte = bytes([13])
        send_cmd = cmd + HiByte
        self.serial_port.write(send_cmd)
        
        ret = self.serial_port.readline()
        if len(ret) > 0: 
            if ret[0] < 128:
                HiByte = ret[0]
                LoByte = ret[1]
                grating.number_of_grating = HiByte * 256 + LoByte
            else:
                grating.number_of_grating = 0
        else:
            grating.number_of_grating = 0
        
        if print_grating_info == True:
            print("grooves/mm = " + str(grating.grooves))
            print("blaze wavelength = " + str(grating.blaze) + ' nm')
            print("current grating number = " + str(grating.current_grating_nbr))
            print("number of grating = " + str(grating.number_of_grating))
            
        return grating
    
    
    def query_speed(self: object, 
                   print_speed: bool = False) -> int:    
        """Read the speed at which the monochromator may scan.
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
            print_speed (bool):
                a boolean to print or not (default) the result
                
        Returns:
            speed (int):
                the speed at which the monochromator may scan (Å/sec).
        """
        
        cmd = bytes([56])
        HiByte = bytes([5])
        send_cmd = cmd + HiByte
        self.serial_port.write(send_cmd)
        
        receive_serial = self.serial_port.readline()
        HiByte = receive_serial[0]
        LoByte = receive_serial[1]
        speed = HiByte * 256 + LoByte
    
        if print_speed == True:
            print("speed = " + str(speed) + ' Å/sec')
            
        return speed
    
    
    def query_size(self: object, 
                       print_size: bool = False) -> int:    
        """Read the step size and the direction of the grating moving.
            If size is positive : rotation of the grating will increase the position in wavelength
            If size is negative : rotation of the grating will decrease the position in wavelength
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
            print_size (bool):
                a boolean to print or not (default) the result
                
        Returns:
            size (int):
                the size of the step
        """
        
        cmd = bytes([56])
        HiByte = bytes([6])
        send_cmd = cmd + HiByte
        self.serial_port.write(send_cmd)
        
        receive_serial = self.serial_port.readline()
        HiByte = receive_serial[0]
        LoByte = receive_serial[1]
        size = HiByte * 256 + LoByte
        
        if size >= 128:
            size = 128 - size
        
        if print_size == True:
            unit = self.query_unit(self)
            if unit == 'A':
                unit_to_print = 'Å'
            else:
                unit_to_print = unit
            print("step size  = " + str(size) + ' ' + unit_to_print)
            
        return size
    
    
    def cmd_unit(self: object, 
                 unit: str = 'nm',
                 print_unit: bool = False):
        """This command allows the selection of units in the GOTO, SCAN, SIZE, and CALIBRATE commands of the current grating.
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
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
            self.serial_port.write(send_cmd) 
            
            ret = self.serial_port.readline()
            if len(ret) > 0:
                if ret[0] >= 128:
                    print('Command not accepted')    
                elif print_unit == True:
                    unit = self.query_unit(self)
                    if unit == 'A':
                        unit_to_print = 'Å'
                    else:
                        unit_to_print = unit
                        
                    print("Unit set to : " + unit_to_print)    
            else:
                print('problem to read the return after the unit command')
        
    
    def cmd_size(self: object, 
                 size: int = 1,
                 unit: str = 'nm',
                 print_size: bool = False):
        """This command determines the change in magnitude and the direction of the grating position after a STEP command.
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
            size (int = 1 default):
                the step size (in the preset unit) and the direction of the grating moving.
                To increase the position, set a value in the range[0:127].
                To decrease the position, set a value in the range[0:-127].
            unit (str):
                the current unit (A, nm or µm)
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
            self.serial_port.write(send_cmd) 
            
            ret = self.serial_port.readline()
            if len(ret) > 0:
                if ret[0] >= 128:
                    print('Command not accepted')    
                elif print_size == True:
                    if unit == 'A':
                        unit_to_print = 'Å'
                    else:
                        unit_to_print = unit
                        
                    print("step size set to : " + str(size) + ' ' + unit_to_print)  
            else:
                print('problem to read the return after the size command')
    
    def cmd_speed(self: object, 
                 speed: int = 1000,
                 print_speed: bool = False):
        """Set the speed at which the grating may scan.
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
            speed (int = 1000 Å/s default):
                Values of speed are grating dependent. The function will find the nearest valid value depending of the grating.   
            print_speed (bool):
                A boolean to print or not (default) the result
                
        Returns:
            Nothing, just display a message if the command is not accepted or the new value if accepted
        """
        
              
        stop = False            
        # possible valid values of the speed
        self.serial_port.readline()# used to flush the buffer
        current_grating  = self.query_grating(self, grating)
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
            self.serial_port.write(send_cmd) 
            
            ret = self.serial_port.readline()
            if len(ret) > 0:
                if ret[0] >= 128:
                    print('Command not accepted')                           
                elif print_speed == True:                    
                    current_speed = self.query_speed(self)
                    print("speed set to : " + str(current_speed) + ' Å/s')      
    
    
    def cmd_step(self: object,
                 print_position: bool = False):
        """Moove the grating by a preset amount defined by the SIZE command.
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
            print_position (bool):
                a boolean to print or not (default) the result
                
        Returns:
            Nothing, just display a message if the command is not accepted or the new position after the moving if accepted
        """
        
        cmd = bytes([54])
        send_cmd = cmd    
        self.serial_port.write(send_cmd) 
        
        ret = self.serial_port.readline()
        if len(ret) > 0:
            if ret[0] >= 128:
                print('Command not accepted')    
            elif print_position == True:
                position = self.query_position(self)
                unit = self.query_unit(self)
                if unit == 'A':
                    unit_to_print = 'Å'
                else:
                    unit_to_print = unit
                print("wavelength position = " + str(position) + ' ' + unit_to_print)    
         
    
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
        
        cmd = bytes([26])
        HiByte = bytes([grating_nbr])
        send_cmd = cmd + HiByte    
        self.serial_port.write(send_cmd) 
        
        # time.sleep(15)
        ret = self.serial_port.readline()
        if len(ret) > 0:
            if ret[0] >= 128:
                print('Command not accepted')    
            else:
                print('grating change, please wait...')       
                while True:
                    grating = self.query_grating(self, grating)  
                    current_grating_nbr = grating.current_grating_nbr
                    if current_grating_nbr == grating_nbr:
                        if print_select:
                            print('grating nrb set to : ' + str(grating.current_grating_nbr))
                            
                        current_position = self.query_position(self)   
                        if current_position == 0:
                            if print_select:
                                print('grating at home')
                            break
               
    
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
            
        cmd = bytes([16])
        HiByte = bytes([int(np.floor(position/256))])
        LoByte = bytes([int(position%256)])
        send_cmd = cmd + HiByte + LoByte    
        self.serial_port.write(send_cmd)

        ret = self.serial_port.readline()
        if len(ret) > 0:
            if ret[0] >= 128:
                print('Command not accepted') 
        
    
    def cmd_scan(self: object,
                 start_position: int = 400,
                 end_position: int = 800,
                 unit: str = 'nm'):
        """This command moves the grating between a START position and an END position
        
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
            start_position (int=400 default):
                the start position of the scan
            end_position (int=800 default):
                the end position of the scan
            unit (str): "µm", "nm" (default) or "A".
                the unit used in the GOTO, SCAN, SIZE, and CALIBRATE commands.
        
        Returns:
            Nothing, just display a message if the command is not accepted
        """
    
        self.serial_port.readline()# used to flush the buffer
        current_unit = self.query_unit(self.serial_port)
        if current_unit != unit:
            self.cmd_unit(self.serial_port, unit = unit)
    
        cmd = bytes([12])
        start_HiByte = bytes([int(np.floor(start_position/256))])
        start_LoByte = bytes([int(start_position%256)])
        end_HiByte = bytes([int(np.floor(end_position/256))])
        end_LoByte = bytes([int(end_position%256)])
        send_cmd = cmd + start_HiByte + start_LoByte + end_HiByte + end_LoByte
        self.serial_port.write(send_cmd) 
    
        ret = self.serial_port.readline()
        if len(ret) > 0:
            if ret[0] >= 128:
                print('Command not accepted')    
    
    
    def cmd_reset(self):
        """This command returns the grating qt home position.
    
        Args:
            spectrograph (obj):
                This itself class containing the serial port communication object and all functions that control the spectrograph.
                
        Returns:
            Nothing, just display a message if the command is accepted or not
        """
        
        cmd = bytes([255])
        HiByte = bytes([255])
        LoByte = bytes([255])
        send_cmd = cmd + HiByte + LoByte    
        self.serial_port.write(send_cmd) 
        
        print('grating is coming home, please wait ...')        
        
        while True:
            current_position = self.query_position(self)   
            if current_position == 0:
                print('grating at home')
                break


    def close_serial(serial_port):
        """Close the serial port
        
        Args:
            serial_port (obj): 
                A object to communicate with the spectrograph by the RS232 serial port.
                
        Returns:
            Nothing, just display a message if the command is accepted or not
        """        
        
        serial_port.close()
        print("Spectrograph disconnected")       
    
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





























