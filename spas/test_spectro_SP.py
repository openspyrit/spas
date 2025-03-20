#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:27:41 2025

@author: mahieu
"""

import sys
sys.path.append('/home/mahieu/openspyrit/spas')

from spas.spectro_SP_module import Spectro_SP_parameters, init_spectro_SP, setup_spectro_SP


spectro_SP = init_spectro_SP()


Spectro_SP_params = Spectro_SP_parameters(spectro_SP.serial_port)

Spectro_SP_params = setup_spectro_SP(Spectro_SP_params = Spectro_SP_parameters,
                                    spectro_SP = spectro_SP,
                                    position = 600,
                                    print_position = True,
                                    unit = 'nm',
                                    print_unit = True,                   
                                    grating_nbr = 1,
                                    print_select = True,
                                    speed = 3000,
                                    print_speed = True,
                                    size = 10,
                                    print_size = True)