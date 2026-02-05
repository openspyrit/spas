# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:23:28 2025

@author: equipe-onli
"""

import numpy as np
import os
os.chdir('E:\\openspyrit\\spas\\scripts')
################ input #######################
Np = 512
pattern_thickness = 16
############## begin ##########################
pattern_dim = '1D'
scan_mode = 'Walsh'
pattern_order_source = '../stats/' + pattern_dim + '/pattern_order_' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np) + '.npz'

pattern_order=np.arange(Np, dtype=np.uint16)

np.savez(pattern_order_source[:len(pattern_order_source)-4], pattern_order = pattern_order, pos_neg = False)

############# read pattern order #############
a=np.load(pattern_order_source)
print(a['pattern_order'])