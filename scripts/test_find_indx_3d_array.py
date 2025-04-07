# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 10:17:58 2025

@author: admin
"""

import numpy as np

arr = np.random.randint(0, 100, size=(5, 6, 7))
d0, d1, d2 = arr.shape

print('shape = ' + str(arr.shape))

indx_t = arr.argmax()
val_max = np.max(arr)
print('maxi = ' + str(val_max))

indx_0 = np.floor(indx_t / (d1 * d2))
rest_0 = indx_t - (d1 * d2)

indx_1 = np.floor(rest_0/d2)
rest_1 = rest_0 - (d2 * indx_1)

indx_2 = rest_1

print('coord = ' + str([indx_0, indx_1, indx_2]))