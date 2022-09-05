# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:04:42 2022

@author: mahieu
"""
import numpy as np


# this is the example (reference)
pattern_order_source = 'C:/openspyrit/spas/stats/pattern_order.npz'
file = np.load(pattern_order_source)



# file.pattern_order = pattern_order
print('file :')
print(file['pattern_order'])
print(file.__dict__)



#########################################
# Create a new pattern order file
# temp_path = 'D:/hspc/patterns/Antonio/pattern_order.npz'
# pattern_order = np.array(list(range(0, 2000)), dtype=np.uint16)

temp_path = 'D:/hspc/patterns/number_in_corner/pattern_order.npz'
pattern_order = np.array(list(range(0, 2000)), dtype=np.uint16)

pos_neg = True

# Save the arrays:
np.savez_compressed(temp_path, pattern_order = pattern_order, pos_neg = pos_neg)

file2 = np.load(temp_path)

print(file2['pattern_order'])
len(file2['pattern_order'])
















