# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:37:50 2026

@author: equipe-onli
"""
from spyrit.misc.walsh_hadamard import walsh_matrix as wh
import numpy as np
 
Np = 8
mat = wh(Np)
mat[mat < 0] = 0
pat_mat = np.empty((Np*2,Np), dtype=np.int16)


for i in range(Np):
    pat_mat[i*2,:] = mat[i,:]
    pat_mat[i*2+1,:] = abs(mat[i,:] - 1)
    
    
print(pat_mat)