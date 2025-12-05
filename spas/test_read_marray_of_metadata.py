# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 16:56:28 2025

@author: admin
"""


def to_float(str_arr):
    arr = []
    for s in str_arr:
        try:
            num = float(s)
            arr.append(num)
        except ValueError:
            pass
    return arr

def to_int(str_arr):
    arr = []
    for s in str_arr:
        try:
            num = int(s)
            arr.append(num)
        except ValueError:
            pass
    return arr


Lc                       = [(550, 2), (600, 2)]

LLc = [] * 2
Lc_str = str(Lc)
print(Lc_str)
a = (Lc_str.strip('[').strip(']').split('), ('))
print(a)
for ia in range(len(a)):
    if ia % 2 == 0:
        b = a[ia] + ')'
    else:
        b = '(' + a[ia]

    print(b)
    
    c = b.strip('(').strip(')').split(', ')
    print(c)
    
    d = to_float(c)
    print(d)
    
    # for id in d:
    #     d[id] = int(d[id])
    
    # LLc[ia][:] = [d]
    LLc.append(d)

LLLc = [] * 2    
for sublist in LLc:
    sublist[:] = map(int, sublist[:])
    print(sublist)
    LLLc.append(sublist)