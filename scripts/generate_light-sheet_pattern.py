# -*- coding: utf-8 -*-
"""
Generate the light-sheet patterns of the DMD for the SPIM 1D (png) and their pattern order (npz).

Geometry (same as the existing Walsh_4x* and Walsh_16x* patterns) :
    - DMD 768 x 1024, the sheet is drawn along the diagonal (45°), centred on row 384, column 511.
    - u = row + col is the thickness direction, v = col - row the length direction.
    - the sheet is 256 micromirrors long (one element every 2 in v, on the lines of same parity as u),
      whatever the number of patterns Np.
    - thickness t : 'Walsh_sparse' -> t lines u = 895-(t-1), 895-(t-1)+2, ..., 895+(t-1)
                                      (one line out of two, checkerboard)
                    'Walsh'        -> the 2t-1 lines from 895-(t-1) to 895+(t-1) (sparse lines + the
                                      lines in between, a pixel is ON if its 4 neighbours are ON)
    - Np patterns = Np/2 Walsh functions (sequency order, Walsh-Hadamard of order 256) split in
      positive / negative parts : pattern 2k = (w_k > 0), pattern 2k+1 = (w_k < 0).
      Np = 512 : full resolution (1 element = 1 diagonal step), 256 : 2 steps, 128 : 4 steps.
      Np = 1 : the 2 patterns white (sheet ON) and black, pos_neg = True (as Walsh_16x1).

Outputs :
    <patterns_root>/<scan_mode>_<t>x<Np>/<scan_mode>_<t>x<Np>_<k>.png
    <stats_root>/pattern_order_<scan_mode>_<t>x<Np>.npz   (pattern_order = 0..Np-1, pos_neg = False)
These names are the ones read by DMD_module.setup_DMD / play_one_pattern.

With t = 1 the sheet is one single line : 'Walsh' and 'Walsh_sparse' give the same patterns.
"""

import os
import numpy as np
from PIL import Image

#%% input
pattern_dim       = '1D'                         # only '1D' for the moment
scan_modes        = ['Walsh', 'Walsh_sparse']    # 'Walsh' and/or 'Walsh_sparse'
sizes             = [128, 256, 512]              # Np : 128, 256 or 512 (1 = white / black)
thicknesses       = [1, 2, 8, 32]                # t : 1, 2, 4, 8, 16, 32 (4 and 16 already exist)
overwrite         = False                        # False : a folder that already exists is skipped
patterns_root     = r'C:\openspyrit\spas\Patterns' + '\\' + pattern_dim
stats_root        = r'C:\openspyrit\spas\stats' + '\\' + pattern_dim

#%% geometry of the DMD and of the sheet
DMD_HEIGHT   = 768
DMD_WIDTH    = 1024
U_CENTRE     = 895    # row + col of the centre line of the sheet (row 384, col 511)
V_FIRST      = -128   # col - row of the first element of the sheet
SHEET_LENGTH = 256    # number of elements along the sheet (diagonal steps)


def walsh_sequency(n: int) -> np.ndarray:
    """Walsh-Hadamard matrix of order n (power of 2), rows sorted by sequency (number of sign changes)."""
    H = np.array([[1]], dtype=np.int8)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    changes = np.sum(H[:, 1:] != H[:, :-1], axis=1)
    return H[np.argsort(changes, kind='stable')]


def sheet_image(profile: np.ndarray, thickness: int, sparse: bool) -> np.ndarray:
    """
    DMD image (uint8 0/255) of the sheet modulated along its length by profile (bool, SHEET_LENGTH).
    """
    im = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    v = V_FIRST + 2 * np.arange(SHEET_LENGTH)
    v_on = v[profile]
    u_first = U_CENTRE - (thickness - 1)
    for u in range(u_first, U_CENTRE + thickness, 2):      # sparse lines (same parity as u_first)
        im[(u - v_on) // 2, (u + v_on) // 2] = 255
    if not sparse and thickness > 1:                        # lines in between : 4 neighbours ON
        on = im == 255
        fill = np.zeros_like(on)
        fill[1:-1, 1:-1] = (~on[1:-1, 1:-1] & on[:-2, 1:-1] & on[2:, 1:-1] & on[1:-1, :-2] & on[1:-1, 2:])
        im[fill] = 255
    return im


def generate(scan_mode: str, Np: int, thickness: int, patterns_root: str, stats_root: str,
             overwrite: bool = False, verbose: bool = True) -> str:
    """
    Write the Np png patterns and the pattern order of one set. Returns the folder of the patterns.
    """
    if scan_mode not in ('Walsh', 'Walsh_sparse'):
        raise ValueError(f"scan_mode = {scan_mode} : 'Walsh' or 'Walsh_sparse'")
    if Np not in (1, 128, 256, 512):
        raise ValueError(f'Np = {Np} : 1, 128, 256 or 512')
    if thickness < 1:
        raise ValueError(f'thickness = {thickness} must be >= 1')

    prefix = f'{scan_mode}_{thickness}x{Np}'
    folder = os.path.join(patterns_root, prefix)
    order_file = os.path.join(stats_root, f'pattern_order_{prefix}.npz')
    if os.path.isdir(folder) and not overwrite:
        if verbose:
            print(f'{prefix} : already exists, skipped (overwrite = False)')
        return folder
    os.makedirs(folder, exist_ok=True)
    os.makedirs(stats_root, exist_ok=True)

    sparse = scan_mode == 'Walsh_sparse'
    if Np == 1:     # white / black
        profiles = [np.ones(SHEET_LENGTH, bool), np.zeros(SHEET_LENGTH, bool)]
        pos_neg = True
    else:
        W = walsh_sequency(SHEET_LENGTH)[:Np // 2]
        profiles = [p for w in W for p in (w > 0, w < 0)]
        pos_neg = False

    for k, profile in enumerate(profiles):
        Image.fromarray(sheet_image(profile, thickness, sparse)).save(os.path.join(folder, f'{prefix}_{k}.png'))
    np.savez(order_file, pattern_order = np.arange(len(profiles), dtype=np.uint16), pos_neg = pos_neg)
    if verbose:
        print(f'{prefix} : {len(profiles)} patterns -> {folder}')
    return folder


#%% generate
if __name__ == '__main__':
    if pattern_dim != '1D':
        raise ValueError('only pattern_dim = 1D is implemented')
    for scan_mode in scan_modes:
        for Np in sizes:
            for t in thicknesses:
                generate(scan_mode, Np, t, patterns_root, stats_root, overwrite = overwrite)
