# -*- coding: utf-8 -*-
"""
Read and display the light-sheet patterns of the DMD (png made by generate_light-sheet_pattern.py).

The patterns are read in <patterns_root>/<scan_mode>_<t>x<Np>/<scan_mode>_<t>x<Np>_<k>.png.
Figure 1 : the chosen patterns, zoomed on the sheet (crop) or on the whole DMD (crop = False).
Figure 2 : profile along the sheet of the same patterns (centre line of the sheet).
"""

import os
import numpy as np
import imageio.v3 as iio
from matplotlib import pyplot as plt

#%% input
pattern_dim   = '1D'
scan_mode     = 'Walsh_sparse'    # 'Walsh' or 'Walsh_sparse'
thickness     = 16                # 1, 2, 4, 8, 16, 32
Np            = 512               # 128, 256, 512 (or 1)
indices       = [511]#[0, 1, 2, 3, 4, 5, 33, 127]   # patterns to display (None : all, up to 32)
crop          = True              # True : zoom on the sheet, False : whole DMD (768 x 1024)
patterns_root = r'C:\openspyrit\spas\Patterns' + '\\' + pattern_dim

#%% read
prefix = f'{scan_mode}_{thickness}x{Np}'
folder = os.path.join(patterns_root, prefix)
if not os.path.isdir(folder):
    raise FileNotFoundError(f'{folder} does not exist (generate it with generate_light-sheet_pattern.py)')
n_files = len([f for f in os.listdir(folder) if f.endswith('.png')])
if indices is None:
    indices = list(range(min(n_files, 32)))
indices = [k for k in indices if k < n_files]
patterns = {k: iio.imread(os.path.join(folder, f'{prefix}_{k}.png')) for k in indices}

# bounding box of the sheet = pattern 0 (whole sheet ON), with a margin
r, c = np.nonzero(patterns[0] if 0 in patterns else iio.imread(os.path.join(folder, f'{prefix}_0.png')))
m = 10
r0, r1, c0, c1 = r.min() - m, r.max() + m + 1, c.min() - m, c.max() + m + 1
u_line = int(np.min(r + c))       # first line of the sheet (always a sparse line)

#%% display the patterns
ncol = min(4, len(indices))
nrow = int(np.ceil(len(indices) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), squeeze=False, constrained_layout=True)
for ax, k in zip(axs.flat, indices):
    im = patterns[k]
    ax.imshow(im[r0:r1, c0:c1] if crop else im, cmap='gray', vmin=0, vmax=255, interpolation='nearest',
              extent=(c0, c1, r1, r0) if crop else None)
    ax.set_title(f'n° {k}  ({100 * np.mean(im > 0):.2f} % ON)', fontsize=9)
    ax.set_xlabel('column', fontsize=8); ax.set_ylabel('row', fontsize=8)
    ax.tick_params(labelsize=7)
for ax in list(axs.flat)[len(indices):]:
    ax.axis('off')
fig.suptitle(f'{prefix}  ({n_files} patterns, {folder})', fontsize=10)

#%% profile along the sheet (line u = row + col = u_line, v = col - row)
v = np.arange(-1024, 1024)
rows, cols = (u_line - v) // 2, (u_line + v) // 2
ok = ((u_line - v) % 2 == 0) & (rows >= 0) & (rows < 768) & (cols >= 0) & (cols < 1024)
sheet0 = patterns[0] if 0 in patterns else iio.imread(os.path.join(folder, f'{prefix}_0.png'))
sel = np.nonzero(sheet0[rows[ok], cols[ok]])[0]     # elements of the sheet, 0 = first one
fig2, ax2 = plt.subplots(figsize=(9, 0.6 + 0.45 * len(indices)), constrained_layout=True)
for i, k in enumerate(indices):
    prof = patterns[k][rows[ok], cols[ok]] > 0
    ax2.step(np.arange(prof.size) - sel.min(), prof * 0.8 + i, where='mid', lw=1)
ax2.set_xlim(-5, sel.max() - sel.min() + 5)
ax2.set_yticks(np.arange(len(indices)) + 0.4, [f'n° {k}' for k in indices], fontsize=8)
ax2.set_xlabel('position along the sheet (micromirror)')
ax2.set_title(f'{prefix} : profile along the sheet (line row + col = {u_line})', fontsize=10)
plt.show()
