#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 2026
@author: mahieu

Wavelength calibration of the spectral arm of the SPIM (Shamrock 500i + Zyla 4.2) with the
mercury-argon calibration lamp Ocean Optics HG-1.

Principle:
    1. for each central wavelength Lc of a list, a snapshot of the lamp is acquired and the lines
       are detected (sub-pixel position on the full sensor, unbinned),
    2. the lines are identified with the theoretical calibration of the spectrograph (SDK) as a
       first guess, then with the fitted model itself (2 passes),
    3. a single 2D polynomial lambda(x, Lc) is fitted on all the lines of all the Lc
       (it also covers the Lc where the lamp has few lines, e.g. 600-650 nm),
    4. the coefficients are written in WAVELENGTH_CALIB (in this module), used by
       define_wavelengths_matrix (also imported by spas.acquisition_SPIM1D).

Used by the script main_SPIM1D_spectral_calib.py
"""

import ast
import os
import re
import datetime
import json

import numpy as np
from scipy import signal, ndimage, interpolate, special
from matplotlib import pyplot as plt

# Lines of the Ocean Optics HG-1 lamp (nm, first order), from the HG-1 manual
HG1_LINES = {
    'Hg': [253.652, 296.728, 302.150, 313.155, 334.148, 365.015, 404.656, 407.783, 435.833,
           546.074, 576.960, 579.066],
    'Ar': [696.543, 706.722, 710.748, 727.294, 738.393, 750.387, 763.511, 772.376, 794.818,
           800.616, 811.531, 826.452, 842.465, 852.144, 866.794, 912.297, 922.450],
}

# Second order of the strong Hg lines (without order sorting filter, a line at L appears at 2*L).
# They are valid calibration points (the grating equation depends only on m*lambda).
HG1_LINES_2ND_ORDER = [2 * L for L in [365.015, 404.656, 407.783, 435.833]]

CALIB_BEGIN = '# >>> WAVELENGTH CALIBRATION >>>'
CALIB_END = '# <<< WAVELENGTH CALIBRATION <<<'

N_SENSOR = 2048        # pixels of the Zyla along the dispersion axis
RES_RANGE = (380, 930)  # (nm) range of the mean resolution (resolution_th of Spectrograph_Parameters)
X0 = (N_SENSOR - 1) / 2
X_SCALE = N_SENSOR / 2


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# >>> WAVELENGTH CALIBRATION >>>
# Written by write_wavelength_calib (script main_SPIM1D_spectral_calib.py).
# Do not edit by hand. One entry per grating number. Model (see wavelengths_from_calib):
#     lambda = Lc + sum_ij coeffs[i][j] * u**i * v**j
#     u = (x - x0) / x_scale    x : pixel of the full sensor (unbinned, 0 = first column)
#     v = (Lc - Lc0) / Lc_scale Lc : central wavelength of the spectrograph (nm)
WAVELENGTH_CALIB = {}
# <<< WAVELENGTH CALIBRATION <<<


def physical_pixels(width: int, offsetX: int = 1, binningX: int = 1) -> np.ndarray:
    """
    Position of the center of each (binned) column of the spectral camera, in pixels of the
    full sensor (unbinned, 0 = first column of the sensor).

    Parameters
    ----------
    width : int
        number of (binned) columns of the image (AOIWidth).
    offsetX : int, optional
        first column of the AOI on the sensor, 1-based (AOILeft). The default is 1.
    binningX : int, optional
        binning in the width direction (AOIHBin). The default is 1.
    """
    offsetX = 1 if offsetX is None else int(offsetX)
    binningX = 1 if binningX is None else int(binningX)
    return (offsetX - 1) + np.arange(int(width)) * binningX + (binningX - 1) / 2


def wavelengths_from_calib(x, Lc: float, calib: dict) -> np.ndarray:
    """
    Wavelength (nm) of the sensor pixels x for the central wavelength Lc, from a calibration
    entry of WAVELENGTH_CALIB.
    """
    u = (np.asarray(x, dtype=float) - calib['x0']) / calib['x_scale']
    v = (Lc - calib['Lc0']) / calib['Lc_scale']
    coeffs = np.asarray(calib['coeffs'], dtype=float)
    lam = np.full(u.shape, float(Lc))
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            lam += coeffs[i, j] * u**i * v**j
    return lam


def dispersion_from_calib(x, Lc: float, calib: dict) -> np.ndarray:
    """Local dispersion |d lambda / d x| (nm per pixel of the sensor) at the pixels x for the central wavelength Lc."""
    x = np.asarray(x, dtype=float)
    return np.abs(wavelengths_from_calib(x + 0.5, Lc, calib) - wavelengths_from_calib(x - 0.5, Lc, calib))


def fwhm_gauss_box(fwhm, box):
    """
    FWHM of a Gaussian of width fwhm convolved with a box of width box (same unit), e.g. a line
    measured with a binning along the wavelength. Exact (bisection on the half maximum), not the
    quadratic sum, which is wrong when the box is wider than the line.
    """
    fwhm, box = np.broadcast_arrays(np.asarray(fwhm, dtype=float), np.asarray(box, dtype=float))
    s2 = np.sqrt(2) * fwhm / (2 * np.sqrt(2 * np.log(2)))     # sigma * sqrt(2)
    profile = lambda h: (special.erf((h + box / 2) / s2) - special.erf((h - box / 2) / s2)) / (2 * special.erf(box / 2 / s2))
    low, high = np.zeros(fwhm.shape), (fwhm + box) * np.ones(fwhm.shape)
    for _ in range(50):
        mid = (low + high) / 2
        above = profile(mid) > 0.5
        low, high = np.where(above, mid, low), np.where(above, high, mid)
    return 2 * (low + high) / 2


def resolution_from_calib(x, Lc: float, calib: dict) -> np.ndarray:
    """Spectral resolution (FWHM of a line, nm) at the pixels x of the sensor (unbinned) for the central
    wavelength Lc, from the 'resolution' part of a calibration entry of WAVELENGTH_CALIB."""
    res = calib['resolution']
    u = (np.asarray(x, dtype=float) - calib['x0']) / calib['x_scale']
    v = (Lc - calib['Lc0']) / calib['Lc_scale']
    coeffs = np.asarray(res['coeffs'], dtype=float)
    fwhm = np.zeros(u.shape)
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            fwhm += coeffs[i, j] * u**i * v**j
    return fwhm


def spectral_resolution(cam_spec_params, Lc: list = [], include_binning: bool = True, verbose: bool = False):
    """
    Spectral resolution (FWHM of a line, nm) of each column of the spectral camera, for each
    (central wavelength, grating) of Lc. Same use as define_wavelengths_matrix.

    The resolution is measured on the lines of the HG-1 lamp with the camera without binning (slit width
    and focus of the calibration). With include_binning, the binning along the wavelength is added
    (the line, taken as a Gaussian, is convolved with a box of binningX pixels : see fwhm_gauss_box).

    Returns
    -------
    A matrix (len(Lc), width) of the resolution (nm), None if a grating has no measured resolution.
    """
    px_width = int(cam_spec_params.width)
    binningX = getattr(cam_spec_params, 'binningX', 1) or 1
    x = physical_pixels(px_width, getattr(cam_spec_params, 'offsetX', 1), binningX)

    r = np.empty([len(Lc), px_width])
    for iLc in range(len(Lc)):
        Lcc = float(Lc[iLc][0])
        Grating = int(Lc[iLc][1])
        calib = WAVELENGTH_CALIB.get(Grating)
        if calib is None or 'resolution' not in calib:
            print('Warning, no measured resolution for the grating ' + str(Grating) + ' (WAVELENGTH_CALIB)')
            return None
        fwhm = resolution_from_calib(x, Lcc, calib)
        if include_binning and binningX > 1:
            fwhm = fwhm_gauss_box(fwhm, binningX * dispersion_from_calib(x, Lcc, calib))
        r[iLc, :] = fwhm
        if verbose:
            print('Lc = ' + str(Lcc) + ' nm, grating ' + str(Grating) + ' : resolution ' + str(round(fwhm.min(), 3)) +
                  ' - ' + str(round(fwhm.max(), 3)) + ' nm (center ' + str(round(fwhm[px_width // 2], 3)) + ' nm)' +
                  (', binning ' + str(binningX) + ' included' if include_binning and binningX > 1 else ''))
    return r


def _define_wavelengths_legacy(cam_spec_params, Lcc: float, Grating: int) -> np.ndarray:
    """
    Old calibration (before the HG-1 calibration of the Shamrock 500i / Zyla), kept as a fallback
    for a grating which is not in WAVELENGTH_CALIB. The binning is not taken into account.
    """
    if Grating == 2:
        Lc_array = np.array([365, 435, 546, 577, 696, 912])
        coeff_array = np.array([3.308, 3.311, 3.319, 3.324, 3.341, 3.365])
        ord_origine = np.array([-5.5, -9.56, -7.58, -3.09, -6.92, -3.89])
    else:
        Lc_array = np.array([     365,   405,   436,   546,    696,   912])
        coeff_array = np.array([13.26, 14.18, 14.12, 14.94,  15.89, 18.36])
        ord_origine = np.array([ 3.94, -0.69, -6.5,  -3.53,  -9.63, -5.88])

    p = np.poly1d(np.polyfit(Lc_array, coeff_array, 2))
    px_width = int(cam_spec_params.width)
    L_offset = cam_spec_params.offsetX / p(Lcc)
    spam_lambda = px_width / p(Lcc)
    xnew2 = np.linspace(Lcc - spam_lambda / 2 + L_offset, Lcc + spam_lambda / 2 + L_offset, px_width)
    ynew = interpolate.interp1d(Lc_array, ord_origine, fill_value='extrapolate')(xnew2)
    i = np.arange(px_width)
    return (i - int(px_width/2) - ynew) / p(xnew2) + Lcc + L_offset


def define_wavelengths_matrix(cam_spec_params, Lc: list = [], display_figure: bool = False, verbose: bool = False):
    """
    define the wavelength vector for each central wavelength of the spectrograph

    The calibration is read in WAVELENGTH_CALIB (measured with the Hg-Ar lamp HG-1 by the script
    main_SPIM1D_spectral_calib.py). The ROI (offsetX) and the binning of the spectral camera are
    taken into account. If the grating is not calibrated, the old calibration is used.

    Parameters
    ----------
    cam_spec_params: class.
        the class containing the parameters of the spectral camera (width, offsetX, binningX)
    Lc : list, optional
        The list of (central wavelength (nm), grating number) of the spectrograph. The default is [].
    display_figure : bool, optional
        plot the wavelength vs the column of the camera. The default is False.
    verbose : bool, optional
        print the wavelength range. The default is False.

    Returns
    -------
    A matrix (len(Lc), width) of the wavelength vector for each Lc.

    """
    px_width = int(cam_spec_params.width)
    x = physical_pixels(px_width, getattr(cam_spec_params, 'offsetX', 1), getattr(cam_spec_params, 'binningX', 1))

    w = np.empty([len(Lc), px_width])
    for iLc in range(len(Lc)):
        Lcc = float(Lc[iLc][0])
        Grating = int(Lc[iLc][1])

        calib = WAVELENGTH_CALIB.get(Grating)
        if calib is not None and 'coeffs' in calib:
            w[iLc, :] = wavelengths_from_calib(x, Lcc, calib)
            if not (calib['Lc_range'][0] <= Lcc <= calib['Lc_range'][1]):
                print('Warning, Lc = ' + str(Lcc) + ' nm is outside the calibrated range ' + str(calib['Lc_range']) +
                      ' nm of the grating ' + str(Grating) + ' : the wavelengths are extrapolated')
        else:
            print('Warning, the grating ' + str(Grating) + ' is not calibrated (WAVELENGTH_CALIB), the old calibration is used')
            w[iLc, :] = _define_wavelengths_legacy(cam_spec_params, Lcc, Grating)

        if verbose:
            print('Lc = ' + str(Lcc) + ' nm, grating ' + str(Grating) + ' : ' + str(round(w[iLc, 0], 2)) + ' -> ' +
                  str(round(w[iLc, -1], 2)) + ' nm, ' + str(round(abs(w[iLc, -1] - w[iLc, 0]) / max(px_width - 1, 1), 4)) + ' nm/column')

        if display_figure:
            plt.figure()
            plt.plot(w[iLc, :])
            plt.xlabel('column of the camera')
            plt.ylabel('wavelength (nm)')
            plt.title('Lc = ' + str(Lcc) + ' nm, grating ' + str(Grating))
            plt.grid()

    return w



def reference_lines(second_order: bool = True):
    """Sorted array of the reference wavelengths (nm) and the array of their labels."""
    lam = [(L, 'Hg') for L in HG1_LINES['Hg']] + [(L, 'Ar') for L in HG1_LINES['Ar']]
    if second_order:
        lam += [(L, 'Hg 2nd order') for L in HG1_LINES_2ND_ORDER]
    lam.sort()
    return np.array([L for L, _ in lam]), np.array([name for _, name in lam])


def extract_spectrum(image: np.ndarray, row_threshold: float = 0.5):
    """
    1D spectrum of the lamp: mean of the rows where the lamp light is (the image of the fiber
    or of the slit does not fill the whole height of the sensor).

    Returns
    -------
    spectrum (np.ndarray), rows (np.ndarray) : the spectrum and the index of the rows used.
    """
    image = np.asarray(image, dtype=float)
    row_profile = np.percentile(image, 99, axis=1)   # the lines are bright on a few columns only
    background = np.percentile(row_profile, 5)
    level = background + row_threshold * (row_profile.max() - background)
    rows = np.where(row_profile >= level)[0]
    spectrum = image[rows, :].mean(axis=0)
    return spectrum, rows


def find_lines(spectrum: np.ndarray, x: np.ndarray, min_snr: float = 15, min_rel_height: float = 0.01,
               baseline_size: int = 151, max_fwhm_px: float = None):
    """
    Detection of the lines of the spectrum, with a sub-pixel position (centroid above half maximum)
    and the FWHM (linear interpolation of the 2 crossings of the half maximum : no line shape assumed).

    Parameters
    ----------
    spectrum : np.ndarray
        the 1D spectrum (one value per column of the camera).
    x : np.ndarray
        the position of each column on the full sensor (physical_pixels).
    min_snr : float
        minimum height of a line, in units of the noise of the baseline.
    min_rel_height : float
        minimum height of a line relative to the highest line.
    baseline_size : int
        minimum size (columns) of the running low percentile giving the baseline.
    max_fwhm_px : float
        expected maximum FWHM of the lines (pixels of the sensor), e.g. the image of the slit (slit width /
        pixel width) : the window of the baseline is at least 4 times wider, so that wide lines are not removed.

    Returns
    -------
    list of dict(x, height, fwhm, isolated) sorted by position. x and fwhm are in pixels of the sensor,
    isolated is False if another peak is closer than 3 FWHM (blended line : FWHM not reliable).
    """
    spectrum = np.asarray(spectrum, dtype=float)
    if max_fwhm_px is not None:
        dx = abs(x[1] - x[0]) if len(x) > 1 else 1
        baseline_size = max(baseline_size, int(4 * max_fwhm_px / dx))
    baseline = ndimage.percentile_filter(spectrum, 10, size=min(baseline_size, len(spectrum)), mode='nearest')
    s = spectrum - baseline
    noise = 1.4826 * np.median(np.abs(s - np.median(s)))
    noise = max(noise, 1e-6)
    threshold = max(min_snr * noise, min_rel_height * s.max())

    peaks, props = signal.find_peaks(s, height=threshold, prominence=threshold)
    lines = []
    for p in peaks:
        half = s[p] / 2
        left = p
        while left > 0 and s[left - 1] > half:
            left -= 1
        right = p
        while right < len(s) - 1 and s[right + 1] > half:
            right += 1
        if left == 0 or right == len(s) - 1:
            continue   # line cut by the edge of the sensor
        idx = np.arange(left, right + 1)
        weights = s[idx] - half
        xc = np.sum(weights * x[idx]) / np.sum(weights)
        x_left = x[left - 1] + (half - s[left - 1]) / (s[left] - s[left - 1]) * (x[left] - x[left - 1])
        x_right = x[right] + (s[right] - half) / (s[right] - s[right + 1]) * (x[right + 1] - x[right])
        lines.append({'x': float(xc), 'height': float(s[p]), 'fwhm': float(abs(x_right - x_left))})
    for k, l in enumerate(lines):
        others = [abs(o['x'] - l['x']) for j, o in enumerate(lines) if j != k]
        l['isolated'] = bool(not others or min(others) > 3 * l['fwhm'])
    return lines


def _match(lam_peaks, lam_ref, tol):
    """Index of the nearest reference line for each peak (-1 if farther than tol), each reference used once."""
    match = -np.ones(len(lam_peaks), dtype=int)
    if len(lam_ref) == 0:
        return match
    idx = np.searchsorted(lam_ref, lam_peaks)
    for k, L in enumerate(lam_peaks):
        cand = [i for i in (idx[k] - 1, idx[k]) if 0 <= i < len(lam_ref)]
        best = min(cand, key=lambda i: abs(lam_ref[i] - L))
        if abs(lam_ref[best] - L) <= tol:
            match[k] = best
    # a reference line matched by several peaks : keep the nearest one
    for i in np.unique(match[match >= 0]):
        ks = np.where(match == i)[0]
        if len(ks) > 1:
            keep = ks[np.argmin(np.abs(lam_peaks[ks] - lam_ref[i]))]
            match[ks[ks != keep]] = -1
    return match


def identify_lines(x_peaks, lam_guess, lam_ref, max_shift: float = 10, tol: float = 0.6):
    """
    Identification of the peaks from a first guess of their wavelength (theoretical calibration):
    search of the global shift that matches the most peaks, then linear correction and new match.

    Returns
    -------
    match (np.ndarray of int) : index of the reference line of each peak, -1 if not identified.
    """
    x_peaks = np.asarray(x_peaks, dtype=float)
    lam_guess = np.asarray(lam_guess, dtype=float)
    if len(x_peaks) == 0:
        return np.zeros(0, dtype=int)

    shifts = [L - g for g in lam_guess for L in lam_ref if abs(L - g) <= max_shift]
    if not shifts:
        return -np.ones(len(x_peaks), dtype=int)
    best = None
    for d in shifts:
        m = _match(lam_guess + d, lam_ref, tol)
        n = np.sum(m >= 0)
        err = np.sum(np.abs(lam_guess[m >= 0] + d - lam_ref[m[m >= 0]])) if n else np.inf
        if best is None or n > best[0] or (n == best[0] and err < best[1]):
            best = (n, err, d, m)
    match = best[3]

    # linear correction of the guess with the matched lines, then new match
    ok = match >= 0
    if np.sum(ok) >= 3:
        p = np.polyfit(lam_guess[ok], lam_ref[match[ok]], 1)
        match = _match(np.polyval(p, lam_guess), lam_ref, tol / 2)
    return match


def _design(x, Lc, deg_x, deg_Lc, Lc0, Lc_scale):
    u = (np.asarray(x, dtype=float) - X0) / X_SCALE
    v = (np.asarray(Lc, dtype=float) - Lc0) / Lc_scale
    return np.stack([u**i * v**j for i in range(deg_x + 1) for j in range(deg_Lc + 1)], axis=1)


def fit_model(x, Lc, lam, deg_x: int = 3, deg_Lc: int = 2, clip_nm: float = 0.5, clip_sigma: float = 3):
    """
    Least square fit of lambda - Lc = sum_ij c_ij u**i v**j on all the identified lines,
    with an iterative rejection of the outliers.

    Returns
    -------
    calib (dict) : an entry of WAVELENGTH_CALIB (without the metadata), keep (np.ndarray of bool).
    """
    x, Lc, lam = np.asarray(x, float), np.asarray(Lc, float), np.asarray(lam, float)
    Lc0 = float(round((Lc.min() + Lc.max()) / 2, 1))
    Lc_scale = float(max(round((Lc.max() - Lc.min()) / 2, 1), 1.0))
    # the degree in Lc can not be higher than the number of Lc - 1
    deg_Lc = min(deg_Lc, len(np.unique(Lc)) - 1)

    keep = np.ones(len(x), dtype=bool)
    for _ in range(10):
        A = _design(x[keep], Lc[keep], deg_x, deg_Lc, Lc0, Lc_scale)
        if A.shape[0] <= A.shape[1]:
            raise RuntimeError('Not enough identified lines (' + str(A.shape[0]) + ') for ' + str(A.shape[1]) +
                               ' coefficients : add central wavelengths or decrease deg_x / deg_Lc')
        c, *_ = np.linalg.lstsq(A, lam[keep] - Lc[keep], rcond=None)
        res = lam - Lc - _design(x, Lc, deg_x, deg_Lc, Lc0, Lc_scale) @ c
        rms = np.sqrt(np.mean(res[keep]**2))
        new_keep = np.abs(res) <= max(min(clip_nm, clip_sigma * rms), 1e-3)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    coeffs = c.reshape(deg_x + 1, deg_Lc + 1)
    calib = {'x0': X0, 'x_scale': X_SCALE, 'Lc0': Lc0, 'Lc_scale': Lc_scale,
             'coeffs': [[float(f'{v:.8g}') for v in row] for row in coeffs],
             'rms_nm': float(f'{rms:.4g}'),
             'Lc_range': [float(Lc.min()), float(Lc.max())]}
    return calib, keep


def fit_resolution(table: dict, calib: dict, deg_x: int = 2, deg_Lc: int = 1, clip_sigma: float = 3):
    """
    Fit of the FWHM (nm) of the isolated, not saturated lines : FWHM = sum_ij r_ij u**i v**j
    (same u, v as the wavelength model), with an iterative rejection of the outliers (blends).

    Returns
    -------
    resolution (dict) : the 'resolution' part of the calibration entry, keep (np.ndarray of bool) : lines used.
    """
    ok = table['used'] & table['res_ok'] & np.isfinite(table['fwhm_nm'])
    x, Lc, fwhm = table['x'], table['Lc'], table['fwhm_nm']
    deg_Lc = max(min(deg_Lc, len(np.unique(Lc[ok])) - 1), 0)
    keep = ok.copy()
    for _ in range(10):
        A = _design(x[keep], Lc[keep], deg_x, deg_Lc, calib['Lc0'], calib['Lc_scale'])
        if A.shape[0] <= A.shape[1]:
            print('Warning, not enough isolated lines (' + str(A.shape[0]) + ') to fit the resolution')
            return None, keep
        c, *_ = np.linalg.lstsq(A, fwhm[keep], rcond=None)
        res = fwhm - _design(x, Lc, deg_x, deg_Lc, calib['Lc0'], calib['Lc_scale']) @ c
        rms = np.sqrt(np.mean(res[keep]**2))
        new_keep = ok & (np.abs(res) <= max(clip_sigma * rms, 1e-4))
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    f = lambda v: float(f'{v:.4g}')
    resolution = {'coeffs': [[float(f'{v:.6g}') for v in row] for row in c.reshape(deg_x + 1, deg_Lc + 1)],
                  'rms_nm': f(rms), 'n_lines': int(np.sum(keep)),
                  'median_nm': f(np.median(fwhm[keep])), 'min_nm': f(np.min(fwhm[keep])), 'max_nm': f(np.max(fwhm[keep])),
                  'median_px': f(np.median(table['fwhm_px'][keep]))}
    return resolution, keep


def calibrate(measurements: list, second_order: bool = True, deg_x: int = 3, deg_Lc: int = 2,
              max_shift: float = 10, tol: float = 0.6, slit_width_um: float = None, verbose: bool = True):
    """
    Identification of the lines and fit of the model on all the measurements.

    Parameters
    ----------
    measurements : list of dict
        one dict per Lc with the keys 'Lc' (nm), 'lines' (find_lines) and 'sdk_lambda'
        (theoretical wavelength of the N_SENSOR pixels, Shamrock.get_calibration).
    second_order : bool
        use also the 2nd order of the Hg lines (no order sorting filter).
    slit_width_um : float
        width (µm) of the entrance slit during the calibration : the resolution is stored for this width
        (calib['resolution_by_slit'], read by Spectrograph_Parameters.resolution_th).

    Returns
    -------
    calib (dict), table (dict of np.ndarray : Lc, x, lam_ref, label, residual, used).
    """
    lam_ref, labels = reference_lines(second_order)
    xs = np.arange(N_SENSOR)

    # 1st pass : theoretical calibration of the SDK, both orientations of the camera are tested
    best = None
    for flip in (False, True):
        rows = []
        for m in measurements:
            sdk = np.asarray(m['sdk_lambda'], dtype=float)
            if flip:
                sdk = sdk[::-1]
            xp = np.array([l['x'] for l in m['lines']])
            match = identify_lines(xp, np.interp(xp, xs, sdk), lam_ref, max_shift, tol)
            rows += [(m['Lc'], xp[k], lam_ref[i]) for k, i in enumerate(match) if i >= 0]
        if best is None or len(rows) > len(best[1]):
            best = (flip, rows)
    flip, rows = best
    if verbose:
        print('1st pass : ' + str(len(rows)) + ' lines identified, camera ' + ('flipped' if flip else 'not flipped') +
              ' relative to the SDK calibration')
    if len(rows) == 0:
        raise RuntimeError('No line identified : check the lamp, the exposure time and the spectra')
    Lc, x, lam = (np.array(c) for c in zip(*rows))
    calib, _ = fit_model(x, Lc, lam, deg_x, deg_Lc)

    # 2nd pass : identification with the fitted model (all the peaks, tighter tolerance)
    for _ in range(2):
        rows = []
        for m in measurements:
            xp = np.array([l['x'] for l in m['lines']])
            if len(xp) == 0:
                continue
            match = _match(wavelengths_from_calib(xp, m['Lc'], calib), lam_ref, max(4 * calib['rms_nm'], 0.2))
            rows += [(m['Lc'], xp[k], lam_ref[i], labels[i], m['lines'][k]['fwhm'],
                      m['lines'][k]['isolated'] and not m.get('saturated', False)) for k, i in enumerate(match) if i >= 0]
        Lc, x, lam, lab, fwhm_px, res_ok = (np.array(c) for c in zip(*rows))
        calib, keep = fit_model(x, Lc, lam, deg_x, deg_Lc)

    residual = np.array([lam[k] - wavelengths_from_calib(np.array([x[k]]), Lc[k], calib)[0] for k in range(len(x))])
    calib['n_lines'] = int(np.sum(keep))
    calib['second_order'] = bool(second_order)
    fwhm_px = fwhm_px.astype(float)
    fwhm_nm = np.array([fwhm_px[k] * dispersion_from_calib(np.array([x[k]]), Lc[k], calib)[0] for k in range(len(x))])
    table = {'Lc': Lc, 'x': x, 'lam_ref': lam, 'label': lab, 'residual': residual, 'used': keep,
             'fwhm_px': fwhm_px, 'fwhm_nm': fwhm_nm, 'res_ok': res_ok.astype(bool)}
    resolution, res_used = fit_resolution(table, calib)
    table['res_used'] = res_used
    if resolution is not None:
        resolution['mean_380_930_nm'] = mean_resolution(calib, resolution)
        resolution['slit_width_um'] = None if slit_width_um is None else float(slit_width_um)
        calib['resolution'] = resolution
        if slit_width_um is not None:
            calib['resolution_by_slit'] = {float(slit_width_um): {'lamp': resolution['mean_380_930_nm']}}

    if verbose:
        print('2nd pass : ' + str(np.sum(keep)) + ' lines used (' + str(np.sum(~keep)) + ' rejected), rms = ' +
              str(calib['rms_nm']) + ' nm, max = ' + str(round(np.max(np.abs(residual[keep])), 4)) + ' nm')
        for L in np.unique(Lc):
            k = Lc == L
            print('    Lc = ' + str(L) + ' nm : ' + str(np.sum(keep & k)) + ' lines ' +
                  str(np.round(lam[keep & k], 2).tolist()))
        if resolution is not None:
            print('resolution (FWHM) : median ' + str(resolution['median_nm']) + ' nm (' + str(resolution['median_px']) +
                  ' pixels), ' + str(resolution['min_nm']) + ' - ' + str(resolution['max_nm']) + ' nm on ' +
                  str(resolution['n_lines']) + ' isolated lines, rms of the model ' + str(resolution['rms_nm']) + ' nm')
            print('mean resolution between ' + str(RES_RANGE[0]) + ' and ' + str(RES_RANGE[1]) + ' nm : ' +
                  str(resolution['mean_380_930_nm']) + ' nm (slit ' + str(slit_width_um) + ' µm)')
    return calib, table


def mean_resolution(calib: dict, resolution: dict, lam_range: tuple = None) -> float:
    """Mean of the resolution model (nm) over the sensor and the calibrated Lc, for the wavelengths in
    lam_range (default RES_RANGE = 380 - 930 nm)."""
    lam_range = RES_RANGE if lam_range is None else lam_range
    entry = dict(calib, resolution=resolution)
    xs = np.arange(0, N_SENSOR, 4)
    values = []
    for L in np.arange(calib['Lc_range'][0], calib['Lc_range'][1] + 1e-9, 5.0):
        lam = wavelengths_from_calib(xs, L, entry)
        k = (lam >= lam_range[0]) & (lam <= lam_range[1])
        values.append(resolution_from_calib(xs[k], L, entry))
    return float(f'{np.mean(np.concatenate(values)):.4g}')


def measured_resolution(grating_nbr: int, slit_width_um: float, method: str = None, verbose: bool = False):
    """
    Measured spectral resolution (nm, mean between 380 and 930 nm) for the grating and the slit width,
    from calib['resolution_by_slit'] = {slit width (µm): {'lamp': (nm), 'laser': (nm), 'used': 'lamp' or 'laser'}}.
    method : 'lamp' or 'laser', None (default) : the method chosen when it was written ('used').
    None if not measured for this slit width. No interpolation between two widths : when the slit is wider
    than the image of the light in its plane (fiber, light sheet), the resolution does not depend on the slit anymore.
    """
    calib = WAVELENGTH_CALIB.get(int(grating_nbr))
    if calib is None or slit_width_um is None or not calib.get('resolution_by_slit'):
        return None
    by_slit = calib['resolution_by_slit']
    slits = sorted(float(k) for k in by_slit)
    match = [k for k in by_slit if np.isclose(float(k), slit_width_um, rtol=0.01)]
    if not match:
        if verbose:
            print('Warning, resolution not measured for a slit of ' + str(slit_width_um) + ' µm (measured : ' +
                  str(slits) + ' µm) : run main_SPIM1D_spectral_calib.py with this slit width')
        return None
    values = by_slit[match[0]]
    if not isinstance(values, dict):                  # old format : one value
        return float(values)
    if method is None:
        method = values.get('used') or ('laser' if 'laser' in values else 'lamp')
    if method not in values:
        if verbose:
            print('Warning, resolution not measured with the method ' + method + ' for a slit of ' + str(slit_width_um) + ' µm')
        return None
    if verbose:
        print('resolution measured with the ' + method + ' (slit ' + str(slit_width_um) + ' µm) : ' + str(values[method]) + ' nm')
    return float(values[method])


def measure_laser_line(spectrum: np.ndarray, x: np.ndarray, margin: float = 3, width: float = 3):
    """
    Position and FWHM of the laser line (the highest peak of the spectrum), with a local linear background
    fitted on both sides of the line, between margin and margin + width FWHM (the fluorescence of the
    cuvette is a broad background under the line).

    Returns
    -------
    dict(x, fwhm, height) in pixels of the sensor, None if the line is cut by the edge of the sensor.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    n = len(spectrum)

    def crossings(s, p):
        half = s[p] / 2
        left, right = p, p
        while left > 0 and s[left - 1] > half:
            left -= 1
        while right < n - 1 and s[right + 1] > half:
            right += 1
        if left == 0 or right == n - 1:
            return None
        x_left = x[left - 1] + (half - s[left - 1]) / (s[left] - s[left - 1]) * (x[left] - x[left - 1])
        x_right = x[right] + (s[right] - half) / (s[right] - s[right + 1]) * (x[right + 1] - x[right])
        idx = np.arange(left, right + 1)
        xc = np.sum((s[idx] - half) * x[idx]) / np.sum(s[idx] - half)
        return x_left, x_right, xc, left, right

    p = int(np.argmax(ndimage.uniform_filter1d(spectrum, 3)))
    s = spectrum - np.percentile(spectrum, 5)
    c = crossings(s, p)
    for _ in range(3):   # local background, then new FWHM
        if c is None:
            return None
        f = max(c[4] - c[3] + 1, 2)
        side = np.r_[np.arange(p - int((margin + width) * f), p - int(margin * f)),
                     np.arange(p + int(margin * f), p + int((margin + width) * f))]
        side = side[(side >= 0) & (side < n)]
        if len(side) < 4:
            return None
        a, b = np.polyfit(side, spectrum[side], 1)
        s = spectrum - (a * np.arange(n) + b)
        c = crossings(s, p)
    if c is None:
        return None
    return {'x': float(c[2]), 'fwhm': float(abs(c[1] - c[0])), 'height': float(s[p])}


def laser_resolution(measurements: list, grating_nbr: int = 1, linewidths: dict = None,
                     lam_range: tuple = None, verbose: bool = True):
    """
    Spectral resolution measured with laser lines (light scattered by the cuvette, light sheet of the SPIM).

    For each laser, the central wavelength Lc is scanned so that the line moves along the sensor.
    The FWHM in pixels is converted in nm with the dispersion of WAVELENGTH_CALIB if the grating is
    calibrated, otherwise with the dispersion measured by the scan itself (|d Lc / d x| at fixed lambda).
    The width of the laser line is removed quadratically : sqrt(FWHM**2 - linewidth**2).

    Parameters
    ----------
    measurements : list of dict
        one dict per (laser, Lc) with the keys 'laser_nm', 'Lc', 'x', 'spectrum', 'saturated'.
    linewidths : dict
        {laser wavelength (nm): FWHM of the laser line (nm)}. The default is None : 0 for all the lasers.
    lam_range : tuple
        range of the mean resolution, default RES_RANGE (380 - 930 nm).

    Returns
    -------
    result (dict), table (dict of np.ndarray : laser_nm, Lc, x, fwhm_px, dispersion, fwhm_nm, lam_measured).
    """
    lam_range = RES_RANGE if lam_range is None else lam_range
    linewidths = linewidths or {}
    calib = WAVELENGTH_CALIB.get(int(grating_nbr))
    calibrated = calib is not None and 'coeffs' in calib

    rows = []
    for m in measurements:
        if m.get('saturated', False):
            if verbose:
                print('laser ' + str(m['laser_nm']) + ' nm, Lc = ' + str(m['Lc']) + ' nm : saturated, not used')
            continue
        line = measure_laser_line(m['spectrum'], m['x'])
        if line is None:
            if verbose:
                print('laser ' + str(m['laser_nm']) + ' nm, Lc = ' + str(m['Lc']) + ' nm : line cut by the edge, not used')
            continue
        rows.append((float(m['laser_nm']), float(m['Lc']), line['x'], line['fwhm']))
    if not rows:
        raise RuntimeError('No laser line measured : check the laser, the shutter, the notch filters and the exposure time')
    laser_nm, Lc, x, fwhm_px = (np.array(c) for c in zip(*rows))

    disp = np.full(len(x), np.nan)
    if calibrated:
        disp = np.array([dispersion_from_calib(np.array([x[k]]), Lc[k], calib)[0] for k in range(len(x))])
    else:
        for L in np.unique(laser_nm):   # at fixed lambda : Lc = f(x), dispersion = |d Lc / d x|
            k = laser_nm == L
            if np.sum(k) >= 2:
                pfit = np.polyfit(x[k], Lc[k], min(2, np.sum(k) - 1))
                disp[k] = np.abs(np.polyval(np.polyder(pfit), x[k]))
    lw = np.array([float(linewidths.get(L, 0)) for L in laser_nm])
    fwhm_nm = np.sqrt(np.maximum((fwhm_px * disp)**2 - lw**2, 0))
    lam_measured = (np.array([wavelengths_from_calib(np.array([x[k]]), Lc[k], calib)[0] for k in range(len(x))])
                    if calibrated else np.full(len(x), np.nan))

    ok = np.isfinite(fwhm_nm) & (laser_nm >= lam_range[0]) & (laser_nm <= lam_range[1])
    if not ok.any():
        raise RuntimeError('No resolution measured : at least 2 Lc per laser (without lamp calibration), laser in ' + str(lam_range) + ' nm')
    f = lambda v: float(f'{v:.4g}')
    per_laser = {}
    for L in np.unique(laser_nm[ok]):
        k = ok & (laser_nm == L)
        per_laser[f(L)] = {'mean_nm': f(np.mean(fwhm_nm[k])), 'min_nm': f(np.min(fwhm_nm[k])),
                           'max_nm': f(np.max(fwhm_nm[k])), 'n': int(np.sum(k)), 'linewidth_nm': f(linewidths.get(L, 0))}
        if calibrated:
            per_laser[f(L)]['lambda_measured_nm'] = f(np.mean(lam_measured[k]))
    result = {'date': datetime.date.today().isoformat(), 'method': 'laser',
              'mean_nm': f(np.mean([v['mean_nm'] for v in per_laser.values()])),   # mean of the lasers
              'per_laser': per_laser, 'lam_range': list(lam_range),
              'dispersion_from': 'calibration' if calibrated else 'laser scan'}
    table = {'laser_nm': laser_nm, 'Lc': Lc, 'x': x, 'fwhm_px': fwhm_px, 'dispersion': disp,
             'fwhm_nm': fwhm_nm, 'lam_measured': lam_measured, 'used': ok}

    if verbose:
        for L, v in per_laser.items():
            print('laser ' + str(L) + ' nm : resolution ' + str(v['mean_nm']) + ' nm (' + str(v['min_nm']) + ' - ' +
                  str(v['max_nm']) + ' nm, ' + str(v['n']) + ' positions on the sensor)' +
                  ('' if 'lambda_measured_nm' not in v else ', measured at ' + str(v['lambda_measured_nm']) +
                   ' nm with the calibration (tolerance of the laser wavelength : a few nm)'))
        print('mean resolution (laser) : ' + str(result['mean_nm']) + ' nm, dispersion from the ' + result['dispersion_from'])
    return result, table


def plot_laser_resolution(measurements: list, result: dict, table: dict, save_path: str = None):
    """FWHM vs the position on the sensor for each laser, and the laser lines."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for L in np.unique(table['laser_nm']):
        k = (table['laser_nm'] == L) & table['used']
        ax[0].plot(table['x'][k], table['fwhm_nm'][k], 'o-', label=str(L) + ' nm')
    ax[0].axhline(result['mean_nm'], color='k', ls='--', lw=0.8, label='mean ' + str(result['mean_nm']) + ' nm')
    ax[0].set_xlabel('position of the line on the sensor (pixel)')
    ax[0].set_ylabel('FWHM (nm)')
    ax[0].set_title('resolution measured with the laser')
    ax[0].legend(fontsize=8)
    ax[0].grid()
    for m in measurements:
        ax[1].plot(m['x'], m['spectrum'] / max(np.max(m['spectrum']), 1), lw=0.7)
    ax[1].set_xlabel('position on the sensor (pixel)')
    ax[1].set_ylabel('normalized intensity')
    ax[1].set_title('laser lines (one curve per Lc)')
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, 'spectral_resolution_laser.png'), dpi=120)


def save_laser_resolution(path: str, measurements: list, result: dict, table: dict):
    """Save the spectra and the lines (compressed npz) and the result (json, same name)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path,
             laser_nm=np.array([m['laser_nm'] for m in measurements]),
             Lc=np.array([m['Lc'] for m in measurements]),
             expos_time=np.array([m['expos_time'] for m in measurements]),
             x=np.array([m['x'] for m in measurements]),
             spectra=np.array([m['spectrum'] for m in measurements]),
             images=np.array([np.clip(np.round(m['image']), 0, 65535).astype(np.uint16) for m in measurements]),
             result=repr(result), **{'table_' + k: v for k, v in table.items()})
    json_path = os.path.splitext(path)[0] + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, default=lambda v: v.item() if hasattr(v, 'item') else str(v))
    print('laser resolution saved in : ' + path + ' and ' + json_path)


def plot_calibration(measurements: list, calib: dict, table: dict, save_path: str = None):
    """Spectra with the identified lines, residuals, and dispersion vs Lc."""
    lam_ref, labels = reference_lines(calib.get('second_order', True))
    figs = []

    n = len(measurements)
    fig, axs = plt.subplots(n, 1, figsize=(10, 2.2 * n), squeeze=False)
    for ax, m in zip(axs[:, 0], measurements):
        lam = wavelengths_from_calib(m['x'], m['Lc'], calib)
        ax.plot(lam, m['spectrum'], lw=0.8)
        for L, name in zip(lam_ref, labels):
            if lam.min() <= L <= lam.max():
                ax.axvline(L, color='r' if 'Hg' in name else 'g', ls=':' if '2nd' in name else '--', lw=0.6)
        k = (table['Lc'] == m['Lc']) & table['used']
        ax.set_title('Lc = ' + str(m['Lc']) + ' nm, ti = ' + str(round(m['expos_time'], 4)) + ' s, ' +
                     str(np.sum(k)) + ' lines', fontsize=9)
        ax.set_xlim(lam.min(), lam.max())
    axs[-1, 0].set_xlabel('wavelength (nm)  [red: Hg, green: Ar, dotted: 2nd order]')
    fig.tight_layout()
    figs.append(('spectra', fig))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    u = table['used']
    sc = ax[0].scatter(table['lam_ref'][u], table['residual'][u], c=table['Lc'][u], s=12)
    ax[0].scatter(table['lam_ref'][~u], table['residual'][~u], marker='x', c='k', s=20, label='rejected')
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set_xlabel('wavelength (nm)')
    ax[0].set_ylabel('residual (nm)')
    ax[0].set_title('rms = ' + str(calib['rms_nm']) + ' nm, ' + str(calib['n_lines']) + ' lines')
    ax[0].legend()
    plt.colorbar(sc, ax=ax[0], label='Lc (nm)')
    Lc_fine = np.linspace(calib['Lc_range'][0], calib['Lc_range'][1], 100)
    disp = [np.mean(np.diff(wavelengths_from_calib(np.array([X0 - 0.5, X0 + 0.5]), L, calib))) for L in Lc_fine]
    span = [np.ptp(wavelengths_from_calib(np.array([0, N_SENSOR - 1]), L, calib)) for L in Lc_fine]
    ax[1].plot(Lc_fine, np.abs(disp), 'b')
    ax[1].set_xlabel('Lc (nm)')
    ax[1].set_ylabel('dispersion at the center (nm/pixel)', color='b')
    ax1b = ax[1].twinx()
    ax1b.plot(Lc_fine, span, 'r')
    ax1b.set_ylabel('spectral range of the sensor (nm)', color='r')
    ax[1].grid()
    fig.tight_layout()
    figs.append(('residuals', fig))

    if 'resolution' in calib:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        r = table['res_used']
        sc = ax[0].scatter(table['lam_ref'][r], table['fwhm_nm'][r], c=table['x'][r], cmap='coolwarm', s=14)
        ax[0].scatter(table['lam_ref'][~r & table['used']], table['fwhm_nm'][~r & table['used']], marker='x', c='k', s=20,
                      label='not used (blended, saturated or outlier)')
        plt.colorbar(sc, ax=ax[0], label='position on the sensor (pixel)')
        ax[0].set_xlabel('wavelength (nm)')
        ax[0].set_ylabel('FWHM (nm)')
        ax[0].set_title('resolution : median ' + str(calib['resolution']['median_nm']) + ' nm')
        ax[0].legend(fontsize=8)
        xs = np.arange(N_SENSOR)
        for L in np.linspace(calib['Lc_range'][0], calib['Lc_range'][1], 5):
            ax[1].plot(xs, resolution_from_calib(xs, L, calib), label='Lc = ' + str(round(L)) + ' nm')
        ax[1].scatter(table['x'][r], table['fwhm_nm'][r], c='k', s=6)
        ax[1].set_xlabel('position on the sensor (pixel)')
        ax[1].set_ylabel('FWHM (nm)')
        ax[1].set_title('model of the resolution')
        ax[1].legend(fontsize=8)
        ax[1].grid()
        fig.tight_layout()
        figs.append(('resolution', fig))

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        for name, f in figs:
            f.savefig(os.path.join(save_path, 'spectral_calib_' + name + '.png'), dpi=120)


def save_calibration(path: str, measurements: list, calib: dict, table: dict):
    """Save the raw images and spectra, the lines and the calibration (compressed npz) for traceability,
    and the calibration alone in a json file (same name)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path,
             Lc=np.array([m['Lc'] for m in measurements]),
             expos_time=np.array([m['expos_time'] for m in measurements]),
             x=np.array([m['x'] for m in measurements]),
             spectra=np.array([m['spectrum'] for m in measurements]),
             images=np.array([np.clip(np.round(m['image']), 0, 65535).astype(np.uint16) for m in measurements]),
             sdk_lambda=np.array([m['sdk_lambda'] for m in measurements]),
             calib=repr(calib), **{'table_' + k: v for k, v in table.items()})
    json_path = os.path.splitext(path)[0] + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(calib, f, indent=4, default=lambda v: v.item() if hasattr(v, 'item') else str(v))
    print('calibration saved in : ' + path + ' and ' + json_path)


def _format_dict(d: dict, indent: int) -> list:
    pad = ' ' * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(pad + repr(k) + ': {')
            lines += _format_dict(v, indent + 4)
            lines.append(pad + '},')
        elif k == 'coeffs':
            lines.append(pad + "'coeffs': [")
            for row in v:
                lines.append(pad + '    [' + ', '.join(repr(c) for c in row) + '],')
            lines.append(pad + '],')
        else:
            lines.append(pad + repr(k) + ': ' + repr(v) + ',')
    return lines


def _format_calib(calibs: dict) -> str:
    lines = ['WAVELENGTH_CALIB = {']
    for g in sorted(calibs):
        lines.append('    ' + str(g) + ': {')
        lines += _format_dict(calibs[g], 8)
        lines.append('    },')
    lines.append('}')
    return '\n'.join(lines)


def _read_calibs(module_path: str):
    """Read WAVELENGTH_CALIB in the file : (src, start, end, prefix of the block, calibs, newline)."""
    with open(module_path, encoding='utf-8', newline='') as f:
        src = f.read()
    newline = '\r\n' if '\r\n' in src else '\n'
    # the markers at the beginning of a line (not the constants CALIB_BEGIN / CALIB_END above)
    begin = re.search('^' + re.escape(CALIB_BEGIN) + r'\s*$', src, re.M)
    end = re.search('^' + re.escape(CALIB_END) + r'\s*$', src, re.M)
    if begin is None or end is None or end.start() < begin.start():
        raise RuntimeError('markers of the wavelength calibration not found in ' + module_path)
    block = src[begin.start():end.start()]
    m = re.search(r'^WAVELENGTH_CALIB = ', block, re.M)
    calibs = ast.literal_eval(block[m.end():].strip())
    return src, begin.start(), end.start(), block[:m.start()], calibs, newline


def _write_calibs(module_path: str, calibs: dict):
    """Write WAVELENGTH_CALIB in the file (and in memory if it is this module)."""
    src, start, end, prefix, _, newline = _read_calibs(module_path)
    src = src[:start] + prefix + _format_calib(calibs).replace('\n', newline) + newline + src[end:]
    with open(module_path, 'w', encoding='utf-8', newline='') as f:
        f.write(src)
    if os.path.normcase(os.path.abspath(module_path)) == os.path.normcase(os.path.abspath(__file__)):
        WAVELENGTH_CALIB.clear()
        WAVELENGTH_CALIB.update(calibs)


def _merge_resolution(old: dict, new: dict, use_method: str = None) -> dict:
    """Merge the resolutions per slit width {slit: {'lamp': , 'laser': , 'used': }}."""
    by_slit = {float(k): (dict(v) if isinstance(v, dict) else {'lamp': v}) for k, v in (old or {}).items()}
    for k, v in (new or {}).items():
        entry = by_slit.setdefault(float(k), {})
        entry.update(v)
        method = [m for m in v if m in ('lamp', 'laser')]
        if use_method in method or (method and 'used' not in entry):
            entry['used'] = use_method if use_method in method else method[0]
    return {k: by_slit[k] for k in sorted(by_slit)}


def write_wavelength_calib(grating_nbr: int, calib: dict, use_resolution: bool = True,
                           module_path: str = None, verbose: bool = True):
    """
    Write the calibration of the grating in WAVELENGTH_CALIB (in this module, between the lines
    CALIB_BEGIN and CALIB_END). The calibrations of the other gratings and the resolutions measured
    with the other slit widths or with the laser are kept.
    use_resolution : the resolution measured with the lamp becomes resolution_th (Spectrograph_Parameters)
    for this slit width. False : the method already chosen is kept (e.g. 'laser').
    WAVELENGTH_CALIB is also updated in memory : define_wavelengths_matrix uses the new calibration
    immediately, without restarting the kernel.
    """
    if module_path is None:
        module_path = os.path.abspath(__file__)
    calibs = _read_calibs(module_path)[4]
    old = calibs.get(int(grating_nbr), {})
    calib = dict(calib)
    by_slit = _merge_resolution(old.get('resolution_by_slit'), calib.get('resolution_by_slit'),
                                'lamp' if use_resolution else None)
    if 'laser_resolution' in old and 'laser_resolution' not in calib:
        calib['laser_resolution'] = old['laser_resolution']
    if by_slit:
        calib['resolution_by_slit'] = by_slit
    calibs[int(grating_nbr)] = calib
    _write_calibs(module_path, calibs)
    if verbose:
        print('calibration of the grating ' + str(grating_nbr) + ' written in ' + module_path)


def write_resolution(grating_nbr: int, slit_width_um: float, result: dict, use_resolution: bool = True,
                     module_path: str = None, verbose: bool = True):
    """
    Write the resolution measured with the laser (laser_resolution) for the grating and the slit width in
    WAVELENGTH_CALIB, without changing the wavelength calibration.
    use_resolution : this value becomes resolution_th (Spectrograph_Parameters) for this slit width.
    """
    if module_path is None:
        module_path = os.path.abspath(__file__)
    calibs = _read_calibs(module_path)[4]
    entry = dict(calibs.get(int(grating_nbr), {}))
    entry['resolution_by_slit'] = _merge_resolution(entry.get('resolution_by_slit'),
                                                    {float(slit_width_um): {'laser': result['mean_nm']}},
                                                    'laser' if use_resolution else None)
    laser = dict(entry.get('laser_resolution', {}))
    laser[float(slit_width_um)] = {k: result[k] for k in ('date', 'mean_nm', 'per_laser', 'dispersion_from', 'lam_range')}
    entry['laser_resolution'] = {k: laser[k] for k in sorted(laser)}
    calibs[int(grating_nbr)] = entry
    _write_calibs(module_path, calibs)
    if verbose:
        print('resolution measured with the laser (' + str(result['mean_nm']) + ' nm, slit ' + str(slit_width_um) +
              ' µm) written in ' + module_path + (' -> resolution_th' if use_resolution else ''))


def new_calib_entry(fit: dict, grooves: int, lamp: str = 'HG-1 Ocean Optics (Hg-Ar)', **metadata) -> dict:
    """Entry of WAVELENGTH_CALIB : metadata + fitted model."""
    entry = {'date': datetime.date.today().isoformat(), 'lamp': lamp, 'grooves': int(grooves)}
    entry.update(metadata)
    entry.update(fit)
    return entry
