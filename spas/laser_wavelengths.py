# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 2026

@author: mahieu

Real wavelengths of the lasers of the combiner, measured with a spectrometer (2026-11-23). The lasers are
called by their nominal wavelength (combiner, scripts) ; the real wavelength is used to analyse the data
(check of the spectral calibration with the laser lines) and stored in the metadata of the acquisitions.
"""

MEASURED_WAVELENGTHS_DATE = '2026-11-23'

# {nominal wavelength (nm): wavelength measured with the spectrometer (nm)}
MEASURED_WAVELENGTHS_NM = {
    405: 405.76,
    473: 469.4,
    515: 510.43,
    561: 561.7,
    594: 594.3,
    638: 639.5,
}

TOLERANCE_NM = 5   # max difference between the wavelength asked and the nominal wavelength of a laser


def measured_wavelength(nominal_nm: float, verbose: bool = False) -> float:
    """
    Real wavelength (nm) of the laser of nominal wavelength nominal_nm (closest laser within TOLERANCE_NM).
    If the laser has not been measured, the nominal wavelength is returned.
    """
    closest = min(MEASURED_WAVELENGTHS_NM, key = lambda w: abs(w - nominal_nm))
    if abs(closest - nominal_nm) <= TOLERANCE_NM:
        return MEASURED_WAVELENGTHS_NM[closest]
    if verbose:
        print('Warning, laser ' + f'{nominal_nm:g}' + ' nm : real wavelength not measured, nominal wavelength used')
    return float(nominal_nm)
