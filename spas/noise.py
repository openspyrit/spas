from dataclasses import dataclass

import numpy as np

@dataclass
class noiseClass:
    """Acquisition noise parameters.

    Holds acquisition noise parameters specific to an acquisition setup.
    Noise is described as mix of a Poisson and a Gaussian distribution.
    Based on the EMVA 1288 standard.

    Attributes:
        mu (np.ndarray): 
            Mean value of the dark noise, e.g. the Gaussian component of the
            noise. Each value corresponds to a different wavelength associated.
        sigma (np.ndarray):
            Standard deviation of the dark noise, e.g. the Gaussian component of
            the noise. Each value corresponds to a different wavelength 
            associated.
        k (np.ndarray):
            Proportionality factor multiplying the Poisson component of the
            noise. Each value corresponds to a different wavelength associated.
    """

    mu: np.ndarray
    sigma: np.ndarray
    K: np.ndarray


def load_noise(path):
    """Loads noise calibration parameters"""

    data = np.load(path)

    return noiseClass(data['mu'], data['sigma'], data['k'])