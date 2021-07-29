from dataclasses import dataclass

import numpy as np

@dataclass
class noiseClass:
    mu: np.ndarray
    sigma: np.ndarray
    K: np.ndarray


def load_noise(path):
    """Loads noise calibration parameters"""

    data = np.load(path)

    return noiseClass(data['mu'], data['sigma'], data['k'])