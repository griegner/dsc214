import numpy as np


def gen_ar2_coeffs(oscillatory=False, random_seed=0):
    """generate coefficients for an stationary AR(2) process"""
    rng = np.random.default_rng(seed=random_seed)
    phi1 = rng.uniform(0, 2)
    if oscillatory:
        phi2 = rng.uniform(-1, -0.25 * phi1**2)
    else:
        phi2 = rng.uniform(np.max([-1, -0.25 * phi1**2]), np.min([1 + phi1, 1 - phi1]))
    return np.array([phi1, phi2])
