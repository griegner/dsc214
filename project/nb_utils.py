import numpy as np
from ripser import ripser
from scipy import sparse


def gen_ar2_coeffs(oscillatory=False, phi1="both", random_seed=0):
    """generate coefficients for an stationary AR(2) process"""
    rng = np.random.default_rng(seed=random_seed)
    if phi1 == "positive":
        phi1 = rng.uniform(0, 1.9)
    elif phi1 == "negative":
        phi1 = rng.uniform(-1.9, 0)
    else:
        phi1 = rng.uniform(-1.9, 1.9)
    if oscillatory:
        phi2 = rng.uniform(-1, -0.25 * phi1**2)
    else:
        phi2 = rng.uniform(np.max([-1, -0.25 * phi1**2]), np.min([1 + phi1, 1 - phi1]))
    return np.array([phi1, phi2])


def sublevel_set_filtration(X):
    """https://github.com/itsmeafra/Sublevel-Set-TDA?tab=readme-ov-file"""

    N = len(X)

    # sublevelset filtration
    # add edges between adjacent points in the time series, with the "distance"
    # along the edge equal to the max value of the points it connects
    I = np.arange(N - 1)
    J = np.arange(1, N)
    V = np.maximum(X[0:-1], X[1::])

    # add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, X))

    # create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    dgm0 = ripser(D, maxdim=0, distance_matrix=True)["dgms"][0]
    dgm0 = dgm0[dgm0[:, 1] - dgm0[:, 0] > 1e-3, :]

    # remove infinity points
    dgm0 = np.vstack(([0, 0], dgm0))
    dgm0 = np.hstack((dgm0, np.zeros((len(dgm0), 1))))
    where_are_inf = np.isinf(dgm0)
    dgm0 = dgm0[~where_are_inf[:, 1]]

    return dgm0[None, ...]
