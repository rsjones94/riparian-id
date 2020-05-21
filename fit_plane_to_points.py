import numpy as np
import scipy.linalg

def fit_plane(data):
    """
    Fits a plane to a set of 3d data

    Args:
        data: np array where each row is [x, y, z]

    Returns:
        A tuple containing the coefficients of the regression. Z = C[0]*X + C[1]*Y + C[2]
    """

    # best-fit linear plane
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

    return C[0], C[1], C[2]
    # Z = C[0]*X + C[1]*Y + C[2]