import numpy as np

def wrap_pi(angle):
    """Wrap angle to [-pi, pi)."""
    return np.mod(angle + np.pi, (2 * np.pi)) - np.pi

def fit_cosine_fourier_series(xdata, ydata, period=1, order = 10):
    """Fit cosine fourier series for data and returns coefficients

    Returns cosine fourier coefficients of size (order + 1)

    Given series of x and y data, we write linear systems of equation as
        y = A(x) * beta
    where
        y = [y1, ... ys]
        x = [x1, ... xs]
        A(x) = [[1, cos(2pi*x1/P), ..., cos(2pi*n*x1/P)]
                   ...
                [1, cos(2pi*xs/P), ..., cos(2pi*n*xs/P)]]
        beta = [a0, a1, a2, ..., an], fourier coefficients
    """

    cos_rows = [np.cos(2*np.pi*i*xdata/period) for i in range(order+1)]
    A_matrix = np.array(cos_rows).T

    # solve for x
    pinv = np.linalg.pinv(A_matrix)
    beta = pinv.dot(ydata)

    return beta