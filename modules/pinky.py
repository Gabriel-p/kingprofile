
import numpy as np
from scipy.interpolate import interp1d, interp2d

"""
Author: Pablo Romano (https://github.com/pgromano)

Source: https://github.com/pgromano/pinky
"""


class Pinky(object):
    def __init__(self, P=None, **kwargs):
        self.P = P
        self.extent = kwargs.get('extent', np.array([-1, 1, -1, 1]))
        self.bins = kwargs.get('bins', 100)

    def sample(self, n_samples=1, **kwargs):
        if self.P is None:
            raise AttributeError('''
            Probability distribution not defined.''')
        return sample(self.P, n_samples, extent=self.extent, **kwargs)


def sample(P_num, n_samples, **kwargs):
    # Get key word arguments
    extent = kwargs.get('extent', np.array([-1, 1, -1, 1]))
    noise = kwargs.get('noise', False)

    if np.any(np.array(P_num.shape) <= 100):
        r = kwargs.get('r', 10)
    else:
        r = kwargs.get('r', 1)

    # Reshape extent to more convenient shape
    extent = [[extent[0],extent[1]],[extent[2],extent[3]]]

    # Check valid probability distribution was passed
    _check_dist(P_num)

    # Build axes
    axes = [np.linspace(extent[i][0], extent[i][1], P_num.T.shape[i]) for i in range(2)]
    tmp_axes = [np.linspace(0, 1, axes[i].shape[0]) for i in range(2)]

    # Create forward scale transformers,  range --> [0,1]
    Xf = interp1d(axes[0], tmp_axes[0])
    Yf = interp1d(axes[1], tmp_axes[1])

    # Create reverse scale transformers, [0, 1] --> range
    Xr = interp1d(tmp_axes[0], axes[0])
    Yr = interp1d(tmp_axes[1], axes[1])

    # Build interpolated P distribution
    if r < 1:
        raise AttributeError('''
        Resolution factor must be int type >= 1.''')
    if type(r) != int:
        raise TypeError('''
        Resolution factor should be of type integer.''')

    # Interpolate probability density on reduced scale coordinates
    P = interp2d(Yf(axes[1]), Xf(axes[0]), P_num.T)

    # Redefining axes with resolution factor
    tmp_axes = [np.linspace(0, 1, int(r*P_num.T.shape[i])) for i in range(2)]

    # Column distribution
    P_col = P(tmp_axes[1], tmp_axes[0]).sum(1)

    # Randomly sample values along x
    idx = _gendist(P_col, n_samples)
    x0 = Xr(tmp_axes[0])[idx]

    # Randomly sample values along y
    try:
        idx = [_gendist(P(tmp_axes[1], Xf(i)), 1) for i in x0]
    except:
        idx = _gendist(P(tmp_axes[1], Xf(x0)), 1)
    y0 = Yr(tmp_axes[1])[np.squeeze(idx)]

    if noise is False:
        return np.column_stack([x0,y0])
    else:
        return _add_noise(np.column_stack([x0,y0]))

def _add_noise(X):

    eps = _variance(X)
    return X+eps

def _variance(X):
    return np.random.randn(*X.shape)*X.std(0)**2

def _gendist(P, n):
    if np.any(P < 0):
        raise AttributeError('''
        All elements of P must be positive.''')

    # Normalize P
    Pnorm = np.append(0,P)/P.sum()

    # Create cumulative distribution
    Pcum = np.cumsum(Pnorm)

    # Create random matrix
    R = np.random.rand(1, n)

    # Calculate output matrix
    V = np.arange(P.shape[0], dtype=int)
    idx = np.digitize(R,Pcum) - 1
    return np.squeeze(V[idx.astype(int)].reshape(n))


def _check_dist(P):
    if len(P.shape) != 2:
        raise AttributeError('''
        Input distribution must be an N x M matrix.''')
    if np.any(P<0):
        raise AttributeError('''
        All input probability values must be positive.''')
