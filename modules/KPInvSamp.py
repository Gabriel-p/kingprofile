
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from modules import pinky


def main(method, N_clust, CI, rc, rt, ecc, theta, xmax, ymax, cl_cent):
    """
    """

    # Estimate number of field stars, given CI, N_clust, and rt
    area_frame = xmax * ymax
    area_cl = np.pi * rt**2

    # Generate positions for field stars
    N_fl = estimateNfield(N_clust, CI, area_frame, area_cl)
    x_fl = np.random.uniform(0., xmax, N_fl)
    y_fl = np.random.uniform(0., ymax, N_fl)

    if method == 'KP_4':
        x_cl, y_cl = invTrnsfSmpl_ellip(cl_cent, rc, rt, ecc, theta, N_clust)
    else:
        # Sample King's profile with fixed rc, rt values.
        x_cl, y_cl = invTrnsfSmpl(cl_cent, rc, rt, N_clust)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # # definitions for the axes
    # left, width = 0.1, 0.65
    # bottom, height = 0.1, 0.65
    # spacing = 0.005
    # rect_scatter = [left, bottom, width, height]
    # rect_histx = [left, bottom + height + spacing, width, 0.2]
    # rect_histy = [left + width + spacing, bottom, 0.2, height]
    # # start with a rectangular Figure
    # plt.figure(figsize=(8, 8))
    # ax_scatter = plt.axes(rect_scatter)
    # ax_scatter.tick_params(direction='in', top=True, right=True)
    # # the scatter plot:
    # # ax_scatter.scatter(x, y)
    # ax_scatter.scatter(x_cl, y_cl, c='b', alpha=.5)
    # ax_scatter.scatter(x_cl4, y_cl4, c='r', alpha=.5)

    # ax_histx = plt.axes(rect_histx)
    # ax_histx.tick_params(direction='in', labelbottom=False)
    # sns.distplot(x_cl, hist=False, kde=True, color='b')
    # sns.distplot(x_cl4, hist=False, kde=True, color='r')

    # ax_histy = plt.axes(rect_histy)
    # ax_histy.tick_params(direction='in', labelleft=False)
    # sns.distplot(y_cl, hist=False, kde=True, color='b', vertical=True)
    # sns.distplot(y_cl4, hist=False, kde=True, color='r', vertical=True)

    # ax_histx.set_xlim(ax_scatter.get_xlim())
    # ax_histy.set_ylim(ax_scatter.get_ylim())

    # plt.show()

    return x_cl, y_cl, x_fl, y_fl


def estimateNfield(N_membs, CI, tot_area, cl_area):
    """
    Estimate the total number of field stars that should be generated so
    that the CI is respected.
    """

    # Number of field stars in the cluster area
    N_field_in_clreg = N_membs / (1. / CI - 1.)

    # Field stars density
    field_dens = N_field_in_clreg / cl_area

    # Total number of field stars in the entire frame
    N_field = int(field_dens * tot_area)

    return N_field


def invTrnsfSmpl(cl_cent, rc, rt, N_samp):
    """
    Sample King's profile using the inverse CDF method.
    """
    r_0rt = np.linspace(0., rt, 100)
    # The CDF is defined as: $F(r)= \int_{r_low}^{r} PDF(r) dr$
    # Sample the CDF
    CDF_samples = []
    for r in r_0rt:
        CDF_samples.append(quad(rKP, 0., r, args=(rc, rt))[0])

    # Normalize CDF
    CDF_samples = np.array(CDF_samples) / CDF_samples[-1]

    # Inverse CDF
    inv_cdf = interp1d(CDF_samples, r_0rt)

    # Sample the inverse CDF
    cl_dists = inv_cdf(np.random.rand(N_samp))

    # Generate positions for cluster members, given heir KP distances to
    # the center.
    phi = np.random.uniform(0., 1., N_samp) * 2 * np.pi
    x_cl = cl_cent[0] + cl_dists * np.cos(phi)
    y_cl = cl_cent[1] + cl_dists * np.sin(phi)

    return x_cl, y_cl


def KingProf(r_in, rc, rt):
    """
    King (1962) profile.
    """
    return ((1. / np.sqrt(1. + (r_in / rc) ** 2)) -
            (1. / np.sqrt(1. + (rt / rc) ** 2))) ** 2


def rKP(r, rc, rt):
    r"""r \times King's profile"""
    return r * KingProf(r, rc, rt)


# def centDens(arr, N_memb, rc, rt):
#     """
#     Central density constant. Integrate up to rt.
#     """
#     # aa = 2. * np.pi * quad(rKP, 0., rt, args=(rc, rt))[0]
#     return N_memb / aa

def centDens(arr, N_memb, rc, rt, ecc=0.):
    """
    Central density constant. Integrate up to rt.

    np.trapz() is substantially faster than scipy.quad()

    https://math.stackexchange.com/a/1891110/37846
    """
    b = np.sqrt(1. - ecc**2)
    integ = np.trapz(2. * np.pi * (b * arr) * KingProf(arr, rc, rt), arr)
    return N_memb / integ


def inEllipse(xy_in, cl_cent, rt, ecc, theta):
    """
    Source: https://stackoverflow.com/a/59718947/1391441
    """
    # Transpose
    xy = xy_in.T

    # The tidal radius 'rt' is made to represent the width ('a')
    # Width (squared)
    a2 = rt**2
    # Height (squared)
    b2 = a2 * (1. - ecc**2)

    # distance between the center and the foci
    foc_dist = np.sqrt(np.abs(b2 - a2))
    # vector from center to one of the foci
    foc_vect = np.array([foc_dist * np.cos(theta), foc_dist * np.sin(theta)])
    # the two foci
    el_foc1 = cl_cent + foc_vect
    el_foc2 = cl_cent - foc_vect

    # For each x,y: calculate z as the sum of the distances to the foci;
    # np.ravel is needed to change the array of arrays (of 1 element) into a
    # single array. Points are exactly on the ellipse when the sum of distances
    # is equal to the width
    z = np.ravel(np.linalg.norm(xy - el_foc1, axis=-1) +
                 np.linalg.norm(xy - el_foc2, axis=-1))

    # Mask that identifies the points inside the ellipse
    in_ellip_msk = z <= 2. * rt  # np.sqrt(max(a2, b2)) * 2.

    return in_ellip_msk


def invTrnsfSmpl_ellip(cl_cent, rc, rt, ecc, theta, Nsample):
    """
    Sample King's profile using the inverse CDF method.
    """

    c1 = 1 / rc**2
    c2 = 1 - ecc**2
    c3 = -1. / np.sqrt(1 + (rt / rc)**2)

    def KP_ellip(x, y, rc, rt):
        return (1. / np.sqrt(1 + c1 * (x**2 + (y**2 / c2))) + c3)**2

    width = rt
    height = width * np.sqrt(1 - ecc**2)
    x = np.linspace(-width, width, 50)
    y = np.linspace(-height, height, 50)
    XX, YY = np.meshgrid(x, y)
    P = KP_ellip(XX, YY, rc, rt)
    p = pinky.Pinky(P=P, extent=[-width, width, -height, height])

    in_cent, in_theta = (0., 0.), 0.
    xy_in_ellipse = []
    while True:
        sampled_points = p.sample(Nsample)
        msk = inEllipse(sampled_points.T, in_cent, rt, ecc, in_theta)
        xy_in_ellipse += list(sampled_points[msk])
        if len(xy_in_ellipse) >= Nsample:
            break
    samples = np.array(xy_in_ellipse)[:Nsample]

    # Rotate sample via rotation matrix
    R = np.array(
        ((np.cos(theta), -np.sin(theta)),
            (np.sin(theta), np.cos(theta))))
    xy_clust = R.dot(samples.T)

    # Shift to center
    xy_clust = (xy_clust.T + cl_cent).T
    x_cl, y_cl = xy_clust

    return x_cl, y_cl
