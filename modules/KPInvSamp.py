
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


def main(N_clust, CI, rc, rt, xmax, ymax, cx, cy):
    """
    """

    # Estimate number fo field stars, given CI, N_clust, and rt
    area_frame = xmax * ymax
    area_cl = np.pi * rt**2

    # Generate positions for field stars
    N_fl = estimateNfield(N_clust, CI, area_frame, area_cl)
    x_fl = np.random.uniform(0., xmax, N_fl)
    y_fl = np.random.uniform(0., ymax, N_fl)

    # Sample King's profile with fixed rc, rt values.
    cl_dists = invTrnsfSmpl(rc, rt, N_clust)

    # Generate positions for cluster members, given heir KP distances to the
    # center.
    theta = np.random.uniform(0., 1., N_clust) * 2 * np.pi
    x_cl = cx + cl_dists * np.cos(theta)
    y_cl = cy + cl_dists * np.sin(theta)

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


def invTrnsfSmpl(rc, rt, N_samp):
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
    samples = inv_cdf(np.random.rand(N_samp))

    return samples


def rKP(r, rc, rt):
    """King's profile"""
    return r * KingProf(r, rc, rt)


def centDens(arr, N_memb, rc, rt):
    """
    Central density constant. Integrate up to rt.
    """
    aa = np.trapz(2. * np.pi * arr * KingProf(arr, rc, rt), arr)
    # aa = 2. * np.pi * quad(rKP, 0., rt, args=(rc, rt))[0]
    return N_memb / aa


def KingProf(r_in, rc, rt):
    """
    King (1962) profile.
    """
    return ((1. / np.sqrt(1. + (r_in / rc) ** 2)) -
            (1. / np.sqrt(1. + (rt / rc) ** 2))) ** 2
