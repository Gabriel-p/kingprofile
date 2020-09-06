
import numpy as np
from scipy import spatial
from scipy.optimize import differential_evolution
from .KPInvSamp import centDens, KingProf, inEllipse


def main(method, area_frame, N_clust, cl_cent, rc0, rt0, xy, corr_fact):
    """
    """
    xy_cent_dist = spatial.distance.cdist([cl_cent], xy)[0]

    rt_max = 2. * rt0
    msk = xy_cent_dist <= rt_max
    r_in = xy_cent_dist[msk]
    xy_in = xy[msk].T
    r_in.sort()

    # Tidal radius array. Used for integrating
    rt_rang = np.linspace(0., rt_max, 1000)

    area_rtmax = np.pi * rt_max**2
    fd = np.sum(~msk) / (area_frame - area_rtmax)
    N_memb = max(1, msk.sum() - fd * area_rtmax)

    ecc, theta = 0., 0.
    if method == 'KP_2':
        rc, rt = gridBruteForce(rt0, rt_max, rt_rang, r_in, fd, N_memb)
    elif method == 'KP_4':
        rc, rt, ecc, theta = diffEvol(
            cl_cent, rt_max, rt_rang, xy_in, r_in, fd, N_memb)
    elif method == 'optmRad':
        rc, rt = optmRad(rt0, rt_max, r_in, fd, N_memb, corr_fact)

    return N_memb, rc, rt, ecc, theta


def diffEvol(cl_cent, rt_max, rt_rang, xy_in, r_in, fd, N_memb):
    """
    """

    # rc, rt, ecc, theta
    bounds = ((.05 * rt_max, rt_max), (.05 * rt_max, rt_max),
              (.5, .9), (-np.pi / 2., np.pi / 2.))
    # Minimize lnlike
    result = differential_evolution(
        lnlike, bounds,
        args=(rt_max, cl_cent, fd, N_memb, xy_in, r_in, rt_rang),
        maxiter=2000, popsize=50, tol=0.00001)

    return result.x


def lnlike(pars, rt_max, cl_cent, fd, N_memb, xy_in, r_in, rt_rang):
    """
    As defined in Pieres et al. (2016)
    """

    rc, rt, ecc, theta = pars
    # Prior.
    if rt <= rc or rc <= 0. or rt > rt_max or ecc < .0 or ecc > 1. or\
            theta < -np.pi / 2. or theta > np.pi / 2.:
        return np.inf

    x, y = xy_in
    # Identify stars inside this ellipse
    in_ellip_msk = inEllipse(xy_in, cl_cent, rt, ecc, theta)
    # N_in_region = in_ellip_msk.sum()

    # General form of the ellipse
    # https://math.stackexchange.com/a/434482/37846
    dx, dy = x - cl_cent[0], y - cl_cent[1]
    x1 = dx * np.cos(theta) + dy * np.sin(theta)
    y1 = dx * np.sin(theta) - dy * np.cos(theta)
    # The 'width' ('a') is used instead of the radius 'r' (sqrt(x**2+y**2))
    # in King's profile, for each star
    a_xy = np.sqrt(x1**2 + y1**2 / (1. - ecc**2))

    # Values outside the ellipse contribute only 'fd' to the likelihood.
    a_xy[~in_ellip_msk] = rt
    KP = KingProf(a_xy, rc, rt)

    # Central density
    i = np.searchsorted(rt_rang, rt)
    k = centDens(rt_rang[:i], N_memb, rc, rt, ecc)

    # Likelihood
    li = k * KP + fd

    # Sum of log-likelighood
    sum_log_lkl = np.log(li).sum()

    return -sum_log_lkl


def gridBruteForce(rt0, rt_max, rt_rang, r_in, fd, N_memb):
    """
    """

    log_fd = np.log(fd * np.ones(r_in.size))
    res = []
    for rc in np.linspace(10., rt0, 100):
        rt_all = np.linspace(rc + 5., rt_max, 100)
        i_rang = np.searchsorted(rt_rang, rt_all)
        i_ri = np.searchsorted(r_in, rt_all)
        for i, rt in enumerate(rt_all):

            k = centDens(rt_rang[:i_rang[i]], N_memb, rc, rt)

            # r_in2 = np.clip(r_in, a_min=0., a_max=rt)
            li = k * KingProf(r_in[:i_ri[i]], rc, rt) + fd

            # Values outside the tidal radius contribute 'fd'.
            lkl = np.log(li).sum() + log_fd[i_ri[i]:].sum()

            res.append([rc, rt, lkl])

    res = np.array(res).T
    i = np.argmax(res[-1])
    rc, rt = res[0][i], res[1][i]

    return rc, rt


def optmRad(rt0, rt_max, r_in, fd, N_memb, corr_fact):
    """
    For some reason this method returns a radius that is around 70% of
    the real r_t:

    r_optm = 0.7 * r_t --> r_t = r_optm / 0.7

    Hence, we apply the 'corr_fact' factor below.
    """
    rad_radii = np.linspace(5., rt_max, 500)

    data = []
    for i, rad in enumerate(rad_radii):

        # Stars within radius.
        n_in_cl_reg = (r_in <= rad).sum()
        if n_in_cl_reg == 0:
            continue

        n_fl = fd * np.pi * rad**2
        n_memb = n_in_cl_reg - n_fl
        data.append([rad, n_memb, n_fl])

    # rads, N_membs, N_field, CI
    data = np.clip(data, a_min=0., a_max=None).T

    # Normalizing separately is important. Otherwise it selects low radii
    # values.
    N_membs = data[1] / data[1].max()
    N_field = data[2] / data[2].max()
    idx = np.argmax(N_membs - N_field)
    rt_x = data[0][idx] / corr_fact

    rt_rang = np.linspace(0., rt_x, 1000)
    i_ri = np.searchsorted(r_in, rt_x)
    log_fd_sum = np.log(fd * np.ones(r_in[i_ri:].size)).sum()

    # This method does not estimate a 'core radius', so we obtain it from
    # the King Profile assuming that 'rt_x' is the tidal radius.
    rc_x, lkl_max = 0, -np.inf
    for rc in np.linspace(5., rt_x, 500):
        k = centDens(rt_rang, N_memb, rc, rt_x)
        li = k * KingProf(r_in[:i_ri], rc, rt_x) + fd
        lkl = np.log(li).sum() + log_fd_sum

        if lkl > lkl_max:
            rc_x = rc
            lkl_max = lkl

    return rc_x, rt_x
