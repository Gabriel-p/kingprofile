
import numpy as np
from scipy import spatial
from .KPInvSamp import centDens, KingProf


def main(method, area_frame, N_clust, cx, cy, rc0, rt0, xy):
    """
    """
    xy_cent_dist = spatial.distance.cdist([(cx, cy)], xy)[0]

    rt_max = 2. * rt0
    msk = xy_cent_dist <= rt_max
    r_in = xy_cent_dist[msk]
    r_in.sort()

    # Tidal radius array. Used for integrating
    rt_rang = np.linspace(0., rt_max, 1000)

    area_rtmax = np.pi * rt_max**2
    fd = np.sum(~msk) / (area_frame - area_rtmax)
    N_memb = max(1, msk.sum() - fd * area_rtmax)

    if method == 'gridBF':
        rc, rt = gridBruteForce(rt0, rt_max, rt_rang, r_in, fd, N_memb)
    elif method == 'optmRad':
        rc, rt = optmRad(rt0, rt_max, r_in, fd, N_memb)

    return N_memb, rc, rt


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


def optmRad(rt0, rt_max, r_in, fd, N_memb, corr_fact=0.7):
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

    rc_x, lkl_max = 0, -np.inf
    for rc in np.linspace(5., rt_x, 500):
        k = centDens(rt_rang, N_memb, rc, rt_x)
        li = k * KingProf(r_in[:i_ri], rc, rt_x) + fd
        lkl = np.log(li).sum() + log_fd_sum

        if lkl > lkl_max:
            rc_x = rc
            lkl_max = lkl

    return rc_x, rt_x
