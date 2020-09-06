
import numpy as np
from modules import IO
from modules import methodCall
from modules import makePlot
import time as t


# Reproducibility
# np.random.seed(12345)

# Fixed synthetic cluster's parameters
xmax, ymax = 2000., 2000.
area_frame = xmax * ymax
cl_cent = (.5 * xmax, .5 * ymax)
rt_fix = 250.


def main(plotResults=False):
    """
    """

    method, mode, corr_fact, N_clust, xy_grid, Nrepeat = IO.readParams()

    if mode == 'create_data':
        IO.genSynthClusters(
            method, mode, xy_grid, Nrepeat, xmax, ymax, cl_cent, rt_fix,
            N_clust)
        return
    else:
        xx_grid, yy_grid, Nrepeat, xy_data, ecc_theta = IO.genSynthClusters(
            method, mode, None, None, xmax, ymax, cl_cent, rt_fix,
            N_clust)

    Nm_delta_rel_median, rc_delta_rel_median, rt_delta_rel_median,\
        ecc_delta_rel_median = [
            np.empty((xx_grid.size, yy_grid.size)) for _ in range(4)]
    rc_delta_rel, rt_delta_rel = [
        [[] for _ in range(xy_grid)] for _ in range(2)]
    N_i, N_tot, elapsed, start_t = 0, xy_grid**2, 0., t.time()

    ecc_rel_all, theta_rel_all = [], []
    for i, kcp in enumerate(xx_grid):
        rc = rt_fix / 10 ** kcp

        for j, CI in enumerate(yy_grid):
            Nm_vals, rc_vals, rt_vals, ecc_vals = [], [], [], []
            for k in range(Nrepeat):

                xy = np.array(xy_data[i][j][k]).T

                N_mx, rc_x, rt_x, ecc_x, theta_x = methodCall.main(
                    method, area_frame, N_clust, cl_cent, rc, rt_fix, xy,
                    corr_fact)
                ecc, theta = ecc_theta[i][j][k]

                if plotResults:
                    resPlot(
                        cl_cent, xy, rt_fix, ecc, theta, rt_x, ecc_x, theta_x)

                Nm_vals.append((N_clust - N_mx) / N_clust)
                rc_vals.append((rc - rc_x) / rc)
                rt_vals.append((rt_fix - rt_x) / rt_fix)

                if mode == 'KP_4':
                    ecc_rel_all.append([ecc, (ecc - ecc_x) / ecc])
                    theta_diff = abs(theta - theta_x)
                    if theta_diff > 90.:
                        theta_diff = 180. - theta_diff
                    theta_rel_all.append([
                        theta, theta_diff / max(0.01, abs(theta))])
                    ecc_vals.append((ecc - ecc_x) / ecc)
                else:
                    theta_rel_all.append(0)
                    ecc_vals.append(0)

            rc_delta_rel[i] += rc_vals
            rt_delta_rel[i] += rt_vals

            # Why i,j are inverted below:
            # pcolormesh() docs: "An array C with shape (nrows, ncolumns) is
            # plotted with the column number as X and the row number as Y"
            Nm_delta_rel_median[j][i] = np.median(Nm_vals)
            rc_delta_rel_median[j][i] = np.median(rc_vals)
            rt_delta_rel_median[j][i] = np.median(rt_vals)
            ecc_delta_rel_median[j][i] = np.median(ecc_vals)

            elapsed += t.time() - start_t
            N_i += 1
            start_t = t.time()
            # Estimate the total amount of time this process will take
            time_tot = ((N_tot * elapsed) / N_i)
            # Subtract the elapsed time to estimate the time left
            t_left = (time_tot - elapsed) / 60.
            print(
                "{} ({:.1f} m) | kcp={:.2f} (rc={:.0f}), CI={:.2f}".format(
                    N_tot - N_i, t_left, kcp, rc, CI))

    IO.writeRes(
        method, xx_grid, yy_grid, Nm_delta_rel_median,
        rc_delta_rel, rt_delta_rel,
        rc_delta_rel_median, rt_delta_rel_median, ecc_delta_rel_median,
        ecc_rel_all, theta_rel_all)

    makePlot.main(
        method, xx_grid, yy_grid, rc_delta_rel, rt_delta_rel,
        rc_delta_rel_median, rt_delta_rel_median, ecc_delta_rel_median,
        ecc_rel_all, theta_rel_all)


def resPlot(cl_cent, xy, rt, ecc, theta, r_rt, r_ecc, r_theta):
    """
    """
    import matplotlib.pyplot as plt
    from matplotlib import patches as mpatches
    ax = plt.subplot(111)
    a2 = rt**2
    b2 = a2 * (1. - ecc**2)
    ellipse = mpatches.Ellipse(
        xy=cl_cent, width=2. * np.sqrt(a2), height=2. * np.sqrt(b2),
        angle=np.rad2deg(theta),
        facecolor='None', edgecolor='green', linewidth=2, ls='--',
        transform=ax.transData, zorder=6)
    ax.add_patch(ellipse)

    a2 = r_rt**2
    b2 = a2 * (1. - r_ecc**2)
    ellipse = mpatches.Ellipse(
        xy=cl_cent, width=2. * np.sqrt(a2), height=2. * np.sqrt(b2),
        angle=np.rad2deg(r_theta),
        facecolor='None', edgecolor='black', linewidth=2,
        transform=ax.transData, zorder=6)
    ax.add_patch(ellipse)

    plt.scatter(*xy.T, c='k', s=15, alpha=.5)
    # plt.scatter(*xy_clust, c='g', s=15, zorder=5)
    # plt.scatter(*xy_in, c='r', s=5, zorder=1)
    plt.axvline(1000.)
    plt.axhline(1000.)
    plt.scatter(*cl_cent, marker='x', c='r', s=30, zorder=7)
    plt.xlim(cl_cent[0] - 2. * rt, cl_cent[0] + 2. * rt)
    plt.ylim(cl_cent[1] - 2. * rt, cl_cent[1] + 2. * rt)
    plt.show()


if __name__ == '__main__':
    main()
