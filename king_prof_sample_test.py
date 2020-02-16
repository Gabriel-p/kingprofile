
from astropy.io import ascii
from astropy.table import Table
import pickle
import numpy as np
from modules import KPInvSamp
from modules import methodCall
from modules import makePlot
import time as t


# Reproducibility
# np.random.seed(12345)


def main():
    """
    """

    method, create_data, xy_grid, Nrepeat = readParams()

    # Fixed synthetic cluster's parameters
    xmax, ymax = 2000., 2000.
    area_frame = xmax * ymax
    cx, cy = .5 * xmax, .5 * ymax
    rt_fix = 250.
    N_clust = 200

    if create_data:
        genSynthClusters(
            rt_fix, xmax, ymax, cx, cy, N_clust, create_data, xy_grid, Nrepeat)
        return
    else:
        xx_grid, yy_grid, Nrepeat, xy_data = genSynthClusters(
            rt_fix, xmax, ymax, cx, cy, N_clust, create_data)

    Nm_delta_rel, kcp_delta_rel, rc_delta_rel, rt_delta_rel = [
        np.empty((xx_grid.size, yy_grid.size)) for _ in range(4)]
    kcp_rc_delta, kcp_rt_delta, CI_rc_delta, CI_rt_delta = [
        [[] for _ in range(xy_grid)] for _ in range(4)]
    N_i, N_tot, elapsed, start_t = 0, xy_grid**2, 0., t.time()
    all_rts = []
    for i, kcp in enumerate(xx_grid):
        rc = rt_fix / 10 ** kcp

        for j, CI in enumerate(yy_grid):

            Nm_vals, kcp_vals, rc_vals, rt_vals, = [], [], [], []
            for k in range(Nrepeat):

                xy = np.array(xy_data[i][j][k]).T

                N_mx, rc_x, rt_x = methodCall.main(
                    method, area_frame, N_clust, cx, cy, rc, rt_fix, xy)
                kcp_x = np.log10(rt_x / rc_x)

                Nm_vals.append((N_clust - N_mx) / N_clust)
                kcp_vals.append((kcp - kcp_x) / kcp)
                rc_vals.append((rc - rc_x) / rc)
                rt_vals.append((rt_fix - rt_x) / rt_fix)

            all_rts += rt_vals
            kcp_rc_delta[i] += rc_vals
            kcp_rt_delta[i] += rt_vals
            CI_rc_delta[j] += rc_vals
            CI_rt_delta[j] += rt_vals

            # Why i,j are inverted below:
            # pcolormesh() docs: "An array C with shape (nrows, ncolumns) is
            # plotted with the column number as X and the row number as Y"
            Nm_delta_rel[j][i] = np.median(Nm_vals)
            kcp_delta_rel[j][i] = np.median(kcp_vals)
            rc_delta_rel[j][i] = np.median(rc_vals)
            rt_delta_rel[j][i] = np.median(rt_vals)

            elapsed += t.time() - start_t
            N_i += 1
            start_t = t.time()
            # Estimate the total ampunt of time this process will take
            time_tot = ((N_tot * elapsed) / N_i)
            # Subtract the elapsed time to estimate the time left
            t_left = (time_tot - elapsed) / 60.
            print(
                "{} ({:.1f} m) | kcp={:.2f} (rc={:.0f}), CI={:.2f}".format(
                    N_tot - N_i, t_left, kcp, rc, CI))

    writeRes(
        method, xx_grid, yy_grid, kcp_rc_delta, kcp_rt_delta, CI_rc_delta,
        CI_rt_delta, Nm_delta_rel, kcp_delta_rel, rc_delta_rel,
        rt_delta_rel)
    makePlot.main(
        method, xx_grid, yy_grid, kcp_rc_delta, kcp_rt_delta, CI_rc_delta,
        CI_rt_delta, Nm_delta_rel, kcp_delta_rel, rc_delta_rel,
        rt_delta_rel)


def readParams():
    """
    """
    pars = ascii.read("params_input.dat")
    create_data = True if pars['create_data'][0] == 'y' else False

    xy_grid, Nrepeat = int(pars['xy_grid'][0]), int(pars['Nrepeat'][0])

    return pars['method'][0], create_data, xy_grid, Nrepeat


def genSynthClusters(
    rt_fix, xmax, ymax, cx, cy, N_clust, create_data, xy_grid=None,
        Nrepeat=None):
    """
    """
    if create_data:

        xx_grid = np.linspace(.04, .99, xy_grid)  # kcp
        yy_grid = np.linspace(.04, .9, xy_grid)   # CI

        xy_data = [[[
            [] for _ in range(Nrepeat)] for _ in xx_grid] for _ in yy_grid]
        for i, kcp in enumerate(xx_grid):
            rc = rt_fix / 10 ** kcp

            for j, CI in enumerate(yy_grid):

                for k in range(Nrepeat):

                    x_cl, y_cl, x_fl, y_fl = KPInvSamp.main(
                        N_clust, CI, rc, rt_fix, xmax, ymax, cx, cy)

                    xy_data[i][j][k] +=\
                        x_cl.tolist() + x_fl.tolist(), y_cl.tolist() +\
                        y_fl.tolist()

        with open(r"xy_data.pickle", "wb") as output_file:
            pickle.dump([xx_grid, yy_grid, Nrepeat, xy_data], output_file)
        print("Syntetic clusters saved to file")

    else:
        # xy_data = np.load('xy_data.npz')
        with open(r"xy_data.pickle", "rb") as input_file:
            xx_grid, yy_grid, Nrepeat, xy_data = pickle.load(input_file)
        print("Syntetic clusters read from file\n")

        return xx_grid, yy_grid, Nrepeat, xy_data


def writeRes(
    method, xx_grid, yy_grid, kcp_rc_delta, kcp_rt_delta, CI_rc_delta,
        CI_rt_delta, Nm_delta, kcp_delta, rc_delta_rel, rt_delta_rel):
    """
    """
    with open("output/plot_{}.pickle".format(method), "wb") as f:
        pickle.dump([
            method, xx_grid, yy_grid, kcp_rc_delta, kcp_rt_delta, CI_rc_delta,
            CI_rt_delta, Nm_delta, kcp_delta, rc_delta_rel, rt_delta_rel], f)

    data_out = {
        "Nm_delta": Nm_delta.flatten(), "rc_delta_rel": rc_delta_rel.flatten(),
        "rt_delta_rel": rt_delta_rel.flatten()}
    data_out = Table(data_out)
    ascii.write(
        data_out, "output/results_{}.dat".format(method), format="csv",
        overwrite=True)


if __name__ == '__main__':
    main()
