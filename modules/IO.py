
import numpy as np
from astropy.io import ascii
from astropy.table import Table
import configparser
import pickle
from modules import KPInvSamp


def readParams():
    """
    """
    in_params = configparser.ConfigParser()
    in_params.read('params.ini')

    analy_method = in_params['Analysis method']
    method, mode, corr_fact = analy_method.get('method'),\
        analy_method.get('mode'), analy_method.getfloat('corr_fact')

    if method not in ('optmRad', 'KP_2', 'KP_4'):
        raise ValueError("'{}' method not recognized".format(method))

    input_data = in_params['Input data']
    N_clust, xy_grid, Nrepeat = input_data.getint('N_clust'),\
        input_data.getint('xy_grid'), input_data.getint('Nrepeat')

    return method, mode, corr_fact, N_clust, xy_grid, Nrepeat


def genSynthClusters(
    method, mode, xy_grid, Nrepeat, xmax, ymax, cl_cent, rt_fix,
        N_clust):
    """
    """
    in_file = "input/xy_data_{}.pickle".format(method)

    if mode == 'create_data':
        print("Generating {} synthetic files".format(xy_grid**2 * Nrepeat))

        xx_grid = np.linspace(.04, .99, xy_grid)  # kcp
        yy_grid = np.linspace(.04, .9, xy_grid)   # CI

        xy_data = [[[
            [] for _ in range(Nrepeat)] for _ in xx_grid] for _ in yy_grid]
        ecc_theta = [[[
            [] for _ in range(Nrepeat)] for _ in xx_grid] for _ in yy_grid]
        for i, kcp in enumerate(xx_grid):
            rc = rt_fix / 10 ** kcp
            for j, CI in enumerate(yy_grid):
                for k in range(Nrepeat):

                    if method == 'KP_4':
                        ecc = np.random.uniform(.4, .95)
                        theta = np.random.uniform(-np.pi / 2., np.pi / 2.)
                    else:
                        ecc, theta = 0., 0.
                    ecc_theta[i][j][k] += (ecc, theta)

                    x_cl, y_cl, x_fl, y_fl = KPInvSamp.main(
                        method, N_clust, CI, rc, rt_fix, ecc, theta, xmax,
                        ymax, cl_cent)

                    xy_data[i][j][k] +=\
                        x_cl.tolist() + x_fl.tolist(), y_cl.tolist() +\
                        y_fl.tolist()

        with open(in_file, "wb") as output_file:
            pickle.dump([xx_grid, yy_grid, Nrepeat, xy_data, ecc_theta],
                        output_file)
        print("Synthetic clusters saved to file")

    else:
        with open(in_file, "rb") as input_file:
            xx_grid, yy_grid, Nrepeat, xy_data, ecc_theta = pickle.load(
                input_file)
        print("Synthetic clusters read from file\n")

        return xx_grid, yy_grid, Nrepeat, xy_data, ecc_theta


def writeRes(
    method, xx_grid, yy_grid, Nm_delta_rel_median, rc_delta_rel, rt_delta_rel,
    rc_delta_rel_median, rt_delta_rel_median, ecc_delta_rel_median,
        ecc_rel_all, theta_rel_all):
    """
    """
    with open("output/plot_{}.pickle".format(method), "wb") as f:
        pickle.dump([
            method, xx_grid, yy_grid, rc_delta_rel, rt_delta_rel,
            rc_delta_rel_median, rt_delta_rel_median, ecc_delta_rel_median,
            ecc_rel_all, theta_rel_all], f)

    data_out = {
        "Nm_delta": Nm_delta_rel_median.flatten(),
        "rc_delta_rel_median": rc_delta_rel_median.flatten(),
        "rt_delta_rel_median": rt_delta_rel_median.flatten(),
        # "ecc_delta_rel_median": ecc_delta_rel_median.flatten(),
        # "theta_delta_rel_median": theta_delta_rel_median.flatten()
    }
    data_out = Table(data_out)
    ascii.write(
        data_out, "output/results_{}.dat".format(method), format="csv",
        overwrite=True)
    print("Data saved to file")
