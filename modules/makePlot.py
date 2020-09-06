
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


def main(
    method, xx_grid, yy_grid, rc_delta_rel, rt_delta_rel,
    rc_delta_rel_median, rt_delta_rel_median, ecc_delta_rel_median,
        ecc_rel_all, theta_rel_all, main_call=False):
    """
    """
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(8, 6)

    tcks = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

    # rc
    # density map
    ax = plt.subplot(gs[0:2, 0:2])
    densMapPlot(ax, xx_grid, yy_grid, rc_delta_rel_median, "r_{c}")
    # CI Box plot
    ax = plt.subplot(gs[0:2, 2:4])
    boxPlot(ax, yy_grid, rc_delta_rel, "r_{c}", "CI", tcks)
    # kcp Box plot
    ax = plt.subplot(gs[0:2, 4:6])
    boxPlot(ax, xx_grid, rc_delta_rel, "r_{c}", "kcp", tcks)

    # rt
    # density map
    ax = plt.subplot(gs[2:4, 0:2])
    densMapPlot(ax, xx_grid, yy_grid, rt_delta_rel_median, "r_{t}")
    # CI Box plot
    ax = plt.subplot(gs[2:4, 2:4])
    boxPlot(ax, yy_grid, rt_delta_rel, "r_{t}", "CI", tcks)
    # kcp Box plot
    ax = plt.subplot(gs[2:4, 4:6])
    boxPlot(ax, xx_grid, rt_delta_rel, "r_{t}", "kcp", tcks)

    if method == 'KP_4':

        def boxPlotArray(xvals, yvals, rmin, rmax, steps=11):
            bp_vals = []
            _range = np.linspace(rmin, rmax, steps)
            for i, ra in enumerate(_range):
                if i + 1 == steps:
                    break
                msk = (xvals >= ra) & (xvals < _range[i + 1])
                bp_vals.append(yvals[msk])
            yarr = np.array(bp_vals)
            grid = .5 * (_range[1:] + _range[:-1])
            return grid, yarr

        ecc_t, ecc_rel = np.array(ecc_rel_all).T
        theta_t, theta_rel = np.array(theta_rel_all).T

        # ecc
        ecc_steps, ecc_bp = boxPlotArray(ecc_t, ecc_rel, 0.4, 0.95)
        _, theta_bp = boxPlotArray(ecc_t, theta_rel, 0.4, 0.95)
        tcks = np.round(ecc_steps, 2)
        # density map
        ax = plt.subplot(gs[4:6, 0:2])
        densMapPlot(ax, xx_grid, yy_grid, ecc_delta_rel_median, "ecc")
        # ecc vs delta_ecc Box plot
        ax = plt.subplot(gs[4:6, 2:4])
        boxPlot(ax, ecc_steps, ecc_bp, "ecc", "ecc", tcks)
        # ecc vs delta_theta Box plot
        ax = plt.subplot(gs[4:6, 4:6])
        boxPlot(ax, ecc_steps, theta_bp, r"\theta", "ecc", tcks, True)

    fig.tight_layout()
    if main_call:
        out_n = "../output/king_lkl_{}.png".format(method)
    else:
        out_n = "output/king_lkl_{}.png".format(method)
    plt.savefig(out_n, dpi=150, bbox_inches='tight')
    print("Finished")


def densMapPlot(ax, xx_grid, yy_grid, par_delta, p_n):
    ax.minorticks_on()
    ax.set_title(r"$\Delta " + p_n + r"$")
    #
    norm = normRange(par_delta)
    Xn, Yn, data_interp = data2Dinterp(xx_grid, yy_grid, par_delta)
    plt.pcolormesh(
        Xn, Yn, data_interp, cmap=plt.cm.RdBu, norm=norm, shading='auto')
    plt.xlabel(r"$K_{cp}$")
    plt.ylabel(r"$CI$")
    cbar = plt.colorbar(pad=.01, fraction=.02, aspect=40)
    cbar.ax.tick_params(labelsize=9)


def boxPlot(ax, xgrid, par_delta, p_n, pID, tcks, thetaplot=False):
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='gray', linestyle=':', lw=.5)
    plt.boxplot(par_delta, positions=xgrid, widths=.03)
    plt.ylabel(r"$\Delta " + p_n + r"$")
    if pID == 'CI':
        plt.xlabel(r"$CI$")
        plt.xlim(.0, .95)
    elif pID == 'kcp':
        plt.xlabel(r"$K_{cp}$")
        plt.xlim(.0, 1.04)
    elif pID == 'ecc':
        plt.xlabel(r"$ecc$")
        plt.xlim(tcks[0] - .05, tcks[-1] + .05)
    ax.set_xticks(tcks)
    ax.set_xticklabels(tcks)
    if thetaplot:
        plt.ylim(0., 1.)
    else:
        plt.axhline(0., color='g', linestyle='--')
        plt.ylim(-1., 1.)


def data2Dinterp(xx_grid, yy_grid, delta):
    """
    """
    f = interp2d(xx_grid, yy_grid, delta, kind='cubic')
    xnew = np.linspace(xx_grid.min(), xx_grid.max(), 100)  # kcp
    ynew = np.linspace(yy_grid.min(), yy_grid.max(), 100)  # CI
    data_interp = f(xnew, ynew)
    Xn, Yn = np.meshgrid(xnew, ynew)

    return Xn, Yn, data_interp


def normRange(delta, mmin=-.5, mmax=.5):
    """
    """
    # vmin, vcent, vmax = max(delta.min(), mmin), 0., min(delta.max(), mmax)
    # if not (vmin < vcent and vcent < vmax):
    #     vmin, vmax = delta.min(), delta.max()
    #     vcent = .5 * (vmin + vmax)
    vmin, vcent, vmax = mmin, 0., mmax
    norm = mcolors.TwoSlopeNorm(vcenter=vcent, vmin=vmin, vmax=vmax)

    return norm


if __name__ == '__main__':
    import pickle

    method = 'KP_4'
    with open("../output/plot_{}.pickle".format(method), "rb") as f:
        method, xx_grid, yy_grid, rc_delta_rel, rt_delta_rel,\
            rc_delta_rel_median, rt_delta_rel_median, ecc_delta_rel_median,\
            ecc_rel_all, theta_rel_all = pickle.load(f)
    main(
        method, xx_grid, yy_grid, rc_delta_rel, rt_delta_rel,
        rc_delta_rel_median, rt_delta_rel_median, ecc_delta_rel_median,
        ecc_rel_all, theta_rel_all, True)
