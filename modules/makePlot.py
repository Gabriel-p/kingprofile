
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


def main(
    method, xx_grid, yy_grid, kcp_rc_delta, kcp_rt_delta, CI_rc_delta,
        CI_rt_delta, Nm_delta, kcp_delta, rc_delta, rt_delta, main_call=False):
    """
    """

    #
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(6, 6)

    # ax = plt.subplot(gs[0:2, 0:2])
    # ax.minorticks_on()
    # ax.set_title(r"$\Delta N_{memb}$ [(True - Estimated) / True]")
    # #
    # norm = normRange(Nm_delta)
    # Xn, Yn, data_interp = data2Dinterp(xx_grid, yy_grid, Nm_delta)
    # plt.pcolormesh(Xn, Yn, data_interp, cmap=plt.cm.RdBu, norm=norm)
    # # plt.pcolormesh(xx_grid, yy_grid, Nm_delta, cmap=plt.cm.RdBu, norm=norm)
    # plt.xlabel(r"$K_{cp}$")
    # plt.ylabel(r"$CI$")
    # cbar = plt.colorbar(pad=.01, fraction=.02, aspect=40)
    # cbar.ax.tick_params(labelsize=9)

    #
    ax = plt.subplot(gs[0:2, 0:2])
    ax.minorticks_on()
    ax.set_title(r"$\Delta r_{c}$")
    #
    norm = normRange(rc_delta)
    Xn, Yn, data_interp = data2Dinterp(xx_grid, yy_grid, rc_delta)
    plt.pcolormesh(Xn, Yn, data_interp, cmap=plt.cm.RdBu, norm=norm)
    # plt.pcolormesh(xx_grid, yy_grid, rc_delta, cmap=plt.cm.RdBu, norm=norm)
    plt.xlabel(r"$K_{cp}$")
    plt.ylabel(r"$CI$")
    cbar = plt.colorbar(pad=.01, fraction=.02, aspect=40)
    cbar.ax.tick_params(labelsize=9)

    #
    ax = plt.subplot(gs[0:2, 2:4])
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='gray', linestyle=':', lw=.5)
    plt.boxplot(CI_rc_delta, positions=yy_grid, widths=.03)
    plt.axhline(0., color='g', linestyle='--')
    plt.ylabel(r"$\Delta r_{c}$")
    plt.xlabel(r"$CI$")
    plt.xlim(.0, .95)
    tcks = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    ax.set_xticks(tcks)
    ax.set_xticklabels(tcks)
    plt.ylim(-1., 1.)

    #
    ax = plt.subplot(gs[0:2, 4:6])
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='gray', linestyle=':', lw=.5)
    plt.boxplot(kcp_rc_delta, positions=xx_grid, widths=.03)
    plt.axhline(0., color='g', linestyle='--')
    plt.ylabel(r"$\Delta r_{c}$")
    plt.xlabel(r"$K_{cp}$")
    plt.xlim(.0, 1.04)
    tcks = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    ax.set_xticks(tcks)
    ax.set_xticklabels(tcks)
    plt.ylim(-1., 1.)

    #
    # ax = plt.subplot(gs[2:4, 0:2])
    # ax.minorticks_on()
    # ax.set_title(r"$\Delta k_{cp}$")
    # #
    # norm = normRange(kcp_delta)
    # Xn, Yn, data_interp = data2Dinterp(xx_grid, yy_grid, kcp_delta)
    # plt.pcolormesh(Xn, Yn, data_interp, cmap=plt.cm.RdBu, norm=norm)
    # # plt.pcolormesh(xx_grid, yy_grid, kcp_delta, cmap=plt.cm.RdBu, norm=norm)
    # plt.xlabel(r"$K_{cp}$")
    # cbar = plt.colorbar(pad=.01, fraction=.02, aspect=40)
    # cbar.ax.tick_params(labelsize=9)

    #
    ax = plt.subplot(gs[2:4, 0:2])
    ax.minorticks_on()
    ax.set_title(r"$\Delta r_{t}$")
    #
    norm = normRange(rt_delta)
    Xn, Yn, data_interp = data2Dinterp(xx_grid, yy_grid, rt_delta)
    plt.pcolormesh(Xn, Yn, data_interp, cmap=plt.cm.RdBu, norm=norm)
    # plt.pcolormesh(xx_grid, yy_grid, rt_delta, cmap=plt.cm.RdBu, norm=norm)
    plt.xlabel(r"$K_{cp}$")
    plt.ylabel(r"$CI$")
    cbar = plt.colorbar(pad=.01, fraction=.02, aspect=40)
    cbar.ax.tick_params(labelsize=9)

    #
    ax = plt.subplot(gs[2:4, 2:4])
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='gray', linestyle=':', lw=.5)
    plt.boxplot(CI_rt_delta, positions=yy_grid, widths=.03)
    plt.axhline(0., color='g', linestyle='--')
    plt.ylabel(r"$\Delta r_{t}$")
    plt.xlabel(r"$CI$")
    plt.xlim(.0, .95)
    tcks = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    ax.set_xticks(tcks)
    ax.set_xticklabels(tcks)
    plt.ylim(-1., 1.)

    #
    ax = plt.subplot(gs[2:4, 4:6])
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='gray', linestyle=':', lw=.5)
    plt.boxplot(kcp_rt_delta, positions=xx_grid, widths=.03)
    plt.axhline(0., color='g', linestyle='--')
    plt.ylabel(r"$\Delta r_{t}$")
    plt.xlabel(r"$K_{cp}$")
    plt.xlim(.0, 1.04)
    tcks = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    ax.set_xticks(tcks)
    ax.set_xticklabels(tcks)
    plt.ylim(-1., 1.)

    fig.tight_layout()
    if main_call:
        out_n = "../output/king_lkl_{}.png".format(method)
    else:
        out_n = "output/king_lkl_{}.png".format(method)
    plt.savefig(out_n, dpi=150, bbox_inches='tight')
    print("Finished")


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
    norm = mcolors.DivergingNorm(vcenter=vcent, vmin=vmin, vmax=vmax)

    return norm


if __name__ == '__main__':
    import pickle

    method = 'optmRad' # 'gridBF'
    with open("../output/plot_{}.pickle".format(method), "rb") as f:
        method, xx_grid, yy_grid, kcp_rc_delta, kcp_rt_delta, CI_rc_delta,\
            CI_rt_delta, Nm_delta, kcp_delta, rc_delta, rt_delta =\
            pickle.load(f)
    main(
        method, xx_grid, yy_grid, kcp_rc_delta, kcp_rt_delta, CI_rc_delta,
        CI_rt_delta, Nm_delta, kcp_delta, rc_delta, rt_delta, True)
