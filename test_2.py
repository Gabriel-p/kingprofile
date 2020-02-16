
import numpy as np
import matplotlib.pyplot as plt


def intKP(r, rc, rt):
    """
    Number of members from the integrated KP
    """
    x = (r / rc)**2
    A = np.sqrt(1. + (rt / rc)**2)
    N = rc**2 * (np.log(1 + x) - 4 * (np.sqrt(1. + x) - 1.) / A + x / A**2)
    return N


rt = 250.
x = np.linspace(0., rt, 100)

r_optm_kcp = []
for rc in np.linspace(25., 230., 50):
    N_memb = []
    for r in x:
        N_memb.append(intKP(r, rc, rt))

    N_memb /= np.array(N_memb).max()
    N_fl = x**2
    N_fl /= N_fl.max()
    imax = np.argmax(N_memb - N_fl)
    r_optm = x[imax]
    r_optm_kcp.append([np.log10(rt / rc), (rt - r_optm) / rt])

r_optm_kcp = np.array(r_optm_kcp).T
plt.scatter(*r_optm_kcp)
plt.show()
