
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


xy_grid = 15
xx_grid = np.linspace(0, 15, xy_grid)  # kcp
yy_grid = np.linspace(50, 65, xy_grid)   # CI
Nrepeat = 10

xy_data = [[[
    [] for _ in range(Nrepeat)] for _ in xx_grid] for _ in yy_grid]
for i, xx in enumerate(xx_grid):
    for j, yy in enumerate(yy_grid):
        for k in range(Nrepeat):
            xy_data[i][j][k] += [xx, yy]

rc_delta = np.empty((xx_grid.size, yy_grid.size))
for i, xx in enumerate(xx_grid):
    for j, yy in enumerate(yy_grid):
        vals = []
        for k in range(Nrepeat):

            xy = np.array(xy_data[i][j][k]).T
            vals.append(xy)

        rc_delta[i][j] = np.median(vals)
        print(i, j, np.median(vals))

# norm = mcolors.DivergingNorm(vcenter=0., vmin=vmin, vmax=vmax)
plt.pcolormesh(xx_grid, yy_grid, rc_delta, cmap=plt.cm.RdBu)  # , norm=norm)
plt.colorbar()

plt.show()
