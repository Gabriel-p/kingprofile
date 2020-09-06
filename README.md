# kingprofile

This code encapsulates two methods to estimate the `r_c, r_t`
parameters of a [King (1962)][1] profile.

The first one is described in [Pieres et al. (2016)][2], and
corresponds to maximizing the King profile likelihood through
brute force applied on a grid.
This method is divided into two: 'KP_2' and 'KP_4'. The former
only fits the core and tidal radius, while the latter also fits
the eccentricity and rotation angle.

The second one I developed for ASteCA and estimates an "optimal"
radius based on maximizing the (normalized) difference between the
number of members within the cluster region, and the number of field
stars. When testing it here, I discovered that this method returns
a fraction of the real `r_t` value. This fraction is ~0.7 (maybe 1/sqrt(2)?) 
for all values of `CI` if we average the `K_cp` factor (concentration
parameter), and varies from ~0.22 to ~0.42 for the `K_cp`, averaging over
the `CI`.

## TODO

1. Estimate the field density and number of members parameters from the data at each step, instead of fixing them.
2. Make the center coordinates also free parameters?











____________________________
[1]: http://adsabs.harvard.edu/abs/1962AJ.....67..471K
[2]: http://adsabs.harvard.edu/abs/2016MNRAS.461..519P