# plots

The figures of the article, drawn from the data structures of this package.
Figures 2 to 8 have a function each, and two further figures belong to the
methods note rather than to the reproduction: the observed survival of one
state beside the exponential fitted to it, and the relapse counts of a cohort
beside the Poisson and the negative binomial mass.

matplotlib is an optional dependency, the `plot` extra, so it is imported
inside the functions that draw. Importing this module therefore works in an
install without it, and only a call that has to create a figure raises, naming
the extra to install. Every function takes the axes to draw on and gives them
back, so panels can be composed into a figure of the caller's own.

::: msrelapse.plots
