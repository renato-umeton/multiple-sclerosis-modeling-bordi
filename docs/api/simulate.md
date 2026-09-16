# simulate

Integration of equation (4) with the Euler and Maruyama scheme, and the four
steps that turn a continuous path into the kind of record the article prints:
first passage times out of a well, the hysteresis band that maps a path to the
two clinical states, the downsampling to whole weeks under the rounding rule of
the study, and the run length encoding of the result.

Three choices here are not in the article: the time step, the Brownian bridge
shift of the absorbing level, and the width of the hysteresis band. Each is
documented below with the measurement that fixed it, and the
[theory page](../theory.md) explains what each one is for. The compiled kernels
are optional: `HAS_NUMBA` says whether they are available and a numpy
implementation runs whenever they are not.

::: msrelapse.simulate
