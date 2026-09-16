# Theory

This page derives every equation the package implements, in the order the
article introduces them, and says at each step which choice belongs to the
article and which belongs to this package. The transcription it rests on is
[Paper facts](paper_facts.md), which records the equations, the printed numbers
and the figure measurements they were checked against.

Two conventions hold throughout. The potential is the one the article prints,

$$
V(x) = -\frac{x^2}{2} + \frac{\alpha}{4} x^4 + \beta x ,
$$

with the control parameter $\alpha$ on the quartic term and not on the
quadratic one. And the noise is written with an amplitude $\sigma$ rather than
with the article's variance $\varepsilon$, so that

$$
\sigma^2 = \varepsilon .
$$

The mathematics stays at the level of a course in survival analysis, apart from
the first passage integral, which is quoted rather than derived. A reader who
wants only the trial statistics can start at
[From one patient to a cohort](#from-one-patient-to-a-cohort).

## The state variable and the sign convention

A patient is one real number $x(t)$. The clinical record is a sequence of $+1$
and $-1$: the article encodes a relapse as $+1$, which it calls *no health*,
and a remission as $-1$, which it calls *health*. The model keeps that sign, so
the three stationary points of the potential read as follows.

| Point | Position at $\beta = 0$ | Clinical meaning | Code |
|---|---|---|---|
| $x_1$ | $-1/\sqrt{\alpha}$ | health, remission | $-1$ |
| $x_0$ | $0$ | top of the barrier, not a state | none |
| $x_2$ | $+1/\sqrt{\alpha}$ | no health, relapse | $+1$ |

The article fixes this in three places: the encoding of the clinical series,
the sentence that names $x_1$ health and $x_2$ no health, and the axis labels
of its Figure 2. The package follows it everywhere, and the two codes are held
in `PAPER.state_health` and `PAPER.state_no_health` rather than written into
any module.

## The deterministic core, equation (1)

$$
\frac{dx}{dt} = x \left(1 - \alpha x^2\right)
$$

Its steady states are the three roots $x_0 = 0$, $x_1 = -1/\sqrt{\alpha}$ and
$x_2 = +1/\sqrt{\alpha}$. The two outer ones are stable and the middle one is
unstable, so without noise a patient near $x_1$ stays near $x_1$ forever, a
patient near $x_2$ stays near $x_2$ forever, and a patient balanced on $x_0$
falls one way or the other with even chances. The article states exactly that
as a three item list. A deterministic double well has no relapses in it; every
transition in the model comes from the noise.

## The potential, equations (2) and (3)

Integrating the drift and changing sign gives the symmetric potential,
equation (2),

$$
V(x) = -\int x\left(1 - \alpha x^2\right) dx = -\frac{x^2}{2} + \frac{\alpha}{4} x^4 ,
$$

and adding the constant force $-\beta$ to the drift gives the asymmetric
potential, equation (3),

$$
V(x) = -\frac{x^2}{2} + \frac{\alpha}{4} x^4 + \beta x .
$$

The drift is the negative gradient of the potential exactly:
$-V'(x) = x(1 - \alpha x^2) - \beta$. `msrelapse.model.DoubleWell` holds one
pair of parameters and gives the potential, its derivatives, the stationary
points and the barriers.

### The symmetric case

At $\beta = 0$ everything is in closed form. The wells sit at
$\pm 1/\sqrt{\alpha}$, their depth is $-1/(4\alpha)$, and both barriers are

$$
\Delta V = V(x_0) - V(x_{1,2}) = \frac{1}{4\alpha} .
$$

The barrier therefore **grows as $\alpha$ falls**: it is $0.25$ at
$\alpha = 1$, and $0.357$ at $\alpha = 0.7$, where the wells have also moved
outward to $\pm 1.195$. That direction is what the article's text says and what
its Figure 5(b) shows, and it is the reason the package uses this
parameterisation and not the more familiar $V = x^4/4 - a x^2/2$, whose barrier
moves the other way. The two agree at $\alpha = 1$ and only there.

Clinically $\alpha$ is "the combination of heritable and nonheritable risk":
a lower $\alpha$ is a higher barrier, longer stays in each state and fewer
transitions. The article never estimates $\alpha$ from data. Every patient fit
holds it at $\alpha = 1$, and $\alpha = 0.7$ appears only in two figures.

### The asymmetric case

For $\beta \neq 0$ there is no closed form. The stationary points are the real
roots of

$$
V'(x) = \alpha x^3 - x + \beta = 0 ,
$$

which the package finds numerically and sorts as $x_1 < x_0 < x_2$. The two
barriers are measured from the top of the barrier down,

$$
\Delta V_1 = V(x_0) - V(x_1), \qquad \Delta V_2 = V(x_0) - V(x_2) ,
$$

which is the only definition the article gives, and it gives it in words rather
than as a formula. With $\beta > 0$ the potential is lowered for $x < 0$ and
raised for $x > 0$, so the health well deepens and the no health well grows
shallower, which is the asymmetry the article wants: it is harder to leave
health than to leave a relapse. At the article's $\alpha = 1$, $\beta = 0.08$
this gives $x_1 = -1.0378$, $x_0 = 0.0805$, $x_2 = 0.9573$,
$\Delta V_1 = 0.3348$, $\Delta V_2 = 0.1749$ and a ratio of $1.914$.

Three real roots exist only while the asymmetry stays below the fold value

$$
\beta_{\text{fold}} = \frac{2}{3\sqrt{3\alpha}} ,
$$

which is $0.3849$ at $\alpha = 1$. At that value two of the roots merge in a
saddle-node fold and the shallower well disappears, leaving a single well and
no relapse state at all. `msrelapse.model.fold_beta` returns it. Everything
that needs the two wells refuses a $\beta$ at or above the fold: `DoubleWell`
finds no three roots, so `critical_points` raises, and with it `barriers`,
`barrier_ratio`, `well_bottom`, `curvatures`, the Kramers times and the passage
endpoints. The potential itself, its derivatives and the drift are defined at
every $\beta$ and evaluate as usual, and `DoubleWell.is_bistable` answers the
same question without raising.

## The stochastic equation, equation (4)

$$
dx = \left[x\left(1 - \alpha x^2\right) - \beta\right] dt + \varepsilon^{1/2}\, dW
$$

The article labels the three terms: the change of the health state of a
patient, the mechanistic double-well component, and the external random
stimuli. $dW$ is a Wiener increment, so it has zero mean, variance $dt$, and no
correlation from one instant to the next. The single noise value the article
uses is $\varepsilon = 0.13$, in its Figure 7.

There is no periodic term anywhere in equation (4), and the article says so
twice: this is not stochastic resonance, and the clinical data show no typical
periodicity of relapses. What the noise does instead is supply, now and then, a
run of stimuli of the same sign long enough to carry $x$ over the barrier.
Averaged over time it does nothing, because its mean is zero; what matters is
its tail.

## Mean exit times, equations (5) and (6)

$$
\tau_{x_1} \approx e^{2\Delta V_1/\varepsilon} , \qquad
\tau_{x_2} \approx e^{2\Delta V_2/\varepsilon}
$$

This is the Kramers exponential without its prefactor, cited by the article to
Benzi, Parisi, Sutera and Vulpiani (1983). The article prints no prefactor, no
units and no first passage reference, and the base is $e$.

The missing prefactor is not a detail. An exponential of a dimensionless
exponent is dimensionless, so equations (5) and (6) give a pure number, and
reading that number as a count of weeks silently sets the prefactor to one
week. `msrelapse.model.paper_exit_time` implements the printed form exactly, so
that a reader can see what it gives; the functions below restore what it leaves
out.

## The Kramers time, with its prefactor

For a walker in a well of a smooth potential, in the limit of small noise, the
classical escape rate of Kramers (1940) gives

$$
\tau_{\text{Kramers}} = \frac{2\pi}{\sqrt{V''(x_{\text{well}})\,\left|V''(x_0)\right|}}
\; \exp\!\left(\frac{2\,\Delta V}{\sigma^2}\right) .
$$

The prefactor carries the units. With the model time of equation (4) read as
weeks it is a number of weeks, and it is the shortest escape time this formula
can produce, since the exponential is at least one. That floor has a
consequence the article could not have seen: at $\alpha = 1$ the relapse-side
prefactor never falls below $2\pi/\sqrt{2} = 4.443$ weeks, which is longer than
the 4.3 week relapse the cohort reports. **No Kramers calibration to the
published cohort exists at all.** `msrelapse.model.kramers_prefactor` and
`kramers_time` compute both parts, and `msrelapse.model.calibrate` refuses the
`kramers` method on that pair rather than returning a fit that cannot be one.

## The exact mean first passage time

The Kramers formula is an approximation. For the one dimensional diffusion
$dx = -V'(x)\,dt + \sigma\,dW$ that starts at $s$, is absorbed at $a$ and is
reflected at a boundary far out on the other side, the mean first passage time
is exact and is a double integral. For a passage that runs towards increasing
$x$, which is how a patient leaves health,

$$
T(s) = \frac{2}{\sigma^2}
\int_{s}^{a} dy\; e^{\,2V(y)/\sigma^2}
\int_{-L}^{y} dz\; e^{-2V(z)/\sigma^2} ,
$$

and for a passage towards decreasing $x$, which is how a patient leaves a
relapse, the same expression with the limits mirrored and the reflecting
boundary at $+L$. The article's $x_0$ is the saddle and plays no part here;
`msrelapse.model.mfpt` calls the two endpoints `x0` and `x_absorb`. It
evaluates the integral on a uniform grid that is doubled until the answer stops
moving, places the reflecting boundary where the potential has risen far enough
that the walker never reaches it, and shifts the integrand by the potential at
the saddle so that the two exponentials stay inside float64.

**Where an episode ends changes the answer by a factor of about two.** A walker
that first touches the top of the barrier has not yet chosen a side: it falls
back into the well it came from about half the time, so the well-to-well time
is about twice the bottom-to-saddle time. On the potential calibrated to the
published pair, the bottom-to-saddle times are 100.0 and 4.30 weeks by
construction, while the same passages taken all the way to the far well bottom
take 209.5 and 11.99 weeks, and the Kramers formula, which is a well-to-well
time, gives 194.4 and 10.76 weeks. The article never says which of the two it
means. `msrelapse.model.passage_endpoints` names three readings,
`bottom_to_saddle` (the default), `bottom_to_bottom` and `band`. `calibrate`
and `exit_times` take that same argument, and `mfpt` takes instead the pair of
endpoints `passage_endpoints` returns, as its `x0` and `x_absorb`, so either
way the reading is named in the call rather than assumed.

## The barrier ratio, equation (7)

$$
\frac{\Delta V_1}{\Delta V_2} \approx \frac{\log \tau_{x_1}}{\log \tau_{x_2}}
$$

The base of the logarithm cancels, so there is one number and not two. With the
cohort's printed durations of 100 and 4.3 weeks it gives

$$
\frac{\ln 100}{\ln 4.3} = 3.157 ,
$$

which the article prints as about 3.1. `msrelapse.model.barrier_ratio_from_durations`
is this one line, and the package treats 3.157 as a named heuristic constant
rather than as a definition, for two reasons.

The first is that equation (7) follows from equations (5) and (6) only because
those dropped the prefactor. With the prefactor dropped, the ratio of the two
exponents is the barrier ratio by construction, for every noise amplitude,
which is exactly why $\varepsilon$ cancels out of it. Applied to durations
measured in weeks it assumes that both prefactors equal one week, and the
section above shows that the relapse-side prefactor alone is 4.4 weeks.

The second is that the potential has a barrier ratio of its own, and the two
numbers are not the same. `DoubleWell.barrier_ratio` computes
$\Delta V_1 / \Delta V_2$ from the potential directly. Two different routes
lead from the same pair of durations to a potential:

| Route | What it solves | Result at $\alpha = 1$ |
|---|---|---|
| `beta_from_barrier_ratio(3.157)` | make $\Delta V_1/\Delta V_2$ equal the ratio of the logarithms | $\beta = 0.1375$ |
| `calibrate(100.0, 4.3)` | make the two mean first passage times equal 100 and 4.3 weeks | $\beta = 0.2104$, $\sigma = 0.5078$, whose barrier ratio is 6.59 |

Both are implemented, both are documented, and neither is presented as the
answer. Equation (7) is what the article did; the calibration is what the
stochastic equation actually says.

## Are the exit times exponential?

The article's central observation is that both duration distributions are
exponential with no typical scale, which is what makes relapse onset
memoryless. The model explains it. In the limit of small noise the exit time of
a well, divided by its own mean, converges in distribution to a standard
exponential: this is the exponential leveling result of Day (1983), and it is
the reason the shape of the observed histograms carries no information about
the shape of the potential.

The limit is a limit, though, and it needs a deep well: $2\Delta V/\sigma^2$
has to be large. At the calibration to the published pair it is 3.83 on the
health side and **0.58 on the relapse side**, so the shallow relapse well is
nowhere near the small noise regime and nothing guarantees an exponential
there. Measured rather than assumed, over 4000 paths at a step of 0.01 weeks
and over the seeds 0 to 4, the coefficient of variation of the exit time lies
between 0.9799 and 1.0332 on the health side and between 0.9137 and 0.9569 on
the relapse side, close to the exponential value of 1 but arrived at by
measurement. The recipe is `exit_times(well, sigma, side, n_paths=4000,
dt=0.01, rng=seed)` on the potential the published pair calibrates to, so the
range above can be reproduced digit for digit. The second notebook sweeps
$\sigma$ and puts the measured coefficient of variation beside the exponential
value of 1, together with the gap between the Kramers time, the exact first
passage time and the simulation. At the few hundred paths behind each of its
points that panel rules out an exit time far from exponential without
separating the two wells; it is the regime table beside it that shows how far
the relapse well sits from the small noise limit.

## From one patient to a cohort

Read phenomenologically rather than mechanistically, a record that alternates
between two exponential states is an alternating renewal process: remissions
are exponential with rate $\lambda = 1/\tau_{x_1}$, relapses are exponential
with rate $\mu = 1/\tau_{x_2}$, and the two alternate.
`msrelapse.renewal.alternating_renewal` generates such records, and it is the
null model the double well has to beat rather than a rival to it.

### The onset rate

Two relapse onsets are separated by one remission and one relapse, so onsets
arrive at

$$
\nu = \frac{1}{1/\lambda + 1/\mu} = \frac{\lambda}{1 + \lambda/\mu} ,
$$

which is lower than $\lambda$ because the time spent in a relapse is time in
which no new relapse can begin. At the published pair,
$\nu = 0.00959$ onsets per week, which is 0.50 relapses per year.

### Counts are Poisson, until patients differ

With a fixed onset rate, the number of onsets in a window of $T$ weeks is
Poisson with mean $\nu T$, so its variance equals its mean. Real cohorts are
overdispersed: some patients relapse far more often than others. Give each
patient a rate of their own drawn from a gamma distribution of shape $k$ and
scale $\theta$, and mix,

$$
\Pr(N = n) = \int_0^{\infty} \frac{(\nu T)^n e^{-\nu T}}{n!}\; g_{k,\theta}(\nu)\, d\nu ,
$$

and the mixture is negative binomial, with

$$
\mathbb{E}[N] = k\,\theta\,T , \qquad
\operatorname{Var}[N] = \mathbb{E}[N] + \frac{\mathbb{E}[N]^2}{k} .
$$

The dispersion is $1/k$, and letting $k$ grow with $k\theta$ held fixed
returns the Poisson. This is the
standard model for relapse counts in trials, from Keene, Jones, Lane and
Anderson (2007), and `msrelapse.renewal.nb_from_gamma` is the two line mapping.

### Staying free of relapse

The probability of reaching the end of a window of $T$ weeks with no relapse is
the survival of the waiting time to the first onset. With one shared rate it is

$$
S(T) = e^{-\nu T} ,
$$

and with the gamma mixture it is

$$
S(T) = \left(1 + \theta T\right)^{-k} ,
$$

which decays more slowly, because the survivors are increasingly the patients
whose own rate is low. That heavier tail is the reason a trial powered on a
Poisson assumption is underpowered, and
`msrelapse.stats.sample_size_arr` implements the negative binomial sample size
of Zhu and Lakkis (2014) rather than a Poisson one.

## The weekly rounding rule and the one week correction

The study recorded weeks. Exacerbations shorter than a week were rounded up to
one week, so any week a relapse touches counts as a whole relapse week, and the
1 week minimum relapse in the histogram is an artefact of that rule rather than
a biological floor.

Rounding is not neutral. A relapse of continuous length $L$ weeks starts at a
uniform position inside a week, so on average it covers $L + 1$ whole weeks,
and the remission beside it loses exactly that week. A continuous time engine
aimed at the printed pair therefore produces weekly means that are too long on
the relapse side and too short on the health side. The correction is to aim one
week below and one week above,

$$
\tau^{\text{continuous}}_{\text{relapse}} = \tau_{x_2} - 1 , \qquad
\tau^{\text{continuous}}_{\text{health}} = \tau_{x_1} + 1 ,
$$

which is what `msrelapse.cohort.continuous_targets` returns, health first: 101
and 3.3 weeks for the published pair. It is not a complete correction. Two
relapses separated by less than a week fall in the same week and merge into one
longer weekly episode, which no aim can undo; how often that happens is set by
the width of the hysteresis band and is measured in the documentation of
`msrelapse.cohort.CohortSpec`.

A second measurement artefact sits beside it. The article averaged every
recorded run of a state, including the last remission of each record, which the
end of follow up cut short. Over follow up windows of 40 to 1311 weeks, a
remission whose true mean is 100 weeks rarely fits twice, so that naive average
lands near 80 weeks rather than 100. Reproducing the printed means therefore
takes a generative mean above them: `msrelapse.cohort.naive_mean_targets`
inverts the measurement, and the shipped twin is generated from 134.21 and
4.34415 weeks so that the naive means of the result land on the printed 100 and
4.3.

## Calibration, and whether it is unique

Calibration takes the pair of observed mean durations and returns the pair
$(\beta, \sigma)$ that reproduces them. Two unknowns, two equations,

$$
T_{\text{health}}(\beta, \sigma) = \tau_{x_1} , \qquad
T_{\text{relapse}}(\beta, \sigma) = \tau_{x_2} ,
$$

with $T$ the exact mean first passage time over the endpoints of the chosen
reading of an episode. `msrelapse.model.calibrate` solves it as a least squares
problem in $\beta$ and $\log \sigma$ on the logarithms of the two times, from
three starting values of $\beta$, and keeps the best of the three.

The solution is unique in the region that matters, and for a reason worth
seeing: $\sigma$ sets the overall time scale, shortening both passages together,
while $\beta$ tilts the potential and moves them in opposite directions. The
two targets therefore pin the pair rather than a curve of pairs, and the fit
lands within $10^{-6}$ in relative terms or raises. Three edges bound the
region. $\beta$ is held strictly positive, so two targets that differ by less
than about $3 \times 10^{-5}$ in relative terms have no solution in the box and
the symmetric potential is the answer there. $\beta$ is held below the fold, so
a ratio of durations that would need a deeper tilt than the fold allows has no
solution at all. And the `kramers` method has the prefactor floor described
above, which the published relapse duration is under.

At $\alpha = 1$ the published pair calibrates to

$$
\beta = 0.210435 , \qquad \sigma = 0.507768 , \qquad
\varepsilon = \sigma^2 = 0.2578 .
$$

That noise variance is about twice the $\varepsilon = 0.13$ the article uses in
its own Figure 7. Under an exact first passage reading, then, the article's own
noise value does not reproduce the article's own durations. The package reports
that rather than hiding it, and the figure reproduction uses the printed 0.13
because the figure is what it reproduces.

## Numerical choices

None of the following is in the article, which states no integrator, no time
step, no seed, no initial condition and no realisation count. They are this
package's choices and they are labelled as such wherever they appear.

**Model time is read as weeks.** The article's Figure 7 has an unlabelled time
axis running from 0 to 1000 and gives no conversion to clinical time. Reading
model time as weeks is what makes the calibration above meaningful, and it is
an assumption, not a result.

**Euler and Maruyama.** The scheme is

$$
x_{k+1} = x_k + \left[x_k\left(1 - \alpha x_k^2\right) - \beta\right] dt
        + \sigma \sqrt{dt}\; z_k , \qquad z_k \sim \mathcal{N}(0, 1) .
$$

The noise is additive, so the scheme is of strong order one and the Milstein
correction term is identically zero. There is nothing to add.

**The time step.** The cubic drift is not globally Lipschitz, so a step that is
too long sends a path to infinity instead of into a well. A convergence study
at the calibrated parameters found that half a week blows paths up, which is
where `msrelapse.simulate.MAX_DT` of 0.3 weeks comes from. The default step is
0.02 weeks in `exit_times` and `simulate_weekly`, and 0.01 weeks in
`simulate_paths`, the raw integrator.

**The Brownian bridge correction.** A crossing tested only at the grid points
misses the excursions between them, so an exit time read off the grid runs long
by a term of order $\sqrt{dt}$. Moving the absorbing level towards the walker
by

$$
0.5826 \; \sigma \sqrt{dt}
$$

cancels that term to first order; the constant is
$-\zeta(1/2)/\sqrt{2\pi}$, the expected overshoot of a discretely monitored
Brownian motion. With it, a step of 0.02 weeks stands in for one of 0.001.
`msrelapse.simulate.BRIDGE_CONSTANT` holds it, and both the exit time
measurement and the mapping of a path to states carry it.

**The hysteresis band.** The article gives no rule for turning a continuous
$x(t)$ into a $\pm 1$ series. A bare threshold at the saddle counts every
wobble across the barrier top as a relapse. The package instead places two
thresholds a fraction of the way from the saddle towards each well bottom, and
a state changes only when the path crosses the far one. The fraction has two
named defaults, and they are not the same number.
`msrelapse.model.DEFAULT_BAND_FRACTION` is 0.3, and every band taking function
of `msrelapse.model` and `msrelapse.simulate` uses it.
`msrelapse.cohort.SDE_ENGINE_BAND_FRACTION` is 0.4, and the stochastic engine of
`msrelapse.cohort.CohortSpec` uses it, because a wider band removes more of the
sub-week remissions that the weekly rounding would merge. The two cut a path
into episodes differently, so a path segmented at one of them and a cohort
generated at the other agree only when the same fraction is passed to both.
Passing that same fraction to `calibrate` with `passage='band'` makes the
simulated durations approach the calibration targets.

**The parameters do not vary with time.** $\beta$ and $\sigma$ are constants of
a patient here, as they are in the article. A slowly varying $\beta(t)$ or
$\sigma(t)$, which would let a barrier drift over a follow up and is the natural
way to write a treatment effect or a progression into the model, is out of scope
for version 0.1 and is recorded as future work: the equations allow it, and what
is missing is a calibration procedure and the data to validate it against.

## Other relapsing conditions

Nothing in `model`, `simulate`, `renewal`, `fit` or `stats` knows what the two
states mean. The mechanism is a particle in an asymmetric double well, and the
statistics is that of an alternating renewal process with exponential sojourns,
so any condition recorded as two clinical states with memoryless transitions
fits the same interface: relapsing neuromyelitis optica spectrum disorder,
relapsing inflammatory conditions recorded as episodes, or the asthma and
chronic obstructive pulmonary disease exacerbation counts that Keene et al.
(2007) analyse, which is where the negative binomial of the counts section
comes from in the first place.

What has to be redone for another condition is the calibration and not the
mathematics: the two mean durations handed to `calibrate`, which fix $\beta$
and $\sigma$ at a chosen $\alpha$; the follow up lengths and the cohort size of
`CohortSpec`; the time unit, a week here only because the study recorded weeks;
and the rounding correction of `continuous_targets`, which is a whole-week
correction and has to be rederived in whatever unit the new record keeps. What
does not transfer is every number in `msrelapse.PAPER`. Those belong to the
2013 multiple sclerosis cohort, and a reproduction of this article is not a
statement about any other disease.

## What the article leaves open

The list is short and it is worth having in one place: no integrator, time
step, seed, initial condition or number of realisations; no conversion between
model time and weeks; no statement of whether $\varepsilon$ is a variance per
unit time or per step; no Kramers prefactor and no first passage citation; no
procedure for inverting the barrier ratio to obtain $\beta$; no rule for
turning a simulated path into a binary series; no procedure for estimating
$\alpha$; and no goodness of fit statistic of any kind, anywhere. Section 13 of
[Paper facts](paper_facts.md) records each of them with the search that
established it.

## References

Bordi I, Umeton R, Ricigliano VAG, Annibali V, Mechelli R, Ristori G, Grassi F,
Salvetti M, Sutera A. A mechanistic, stochastic model helps understand multiple
sclerosis course and pathogenesis. International Journal of Genomics.
2013;2013:910321. doi:10.1155/2013/910321.

Benzi R, Parisi G, Sutera A, Vulpiani A. A theory of stochastic resonance in
climatic change. SIAM Journal on Applied Mathematics. 1983;43(3):565.

Kramers HA. Brownian motion in a field of force and the diffusion model of
chemical reactions. Physica. 1940;7:284.

Day MV. On the exponential exit law in the small parameter exit problem.
Stochastics. 1983;8:297.

Zhu H, Lakkis H. Sample size calculation for comparing two negative binomial
rates. Statistics in Medicine. 2014;33:376. doi:10.1002/sim.5947.

Keene ON, Jones MRK, Lane PW, Anderson J. Analysis of exacerbation rates in
asthma and chronic obstructive pulmonary disease. Pharmaceutical Statistics.
2007;6:89.
