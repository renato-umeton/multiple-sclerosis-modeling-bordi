# Bordi et al. 2013: verified paper facts for implementation

Source: Bordi I, Umeton R, Ricigliano VAG, Annibali V, Mechelli R, Ristori G, Grassi F, Salvetti M, Sutera A. "A Mechanistic, Stochastic Model Helps Understand Multiple Sclerosis Course and Pathogenesis". International Journal of Genomics, Volume 2013, Article ID 910321, 10 pages. DOI 10.1155/2013/910321. Publisher Wiley. Received 24 October 2012; Revised 2 January 2013; Accepted 27 January 2013. Academic Editor Brian Wigdahl. Licence CC BY.

Every equation, number and figure value below was checked against the PDF: the text layer was extracted and read directly, and every disputed figure was rendered at 300 to 400 dpi and measured with axis calibration. Where a value is measured rather than printed, this document says so.

## 1. Sign convention, fixed in three places

Clinical encoding (Section 2.1, page 2): "Each subject has been represented by a sequence of plus one (+1) and minus one (-1) corresponding to the states of relapses and remissions, respectively. For clarity, hereafter these two states are denoted as *no health* and *health*."

So `+1` is RELAPSE, called "no health"; `-1` is REMISSION, called "health".

Model counterpart (page 4, right column, immediately after equation (1)): "The above equation has three steady (independent on time) states. They are x0 = 0, x1 = -1/sqrt(a), and x2 = +1/sqrt(a). Note that in case alpha = 1, we have x1 = -1 and x2 = 1. Now, in agreement with the convention used for clinical data, let x1 be the state of *health* and x2 the state of *no health*."

| Model state | Position | Clinical meaning | Clinical code | Depth when beta > 0 |
|---|---|---|---|---|
| x1 | -1/sqrt(alpha) | health, remission | -1 | deeper well |
| x0 | middle root | unstable barrier top | none | not a state |
| x2 | +1/sqrt(alpha) | no health, relapse | +1 | shallower well |

Restated in the Figure 5 caption (page 5) and at the head of Section 3.2 (page 3). Figure 2's y-axis labels the +1 tick "No health sate" (a printed typo for "state") and the -1 tick "Health state". Figure 7(b) confirms the direction: with beta = 0.08 "the variable x still changes erratically its state but spends more time around -1 than around +1".

## 2. Equations exactly as printed

Equation (1), page 4, deterministic mechanistic component:

    dx = x (1 - alpha*x^2) dt

The control parameter sits INSIDE the bracket, so the drift expands to `x - alpha*x^3`. Alpha multiplies x CUBED, never the linear term.

Steady states of (1), unnumbered, inline on page 4:

    x0 = 0,  x1 = -1/sqrt(a),  x2 = +1/sqrt(a)

The radicand is printed as a Latin italic `a` (U+1D44E), not the Greek alpha (U+1D6FC) used everywhere else. This was verified at the byte level. It is a typesetting slip; implement `1/sqrt(alpha)`.

Equation (2), page 4, symmetric potential:

    V(x) = - integral of x(1 - alpha*x^2) dx = -(1/2)*x^2 + alpha*(1/4)*x^4

Equation (3), page 5, asymmetric potential:

    V(x) = - integral of [x(1 - alpha*x^2) - beta] dx = -(1/2)*x^2 + alpha*(1/4)*x^4 + beta*x

Equation (4), page 6, the full stochastic equation of motion, the one actually simulated:

    dx = [x(1 - alpha*x^2) - beta] dt + epsilon^(1/2) dw

printed with three underbraces: `dx` is "change of the health state of a patient", the drift term is "mechanistic component (double-well)", the noise term is "external random stimuli". Followed by "where epsilon^(1/2) dw is the noise of variance epsilon".

Equations (5) and (6), page 6, mean exit times:

    tau_x1 approximately e^(2*dV1/epsilon)
    tau_x2 approximately e^(2*dV2/epsilon)

No prefactor is printed. "where e is the Nepero number", that is Napier's number, so the base is e.

Equation (7), page 6, the barrier-ratio estimator:

    dV1 / dV2 approximately log(tau_x1) / log(tau_x2)

The base of "log" is not given, but because this is a ratio of two logarithms the base cancels exactly. Do not write two separate tests for a natural-log and a base-10 version; they are the same number.

Barrier definition, stated only in words on page 6, left column: "the barriers are denoted as dV1 and dV2 and are the difference in the potential between the states x1 and x0, and x2 and x0, respectively". That is:

    dV1 = V(x0) - V(x1)      barrier out of health
    dV2 = V(x0) - V(x2)      barrier out of no health

No closed form for the barrier is printed anywhere in the paper.

## 3. Which convention to implement, and why

Implement, in exactly this form:

```python
def potential(x, alpha, beta):
    return -0.5 * x**2 + 0.25 * alpha * x**4 + beta * x

def drift(x, alpha, beta):
    return x * (1.0 - alpha * x**2) - beta        # equals -dV/dx
```

This is the numerical check's **convention B**. Convention A (`V = x**4/4 - a*x**2/2 + beta*x`) is NOT this paper and must not be used. The two are algebraically identical at alpha = 1 and only there, which is why every alpha = 1 value agrees under both. They diverge at alpha = 0.7, which the paper does use.

Evidence that B is correct:

1. Printed equation (2) is convention B verbatim: the alpha multiplies the quartic, and there is no alpha on the quadratic.
2. The drift in (4) is exactly the negative gradient of (3): `-dV/dx = x - alpha*x^3 - beta = x(1 - alpha*x^2) - beta`. Verified.
3. Figure 5(b), rendered at 300 dpi. At alpha = 0.7 the minima move OUTWARD to about plus or minus 1.2 and DOWN to about -0.357. Convention B predicts wells at `1/sqrt(0.7) = 1.1952` and depth `-1/(4*0.7) = -0.3571`. Convention A would have moved them INWARD to plus or minus 0.837 and UP to -0.1225. The figure is unambiguous.

Consequences, derived and not printed in the paper:

| Quantity | beta = 0 closed form | alpha = 1 | alpha = 0.7 |
|---|---|---|---|
| Well positions | plus or minus 1/sqrt(alpha) | plus or minus 1 | plus or minus 1.195229 |
| Well depth | -1/(4*alpha) | -0.25 | -0.357143 |
| Barrier dV | 1/(4*alpha) | 0.25 | 0.357143 |

For beta not equal to zero there is no closed form. The three stationary points are the real roots of `alpha*x^3 - x + beta = 0`; find them numerically (`numpy.roots([alpha, 0, -1, beta])`, keep real roots, sort ascending as x1 < x0 < x2). At alpha = 1 three real roots exist only for `beta < 2/(3*sqrt(3)) = 0.384900179459750`; above that the no-health well vanishes in a saddle-node fold. Guard package input against this.

## 4. How the barrier depends on alpha

Text, page 4, right column: "It should be noted that the difference of the potential in x1 (x2) and x0 (denoted by delta-V in the figure) is the barrier that must be overcome to jump from one state to the other; **it increases when alpha is decreased with respect to the reference value alpha = 1** (the opposite occurs when alpha is greater than 1). In the context of MS, such a barrier is though [sic] to be set by the combination of heritable and nonheritable risk."

Figure 5 caption: "delta-V is the barrier whose height is controlled by the parameter alpha."

Table 1, page 7: "alpha: control parameter that sets the barrier to overcome for going from one state to the other."

**Text and figure agree completely**, and both agree with the algebra of equation (2): `dV = 1/(4*alpha)`, monotonically decreasing in alpha. Lower alpha means a higher barrier, longer dwell times and fewer transitions. Clinically alpha encodes "the combination of heritable and nonheritable risk" (page 4) or "the genetic background and environmental exposures of the patient" (page 3).

This resolves the plan's section 1.1 VERIFY flags. Both resolve against the plan's primary guess and in favour of its parenthetical alternative. The plan's provisional forms `dx/dt = a*x - x^3` and `V = x^4/4 - a*x^2/2` are wrong for alpha not equal to 1 and should be deleted, not merely flagged. Note that the plan's regression targets for alpha = 1, beta = 0.08 remain correct unchanged, because alpha = 1 is the crossover point.

The same direction holds in the asymmetric case: at beta = 0.08, lowering alpha from 1 to 0.7 raises dV1 from 0.3348 to 0.4575 and dV2 from 0.1749 to 0.2664, while their ratio falls slightly from 1.914 to 1.718.

Alpha is never estimated from data. Every patient fit fixes alpha = 1; alpha = 0.7 appears only in Figures 5(b) and 6(b).

## 5. The asymmetry term

The potential term is `+beta*x`, arising from `-beta` inside the drift. With beta > 0 the potential is lowered for x < 0 and raised for x > 0, so the LEFT well at x1 (health) deepens and the RIGHT well at x2 (no health) becomes shallower. This is exactly what the paper wants (page 5: "the one associated to the health state is deeper than the other, so that it is more difficult to exit from the health condition than from the relapse"), and measuring Figure 6(a) confirms it: the deep well is on the negative x side.

Parameter values used in the figures:

| Parameter | Values used | Where |
|---|---|---|
| alpha | 1 (reference), 0.7 | 1 in Figs 5a, 6a, 7a, 7b, 8a-c; 0.7 in Figs 5b, 6b only |
| beta | 0, 0.08, 0.25, 0.19, 0.12 | 0 in Figs 5, 7a; 0.08 in Figs 6, 7b; 0.25/0.19/0.12 in Figs 8a/8b/8c |
| epsilon | 0.13 | Figure 7, both panels. The only noise value in the entire paper |

The noise amplitude multiplying `dw` is `sqrt(0.13) = 0.360555`.

## 6. Verified numerical values

All computed in the paper's own convention with barriers anchored at V(x0), and cross-checked against the rendered figures.

| alpha | beta | x1 | x0 | x2 | V(x1) | V(x0) | V(x2) | dV1 | dV2 | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | -1 | 0 | 1 | -0.2500 | 0 | -0.2500 | 0.2500 | 0.2500 | 1.000 |
| 0.7 | 0 | -1.1952 | 0 | 1.1952 | -0.3571 | 0 | -0.3571 | 0.3571 | 0.3571 | 1.000 |
| 1 | 0.08 | -1.037827 | 0.080522 | 0.957305 | -0.331541 | 0.003210 | -0.171670 | 0.334751 | 0.174880 | 1.914174 |
| 0.7 | 0.08 | -1.2334 | 0.0804 | 1.1530 | -0.4543 | 0.0032 | -0.2632 | 0.4575 | 0.2664 | 1.7175 |
| 1 | 0.25 | -1.1072 | 0.2696 | 0.8376 | -0.5140 | 0.0324 | -0.0183 | 0.5464 | 0.0507 | 10.775 |
| 1 | 0.19 | -1.0841 | 0.1977 | 0.8864 | -0.4483 | 0.0184 | -0.0701 | 0.4667 | 0.0885 | 5.273 |
| 1 | 0.12 | -1.0553 | 0.1218 | 0.9335 | -0.3734 | 0.0073 | -0.1338 | 0.3807 | 0.1411 | 2.698 |

Equation (7) applied to the printed durations, with natural logarithms:

| Case | tau_x1 | tau_x2 | ln ratio | printed |
|---|---|---|---|---|
| Cohort | 100 | 4.3 | 3.1572 | 3.1 |
| Patient 23 | 117.7 | 1.5 | 11.7597 | 11.8 |
| Patient 32 | 54.0 | 2.1 | 5.3764 | 5.3 |
| Patient 53 | 47.0 | 4.3 | 2.6396 | 2.7 |

Note the rounding direction is not uniform: 5.3764 is printed 5.3 rather than 5.4, and 2.6396 is printed 2.7 rather than 2.6.

The beta that would deliver each ratio at alpha = 1: 0.1375 for the cohort's 3.157, 0.2565 for 11.8, 0.1905 for 5.3, 0.1201 for 2.7. So the published beta for patients 32 and 53 is consistent to two decimals, but the published beta = 0.25 for patient 23 delivers only 10.775. That is a real inconsistency in the paper rather than a mis-transcription, which rendering Figure 8(a) settles: the plotted curve matches the beta = 0.25 values exactly.

The alternative barrier anchor was falsified as well. Measuring well depths from V = 0 instead of from V(x0) gives ratios 28.04, 6.39 and 2.79 for the three Figure 8 beta values, matching none of the printed values. The V(x0) anchor is correct.

## 7. Clinical data

From Section 2.1 (page 2), verbatim in substance: 70 patients (28 males and 42 females) with definite MS, monitored prospectively at the MS Clinic of Sapienza University of Rome by four experienced MS neurologists. All were free of any disease-modifying therapy (all data antecedent 1993) and received short courses of corticosteroids if deemed necessary during relapses. "Times to disability endpoints were not different between this cohort and published data on the natural history of the disease."

Diagnostic criteria: references [14] Schumacher et al. 1965 and [15] Poser et al. 1983.

Observation window: "The period of interest starts with the first relapse at onset and ends with the last one before the shift to a secondary progressive form." The secondary progressive phase is excluded.

Relapse definition (citing [14]): "the occurrence of new symptoms, the reappearance of former ones, or the worsening of current symptoms of at least 24 hours duration but less than 6 month". Duration is "the interval between onset of the first symptom or sign and maximum improvement of the last symptom or sign".

Time resolution: weekly scale; "shorter exacerbations have been rounded up to one week". The 1-week minimum relapse duration is an artefact of this rule.

Discretisation caveat (page 3): transitions are step functions, which the paper concedes "can be reductive or unrealistic since patients usually report a slow or subacute onset of the relapses. However, this feature does not affect the results presented below."

Aggregate statistics:

| Quantity | Value | Source |
|---|---|---|
| No health (relapse) events | 218 total | Fig 4(a) in-panel text |
| Mean relapse duration | 4.3 weeks | Fig 4(a), Section 3.1, Section 3.4 |
| Relapse duration range | 1 to about 24 weeks | Section 3.1 |
| Health (remission) events | 266 total | Fig 4(b) in-panel text |
| Mean remission duration | about 100 weeks | Fig 4(b), Section 3.1, Section 3.4 |
| Remission duration range | a few weeks to about 1000 weeks | Section 3.1 |
| Relapsing-remitting phase length | 40 to 1311 weeks (about 27 years) | Section 3.1, Fig 3 caption |
| Modal phase length | 25 of 70 patients (about 36 percent) at about 200 weeks | Section 3.1, Fig 3 caption |
| Cohort barrier ratio | dV1/dV2 about 3.1 | Section 3.4 |

A consistency check: 266 x 100 + 218 x 4.3 is about 27,537 patient-weeks, that is about 393 weeks per patient, matching the bin-weighted mean of Figure 3 (about 394 weeks). The two figures are mutually consistent.

**There are no statistical fits in this paper.** A search of the full text returns nothing: no fitted exponential rate, no R squared, no chi square, no Kolmogorov-Smirnov test, no p value, no confidence interval, and no fitted curve overlaid on any histogram. The exponential and no-periodicity claims are purely qualitative, resting on "the absence of any peak in the distributions at some specific time" (page 3). Do not attempt to reproduce a published fit; there is none.

## 8. The three sample patients

Printed in Section 3.4 (Model Solutions), page 6, as a three-line display:

| Patient | tau_x1 (health, weeks) | tau_x2 (no health, weeks) | dV1/dV2 | fitted beta (alpha = 1) |
|---|---|---|---|---|
| no. 23 | 117.7 | 1.5 | 11.8 | 0.25 |
| no. 32 | 54.0 | 2.1 | 5.3 | 0.19 |
| no. 53 | 47.0 | 4.3 | 2.7 | 0.12 |

Fitting procedure (page 6): "the potential V associated with each patient can be estimated by fixing alpha = 1 and changing beta so that the two wells are asymmetric in the predetermined ratio."

Time series, measured at 400 dpi since the paper prints no numbers on Figure 2:

| Patient | Relapse episodes | Approximate onset weeks | Record length |
|---|---|---|---|
| no. 23 | 2 | 13 to 15, 189 | about 356 weeks |
| no. 32 | 8 | 3, 88, 97, 165, 210, 318, 481, 501 | about 505 weeks |
| no. 53 | 4 | 8 to 18, 45 to 48, 201 to 203, 245 to 248 | about 252 weeks |

Each record is consistent with n relapse segments and n+1 remission segments, counting both the leading and the trailing remission: 3 x 117.7 + 2 x 1.5 = 356.1; 9 x 54.0 + 8 x 2.1 = 502.8; 5 x 47.0 + 4 x 4.3 = 252.2. Use these as regression fixtures. The paper never states this censoring convention, and applied cohort-wide it would predict 288 health events rather than the 266 reported.

Interpretation (page 6 into page 7): patient 23 has "a very deep well associated with the health state and a very shallow well associated with the no health state: this configuration lets the patient be affected by a few disease attacks of short duration"; patient 32 "is expected to have more attacks compared to the previous patient"; patient 53 "should experience disease events lasting more time because of the deeper well associated with the no health state".

No age, sex, EDSS, onset date or treatment history is given for any of the three.

## 9. What to reproduce for each figure

**Figure 1** (page 3). Schematic cartoon only, no axes and no numbers. Deep well on the left labelled "Deep well", shallow well on the right labelled "Shallow well", central hump annotated "Potential barriers separating the wells". Panel (a) a particle falling into the deep well; panel (b) a particle hopping over the hump. Documentation art only; do not present it as a plot of equation (2) or (3).

**Figure 2** (page 4). Three binary step-function series. y-axis two levels only, +1 "No health sate" [sic] and -1 "Health state". x-axis "Time (week)" 0 to 500 ticked every 50. Episode counts and onsets as tabulated in section 8 above.

**Figure 3** (page 4). Histogram of relapsing-remitting phase length. y-axis "Counts (number of patients)" 0 to 30 ticked every 5. x-axis "Period of analysis (week)" with tick labels at the bin edges 40, 199, 358, 517, 676, 835, 994, 1153, 1312, giving 8 bins of 159 weeks. Bar heights, measured and summing to exactly 70: **25, 17, 9, 5, 7, 2, 2, 3**.

**Figure 4** (page 5). Two histograms.
Panel (a): y "Counts (no. of no health events)" 0 to 140 ticked every 20; x "Duration of no health events (week)" ticked 1, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 23.5, 25, so bins are 2.5 weeks wide starting at 1; in-panel text "No health events: total = 218" and "Duration of no health events: mean = 4.3 weeks". Measured bar heights, summing to exactly 218: **127, 53, 16, 6, 1, 6, 0, 4, 5**. The last bar is drawn wider than the uniform bin width and ends at about 24 weeks.
Panel (b): y "Counts (no. of health events)" 0 to 200 ticked at 0, 50, 100, 150, 200; x "Duration of health events (week)" 0 to 1200 ticked every 200, bins 100 weeks wide; in-panel text "Health events: total = 266" and "Duration of health events: mean = 100 weeks". Measured bar heights: **192 or 193, 35, 18, 5, 0, 5, 4, 2, 0, 2, 2**, summing to 265 against the printed 266. Treat the first two bins as uncertain by plus or minus 1; unlike Figures 3 and 4(a) these cannot be pinned exactly.

**Figure 5** (page 5). Two plots of the symmetric potential, equation (2). Axes x from -2 to 2 ticked every 1, V from -0.8 to 0.8 ticked every 0.2. Two horizontal dashed lines per panel (upper at V = 0 through x0, lower at the well depth) plus a vertical double-headed arrow labelled delta-V, and point labels x0, x1, x2. Panel (a) alpha = 1: minima at plus or minus 1, V = -0.25, delta-V = 0.25. Panel (b) alpha = 0.7: minima at plus or minus 1.195229, V = -0.357143, delta-V = 0.357143. **This pair is the acceptance test for the potential convention.** Any implementation where lowering alpha moves the wells inward or makes them shallower has the wrong convention.

**Figure 6** (page 7). Two plots of the asymmetric potential, equation (3), same axes. Three horizontal dashed lines per panel (through V(x0), V(x2), V(x1)) and two vertical arrows, dV1 on the left and taller, dV2 on the right and shorter, both anchored at the V(x0) line. Panel titles inside the axes at upper right. (a) alpha = 1, beta = 0.08; (b) alpha = 0.7, beta = 0.08. In both panels the deep well is on the negative x side. One measured caveat: in panel (a) the x1 and x0 dashed lines match the formula to about 0.001, but the x2 dashed line is drawn at V = -0.1821 against the true -0.1717, about 0.010 too low. Trust the formula.

**Figure 7** (page 8). The key simulation figure. Two noisy traces from solutions of equation (4). y-axis "x" from -2 to 2 ticked at -2, -1, 0, 1, 2; x-axis "Time" 0 to 1000 ticked every 100, with **no unit** and no conversion to weeks; parameters printed as a title above each panel. Panel (a) alpha = 1, beta = 0, epsilon = 0.13: roughly balanced occupancy of the two states with residence times of order 100 to 300 time units. Panel (b) alpha = 1, beta = 0.08, epsilon = 0.13: the trace sits near -1 almost throughout with only two brief excursions to +1 (around Time 135 to 150 and 490 to 505). **These traces cannot be reproduced bit for bit**: no integration scheme, time step, seed, initial condition or realisation count is stated anywhere. Reproduce the qualitative behaviour and the occupancy asymmetry only, and document your own choices.

**Figure 8** (page 8). Three plots of equation (3), one per sample patient, axes x from -2 to 2 and V from -0.8 to 0.8. Panel titles above the axes; parameter pair inside the axes at upper right. A single horizontal dashed line per panel marks V(x0), the upper bound of both barriers. All three curves are strongly left skewed, with the right well deepening progressively from (a) to (c). Values as tabulated in section 6. Reproduce by fixing alpha = 1 and using the printed beta; do not derive beta from the printed ratio, because that gives 0.2565 rather than 0.25 for patient 23.

## 10. Table 1 (page 7), full contents

Caption: "Disease characteristics and parameters of the mechanistic stochastic model. The table summarizes analogies and links between MS characteristics (left column) and model parameters in (4), in the right column."

| Disease characteristics during the relapsing-remitting phase | Mechanistic, stochastic model |
|---|---|
| Two states are observed that we referred to as health and no health. | V(x): double-well potential (two minima associated with the two states). |
| The disease develops in patients with a certain degree of heritable and nonheritable risk. | alpha: control parameter that sets the barrier to overcome for going from one state to the other. |
| It is found that the time spent in the state of health is much longer than in the no health state. | beta: asymmetry parameter of the potential. |
| Random variability of biological processes (i.e. randomness in transcription and translation leading to cell-to-cell variations) [8]. | epsilon^(1/2) dw: stochastic perturbation with zero mean and variance epsilon. |
| Genetic background and environmental factors alone are not enough to trigger the transitions between the two states. | The noise provides the required energy for the random transitions. |

## 11. Noise description

Section 2.2 (page 2): the random perturbation is the Wiener process [16], "which consists of a normally distributed white noise (i.e., the amplitude of the noise is normally distributed with given mean and standard deviation, and all frequencies are involved with no time periodicity)".

Section 3.3 (page 6): "Let us suppose that on average the noise has zero effect (i.e., its time mean is zero), has a given variance, and it is time decorrelated (i.e., at a given time the value of the stimulus depends only on the previous time): this is the zero order approximation of the statistics of noise". The parenthetical is a Markov statement rather than a decorrelation statement, but the intended meaning (white, delta-correlated Wiener increments) is clear from the surrounding text.

Why noise is needed (page 6): "Averaged over time it is impossible that x crosses the barrier since the noise, by construction, has zero mean. However, given the statistical property of the stimulus, in due time, we expect that a sequence of small driving occurrences may be equally signed so that x may overcome the barrier and switch from one state to the other. In other words, x is harvesting energy from the stimuli for crossing the barrier."

The paper explicitly disclaims stochastic resonance (page 3 and page 9): there is no periodic forcing term in equation (4), and "In the case of relapsing-remitting MS, as clinical data suggest, there is not a typical periodicity of relapses induced by some external factors."

Deterministic limit, printed as a lettered list on pages 4 to 5: "(a) if, by any chance, at initial time the patient is nearby x1, he will be there forever; (b) if, by any chance, at initial time the patient is nearby x2, he will be there forever; (c) if, by any chance, at initial time the patient is nearby x0, he will have 50% chance to be forever nearby x1 or x2."

## 12. Implementation checklist

1. Potential: `V(x) = -x**2/2 + alpha*x**4/4 + beta*x`.
2. Drift: `f(x) = x*(1 - alpha*x**2) - beta`, which equals `-dV/dx` exactly.
3. SDE: `dx = f(x)*dt + sqrt(epsilon)*dW`, with epsilon = 0.13 for the published simulations.
4. Fixed points: real roots of `alpha*x**3 - x + beta = 0`, sorted ascending as x1 (health) < x0 (saddle) < x2 (no health). At beta = 0 these are `-1/sqrt(alpha)`, 0, `+1/sqrt(alpha)`.
5. Barriers: `dV1 = V(x0) - V(x1)`, `dV2 = V(x0) - V(x2)`. At beta = 0 both equal `1/(4*alpha)`.
6. Mean exit times: `tau_1 = exp(2*dV1/epsilon)`, `tau_2 = exp(2*dV2/epsilon)`, with no prefactor, as printed.
7. Estimator: `dV1/dV2 = log(tau_1)/log(tau_2)`, with tau in WEEKS.
8. Per-patient calibration: fix alpha = 1, solve for beta such that `dV1(beta)/dV2(beta)` equals that patient's log-ratio. Document that this returns 0.2565 for patient 23, against the published 0.25.
9. Guard beta below `2/(3*sqrt(3)) = 0.384900179459750` at alpha = 1, above which bistability is lost.
10. Everything else (time step, integrator, seed, run length in weeks, the rule converting x(t) into a plus-or-minus-one series) must be chosen by the implementer and clearly labelled as not from the paper.

A caution on equations (5) to (7) that matters for any calibration layer. Because the prefactor is dropped, equation (7) is trivially self-consistent with (5) and (6): the ratio of the exponents equals the barrier ratio by construction, for any epsilon. That is why epsilon cancels. But it also means applying (7) to OBSERVED durations in weeks silently assumes the Kramers prefactor equals one week and that the two prefactors cancel, and neither holds. An independent numerical check of the same potential found that the relapse-side Kramers prefactor at alpha = 1 has a hard floor of `2*pi/sqrt(2) = 4.4429` weeks, which is larger than the observed 4.3 weeks, so no Kramers-based calibration to a 4.3-week relapse exists at all. Treat 3.1572 as a named, documented heuristic constant, never as the definition of the barrier ratio and never as a way to derive beta.

A second caution. The same numerical check calibrated the exact mean first passage time to (100 weeks, 4.3 weeks) and obtained beta = 0.210435, sigma = 0.507768. That calibration was performed at a = 1, where the two conventions coincide, so it DOES transfer to this paper's potential. The implied noise variance is `sigma**2 = 0.2578`, roughly twice the paper's epsilon = 0.13. So the paper's own epsilon does not reproduce its own observed durations under an exact first-passage reading. Report this, do not paper over it.

## 13. What the paper does not specify

- No integration scheme, time step, random seed, initial condition or number of realisations for Figure 7. The words "Euler" and "time step" appear nowhere in the paper.
- No conversion between the Figure 7 model time axis (0 to 1000, unitless) and clinical weeks.
- No statement of whether epsilon is a variance per unit time or a per-step variance.
- No Kramers prefactor, and no Kramers or first-passage-time citation. A search for both returns zero hits; equations (5) and (6) are attributed solely to reference [6], Benzi, Parisi, Sutera and Vulpiani, SIAM Journal on Applied Mathematics 43(3):565-578, 1983.
- No procedure, formula or tolerance for inverting the barrier ratio to obtain beta.
- No rule converting a continuous simulated x(t) into a binary plus-or-minus-one episode series, and no quantitative fit of simulated to observed exit-time distributions. The comparison in the paper is purely visual.
- No procedure for estimating alpha from data.
- No per-patient data table, no raw series, no supplementary material.
- No goodness-of-fit statistics of any kind.

## 14. Ethics, licence and availability

Ethics (Section 2.1, page 2, verbatim, including the printed "a analysis"): "Being this a analysis of anonymous clinical data (collected until 1993, before the institution of ethics committee in Italy in 1998), stored for both clinical and research purposes in the database of the University hospital, ethics committee approval and written informed consent of patients are not needed (http://www.garanteprivacy.it/garante/doc.jsp?ID=1884019)."

Licence (page 1): "Copyright (c) 2013 Isabella Bordi et al. This is an open access article distributed under the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited." CC BY, so the equations, figures and numbers may be reused in a derived package with citation.

Authors' Contribution (page 9, a section of its own): "I. Bordi and R. Umeton contributed equally to the work and should be considered co-first authors."

Acknowledgments (page 9): "This work was supported by Fondazione Italiana Sclerosi Multipla (FISM) and by Progetto Strategico 2007, Italian Ministry of Health. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the paper."

Correspondence (page 1): Marco Salvetti, marco.salvetti@uniroma1.it, and Alfonso Sutera, alfonso.sutera@roma1.infn.it.

**Absent from the article.** A search of the complete text of all ten pages for "conflict", "competing", "availab" and "supplement" returns zero hits for every term. There is no data availability statement, no code availability statement, no supplementary material and no conflict of interest statement. The per-patient series are neither deposited nor referenced. A derived package must not claim the paper released data.

The local PDF also carries a Wiley Online Library download stamp in the right margin of pages 2 to 10 and a "Check for updates" badge on page 1. These are retrieval artefacts, not part of the published article; strip them when quoting.

## 15. Printed typographical slips, for transcribers

- Page 4: the steady states are printed `x1 = -1/sqrt(a)` and `x2 = +1/sqrt(a)` with a Latin italic `a` (U+1D44E) under the radical, while the control parameter in the prose is Greek alpha (U+1D6FC). Verified at the byte level. Read as alpha.
- Page 6: "the shapes of the potentials reflect the differences in the duration times ... observed in Figure 1". Figure 2 is meant; Figure 1 is the schematic cartoon.
- Figure 2, all three panels: the +1 tick is labelled "No health sate", missing the t in "state".
- Page 4: "such a barrier is though to be set by", for "thought".
- Page 2: "Being this a analysis of anonymous clinical data", for "an analysis".
- Page 2: "less than 6 month", for "months".
- Reference [27] lists "M. Prinen", almost certainly M. Pirinen.

## 16. Key references for the mathematics

- [6] R. Benzi, G. Parisi, A. Sutera, A. Vulpiani, "A theory of stochastic resonance in climatic change", SIAM Journal on Applied Mathematics 43(3):565-578, 1983. Cited for BOTH the potential computation (equation (2), page 4) and the mean exit time estimate (equation (5), page 6). The single methodological source.
- [13] I. Bordi and A. Sutera, "Drought variability and its climatic implications", Global and Planetary Change 40(1-2):115-127, 2004. Cited as the source of "the equation for a double-well forced by a random perturbation (noise)" and of the mechanistic-plus-stochastic decomposition.
- [16] N. Wiener, "Differential-space", Journal of Mathematics and Physics 2:131-174, 1923. The Wiener process.
- [5] A. Sutera, "On stochastic perturbation and long-term climate behaviour", Quarterly Journal of the Royal Meteorological Society 107(451):137-151, 1981.
- [7] I. Bordi and A. Sutera, "Stochastic perturbation in meteorology", Waves in Random Media 10(3):R1-R30, 2000.
- [14] G. A. Schumacher et al., Annals of the New York Academy of Sciences 122:552-568, 1965. Relapse definition and MS diagnosis.
- [15] C. M. Poser et al., Annals of Neurology 13(3):227-231, 1983. Definite MS.

There is no Kramers reference and no dedicated first-passage-time reference in the 35-item bibliography.
