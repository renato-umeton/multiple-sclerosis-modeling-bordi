# renewal

The phenomenological twin of the stochastic equation. A record that alternates
between two exponentially distributed states is an alternating renewal process,
and this module generates such records, counts the relapse onsets they hold,
and gives the Poisson and negative binomial statistics those counts follow when
relapses are brief and when the onset rate varies from patient to patient.

It produces the same kind of record as `msrelapse.simulate` with no double well
behind it, which is what makes it the null model rather than a rival.

::: msrelapse.renewal
