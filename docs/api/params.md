# params

Every number the article reports, with its unit and its provenance. This module
is the single place in the package where a number taken from the article is
computed with: no other module reads one from a literal, though a docstring or
a doctest elsewhere may quote a printed value as an illustration. Each field
holds the value, the unit it is expressed in and the sentence in the article it
came from, so a reader can check the package against the source one field at a
time.

The module is spelled `msrelapse._params`, but its `PAPER` object is public and
re-exported as `msrelapse.PAPER`. `msrelapse params` prints the whole set from a
terminal.

::: msrelapse._params
