"""Reference implementation of the Bordi et al. 2013 stochastic model of multiple sclerosis.

The package reproduces the double-well model of relapsing-remitting multiple
sclerosis published in I. Bordi, R. Umeton, V. A. G. Ricigliano, V. Annibali,
R. Mechelli, G. Ristori, F. Grassi, M. Salvetti, and A. Sutera, "A mechanistic,
stochastic model helps understand multiple sclerosis course and pathogenesis,"
International Journal of Genomics, 2013, doi 10.1155/2013/910321, in which a
patient moves between a health state and a no health state under a stochastic
forcing, and the residence times observed in the study cohort fix the depth of
the two wells. Every number the article reports is collected in ``PAPER`` with
the citation it comes from.
"""

from msrelapse._params import PAPER

__version__ = "0.1.0"

__all__ = ["PAPER", "__version__"]
