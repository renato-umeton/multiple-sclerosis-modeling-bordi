"""The citation of the article and of this package, in plain text and in BibTeX.

Every result object of the package carries the article it reproduces on a
``citation`` property and repeats it in its repr, so that a number copied out
of a session can be traced back to its source. A raw simulation output, such as
the paths of [`msrelapse.simulate`][msrelapse.simulate], is not a result object
and carries none. The strings are built here once and read from everywhere
else.

No string built by this module writes the DOI of the article down: every one of
them reads it from ``PAPER.paper_doi`` in
[`msrelapse._params`][msrelapse._params], which is the single place in the
package where a number or an identifier of the article may be recorded. The
reference block below quotes the DOI as prose, as the docstring of every other
module of the package does.

The article DOI is the only one here. The software is deposited in no archive
and carries no DOI of its own, so its entry is identified by the repository URL
and carries a note naming the article as the reference to cite.

References
----------
I. Bordi, R. Umeton, V. A. G. Ricigliano, et al., "A mechanistic, stochastic
model helps understand multiple sclerosis course and pathogenesis",
International Journal of Genomics, 2013, doi 10.1155/2013/910321.
"""

from __future__ import annotations

from importlib import metadata
from typing import Final

from msrelapse._params import PAPER

__all__ = [
    "PAPER_BIBTEX",
    "PAPER_REFERENCE",
    "SOFTWARE_BIBTEX",
    "citation",
    "short_citation",
]

_PACKAGE: Final = "msrelapse"

_UNINSTALLED_VERSION: Final = "0+unknown"
"""Version reported when the package is imported without being installed."""

_REPOSITORY: Final = "https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi"

_TITLE: Final = (
    "A mechanistic, stochastic model helps understand multiple sclerosis course and pathogenesis"
)

_AUTHORS: Final = (
    "Bordi, Isabella and Umeton, Renato and Ricigliano, Vito A. G. and "
    "Annibali, Viviana and Mechelli, Rosella and Ristori, Giovanni and "
    "Grassi, Francesca and Salvetti, Marco and Sutera, Alfonso"
)

_SOFTWARE_TITLE: Final = (
    f"{_PACKAGE}: a reference implementation of the Bordi et al. 2013 double well "
    "model of multiple sclerosis"
)


def _software_version() -> str:
    """Return the installed version of the package.

    Returns
    -------
    str
        The version recorded in the installed distribution metadata, or
        ``_UNINSTALLED_VERSION`` when the package is being imported from a
        source tree that was never installed.
    """
    try:
        return metadata.version(_PACKAGE)
    except metadata.PackageNotFoundError:
        return _UNINSTALLED_VERSION


PAPER_REFERENCE: Final = (
    "Bordi I, Umeton R, Ricigliano VAG, Annibali V, Mechelli R, Ristori G, Grassi F, "
    "Salvetti M, Sutera A. A mechanistic, stochastic model helps understand multiple "
    "sclerosis course and pathogenesis. International Journal of Genomics. "
    f"2013;2013:910321. doi:{PAPER.paper_doi.value}."
)
"""str: The article, in the plain text style of a clinical reference list."""

PAPER_BIBTEX: Final = f"""@article{{bordi2013mechanistic,
  author    = {{{_AUTHORS}}},
  title     = {{{_TITLE}}},
  journal   = {{International Journal of Genomics}},
  volume    = {{2013}},
  pages     = {{910321}},
  year      = {{2013}},
  doi       = {{{PAPER.paper_doi.value}}},
  publisher = {{Wiley}}
}}"""
"""str: The article as a BibTeX entry, keyed bordi2013mechanistic."""

_SOFTWARE_NOTE: Final = (
    f"This software reproduces the article, doi {PAPER.paper_doi.value}, "
    "which is the reference to cite."
)
"""What the note field of the entry below says, in one sentence."""

SOFTWARE_BIBTEX: Final = f"""@software{{{_PACKAGE},
  author  = {{Umeton, Renato}},
  title   = {{{_SOFTWARE_TITLE}}},
  year    = {{2026}},
  version = {{{_software_version()}}},
  url     = {{{_REPOSITORY}}},
  note    = {{{_SOFTWARE_NOTE}}}
}}"""
"""str: This package as a BibTeX entry, identified by its repository URL.

The entry carries no DOI. The software is deposited in no archive, and a field
left empty or filled with a stand-in would be read by a reference manager as an
identifier that resolves. The note field carries the article instead, so a
reader who reaches the software entry on its own still reaches the reference to
cite.
"""


def citation() -> str:
    """Return the full citation of the article and of this package.

    Returns
    -------
    str
        ``PAPER_REFERENCE``, a blank line, then the two BibTeX entries.

    Examples
    --------
    >>> citation().startswith("Bordi I")
    True
    """
    return f"{PAPER_REFERENCE}\n\n{PAPER_BIBTEX}\n\n{SOFTWARE_BIBTEX}"


def short_citation() -> str:
    """Return the one line citation carried by the result objects.

    Returns
    -------
    str
        Author, journal, year and the DOI of the article.

    Examples
    --------
    >>> short_citation().startswith("Bordi, Umeton et al.")
    True
    """
    return f"Bordi, Umeton et al., Int J Genomics 2013, doi:{PAPER.paper_doi.value}"
