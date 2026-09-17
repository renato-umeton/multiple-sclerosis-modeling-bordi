# Citing

Cite the article. Cite the software as well if the software did work for you.

## The preferred citation: the article

Bordi I, Umeton R, Ricigliano VAG, Annibali V, Mechelli R, Ristori G, Grassi F,
Salvetti M, Sutera A. A mechanistic, stochastic model helps understand multiple
sclerosis course and pathogenesis. International Journal of Genomics.
2013;2013:910321. doi:10.1155/2013/910321.

The article is open access under CC BY. Isabella Bordi and Renato Umeton
contributed equally and are co-first authors, which the article states in a
section of its own.

## The software

This package is a reference implementation of that article and is versioned
separately. Cite it when the analysis used the code, in addition to the article
and never instead of it. The software is deposited in no archive and carries no
DOI of its own, so the repository URL is its stable reference and its BibTeX
entry carries a note naming the article as the reference to cite.

## From inside a session

Every result the package hands back names where its numbers come from, which is
the point: a number copied out of a session carries its source with it.
Importing the package itself prints nothing.

```python
import msrelapse as ms

ms.cite()                     # prints the article and both BibTeX entries
text = ms.citation()          # the same text, as a string
line = ms.short_citation()    # one line, author, journal, year and DOI
```

Result objects carry the short citation on a `citation` property and repeat it
once in their `repr`, so a number copied out of a terminal session travels with
its source:

```python
import msrelapse as ms

runs = ms.weekly_to_durations(ms.load_synthetic_bordi2013())
fit = ms.fit_durations(runs, state=ms.PAPER.state_no_health.value)
print(fit.citation)
```

The same from the command line:

```bash
msrelapse cite             # the reference and both BibTeX entries
msrelapse cite --bibtex    # only the two BibTeX entries
```

Prefer that output over the block below when you need the exact version
string: `msrelapse cite` fills the `version` field from the installed
distribution metadata.

## CITATION.cff and the cite button

The repository carries a `CITATION.cff` file at its root. GitHub reads it and
shows a **Cite this repository** button in the sidebar of the repository page,
which offers both APA and BibTeX with no further work. The file lists the
software as the thing being cited and the article as the reference the software
implements, so a reader who follows the button reaches both.

Tools that read the file directly include `cffconvert`, which converts it to
BibTeX, APA, RIS and CodeMeta:

```bash
uvx cffconvert --validate
uvx cffconvert -f bibtex
```

## BibTeX

```bibtex
@article{bordi2013mechanistic,
  author    = {Bordi, Isabella and Umeton, Renato and Ricigliano, Vito A. G. and
               Annibali, Viviana and Mechelli, Rosella and Ristori, Giovanni and
               Grassi, Francesca and Salvetti, Marco and Sutera, Alfonso},
  title     = {A mechanistic, stochastic model helps understand multiple sclerosis
               course and pathogenesis},
  journal   = {International Journal of Genomics},
  volume    = {2013},
  pages     = {910321},
  year      = {2013},
  doi       = {10.1155/2013/910321},
  publisher = {Wiley}
}
```

```bibtex
@software{msrelapse,
  author  = {Umeton, Renato},
  title   = {msrelapse: a reference implementation of the Bordi et al. 2013 double
             well model of multiple sclerosis},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi},
  note    = {This software reproduces the article, doi 10.1155/2013/910321, which
             is the reference to cite.}
}
```

The software entry carries no `doi` field, because the software is deposited in
no archive. Its `note` is what sends a reader who meets that entry on its own
back to the article.

## Citing a result, not just the code

If you report a number this package produced, say which record it came from.
A default run reads the synthetic twin, and a figure or a rate from it is a
property of a generator rather than of a cohort of patients. [Data](data.md)
has the sentence to use.
