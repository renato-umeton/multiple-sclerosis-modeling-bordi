from __future__ import annotations

import numpy as np
import pytest

from msrelapse._params import PAPER, Bordi2013


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20130910)


@pytest.fixture
def paper() -> Bordi2013:
    return PAPER
