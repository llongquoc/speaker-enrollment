"""Common selection result type."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Selection:
    """Selected row indices; ``fallback`` counts utterances added only because the
    selection constraint could not be satisfied with ``k`` utterances."""

    indices: np.ndarray
    fallback: int = 0
