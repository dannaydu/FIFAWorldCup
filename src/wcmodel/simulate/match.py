"""Single-match presentation helpers built on a Dixon-Coles score matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd


def top_scorelines(score_matrix: np.ndarray, team_a: str, team_b: str,
                   n: int = 10) -> pd.DataFrame:
    """Most likely exact scorelines from a score matrix."""
    mg = score_matrix.shape[0] - 1
    rows = []
    for x in range(mg + 1):
        for y in range(mg + 1):
            rows.append({"scoreline": f"{team_a} {x}-{y} {team_b}",
                         "prob": float(score_matrix[x, y])})
    return (pd.DataFrame(rows).sort_values("prob", ascending=False)
            .head(n).reset_index(drop=True))


def outcome_probs(score_matrix: np.ndarray) -> dict[str, float]:
    return {
        "team_a_win": float(np.tril(score_matrix, -1).sum()),
        "draw": float(np.trace(score_matrix)),
        "team_b_win": float(np.triu(score_matrix, 1).sum()),
    }


def expected_goals(score_matrix: np.ndarray) -> tuple[float, float]:
    mg = score_matrix.shape[0] - 1
    gx = np.arange(mg + 1)
    ega = float((score_matrix.sum(axis=1) * gx).sum())
    egb = float((score_matrix.sum(axis=0) * gx).sum())
    return ega, egb
