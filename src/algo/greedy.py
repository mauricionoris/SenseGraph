# ----------------------------
# Máxima Cobertura (greedy)
# ----------------------------

from typing import Dict, List, Set
import pandas as pd


def greedy_max_coverage(cover_sets: Dict[str, Set[str]],
                        universe_weights: pd.Series,
                        budget_k: int) -> List[str]:
    """Guloso 1-1/e para função submodular de cobertura ponderada."""
    covered: Set[str] = set()
    chosen: List[str] = []
    remaining = set(cover_sets.keys())
    weights = universe_weights

    for _ in range(budget_k):
        best_gain = 0.0
        best_c = None
        for c in remaining:
            gain_set = cover_sets[c] - covered
            if not gain_set:
                continue
            gain = weights.loc[list(gain_set)].sum()
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c is None or best_gain <= 0:
            break
        chosen.append(best_c)
        covered |= cover_sets[best_c]
        remaining.remove(best_c)
    return chosen
