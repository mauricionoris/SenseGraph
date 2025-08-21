# ----------------------------
# Máxima Cobertura (greedy)
# ----------------------------

from typing import Dict, List, Set
import pandas as pd
import time
import wandb as wb

def greedy_max_coverage(cover_sets: Dict[str, Set[str]],
                        universe_weights: pd.Series,
                        budget_k: int, run: wb.Run) -> List[str]:
    
    """Guloso 1-1/e para função submodular de cobertura ponderada."""
    covered: Set[str] = set()
    chosen: List[str] = []
    remaining = set(cover_sets.keys())
    weights = universe_weights

    start_total = time.time()


    for i in range(budget_k):
        start_iter = time.time()
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
        iter_time = time.time() - start_iter
        run.log({
            "iteration": i,
            "marginal_gain": best_gain,
            "cumulative_coverage": sum(weights.loc[list(covered)]),
            "num_elements_covered": len(covered),
            "time_per_iter": iter_time
        })

    # final
    total_time = time.time() - start_total
    run.log({
        "final_coverage": sum(weights.loc[list(covered)]),
        "coverage_ratio": sum(weights.loc[list(covered)]) / weights.sum(),
        "total_time": total_time
    })

    return chosen, run
