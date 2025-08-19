from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


from common import util



# =============================
# Ambiente de RL
# =============================

class SensorPlacementEnv:
    """Ambiente tabular-mascarado de seleção sequencial sem reposição.
    - observation(): devolve (features dinâmicas N x F_dyn)
    - mask: ações inválidas (já selecionados)
    - reward: ganho marginal de cobertura ponderada
    """
    def __init__(self, candidates: gpd.GeoDataFrame, universe: gpd.GeoDataFrame,
                 cover_sets: Dict[int, Set[int]], budget_k: int,
                 static_feats: np.ndarray, universe_weights: np.ndarray):
        self.cand = candidates.reset_index(drop=True)
        self.U = universe.reset_index(drop=True)
        self.C = list(cover_sets.values())
        self.candidates = list(cover_sets.keys()) 
        self.k = budget_k
        self.static = static_feats.astype(np.float32)  # shape [N, F_static]
        self.w = universe_weights.astype(np.float32)   # shape [|U|]
        self.N = len(self.cand)
        self.reset()

    def reset(self):
        self.t = 0
        self.selected: List[int] = []
        self.covered: Set[int] = set()
        return self._obs()

    def _remaining_gain(self) -> np.ndarray:
        # vetor N: soma dos pesos ainda não cobertos que cada candidato cobriria
        gain = np.zeros(self.N, dtype=np.float32)


        for i in range(self.N):
        #for i, key in enumerate(self.C.keys()):

            if i in self.selected:
                continue
            new = self.C[i] - self.covered
            if new:
                gain[i] = self.w[list(new)].sum()
        return gain

    def _obs(self) -> np.ndarray:
        sel_flag = np.zeros((self.N, 1), dtype=np.float32)
        if self.selected:
            sel_flag[self.selected] = 1.0
        rem_gain = self._remaining_gain().reshape(self.N, 1)
        # normaliza rem_gain
        rg = rem_gain
        if rg.max() > 0:
            rg = rg / (rg.max() + 1e-6)
        dyn = np.concatenate([sel_flag, rg], axis=1)
        feats = np.concatenate([self.static, dyn], axis=1)
        mask = np.zeros(self.N, dtype=np.bool_)
        if self.selected:
            mask[self.selected] = True
        return feats, mask

    def step(self, action: int):
        assert action not in self.selected
        new_set = self.C[action] - self.covered
        reward = float(self.w[list(new_set)].sum()) if new_set else 0.0
        self.covered |= self.C[action]
        self.selected.append(action)
        self.t += 1
        done = (self.t >= self.k)
        obs = self._obs()
        return obs, reward, done, {}
