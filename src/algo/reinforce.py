
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

from shapely.geometry import LineString

from tqdm import trange, tqdm

from common import util
from algo import env

from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp


device = 'cuda' if torch.cuda.is_available() else 'cpu'
  


# =============================
# GCN Policy (sem PyG)
# =============================

class GCNPolicy(nn.Module):
    def __init__(self, in_feats: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)  # logit por nó
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # X: [N, F], A_hat: [N, N]
        H = self.relu(A_hat @ self.fc1(X))
        H = self.relu(A_hat @ self.fc2(H))
        logits = self.out(H).squeeze(-1)  # [N]
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        probs = torch.softmax(logits, dim=0)
        return probs, logits

# =============================
# Features de nó e adjacência
# =============================

def build_node_features(candidates: gpd.GeoDataFrame,
                        centralities: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # mapeia centralidades para cand por nearest (usando índice do gdf_nodes original como proxy)
    # aqui simplificamos: como 'centralities' veio dos nós da rede, e 'candidates' contém parte deles,
    # para linhas com origem em POIs usaremos média das centralidades
    deg = centralities.set_index('node')[['degree','betweenness','closeness']]
    # valores agregados (média) para fallback
    agg = deg.mean().to_dict()
    feats = []
    for _, r in candidates.reset_index().iterrows():
        # se o candidato tem coluna 'node', use as centralidades reais
        if 'node' in candidates.columns and not pd.isna(r.get('node', np.nan)) and r['index'] in deg.index:
            dv = deg.loc[r['index']].values
        else:
            dv = np.array([agg['degree'], agg['betweenness'], agg['closeness']])
        src = r.get('source', 'centrality')
        is_poi = 1.0 if src not in ('centrality',) else 0.0
        feats.append(np.concatenate([dv, np.array([is_poi], dtype=np.float32)]))
    X = np.vstack(feats).astype(np.float32)
    # normalização min-max por coluna
    X = (X - X.min(0, keepdims=True)) / (X.max(0, keepdims=True) - X.min(0, keepdims=True) + 1e-6)
    # adjacência por proximidade geográfica (kNN) + simétrica
    N = len(candidates)
    coords = util.ensure_crs_utm(candidates)
    xy = np.vstack([coords.geometry.x.values, coords.geometry.y.values]).T

    A = kneighbors_graph(xy, n_neighbors=min(10, max(2, N-1)), mode='connectivity', include_self=True)
    A = A.minimum(A.T)  # tornar simétrica
    # A_hat = D^{-1/2} (A + I) D^{-1/2}
    I = sp.eye(N, format='csr')
    A_tilde = (A + I).tocsr()
    degs = np.array(A_tilde.sum(1)).flatten()
    D_inv_sqrt = sp.diags(1.0/np.sqrt(np.maximum(degs, 1e-6)))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    A_hat = torch.from_numpy(A_hat.toarray()).float()
    return X, A_hat

# =============================
# Treinamento REINFORCE
# =============================

def train(env: env.SensorPlacementEnv, A_hat: torch.Tensor, episodes: int = 200, lr: float = 1e-3,
          hidden: int = 64, gamma: float = 1.0, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_feats = env.static.shape[1] + 2  # + sel_flag + rem_gain
    policy = GCNPolicy(in_feats, hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    returns_hist = []
    best = {"ret": -1, "sel": None}

    for ep in trange(episodes, desc="[Reinforce]"):
        feats, mask_np = env.reset()
        mask = torch.from_numpy(mask_np).to(device)
        X = torch.from_numpy(feats).float().to(device)
        A = A_hat.to(device)

        logps = []
        rewards = []
        actions = []

        done = False
        while not done:
            probs, logits = policy(X, A, mask)
            dist = torch.distributions.Categorical(probs)
            a = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(a, device=device))

            (feats_next, mask_next), r, done, _ = env.step(a)
            rewards.append(r)
            logps.append(logp)
            actions.append(a)

            # prepara próxima observação
            X = torch.from_numpy(feats_next).float().to(device)
            mask = torch.from_numpy(mask_next).to(device)

        # REINFORCE com baseline simples (média da recompensa por passo)
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.append(R)
        returns = list(reversed(returns))
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        baseline = ret_tensor.mean()
        loss = -(torch.stack(logps) * (ret_tensor - baseline)).sum()
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        ep_return = float(sum(rewards))
        returns_hist.append(ep_return)
        if ep_return > best["ret"]:
            best = {"ret": ep_return, "sel": env.selected.copy()}

    return policy, returns_hist, best

# =============================
# Export
# =============================

def greedy_env_placement(env: env.SensorPlacementEnv) -> List[int]:

    """
    Versão gulosa 1-1/e para o ambiente SensorPlacementEnv.
    Seleciona sequencialmente os sensores com maior ganho marginal de cobertura ponderada.
    """
    env.reset()
    chosen = []
    covered = set()

    for _ in range(env.k):
        best_gain = 0.0
        best_i = None
        for i in range(env.N):
            if i in chosen:
                continue
            new = env.C[i] - covered
            if not new:
                continue
            gain = env.w[list(new)].sum()
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        # aplica a ação no ambiente
        (_, _), r, _, _ = env.step(best_i)
        chosen.append(best_i)
        covered |= env.C[best_i]

    return [env.candidates[a] for a in chosen]