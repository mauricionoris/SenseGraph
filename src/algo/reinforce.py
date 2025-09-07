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

import wandb as wb
import time


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
    deg = centralities.set_index('node')[['degree','betweenness','closeness']]
    agg = deg.mean().to_dict()
    feats = []
    for _, r in candidates.reset_index().iterrows():
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
    A = A.minimum(A.T)
    I = sp.eye(N, format='csr')
    A_tilde = (A + I).tocsr()
    degs = np.array(A_tilde.sum(1)).flatten()
    D_inv_sqrt = sp.diags(1.0/np.sqrt(np.maximum(degs, 1e-6)))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    A_hat = torch.from_numpy(A_hat.toarray()).float()
    return X, A_hat


# =============================
# Métricas Funcionais
# =============================
def compute_functional_metrics(env) -> Dict[str, float]:
    """
    Compute coverage, redundancy, and efficiency metrics
    based on the current state of the environment.
    Works whether env.candidates is a DataFrame or just a list.
    """
    N = env.N
    weights = env.w

    # Build mapping: node_id -> index
    if hasattr(env, "candidates"):
        if isinstance(env.candidates, (pd.DataFrame, gpd.GeoDataFrame)) and "node" in env.candidates.columns:
            id2idx = {nid: i for i, nid in enumerate(env.candidates["node"].values)}
        else:
            # candidates is just a list/array → identity mapping
            id2idx = {i: i for i in range(N)}
    else:
        id2idx = {i: i for i in range(N)}

    # Covered set
    covered = set().union(*[env.C[i] for i in env.selected]) if env.selected else set()
    covered_idx = [id2idx[v] for v in covered if v in id2idx]

    # Coverage metrics
    coverage_ratio = len(covered_idx) / N if N > 0 else 0.0
    weighted_coverage = (
        weights.iloc[covered_idx].sum() / weights.sum()
        if len(covered_idx) > 0
        else 0.0
    )

    # Redundancy
    cover_count = np.zeros(N, dtype=int)
    for i in env.selected:
        for v in env.C[i]:
            if v in id2idx:
                cover_count[id2idx[v]] += 1

    k_coverage = np.mean(cover_count >= 2) if N > 0 else 0.0
    redundant_index = cover_count.mean() if N > 0 else 0.0

    # Efficiency
    cost = len(env.selected)
    coverage_per_cost = coverage_ratio / cost if cost > 0 else 0.0

    return {
        "coverage_ratio": coverage_ratio,
        "weighted_coverage": weighted_coverage,
        "k_coverage": k_coverage,
        "redundant_index": redundant_index,
        "cost": cost,
        "coverage_per_cost": coverage_per_cost,
    }

# =============================
# Treinamento REINFORCE
# =============================

def train(env: env.SensorPlacementEnv, A_hat: torch.Tensor, episodes: int = 200, lr: float = 1e-3,
          hidden: int = 64, gamma: float = 1.0, seed: int = 42, run: wb.Run = None):

    start_total = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_feats = env.static.shape[1] + 2  # + sel_flag + rem_gain
    policy = GCNPolicy(in_feats, hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    returns_hist = []
    best = {"ret": -1, "sel": None}

    for ep in trange(episodes, desc="[Reinforce]"):
        ep_start = time.time()
        feats, mask_np = env.reset()
        mask = torch.from_numpy(mask_np).to(device)
        X = torch.from_numpy(feats).float().to(device)
        A = A_hat.to(device)

        logps, rewards, actions = [], [], []
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

            X = torch.from_numpy(feats_next).float().to(device)
            mask = torch.from_numpy(mask_next).to(device)

        # REINFORCE com baseline simples
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
        ep_length = len(rewards)
        returns_hist.append(ep_return)
        if ep_return > best["ret"]:
            best = {"ret": ep_return, "sel": env.selected.copy()}

        ep_time = time.time() - ep_start

        # 🔹 Métricas de treinamento
        train_metrics = {
            "episode": ep,
            "ep_return": ep_return,
            "ep_length": ep_length,
            "loss": float(loss.item()),
            "baseline": float(baseline.item()),
            "best_return_so_far": best["ret"],
            "ep_time": ep_time,
        }

        # 🔹 Métricas funcionais
        func_metrics = compute_functional_metrics(env)

        # 🔹 Loga todas
        run.log({**train_metrics, **func_metrics})

    # 🔹 Log final
    total_time = time.time() - start_total
    final_func_metrics = compute_functional_metrics(env)

    run.log({
        "avg_return": sum(returns_hist)/len(returns_hist),
        "best_return": best["ret"],
        "total_time": total_time,
        **final_func_metrics
    })

    return policy, returns_hist, best, run


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
        (_, _), r, _, _ = env.step(best_i)
        chosen.append(best_i)
        covered |= env.C[best_i]

    return [env.candidates[a] for a in chosen]
