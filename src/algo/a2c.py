import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional, Dict
from tqdm import trange
import wandb as wb
import time

# =============================
# Actor-Critic Model (GCN-based)
# =============================

class ActorCriticGCN(nn.Module):
    """
    GCN compartilhado + cabeça de política (logits por nó) + cabeça de valor (V(s) escalar).
    """
    def __init__(self, in_feats: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.actor_out = nn.Linear(hidden, 1)
        self.critic_out = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor, mask: Optional[torch.Tensor] = None):
        H = self.relu(A_hat @ self.fc1(X))
        H = self.relu(A_hat @ self.fc2(H))

        logits = self.actor_out(H).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        probs = torch.softmax(logits, dim=0)

        # Critic: masked mean pooling
        if mask is not None:
            avail = (~mask).float().unsqueeze(-1)
            denom = torch.clamp(avail.sum(dim=0), min=1.0)
            pooled = (H * avail).sum(dim=0) / denom
        else:
            pooled = H.mean(dim=0)

        V = self.critic_out(pooled).squeeze(-1)
        return probs, logits, V

@torch.no_grad()
def _argmax_action(policy_module, X, A, mask):
    probs, _, _ = policy_module(X, A, mask)
    return int(torch.argmax(probs).item())

# =============================
# Functional Metrics
# =============================

def compute_functional_metrics(env) -> Dict[str, float]:
    N = env.N
    weights = env.w

    # Build node ID -> index mapping robustly
    if hasattr(env, "candidates"):
        if isinstance(env.candidates, (list, np.ndarray)):
            id2idx = {i: i for i in range(N)}
        elif isinstance(env.candidates, (pd.DataFrame, gpd.GeoDataFrame)) and "node" in env.candidates.columns:
            id2idx = {nid: i for i, nid in enumerate(env.candidates["node"].values)}
        else:
            id2idx = {i: i for i in range(N)}
    else:
        id2idx = {i: i for i in range(N)}

    covered = set().union(*[env.C[i] for i in env.selected]) if env.selected else set()
    covered_idx = [id2idx[v] for v in covered if v in id2idx]

    coverage_ratio = len(covered_idx) / N if N > 0 else 0.0
    weighted_coverage = weights.iloc[covered_idx].sum() / weights.sum() if covered_idx else 0.0

    # Redundancy
    cover_count = np.zeros(N, dtype=int)
    for i in env.selected:
        for v in env.C[i]:
            if v in id2idx:
                cover_count[id2idx[v]] += 1

    k_coverage = np.mean(cover_count >= 2) if N > 0 else 0.0
    redundant_index = cover_count.mean() if N > 0 else 0.0

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
# Training Loop (A2C)
# =============================

def train(env,
          A_hat: torch.Tensor,
          episodes: int = 200,
          lr: float = 1e-3,
          hidden: int = 64,
          gamma: float = 1.0,
          entropy_coef: float = 1e-2,
          value_coef: float = 0.5,
          seed: int = 42,
          device: str = "cpu",
          greedy_eval_every: Optional[int] = None,
          run: wb.Run = None):
    
    start_total = time.time()
    torch.manual_seed(seed); np.random.seed(seed)

    in_feats = env.static.shape[1] + 2
    ac = ActorCriticGCN(in_feats, hidden).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=lr)

    returns_hist: List[float] = []
    best = {"ret": -1.0, "sel": None}
    A = A_hat.to(device)

    for ep in trange(episodes, desc="[A2C]"):
        ep_start = time.time()
        feats, mask_np = env.reset()
        X = torch.from_numpy(feats).float().to(device)
        mask = torch.from_numpy(mask_np).to(device)

        policy_losses, value_losses, entropies, rewards = [], [], [], []
        done = False

        while not done:
            probs, logits, V_s = ac(X, A, mask)
            dist = torch.distributions.Categorical(probs)
            a = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(a, device=device))
            entropy = dist.entropy()

            (feats_next, mask_next), r, done, _ = env.step(a)
            rewards.append(r)

            X_next = torch.from_numpy(feats_next).float().to(device)
            mask_n = torch.from_numpy(mask_next).to(device)

            with torch.no_grad():
                _, _, V_next = ac(X_next, A, mask_n)
                target = torch.tensor(r, dtype=torch.float32, device=device) + (0.0 if done else gamma) * V_next

            advantage = target - V_s
            policy_losses.append(-logp * advantage.detach())
            value_losses.append(advantage.pow(2))
            entropies.append(entropy)

            X, mask = X_next, mask_n

        policy_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        entropy_loss = -torch.stack(entropies).sum()
        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        ep_return = float(sum(rewards))
        ep_length = len(rewards)
        returns_hist.append(ep_return)
        if ep_return > best["ret"]:
            best = {"ret": ep_return, "sel": env.selected.copy()}

        # compute functional metrics safely
        func_metrics = compute_functional_metrics(env)

        ep_time = time.time() - ep_start

        # 🔹 Log per episode
        run.log({
            "episode": ep,
            "ep_return": ep_return,
            "ep_length": ep_length,
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy_loss": float(entropy_loss.item()),
            "advantage_mean": float(torch.stack([adv.detach() for adv in value_losses]).mean().item()),
            "best_return_so_far": best["ret"],
            "ep_time": ep_time,
            **func_metrics  # log functional metrics too
        })

    # 🔹 Log final
    total_time = time.time() - start_total
    run.log({
        "avg_return": sum(returns_hist)/len(returns_hist),
        "best_return": best["ret"],
        "total_time": total_time,
    })

    return ac, returns_hist, best, run
