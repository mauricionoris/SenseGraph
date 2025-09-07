import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange
import numpy as np
import wandb as wb


class PolicyNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def train(
    env,
    A_hat,
    episodes: int = 20,
    lr: float = 1e-3,
    hidden: int = 128,
    gamma: float = 0.99,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
    run: wb.Run = None,
):
    torch.manual_seed(seed)
    in_dim = env.static.shape[1] + 2   # static + sel_flag + rem_gain
    out_dim = env.N
    policy = PolicyNet(in_dim, hidden, out_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    returns_hist = []
    best = {"ret": -1.0, "sel": None}

    for ep in trange(episodes, desc="[PPO]"):
        obs, mask = env.reset()
        done = False
        log_probs = []
        rewards = []

        # baseline já consome todo o budget → skip
        if len(env.selected) >= env.k:
            returns_hist.append(0.0)
            if run is not None:
                run.log({"episode": ep, "return": 0.0})
            continue

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            probs = policy(obs_t)

            # aplica máscara: zera ações inválidas
            probs = probs.clone()
            probs[mask] = 0.0

            # normaliza e garante simplex válido
            if probs.sum() == 0:
                probs = torch.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()
            probs = torch.clamp(probs, min=1e-8, max=1.0)
            probs = probs / probs.sum()

            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            (obs, mask), reward, done, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)

        # calcular retornos descontados
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # normalização robusta
        if returns.std() > 1e-8:
            adv = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            adv = returns - returns.mean()

        # perda de política
        loss = -torch.stack(log_probs) * adv
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()

        ep_return = float(returns.sum().item())
        returns_hist.append(ep_return)

        # cobertura funcional (peso coberto / total)
        if getattr(env, 'w', None) is not None:
            total_w = float(env.w.sum()) if env.w.sum() > 0 else 1.0
            cov_w = float(env.w[list(env.covered)].sum()) if len(env.covered) else 0.0
            coverage = cov_w / total_w
        else:
            coverage = 0.0

        if ep_return > best["ret"]:
            best = {"ret": ep_return, "sel": env.selected.copy()}

        if run is not None:
            run.log({
                "episode": ep,
                "return": ep_return,
                "avg_adv": float(adv.mean().item()),
                "coverage": float(coverage),
                "selected_count": int(len(env.selected)),
            })

    return policy, returns_hist, best, run
