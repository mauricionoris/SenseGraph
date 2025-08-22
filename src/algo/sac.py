import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional
from tqdm import trange
import wandb as wb

# =============================
# Redes SAC (ator + críticos)
# =============================

class DiscreteActorGCN(nn.Module):
    def __init__(self, in_feats: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.logits_out = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor, mask: Optional[torch.Tensor] = None):
        H = self.relu(A_hat @ self.fc1(X))
        H = self.relu(A_hat @ self.fc2(H))
        logits = self.logits_out(H).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        probs = torch.softmax(logits, dim=0)
        log_probs = torch.log_softmax(logits, dim=0)
        return probs, log_probs


class CriticGCN(nn.Module):
    def __init__(self, in_feats: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q_out = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor, mask: Optional[torch.Tensor] = None):
        H = self.relu(A_hat @ self.fc1(X))
        H = self.relu(A_hat @ self.fc2(H))
        q_vals = self.q_out(H).squeeze(-1)
        if mask is not None:
            q_vals = q_vals.masked_fill(mask, -1e9)
        return q_vals


# =============================
# Treinamento SAC Discreto
# =============================

def train(env,
              A_hat: torch.Tensor,
              episodes: int = 500,
              lr: float = 3e-4,
              hidden: int = 64,
              gamma: float = 0.99,
              alpha: float = 0.2,   # entropia fixa (poderia ser aprendido)
              tau: float = 0.005,   # atualização suave
              seed: int = 42,
              device: str = "cpu", 
              run: wb.Run = None):

    torch.manual_seed(seed); np.random.seed(seed)

    in_feats = env.static.shape[1] + 2
    actor = DiscreteActorGCN(in_feats, hidden).to(device)

    critic1 = CriticGCN(in_feats, hidden).to(device)
    critic2 = CriticGCN(in_feats, hidden).to(device)
    target_critic1 = CriticGCN(in_feats, hidden).to(device)
    target_critic2 = CriticGCN(in_feats, hidden).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    critic1_opt = optim.Adam(critic1.parameters(), lr=lr)
    critic2_opt = optim.Adam(critic2.parameters(), lr=lr)

    returns_hist: List[float] = []
    best = {"ret": -1.0, "sel": None}

    A = A_hat.to(device)

    for ep in trange(episodes, desc="[SAC]"):
        feats, mask_np = env.reset()
        X = torch.from_numpy(feats).float().to(device)
        mask = torch.from_numpy(mask_np).to(device)

        done = False
        rewards = []

        while not done:
            # --- Ator ---
            probs, log_probs = actor(X, A, mask)
            dist = torch.distributions.Categorical(probs)
            a = int(dist.sample().item())

            (feats_next, mask_next), r, done, _ = env.step(a)
            rewards.append(r)

            X_next = torch.from_numpy(feats_next).float().to(device)
            mask_n = torch.from_numpy(mask_next).to(device)

            # --- Críticos alvo ---
            with torch.no_grad():
                next_probs, next_log_probs = actor(X_next, A, mask_n)
                q1_next = target_critic1(X_next, A, mask_n)
                q2_next = target_critic2(X_next, A, mask_n)
                min_q_next = torch.min(q1_next, q2_next)
                soft_value = (next_probs * (min_q_next - alpha * next_log_probs)).sum()
                target_q = torch.tensor(r, dtype=torch.float32, device=device) + (0.0 if done else gamma) * soft_value

            # --- Críticos ---
            q1 = critic1(X, A, mask)[a]
            q2 = critic2(X, A, mask)[a]
            loss_c1 = nn.MSELoss()(q1, target_q)
            loss_c2 = nn.MSELoss()(q2, target_q)
            critic1_opt.zero_grad(); loss_c1.backward(); critic1_opt.step()
            critic2_opt.zero_grad(); loss_c2.backward(); critic2_opt.step()

            # --- Ator ---
            probs, log_probs = actor(X, A, mask)
            q1_pi = critic1(X, A, mask)
            q2_pi = critic2(X, A, mask)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (probs * (alpha * log_probs - min_q_pi)).sum()
            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

            # --- Atualização suave dos críticos alvo ---
            with torch.no_grad():
                for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
                    target_param.data.mul_(1 - tau)
                    target_param.data.add_(tau * param.data)
                for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
                    target_param.data.mul_(1 - tau)
                    target_param.data.add_(tau * param.data)

            # avança estado
            X, mask = X_next, mask_n

        ep_return = float(sum(rewards))
        returns_hist.append(ep_return)
        if ep_return > best["ret"]:
            best = {"ret": ep_return, "sel": env.selected.copy()}

    return actor, returns_hist, best, run
