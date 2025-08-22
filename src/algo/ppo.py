import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional
from tqdm import trange
import wandb as wb

# =============================
# PPO (GCN-based, ações discretas com máscara)
# =============================

class PPOActorCriticGCN(nn.Module):
    """Backbone GCN compartilhado + cabeças ator (logits por nó) e crítico (valor escalar)."""
    def __init__(self, in_feats: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.actor_out = nn.Linear(hidden, 1)
        self.critic_out = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # X: [N, F], A_hat: [N, N], mask: [N] bool
        H = self.relu(A_hat @ self.fc1(X))
        H = self.relu(A_hat @ self.fc2(H))

        logits = self.actor_out(H).squeeze(-1)  # [N]
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        probs = torch.softmax(logits, dim=0)

        # Critic: masked mean pooling sobre nós disponíveis
        if mask is not None:
            avail = (~mask).float().unsqueeze(-1)  # [N,1]
            denom = torch.clamp(avail.sum(dim=0), min=1.0)
            pooled = (H * avail).sum(dim=0) / denom
        else:
            pooled = H.mean(dim=0)
        V = self.critic_out(pooled).squeeze(-1)
        return probs, logits, V


def _compute_gae(rewards: List[float], values: torch.Tensor, gamma: float, lam: float, last_value: float):
    """GAE-Lambda para episódio curto; values: tensor shape [T] no device.
    last_value é V(s_T) (0 se terminal). Retorna (advantages[T], returns[T])."""
    T = len(rewards)
    adv = torch.zeros(T, dtype=torch.float32, device=values.device)
    gae = 0.0
    for t in reversed(range(T)):
        v_next = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def train(env,
              A_hat: torch.Tensor,
              episodes: int = 300,
              lr: float = 3e-4,
              hidden: int = 64,
              gamma: float = 1.0,
              lam: float = 0.95,
              clip_eps: float = 0.2,
              entropy_coef: float = 1e-2,
              value_coef: float = 0.5,
              update_epochs: int = 10,
              minibatch_frac: float = 1.0,  # 1.0 => full batch por episódio
              seed: int = 42,
              device: str = "cpu",
              run: wb.Run = None):
    """
    PPO discreto com máscara e backbone GCN. Atualiza por episódio (horizonte curto k).
    - Usa GAE(γ, λ) e objetivo com clipping.
    - minibatch_frac em (0,1] divide o episódio em minibatches aleatórios.
    Retorna: (modelo, returns_hist, best)
    """
    torch.manual_seed(seed); np.random.seed(seed)

    in_feats = env.static.shape[1] + 2
    policy = PPOActorCriticGCN(in_feats, hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    returns_hist: List[float] = []
    best = {"ret": -1.0, "sel": None}

    A = A_hat.to(device)

    for ep in trange(episodes, desc="[PPO]"):
        # --- Coleta de trajetória (um episódio) ---
        feats, mask_np = env.reset()
        X = torch.from_numpy(feats).float().to(device)
        mask = torch.from_numpy(mask_np).to(device)

        states_X, states_mask = [], []
        actions, logprobs_old, values, rewards = [], [], [], []

        done = False
        while not done:
            probs, logits, V_s = policy(X, A, mask)
            dist = torch.distributions.Categorical(probs)
            a = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(a, device=device))

            (feats_next, mask_next), r, done, _ = env.step(a)

            # armazena
            states_X.append(X)
            states_mask.append(mask)
            actions.append(a)
            logprobs_old.append(logp.detach())
            values.append(V_s.detach())
            rewards.append(float(r))

            X = torch.from_numpy(feats_next).float().to(device)
            mask = torch.from_numpy(mask_next).to(device)

        # valor final para GAE (terminal => 0)
        with torch.no_grad():
            _, _, last_V = policy(X, A, mask)
        values_t = torch.stack(values)  # [T]
        adv, returns = _compute_gae(rewards, values_t, gamma, lam, last_V.detach())

        # normaliza vantagens
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # prepara tensores fixos
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
        logp_old_t = torch.stack(logprobs_old)  # [T]

        T = len(actions)
        mb_size = max(1, int(T * minibatch_frac))

        # --- Otimização por epochs ---
        for _ in range(update_epochs):
            # indices embaralhados
            idx = torch.randperm(T, device=device)
            for start in range(0, T, mb_size):
                sel = idx[start:start + mb_size]

                # reavalia policy/critic para estados selecionados
                new_logps, new_values, entropies = [], [], []
                for j in sel.tolist():
                    probs_j, _, V_j = policy(states_X[j], A, states_mask[j])
                    dist_j = torch.distributions.Categorical(probs_j)
                    new_logps.append(dist_j.log_prob(actions_t[j]))
                    new_values.append(V_j)
                    entropies.append(dist_j.entropy())
                new_logps = torch.stack(new_logps)
                new_values = torch.stack(new_values)
                entropies = torch.stack(entropies)

                ratio = torch.exp(new_logps - logp_old_t[sel])
                adv_sel = adv[sel]
                surr1 = ratio * adv_sel
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_sel
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(new_values, returns[sel])
                entropy_loss = -entropies.mean()

                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        ep_return = float(sum(rewards))
        returns_hist.append(ep_return)
        if ep_return > best["ret"]:
            best = {"ret": ep_return, "sel": env.selected.copy()}

    return policy, returns_hist, best, run
