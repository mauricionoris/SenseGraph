import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional
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

        logits = self.actor_out(H).squeeze(-1)  # [N]
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
    """Escolhe ação determinística (greedy)"""
    probs, _, _ = policy_module(X, A, mask)
    return int(torch.argmax(probs).item())


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
    """
    Treina um A2C on-policy acumulando perdas por episódio.
    """

    start_total = time.time()
    
    torch.manual_seed(seed); np.random.seed(seed)

    in_feats = env.static.shape[1] + 2  # + sel_flag + rem_gain
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

        # avaliação greedy (opcional)
        if greedy_eval_every is not None and (ep + 1) % greedy_eval_every == 0:
            env.reset()
            feats_g, mask_g_np = env._obs()
            Xg = torch.from_numpy(feats_g).float().to(device)
            maskg = torch.from_numpy(mask_g_np).to(device)
            done_g = False; greedy_ret = 0.0
            while not done_g:
                a_g = _argmax_action(ac, Xg, A, maskg)
                (feats_gn, mask_gn), r_g, done_g, _ = env.step(a_g)
                greedy_ret += float(r_g)
                Xg = torch.from_numpy(feats_gn).float().to(device)
                maskg = torch.from_numpy(mask_gn).to(device)
            # poderia salvar greedy_ret para análise

        ep_time = time.time() - ep_start

        # 🔹 Log por episódio
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
        })
                
    
    # 🔹 Log final
    total_time = time.time() - start_total
    run.log({
        "avg_return": sum(returns_hist)/len(returns_hist),
        "best_return": best["ret"],
        "total_time": total_time,
    })


    return ac, returns_hist, best, run
