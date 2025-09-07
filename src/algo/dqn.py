import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Optional, Tuple
from tqdm import trange
import wandb as wb

# =============================
# Q-Network (GCN backbone)
# =============================

class DQNGCN(nn.Module):
    """GCN-like MLP over A_hat with masked Q outputs per node."""
    def __init__(self, in_feats: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q_out = nn.Linear(hidden, 1)  # Q por nó
        self.relu = nn.ReLU()

    def forward(
        self,
        X: torch.Tensor,            # [B, N, F] ou [N, F]
        A_hat: torch.Tensor,        # [N, N]
        mask: Optional[torch.Tensor] = None  # [B, N] ou [N]
    ) -> torch.Tensor:
        # dá suporte a batch ou não-batch
        batched = X.dim() == 3
        if not batched:
            # [N, F] -> [N, H]
            H = self.relu(A_hat @ self.fc1(X))
            H = self.relu(A_hat @ self.fc2(H))
            q_vals = self.q_out(H).squeeze(-1)  # [N]
            if mask is not None:
                q_vals = q_vals.masked_fill(mask, -1e9)
            return q_vals
        else:
            B, N, F = X.shape
            # aplica por batch: (A_hat @ X_b)
            H1 = torch.einsum('ij,bjf->bif', A_hat, X)
            H1 = self.relu(self.fc1(H1))
            H2 = torch.einsum('ij,bjf->bif', A_hat, H1)
            H2 = self.relu(self.fc2(H2))
            q_vals = self.q_out(H2).squeeze(-1)  # [B, N]
            if mask is not None:
                q_vals = q_vals.masked_fill(mask, -1e9)
            return q_vals


# =============================
# Replay Buffer
# =============================

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, mask, action, reward, next_state, next_mask, done):
        self.buffer.append((state, mask, action, reward, next_state, next_mask, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, masks, actions, rewards, next_states, next_masks, dones = map(list, zip(*batch))
        return (states, masks, actions, rewards, next_states, next_masks, dones)

    def __len__(self):
        return len(self.buffer)


# =============================
# Training Loop (DQN) — robusto e alinhado ao padrão
# =============================

def train(
    env,
    A_hat: torch.Tensor,
    episodes: int = 500,
    lr: float = 1e-3,
    hidden: int = 64,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 300,
    buffer_capacity: int = 50000,
    batch_size: int = 64,
    target_update: int = 20,
    grad_clip_norm: Optional[float] = 1.0,
    seed: int = 42,
    device: str = "cpu",
    run: Optional[wb.Run] = None,
):
    """Treina DQN com máscara e baseline no env.

    Retorna: (policy_net, returns_hist, best, run)
    - returns_hist: soma de recompensas por episódio
    - best: dict com melhor retorno e seleção correspondente
    """
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    in_feats = env.static.shape[1] + 2  # static + sel_flag + rem_gain
    policy_net = DQNGCN(in_feats, hidden).to(device)
    target_net = DQNGCN(in_feats, hidden).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    returns_hist: List[float] = []
    best = {"ret": -1.0, "sel": None}

    A = A_hat.to(device)

    steps_done = 0
    epsilon_by_frame = lambda frame: epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * frame / epsilon_decay)

    for ep in trange(episodes, desc="[DQN]"):
        feats, mask_np = env.reset()

        # Se baseline já consumiu todo budget, pular episódio (útil quando RL ajusta só alterações marginais)
        if len(env.selected) >= env.k:
            returns_hist.append(0.0)
            if run is not None:
                run.log({"episode": ep, "return": 0.0, "epsilon": epsilon_by_frame(steps_done)})
            continue

        X = torch.from_numpy(feats).float().to(device)
        mask = torch.from_numpy(mask_np).to(device)

        done = False
        rewards = []
        ep_losses: List[float] = []

        while not done:
            epsilon = float(epsilon_by_frame(steps_done))
            steps_done += 1

            with torch.no_grad():
                q_vals = policy_net(X, A, mask)

            if random.random() < epsilon:
                # escolha aleatória de ação válida
                avail_actions = (~mask).nonzero(as_tuple=True)[0]
                if avail_actions.numel() == 0:
                    # nada a fazer — finaliza
                    break
                a = int(np.random.choice(avail_actions.detach().cpu().numpy()))
            else:
                a = int(torch.argmax(q_vals).item())

            (feats_next, mask_next), r, done, _ = env.step(a)
            rewards.append(r)

            X_next = torch.from_numpy(feats_next).float().to(device)
            mask_n = torch.from_numpy(mask_next).to(device)

            replay_buffer.push(X.detach().cpu().numpy(), mask.detach().cpu().numpy(), a, r,
                               X_next.detach().cpu().numpy(), mask_n.detach().cpu().numpy(), done)

            X, mask = X_next, mask_n

            # Treino
            if len(replay_buffer) >= batch_size:
                states, masks, actions, rewards_b, next_states, next_masks, dones = replay_buffer.sample(batch_size)

                states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)   # [B, N, F]
                masks_t = torch.tensor(np.stack(masks), dtype=torch.bool, device=device)       # [B, N]
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device)            # [B]
                rewards_t = torch.tensor(rewards_b, dtype=torch.float32, device=device)        # [B]
                next_states_t = torch.tensor(np.stack(next_states), dtype=torch.float32, device=device)
                next_masks_t = torch.tensor(np.stack(next_masks), dtype=torch.bool, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                # Q(s,a)
                q_values = policy_net(states_t, A, masks_t)                    # [B, N]
                q_values = q_values.gather(1, actions_t.view(-1, 1)).squeeze(1)  # [B]

                # max Q(s', a') com rede alvo, respeitando máscara
                with torch.no_grad():
                    next_q_values = target_net(next_states_t, A, next_masks_t)  # [B, N]
                    max_next_q = next_q_values.max(1)[0]                        # [B]
                    target = rewards_t + gamma * max_next_q * (1.0 - dones_t)

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip_norm)
                optimizer.step()

                ep_losses.append(float(loss.item()))

        ep_return = float(sum(rewards))
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

        if (ep % target_update) == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if run is not None:
            log_data = {
                "episode": ep,
                "return": ep_return,
                "epsilon": float(epsilon_by_frame(steps_done)),
                "avg_loss": float(np.mean(ep_losses)) if ep_losses else 0.0,
                "coverage": float(coverage),
                "selected_count": int(len(env.selected)),
            }
            run.log(log_data)

    return policy_net, returns_hist, best, run
