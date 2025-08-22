import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Optional
from tqdm import trange
import wandb as wb

# =============================
# Q-Network (GCN backbone)
# =============================

class DQNGCN(nn.Module):
    def __init__(self, in_feats: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q_out = nn.Linear(hidden, 1)  # Q por nó
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor, mask: Optional[torch.Tensor] = None):
        H = self.relu(A_hat @ self.fc1(X))
        H = self.relu(A_hat @ self.fc2(H))
        q_vals = self.q_out(H).squeeze(-1)  # [N]
        if mask is not None:
            q_vals = q_vals.masked_fill(mask, -1e9)  # -inf para ações inválidas
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
# Training Loop (DQN)
# =============================

def train(env,
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
              seed: int = 42,
              device: str = "cpu",
              run: wb.Run = None):

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    in_feats = env.static.shape[1] + 2
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
    epsilon_by_frame = lambda frame: epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * frame / epsilon_decay)

    for ep in trange(episodes, desc="[DQN]"):
        feats, mask_np = env.reset()
        X = torch.from_numpy(feats).float().to(device)
        mask = torch.from_numpy(mask_np).to(device)

        done = False
        rewards = []

        while not done:
            epsilon = epsilon_by_frame(steps_done)
            steps_done += 1

            q_vals = policy_net(X, A, mask)

            if random.random() < epsilon:
                # escolha aleatória de ação válida
                avail_actions = (~mask).nonzero(as_tuple=True)[0]
                a = int(np.random.choice(avail_actions.cpu().numpy()))
            else:
                a = int(torch.argmax(q_vals).item())

            (feats_next, mask_next), r, done, _ = env.step(a)
            rewards.append(r)

            X_next = torch.from_numpy(feats_next).float().to(device)
            mask_n = torch.from_numpy(mask_next).to(device)

            replay_buffer.push(X.cpu().numpy(), mask.cpu().numpy(), a, r, X_next.cpu().numpy(), mask_n.cpu().numpy(), done)

            X, mask = X_next, mask_n

            # Treino
            if len(replay_buffer) >= batch_size:
                states, masks, actions, rewards_b, next_states, next_masks, dones = replay_buffer.sample(batch_size)

                states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
                masks_t = torch.tensor(np.stack(masks), dtype=torch.bool, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
                rewards_t = torch.tensor(rewards_b, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(np.stack(next_states), dtype=torch.float32, device=device)
                next_masks_t = torch.tensor(np.stack(next_masks), dtype=torch.bool, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                # Q(s,a)
                q_values = policy_net(states_t, A, masks_t)
                q_values = q_values.gather(1, actions_t.view(-1,1)).squeeze(1)

                # max Q(s', a') com rede alvo
                with torch.no_grad():
                    next_q_values = target_net(next_states_t, A, next_masks_t)
                    max_next_q = next_q_values.max(1)[0]
                    target = rewards_t + gamma * max_next_q * (1 - dones_t)

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        ep_return = float(sum(rewards))
        returns_hist.append(ep_return)
        if ep_return > best["ret"]:
            best = {"ret": ep_return, "sel": env.selected.copy()}

        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net, returns_hist, best, run
