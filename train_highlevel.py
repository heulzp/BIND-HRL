# train_highlevel.py
# ---------------------------------------------------------------------
# 上层 DQN 训练脚本（充电点选择/是否直达终点）
# ---------------------------------------------------------------------

import os, time, math, random, json
from pathlib import Path
from typing import List, Tuple
from collections import deque, defaultdict
from tqdm import trange

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import Env
from model import QNetwork
from replay_buffer import ReplayMemory, Transition
from options_runner import run_option

# === offline analysis storage (保留) ===
nov_var_list = []          # 每回合TNE新颖度方差
task_uniq_set = set()      # 任务覆盖率（如env提供签名）
tne_loss_curve = []        # 每回合TNE loss
hl_action_hist = []        # 每回合动作直方图

# ======================= 超参数 =======================
# 低层
GAMMA_LOW = 0.95
BATCH_SIZE_LOW = 64
LEARN_EVERY_LOW = 1

# 上层
HIGH_HIDDEN_DIM = 128
HIGH_LR = 1e-3
HIGH_GAMMA = 0.99
HIGH_BATCH_SIZE = 64
HIGH_REPLAY_CAP = 50000
HIGH_TARGET_UPDATE = 20
HIGH_EPS_START = 1.0
HIGH_EPS_END = 0.05
HIGH_EPS_DECAY_STEPS = 50000  # 线性退火步数（上层决策步）

# 子目标执行相关
K_MAX_BASE = 300
K_MAX_SCALE = 1.5
LOW_EPS_FIXED = 0.1

# 外在奖励（沿用原先）
ALPHA_DDIST = 1.0
BETA_STEPS = 0.02
GAMMA_STOP_COST = 5.0
R_GOAL = 200.0
R_FAIL = 200.0

MAX_HIGH_DECISIONS = 6
NUM_EPISODES = 6000
UAV_INDEX = 0

# ======================= TNE/SNE 相关 =======================
USE_SELECTION_BIAS = True   # 选动作前对Q加 κ*g*Norm(SNE) 的偏置；只影响选择，不进学习目标
KAPPA_BIAS = 0.8            # κ
ALPHA_GATE = 5.0            # α
TAU_GATE = 0.6              # τ

INT_COEF = 1.0              # intrinsic reward global scale (eta)
DECAY_T  = 2000             # q-bias linear annealing horizon in episodes (0 disables annealing)

LAMBDA_TNE = 0.40           # 学习阶段内在奖励：TNE权重
LAMBDA_SNE = 0.80           # 学习阶段内在奖励：SNE权重（受 gate 调制）
CLIP_INT_HARD = 6.0         # 内在奖励硬上限

RND_LR = 5e-5               # RND predictor 学习率
RND_EMA = 0.99
RND_CLIP = 6.0

# ======================= 日志 =======================
LOG_TXT_TEMPLATE = "result_run{}.txt"

# ======================= 工具函数 =======================

def bias_qvalues_with_sne(q_plain, nov_sne_all, gate, kappa=0.3, eps=1e-8):
    """Selection bias: add (kappa * gate * normalized SNE) to Q for action selection only.

    Args:
        q_plain (np.ndarray): shape (n_actions,)
        nov_sne_all (np.ndarray): shape (n_actions,), non-negative preferred
        gate (float): [0,1]
        kappa (float): bias strength
    Returns:
        q_sel (np.ndarray): biased Q-values for selection
        q_bias (np.ndarray): bias vector added to q_plain
    """
    import numpy as np
    q_plain = np.asarray(q_plain, dtype=np.float32)
    nov = np.asarray(nov_sne_all, dtype=np.float32)

    # Robust normalize to [0,1]
    nov = np.maximum(nov, 0.0)
    vmin = float(nov.min()) if nov.size else 0.0
    vmax = float(nov.max()) if nov.size else 0.0
    denom = (vmax - vmin) + eps
    nov_hat = (nov - vmin) / denom

    g = float(np.clip(gate, 0.0, 1.0))
    q_bias = (float(kappa) * g) * nov_hat
    q_sel = q_plain + q_bias
    return q_sel, q_bias

def linear_epsilon(step: int, start: float, end: float, total_steps: int) -> float:
    if step >= total_steps:
        return end
    return start + (end - start) * (step / float(total_steps))

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> int:
    very_neg = torch.finfo(q_values.dtype).min
    q_masked = q_values.clone()
    q_masked[mask == 0] = very_neg
    return int(torch.argmax(q_masked, dim=1).item())

# === 轻量 MLP 基类（供 RND 两塔使用） ===
class _MLP(nn.Module):
    def __init__(self, in_dim, hid=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# === TNE: 任务新颖度（两塔 RND） ===
class TNE:
    def __init__(self, input_dim=31, hid=64, lr=1e-4, ema=0.99, clip=3.0, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target = _MLP(input_dim, hid, hid).to(self.device)
        self.predictor = _MLP(input_dim, hid, hid).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = optim.Adam(self.predictor.parameters(), lr=lr)
        self.ema = ema
        self.mean = 0.0
        self.var = 1.0
        self.clip = clip

    def _to_tensor(self, x_np):
        if not isinstance(x_np, np.ndarray):
            x_np = np.asarray(x_np, dtype=np.float32)
        x = torch.from_numpy(x_np.astype(np.float32)).to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x

    @torch.no_grad()
    def novelty(self, x_np):
        x = self._to_tensor(x_np)
        t = self.target(x)
        p = self.predictor(x)
        err = torch.norm(p - t, dim=1)  # L2
        m = self.mean
        s = np.sqrt(self.var) if self.var > 1e-6 else 1.0
        n = ((err.detach().cpu().numpy() - m) / (s + 1e-8)).astype(np.float32)
        n = np.clip(n, 0.0, self.clip)  # 正半轴
        return float(n[0])

    def update_running_stats(self, raw_err_value: float):
        m, v, e = self.mean, self.var, float(raw_err_value)
        new_m = self.ema * m + (1 - self.ema) * e
        new_v = self.ema * v + (1 - self.ema) * (e - new_m) ** 2
        self.mean, self.var = new_m, max(new_v, 1e-6)

    def train_step(self, x_np):
        x = self._to_tensor(x_np)
        t = self.target(x); p = self.predictor(x)
        loss = ((p - t) ** 2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            raw_err = torch.norm(self.predictor(x) - self.target(x), dim=1).mean().item()
            self.update_running_stats(raw_err)
        return float(loss.item())

# === SNE: 子目标新颖度（两塔 RND，与 TNE 平行） ===
class SNE:
    def __init__(self, input_dim=6, hid=64, lr=1e-4, ema=0.99, clip=3.0, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target = _MLP(input_dim, hid, hid).to(self.device)
        self.predictor = _MLP(input_dim, hid, hid).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = optim.Adam(self.predictor.parameters(), lr=lr)
        self.ema = ema
        self.mean = 0.0
        self.var = 1.0
        self.clip = clip

    def _to_tensor(self, x_np):
        if not isinstance(x_np, np.ndarray):
            x_np = np.asarray(x_np, dtype=np.float32)
        x = torch.from_numpy(x_np.astype(np.float32)).to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x

    @torch.no_grad()
    def novelty(self, x_np):
        x = self._to_tensor(x_np)
        t = self.target(x); p = self.predictor(x)
        err = torch.norm(p - t, dim=1)
        m = self.mean
        s = np.sqrt(self.var) if self.var > 1e-6 else 1.0
        n = ((err.detach().cpu().numpy() - m) / (s + 1e-8)).astype(np.float32)
        n = np.clip(n, 0.0, self.clip)
        return float(n[0])

    def update_running_stats(self, raw_err_value: float):
        m, v, e = self.mean, self.var, float(raw_err_value)
        new_m = self.ema * m + (1 - self.ema) * e
        new_v = self.ema * v + (1 - self.ema) * (e - new_m) ** 2
        self.mean, self.var = new_m, max(new_v, 1e-6)

    def train_step(self, x_np):
        x = self._to_tensor(x_np)
        t = self.target(x); p = self.predictor(x)
        loss = ((p - t) ** 2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            raw_err = torch.norm(self.predictor(x) - self.target(x), dim=1).mean().item()
            self.update_running_stats(raw_err)
        return float(loss.item())

# === 门控 & 偏置 ===
def tne_gate(nov_tne_hat: float, alpha=ALPHA_GATE, tau=TAU_GATE) -> float:
    g = 1.0 / (1.0 + math.exp(-alpha * (float(nov_tne_hat) - tau)))
    return float(np.clip(g, 0.0, 1.0))

def linear_anneal(ep: int, T: int) -> float:
    if T is None or T <= 0:
        return 1.0
    return float(max(0.0, 1.0 - ep / float(T)))

def update_ema(prev: float, x: float, ema: float=0.98) -> float:
    return float(ema*float(prev) + (1.0-ema)*float(x))

# ======================= 上层 DQN 包装器 =======================
class HighLevelAgent:
    def __init__(self, state_dim: int, n_actions: int, lr: float):
        self.q_local = QNetwork(state_dim, n_actions, hidden_dim=HIGH_HIDDEN_DIM)
        self.q_target = QNetwork(state_dim, n_actions, hidden_dim=HIGH_HIDDEN_DIM)
        self.q_target.load_state_dict(self.q_local.state_dict())
        self.optim = optim.Adam(self.q_local.parameters(), lr=lr)
        self.crit = nn.SmoothL1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_local.to(self.device); self.q_target.to(self.device)
        self.buffer = ReplayMemory(HIGH_REPLAY_CAP)
        self.learn_steps = 0

    def act(self, state_np: np.ndarray, eps: float) -> int:
        state_t = torch.tensor(state_np[None, :], dtype=torch.float32, device=self.device)
        if random.random() < eps:
            return int(np.random.randint(0, 4))
        with torch.no_grad():
            q = self.q_local(state_t)  # [1,4]
            return int(torch.argmax(q, dim=1).item())

    @torch.no_grad()
    def act_with_bias(self, state_np: np.ndarray, eps: float, q_bias_np: np.ndarray) -> int:
        state_t = torch.tensor(state_np[None, :], dtype=torch.float32, device=self.device)
        if random.random() < eps:
            return int(np.random.randint(0, 4))
        q = self.q_local(state_t).detach().cpu().numpy()[0]  # [4]
        q_biased = q + q_bias_np.astype(np.float32)
        return int(np.argmax(q_biased))

    def push(self, s, a, r, s2, done):
        s_t = torch.tensor(s[None, :], dtype=torch.float32, device=self.device)
        a_t = torch.tensor([[a]], dtype=torch.int64, device=self.device)
        r_t = torch.tensor([[r]], dtype=torch.float32, device=self.device)
        s2_t = torch.tensor(s2[None, :], dtype=torch.float32, device=self.device)
        d_t = torch.tensor([[1.0 if done else 0.0]], dtype=torch.float32, device=self.device)
        transition = Transition(s_t, a_t, r_t, s2_t, d_t)
        self.buffer.push(transition)

    def learn(self, gamma: float, batch_size: int):
        mem_len = len(self.buffer.memory) if hasattr(self.buffer, "memory") else len(self.buffer)
        if mem_len < batch_size:
            return
        transitions = self.buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.cat(batch.state, dim=0)
        if states.dim() == 3 and states.size(1) == 1:
            states = states.squeeze(1)
        next_states = torch.cat(batch.next_state, dim=0)
        if next_states.dim() == 3 and next_states.size(1) == 1:
            next_states = next_states.squeeze(1)
        actions = torch.cat(batch.action, dim=0).view(-1, 1)
        rewards = torch.cat(batch.reward, dim=0).view(-1, 1)
        dones = torch.cat(batch.done, dim=0).view(-1, 1)

        q_all = self.q_local(states)
        q_expected = q_all.gather(1, actions)
        with torch.no_grad():
            q_next = self.q_target(next_states)
            q_next_max = q_next.max(1, keepdim=True)[0]
            q_target = rewards + gamma * q_next_max * (1.0 - dones)
        loss = self.crit(q_expected, q_target)
        self.optim.zero_grad(); loss.backward(); self.optim.step()
        self.learn_steps += 1

    def maybe_update_target(self, period: int):
        if self.learn_steps % period == 0:
            self.q_target.load_state_dict(self.q_local.state_dict())

# ======================= 主训练流程（上层） =======================
def main(log_path="result.txt", run_idx=1):
    global nov_var_list, task_uniq_set, tne_loss_curve, hl_action_hist
    nov_var_list = []; task_uniq_set = set()
    tne_loss_curve = []; hl_action_hist = []

    # 低层网络输入维
    N_STATES_LOW = 42
    N_ACTIONS_LOW = 27
    LR_LOW = 1e-3

    env = Env(n_states=N_STATES_LOW, n_actions=N_ACTIONS_LOW, LEARNING_RATE=LR_LOW)

    # 预估上层状态维
    env.reset()
    s_high = env.get_high_state(UAV_INDEX)
    state_dim_high = int(len(s_high)); n_actions_high = 4
    print(f"[High] state_dim={state_dim_high}, n_actions={n_actions_high}")

    agent = HighLevelAgent(state_dim_high, n_actions_high, lr=HIGH_LR)
    global_step_high = 0

    # === 动态探测 TNE/SNE 输入维并实例化 ===
    goal_xyz = (env.target[0].x, env.target[0].y, env.target[0].z)
    probe_taskvec = env.build_taskvec_25(UAV_INDEX, goal_xyz)      # 作为TNE输入示例
    tne = TNE(input_dim=len(probe_taskvec), hid=64, lr=RND_LR, ema=RND_EMA, clip=RND_CLIP)
    u0 = env.uavs[UAV_INDEX]
    probe_edge = env._edge_features((int(round(u0.x)), int(round(u0.y)), int(round(u0.z))), goal_xyz)  # 6维
    sne = SNE(input_dim=len(probe_edge), hid=64, lr=RND_LR, ema=RND_EMA, clip=RND_CLIP)

    scores_deque = deque(maxlen=100)
    ret_high_deque = deque(maxlen=100)

    t0 = time.time()
    tbar = trange(NUM_EPISODES, desc="HighLevel", unit="ep", dynamic_ncols=True)
    for ep in tbar:
        env.reset()
        decisions = 0
        ep_return_high = 0.0
        ep_steps_low_total = 0
        ep_succ = False
        eps_high = HIGH_EPS_START
        ep_low_score = 0.0

        # === 统计容器 ===
        ep_decisions_cnt = 0
        ep_intr_sum = 0.0
        ep_intr_tne_sum = 0.0
        ep_intr_sne_sum = 0.0
        ep_r_ext_sum = 0.0
        ep_nov_list = []              # TNE nov 历史（用于方差/均值）
        ep_actions = []
        ep_tne_loss_sum = 0.0
        ep_sne_loss_sum = 0.0
        ep_twoleg_list = []
        ep_task_hash_set = set()
        ep_direct_cnt = 0
        ep_charge_counts = {0: 0, 1: 0, 2: 0}

        HL_actions = [0, 0, 0, 0]
        HL_masked_skips = 0
        charge_cnt = 0
        opt_succ = 0; opt_fail = 0; opt_timeout = 0

        step0_stats = {"pick_chg": 0, "pick_goal": 0, "succ_chg": 0, "succ_goal": 0}
        after_chg_stats = defaultdict(lambda: {"pick_chg": 0, "pick_goal": 0, "succ_chg": 0, "succ_goal": 0})

        active_idxs = [i for i in range(len(env.uavs))]
        decisions_per = [0] * len(env.uavs)
        prev_success_chg = [False for _ in range(len(env.uavs))]

        # === 选择偏置诊断 + 门控/新颖度“最后一次/均值” ===
        ep_q_bias_sum = 0.0
        ep_q_bias_cnt = 0
        gate_hist = []
        sne_sel_hist = []
        tne_hist = []

        while (len(active_idxs) > 0) and (decisions < MAX_HIGH_DECISIONS * len(env.uavs)):
            eps_high = linear_epsilon(global_step_high, HIGH_EPS_START, HIGH_EPS_END, HIGH_EPS_DECAY_STEPS)

            for i in list(active_idxs):
                if decisions_per[i] >= MAX_HIGH_DECISIONS:
                    active_idxs.remove(i); continue
                step_idx = int(decisions_per[i])
                if step_idx < 0: step_idx = 0
                if step_idx >= MAX_HIGH_DECISIONS: step_idx = MAX_HIGH_DECISIONS - 1
                s_high = env.get_high_state(i)

                # ---------------- 选动作前：TNE→SNE → (可选) 选择偏置 ----------------
                ux, uy, uz = int(round(env.uavs[i].x)), int(round(env.uavs[i].y)), int(round(env.uavs[i].z))
                cands = [env.chargers[0], env.chargers[1], env.chargers[2], goal_xyz]

                sne_inputs = []
                for k, tgt in enumerate(cands):
                    feats6 = env._edge_features((ux, uy, uz), tgt)
                    sne_inputs.append(np.asarray(feats6, dtype=np.float32))

                with torch.no_grad():
                    q_plain = agent.q_local(torch.tensor(s_high[None,:], dtype=torch.float32, device=agent.device)).cpu().numpy()[0]
                a_tmp = int(np.argmax(q_plain)) if (random.random() >= eps_high) else int(np.random.randint(0,4))
                subgoal_tmp = cands[a_tmp]

                taskvec = env.build_taskvec_25(i, subgoal_tmp)
                nov_tne_hat = tne.novelty(taskvec)
                gate_raw = tne_gate(nov_tne_hat, alpha=ALPHA_GATE, tau=TAU_GATE)
                gate = float(gate_raw)  # gate depends only on task novelty
                # SNE novelty for all candidate actions
                nov_sne_all = [float(sne.novelty(sne_inputs[k])) for k in range(4)]

                if USE_SELECTION_BIAS:
                    anneal = linear_anneal(ep, DECAY_T)
                    kappa_eff = float(np.clip(KAPPA_BIAS * anneal, 0.0, KAPPA_BIAS))
                    q_sel, q_bias = bias_qvalues_with_sne(q_plain, nov_sne_all, gate, kappa=kappa_eff)
                    if random.random() < eps_high:
                        a_high = int(np.random.randint(0,4))
                    else:
                        a_high = int(np.argmax(q_sel))
                    ep_q_bias_sum += float(np.mean(np.abs(q_bias))); ep_q_bias_cnt += 1
                else:
                    a_high = agent.act(s_high, eps_high)

                if a_high in (0, 1, 2, 3):
                    HL_actions[a_high] += 1
                step_idx = decisions_per[i]
                is_pick_chg = (a_high in (0, 1, 2))
                is_pick_goal = (a_high == 3)
                if step_idx == 0:
                    if is_pick_chg:  step0_stats["pick_chg"] += 1
                    if is_pick_goal: step0_stats["pick_goal"] += 1
                else:
                    if prev_success_chg[i] and step_idx <= 2:
                        bucket = after_chg_stats[step_idx]
                        if is_pick_chg:  bucket["pick_chg"] += 1
                        if is_pick_goal: bucket["pick_goal"] += 1

                # 子目标坐标
                if a_high in (0, 1, 2):
                    subgoal = env.chargers[a_high]
                else:
                    subgoal = goal_xyz
                if a_high in (0,1,2):
                    ep_charge_counts[a_high] += 1

                # k_max
                d1 = manhattan((ux, uy, uz), subgoal)
                k_max = int(K_MAX_BASE + K_MAX_SCALE * d1)

                # 执行子目标
                result = run_option(
                    env=env, uav_idx=i, subgoal_xyz=subgoal, k_max=k_max,
                    eps_low=LOW_EPS_FIXED, gamma_low=GAMMA_LOW, batch_size_low=BATCH_SIZE_LOW,
                    learn_every=LEARN_EVERY_LOW,
                )

                if result.get("succ", False):
                    if step_idx == 0:
                        if is_pick_chg:  step0_stats["succ_chg"] += 1
                        if is_pick_goal: step0_stats["succ_goal"] += 1
                    else:
                        if prev_success_chg[i] and step_idx <= 2:
                            bucket = after_chg_stats[step_idx]
                            if is_pick_chg:  bucket["succ_chg"] += 1
                            if is_pick_goal: bucket["succ_goal"] += 1
                prev_success_chg[i] = bool(is_pick_chg and result.get("succ", False))

                decisions += 1
                decisions_per[i] += 1
                global_step_high += 1

                # ---------------- 外在奖励（保持原口径） ----------------
                delta_d = float(result["d_goal_before"] - result["d_goal_after"])
                if result["succ"]:
                    is_charger = any(
                        (abs(subgoal[0] - cx) + abs(subgoal[1] - cy) + abs(subgoal[2] - cz)) == 0
                        for (cx, cy, cz) in getattr(env, "chargers", [])
                    )
                    if is_charger:
                        r_high_ext = ALPHA_DDIST * delta_d - BETA_STEPS * result["steps_low"] - GAMMA_STOP_COST
                        opt_succ += 1; charge_cnt += 1
                        done_i = False
                    else:
                        r_high_ext = R_GOAL + ALPHA_DDIST * delta_d - BETA_STEPS * result["steps_low"]
                        done_i = True; ep_succ = True
                else:
                    r_high_ext = -R_FAIL
                    done_i = True
                    if result["fail_type"] == "timeout": opt_timeout += 1
                    else: opt_fail += 1

                # ---------------- 学习阶段内在奖励（叠加） ----------------
                taskvec_final = env.build_taskvec_25(i, subgoal)
                nov_tne_final = tne.novelty(taskvec_final)   # TNE 标准化+截断
                gate_raw_final = tne_gate(nov_tne_final, alpha=ALPHA_GATE, tau=TAU_GATE)
                gate_final = float(gate_raw_final)  # no phase-conditioned modulation

                sne_in_final = np.asarray(env._edge_features((ux,uy,uz), subgoal), dtype=np.float32)
                nov_sne_final = float(sne.novelty(sne_in_final))  
                raw_int_tne = (LAMBDA_TNE * float(nov_tne_final))
                raw_int_sne = (LAMBDA_SNE * float(gate_final) * float(nov_sne_final))
                base_int = raw_int_tne + raw_int_sne
                anneal_int = 1.0
                tne_scaled = INT_COEF * anneal_int * float(raw_int_tne)
                sne_scaled = INT_COEF * anneal_int * float(raw_int_sne)
                r_int_total = float(np.clip(INT_COEF * anneal_int * float(base_int), 0.0, CLIP_INT_HARD))
                r_high = float(r_high_ext + r_int_total)

                # ---- 日志累积 ----
                ep_decisions_cnt += 1
                ep_intr_sum     += r_int_total
                ep_intr_tne_sum += float(tne_scaled)
                ep_intr_sne_sum += float(sne_scaled)
                ep_r_ext_sum    += float(r_high_ext)
                ep_nov_list.append(float(nov_tne_final))
                if a_high == 3: ep_direct_cnt += 1
                ep_twoleg_list.append(float(taskvec_final[-1]))

                # 新增：门控与新颖度“历史”用于 mean/last
                gate_hist.append(float(gate_final))
                sne_sel_hist.append(float(nov_sne_final))
                tne_hist.append(float(nov_tne_final))

                # ---- 回合累加 ----
                ep_return_high += r_high
                ep_steps_low_total += int(result["steps_low"])
                ep_low_score += float(result.get("total_reward_low", 0.0))

                # ---- 经验写入与学习 ----
                s_high_next = env.get_high_state(i)
                agent.push(s_high, a_high, r_high, s_high_next, done_i)
                agent.learn(HIGH_GAMMA, HIGH_BATCH_SIZE)
                agent.maybe_update_target(HIGH_TARGET_UPDATE)

                # ---- RND 头各自小步更新 ----
                tne_loss = tne.train_step(taskvec_final)
                sne_loss = sne.train_step(sne_in_final)
                ep_tne_loss_sum += float(tne_loss)
                ep_sne_loss_sum += float(sne_loss)

                if done_i and i in active_idxs:
                    active_idxs.remove(i)
            # end for i
        # end while

        # ==== 回合结束：统计 ====
        succ_cnt = crash_cnt = noenergy_cnt = overstep_cnt = 0
        energy_list, margin_list = [], []
        for u in env.uavs:
            if hasattr(u, "cost") and hasattr(u, "bt"):
                energy_list.append(float(u.cost))
                margin_list.append(float(u.bt - u.cost))
            info = getattr(u, "last_info", None)
            if info == 1: succ_cnt += 1
            elif info == 2: crash_cnt += 1
            elif info == 3: noenergy_cnt += 1
            elif info == 5: overstep_cnt += 1

        num_total = len(env.uavs)
        energy_used_avg = (sum(energy_list)/len(energy_list)) if energy_list else 0.0
        residual_margin_avg = (sum(margin_list)/len(margin_list)) if margin_list else 0.0
        wind_level = float(env.WindField[0]) if hasattr(env, "WindField") else 0.0
        lvl = int(getattr(env, "level", 0))

        scores_deque.append(ep_low_score)
        avg_score = float(np.mean(scores_deque)) if len(scores_deque) > 0 else 0.0
        ret_high_deque.append(ep_return_high)
        avg_ret_high = float(np.mean(ret_high_deque)) if len(ret_high_deque) > 0 else 0.0
        ret_per_dec = ep_return_high / max(decisions, 1)

        # === TNE/SNE Episode 指标 ===
        TNE_nov_mean   = (float(np.mean(ep_nov_list)) if ep_nov_list else 0.0)
        TNE_nov_p95    = (float(np.percentile(ep_nov_list, 95)) if ep_nov_list else 0.0)
        TNE_intr_sum   = ep_intr_sum
        TNE_intr_ratio = float(ep_intr_sum / (abs(ep_r_ext_sum) + 1e-6))
        TNE_loss_mean  = ep_tne_loss_sum / max(ep_decisions_cnt, 1)
        SNE_loss_mean  = ep_sne_loss_sum / max(ep_decisions_cnt, 1)
        TNE_task_uniq_ep = len(ep_task_hash_set)
        TNE_direct_rate  = ep_direct_cnt / max(ep_decisions_cnt, 1)
        total_charge = sum(ep_charge_counts.values())
        if total_charge > 0:
            ps = np.array([ep_charge_counts[0], ep_charge_counts[1], ep_charge_counts[2]], dtype=np.float32)
            ps = ps / ps.sum()
            TNE_charge_entropy = float(-np.sum(ps * np.log(ps + 1e-12)))
        else:
            TNE_charge_entropy = 0.0
        TNE_twoleg_mean = (float(np.mean(ep_twoleg_list)) if ep_twoleg_list else 0.0)

        # 选择偏置诊断
        q_bias_mean = (ep_q_bias_sum / max(ep_q_bias_cnt, 1)) if USE_SELECTION_BIAS else 0.0

        # === 新增：门控/新颖度的均值与末次快照 ===
        gate_mean = float(np.mean(gate_hist)) if gate_hist else 0.0
        gate_last = float(gate_hist[-1]) if gate_hist else 0.0
        nov_sne_last = float(sne_sel_hist[-1]) if sne_sel_hist else 0.0
        nov_tne_last = float(tne_hist[-1]) if tne_hist else 0.0

        # === offline 统计 ===
        if len(ep_nov_list) > 1:
            nov_var_list.append(float(np.var(ep_nov_list)))
        task_sig = getattr(env, "get_task_signature", None)
        if callable(task_sig):
            task_uniq_set.add(task_sig())
        if ep_actions:
            from collections import Counter
            hl_action_hist.append(dict(Counter(ep_actions)))
        if ep_decisions_cnt > 0:
            tne_loss_curve.append(ep_tne_loss_sum / ep_decisions_cnt)

        after_compact = {int(k): v for k, v in after_chg_stats.items()
                         if int(k) in (1, 2) and any(v.values())}
        step0_str = json.dumps(step0_stats, ensure_ascii=False, sort_keys=True)
        after_str = json.dumps(after_compact, ensure_ascii=False, sort_keys=True)

        # ---- 写日志 ----
        txt_line = (
            "sum_Episode:{:5d} Episode:{:5d} "
            "HL_decisions:{:2d} HL_picks(chg/goal):{}/{} HL_chg_succ:{:2d} HL_chg_rate:{:4.2f} "
            "HL_masked_skips:{:2d} "
            "HL_return:{:7.2f} HL_AvgReturn:{:7.2f} HL_ret/dec:{:6.2f} "
            "LL_Score:{:7.1f} LL_AvgScore:{:7.2f} "
            # --- TNE/SNE metrics ---
            "TNE_nov_mean:{:4.2f} TNE_nov_p95:{:4.2f} "
            "TNE_intr_sum:{:7.2f} TNE_intr_ratio:{:4.2f} TNE_loss_mean:{:6.4f} SNE_loss_mean:{:6.4f} "
            "TNE_task_uniq_ep:{:4d} TNE_direct_rate:{:4.2f} TNE_charge_entropy:{:4.2f} TNE_twoleg_mean:{:6.3f} "
            # --- 内在奖励拆分与门控/偏置 ---
            "Rint_total:{:6.3f} Rint_tne:{:6.3f} Rint_sne:{:6.3f} Rint_ratio:{:5.3f} "
            "gate_mean/last:{:5.3f}/{:5.3f} nov_tne_last:{:5.3f} nov_sne_last:{:5.3f} q_bias_mean:{:5.3f} "
            # --- 其他 ---
            "eps_high:{:5.3f} eps_low:{:5.3f} low_steps_sum:{:4d} "
            "outcome(s/c/ne/os):{:2d}/{:2d}/{:2d}/{:2d} total:{:2d} "
            "energy_used_avg:{:6.1f} residual_margin_avg:{:6.1f} "
            "level:{:2d} wind:{:4.1f} "
            "seed:{} algo:{} ablation:{}"
        ).format(
            ep + 1, ep + 1,
            decisions, HL_actions[0]+HL_actions[1]+HL_actions[2], HL_actions[3], charge_cnt,
            (charge_cnt / max(HL_actions[0]+HL_actions[1]+HL_actions[2], 1)),
            HL_masked_skips,
            ep_return_high, avg_ret_high, ret_per_dec,
            ep_low_score, avg_score,
            # TNE/SNE metrics
            TNE_nov_mean, TNE_nov_p95,
            TNE_intr_sum, TNE_intr_ratio, TNE_loss_mean, SNE_loss_mean,
            TNE_task_uniq_ep, TNE_direct_rate, TNE_charge_entropy, TNE_twoleg_mean,
            # 内在奖励拆分与门控/偏置
            ep_intr_sum, ep_intr_tne_sum, ep_intr_sne_sum, TNE_intr_ratio,
            gate_mean, gate_last, nov_tne_last, nov_sne_last, q_bias_mean,
            float(linear_epsilon(global_step_high, HIGH_EPS_START, HIGH_EPS_END, HIGH_EPS_DECAY_STEPS)),
            float(LOW_EPS_FIXED), int(ep_steps_low_total),
            succ_cnt, crash_cnt, noenergy_cnt, overstep_cnt, num_total,
            energy_used_avg, residual_margin_avg,
            lvl, wind_level,
            123, "HRL+TNE+SNE", ("bias_on" if USE_SELECTION_BIAS else "bias_off")
        )

        txt_line += f" STEP0:{step0_str} AFTER_CHG:{after_str}"
        elapsed = int(time.time() - t0)
        eps_done = ep + 1
        tbar.set_postfix({
            "lvl": env.level,
            "eps": f"{eps_done}/{NUM_EPISODES}",
            "elapsed": f"{elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}",
        })

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(txt_line + "\n")

    # === 保存 offline 统计 ===
    np.save(f"nov_var_list_run{run_idx}.npy", np.array(nov_var_list, dtype=np.float32))
    with open(f"task_uniq_total_run{run_idx}.txt", "w") as f:
        f.write(str(len(task_uniq_set)))
    np.save(f"tne_loss_curve_run{run_idx}.npy", np.array(tne_loss_curve, dtype=np.float32))
    with open(f"hl_action_hist_run{run_idx}.json", "w", encoding="utf-8") as f:
        json.dump(hl_action_hist, f, ensure_ascii=False)

    print("High-level training done.")

if __name__ == "__main__":
    RUNS = 5
    for run_idx in range(1, RUNS + 1):
        log_path = LOG_TXT_TEMPLATE.format(run_idx)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# ===== RUN {run_idx} START =====\n")
        print(f"=== Running training {run_idx}/{RUNS} ===")
        main(log_path=log_path, run_idx=run_idx)