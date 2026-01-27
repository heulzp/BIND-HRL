##############################################################################
# Environment build  (HRL-ready: charging points on rooftops, Morton ordering)
# ---------------------------------------------------------------
# author by younghow  |  upgraded for HRL (charging task)
# email: younghowkg@gmail.com
##############################################################################

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
import torch.optim as optim
from model import QNetwork
from UAV import *
from torch.autograd import Variable
from replay_buffer import ReplayMemory, Transition

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")  # 使用GPU进行训练


class building:
    def __init__(self, x, y, l, w, h):
        self.x = x  # 建筑中心x坐标
        self.y = y  # 建筑中心y坐标
        self.l = l  # 建筑长半值
        self.w = w  # 建筑宽半值
        self.h = h  # 建筑高度


class sn:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# ----------------------------- Morton 3D helpers -----------------------------
def _interleave_10bits(n: int) -> int:
    """Spread the lower 10 bits of n so that there are 2 zeros between each bit."""
    n &= 0x3FF  # 10 bits
    n = (n | (n << 16)) & 0x30000FF
    n = (n | (n << 8)) & 0x300F00F
    n = (n | (n << 4)) & 0x30C30C3
    n = (n | (n << 2)) & 0x9249249
    return n


def morton3D(x: int, y: int, z: int) -> int:
    """Compute a simple 3D Morton (Z-order) code for small grids (<= 1024 per axis)."""
    return (_interleave_10bits(x) << 0) | (_interleave_10bits(y) << 1) | (_interleave_10bits(z) << 2)


# ----------------------------- Env Definition -----------------------------
class Env(object):
    def __init__(self, n_states, n_actions, LEARNING_RATE):
        # 规划空间
        self.len = 100
        self.width = 100
        self.h = 22
        self.map = np.zeros((self.len, self.width, self.h))
        self.WindField = [30, 0]  # 风场(风速,风向角)
        self.uavs: List[UAV] = []  # 无人机对象集合
        self.bds: List[building] = []  # 建筑集合
        self.target: List[sn] = []  # 终点
        self.n_uav = 15  # 训练环境中的无人机个数
        self.v0 = 40  # 无人机可控风速
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        plt.ion()
        self.level = 1  # 训练难度等级(0-10)

        # ---------------- HRL task params (Module A) ----------------
        self.M_CHARGERS = 3                # 每回合充电点数量
        self.CHARGER_MAP_VAL = 3           # 地图标记：充电点
        self.D_MIN = 8                     # 充电点与起/终/彼此的最小格距（曼哈顿）
        self.instant_charge = True         # 第一阶段：瞬时充满（接口保留）
        self.avg_energy_per_step = 1.0     # 上层粗略能耗估计用（可在线更新）
        # === 初始电量策略（按任务能耗比例抽样）===
        # 基础比例区间（level=0 时约 0.8~1.3 倍 E_goal）
        self.BT_ALPHA_BASE_MIN = 0.7
        self.BT_ALPHA_BASE_MAX = 1.12
        # 难度自适应系数：level 越高，下界/上界都略降 → 更常需要充电
        self.BT_ALPHA_MIN_DROP = 0.15  # 下界最多再降 0.15
        self.BT_ALPHA_MAX_DROP = 0.10  # 上界最多再降 0.10
        self.BT_NOISE_STD = 25.0  # 电量抽样的少量噪声
        self.BT_FLOOR = 50.0  # 电量下限，避免极端为 0

        # 运行期容器（每回合 reset 后填充）
        self.chargers: List[Tuple[int, int, int]] = []
        self.charger_ids: List[int] = []   # Morton 稳定编号
        self.charger_mask: List[int] = []  # 1=可选, 0=屏蔽（不可达等）

        # ---------------- 神经网络（下层 DQN，保持不变） ----------------
        self.q_local = QNetwork(n_states, n_actions, hidden_dim=16).to(device)
        self.q_target = QNetwork(n_states, n_actions, hidden_dim=16).to(device)
        self.mse_loss = torch.nn.MSELoss()
        self.optim = optim.Adam(self.q_local.parameters(), lr=LEARNING_RATE)
        self.n_states = n_states
        self.n_actions = n_actions
        self.replay_memory = ReplayMemory(10000)

    # ---------------- 下层 DQN（保持原样） ----------------
    def get_action(self, state, eps, check_eps=True):
        sample = random.random()
        if check_eps == False or sample > eps:
            with torch.no_grad():
                return self.q_local(Variable(state)).data.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device)

    # env.py -> Env.learn()
    def learn(self, gamma, BATCH_SIZE):
        if len(self.replay_memory.memory) < BATCH_SIZE:
            return
        transitions = self.replay_memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # 1) 拼接并整理到标准形状
        states = torch.cat(batch.state, dim=0)  # [B, state_dim] 或 [B,1,state_dim]
        if states.dim() == 3 and states.size(1) == 1:
            states = states.squeeze(1)  # -> [B, state_dim]

        next_states = torch.cat(batch.next_state, dim=0)
        if next_states.dim() == 3 and next_states.size(1) == 1:
            next_states = next_states.squeeze(1)  # -> [B, state_dim]

        actions = torch.cat(batch.action, dim=0).view(-1, 1).long()  # -> [B,1]
        rewards = torch.cat(batch.reward, dim=0).view(-1, 1)  # -> [B,1]
        dones = torch.cat(batch.done, dim=0).view(-1, 1)  # -> [B,1]

        # 2) 计算 Q(s,a) 与目标，全部保持 [B,1]
        q_all = self.q_local(states)  # [B, n_actions]
        Q_expected = q_all.gather(1, actions)  # [B,1]

        with torch.no_grad():
            q_next = self.q_target(next_states)  # [B, n_actions]
            Q_targets_next = q_next.max(1, keepdim=True)[0]  # [B,1]
            Q_targets = rewards + gamma * Q_targets_next * (1.0 - dones)  # [B,1]

        self.q_local.train(True)
        self.optim.zero_grad()
        loss = self.mse_loss(Q_expected, Q_targets)  # 两边都是 [B,1]，不再 unsqueeze
        loss.backward()
        self.optim.step()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, local, target):
        for target_param, param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(param.data)

    # ---------------- 可视化（保持原样） ----------------
    def render(self, flag=0):
        if flag == 1:
            for ob in self.bds:
                x, y, z = ob.x, ob.y, 0
                dx, dy, dz = ob.l, ob.w, ob.h
                xx = np.linspace(x - dx, x + dx, 2)
                yy = np.linspace(y - dy, y + dy, 2)
                zz = np.linspace(z, z + dz, 2)
                xx2, yy2 = np.meshgrid(xx, yy)
                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z + dz))
                yy2, zz2 = np.meshgrid(yy, zz)
                self.ax.plot_surface(np.full_like(yy2, x - dx), yy2, zz2)
                self.ax.plot_surface(np.full_like(yy2, x + dx), yy2, zz2)
                xx2, zz2 = np.meshgrid(xx, zz)
                self.ax.plot_surface(xx2, np.full_like(yy2, y - dy), zz2)
                self.ax.plot_surface(xx2, np.full_like(yy2, y + dy), zz2)
            for t in self.target:
                self.ax.scatter(t.x, t.y, t.z, c="red")
            # 可选：渲染充电点
            for (cx, cy, cz) in self.chargers:
                self.ax.scatter(cx, cy, cz, c="green")

        for uav in self.uavs:
            self.ax.scatter(uav.x, uav.y, uav.z, c="blue")

    # ---------------- 环境一步（记录结局标记） ----------------


    def step(self, action, i):
        reward, done, info = self.uavs[i].update(action)  # info: 1=到达; 2/3/5=失败
        next_state = self.uavs[i].state()

        # 记录该 UAV 的结局（供高层统计）
        if not hasattr(self.uavs[i], "done_flag"):
            self.uavs[i].done_flag = False
        if not hasattr(self.uavs[i], "last_info"):
            self.uavs[i].last_info = None
        if done:
            self.uavs[i].done_flag = True
            self.uavs[i].last_info = info

        # —— 在线校准 avg_energy_per_step（EMA）——
        beta = 0.05  # 0.02~0.10 之间都可以
        nb = float(self.uavs[i].now_bt)

        # 处理异常与过小值，避免把均值拖崩
        if not math.isfinite(nb):
            nb = self.avg_energy_per_step
        nb = max(0.1, nb)  # 设一个温和下限
        nb = min(10.0, nb)  # 也给个上限，防止偶发极端值

        self.avg_energy_per_step = (1.0 - beta) * self.avg_energy_per_step + beta * nb

        return next_state, reward, done, info

    # ---------------- reset：加入充电点逻辑（模块 A 核心） ----------------
    def reset(self):
        # 清空
        self.uavs = []
        self.bds = []
        self.map = np.zeros((self.len, self.width, self.h))
        self.WindField = []
        # 风场
        self.WindField.append(np.random.normal(40, 5))
        self.WindField.append(2 * math.pi * random.random())

        # 随机建筑
        for i in range(random.randint(self.level, 2 * self.level)):
            self.bds.append(
                building(
                    random.randint(10, self.len - 10),
                    random.randint(10, self.width - 10),
                    random.randint(1, 10),
                    random.randint(1, 10),
                    random.randint(9, 13),
                )
            )
            self.map[
                self.bds[i].x - self.bds[i].l : self.bds[i].x + self.bds[i].l,
                self.bds[i].y - self.bds[i].w : self.bds[i].y + self.bds[i].w,
                0 : self.bds[i].h,
            ] = 1

        # 终点
        while True:
            x = random.randint(60, 90)
            y = random.randint(10, 90)
            z = random.randint(3, 15)
            if self.map[x, y, z] == 0:
                break
        self.target = [sn(x, y, z)]
        self.map[x, y, z] = 2

        # 无人机
        for i in range(self.n_uav):
            while True:
                ux = random.randint(15, 30)
                uy = random.randint(10, 90)
                uz = random.randint(3, 7)
                if self.map[ux, uy, uz] == 0:
                    break
            u = UAV(ux, uy, uz, self)
            # 记录起始点坐标（供高层全局特征）
            u.start = (ux, uy, uz)

            # === 按与终点的距离估算 E_goal，并随 level 动态抽样初始电量 bt ===
            gx, gy, gz = self.target[0].x, self.target[0].y, self.target[0].z
            d_goal = abs(ux - gx) + abs(uy - gy) + abs(uz - gz)
            E_goal = float(d_goal) * float(self.avg_energy_per_step)  # 估算到终点能耗

            # level ∈ [0,10] 时的自适应缩放：level 越高，区间整体往下压
            k = min(1.0, float(self.level) / 10.0)
            alpha_min = self.BT_ALPHA_BASE_MIN - self.BT_ALPHA_MIN_DROP * k
            alpha_max = self.BT_ALPHA_BASE_MAX - self.BT_ALPHA_MAX_DROP * k

            # 从区间内均匀抽样，再加一点高斯噪声
            alpha = np.random.uniform(alpha_min, alpha_max)
            noise = np.random.normal(0.0, self.BT_NOISE_STD)

            # 设定该 UAV 的“电量上限/预算”并清零累计能耗
            u.bt = max(self.BT_FLOOR, alpha * E_goal + noise)
            u.cost = 0.0

            # ——初始化结局标记——
            u.done_flag = False
            u.last_info = None
            self.uavs.append(u)

        # === 新增：充电点生成（屋顶） ===
        self._place_chargers_on_rooftops()

        # 更新状态
        self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])
        return self.state

    # ---------------------- 充电点：放置/可达/编号 ----------------------
    def _place_chargers_on_rooftops(self):
        """在建筑顶面生成 M_CHARGERS 个充电点，满足：间距约束、可达性、Morton 稳定编号。"""
        self.chargers = []
        self.charger_ids = []
        self.charger_mask = []

        # 1) 收集所有“屋顶可落足格”（建筑顶部上的第一个空格）
        rooftop_candidates: List[Tuple[int, int, int]] = []
        for b in self.bds:
            z_top = b.h  # 建筑占据 0..h-1，顶面上方第一个空格是 z=h
            if z_top >= self.h:
                continue
            for xx in range(b.x - b.l, b.x + b.l):
                for yy in range(b.y - b.w, b.y + b.w):
                    if 0 <= xx < self.len and 0 <= yy < self.width:
                        if self.map[xx, yy, z_top] == 0:
                            rooftop_candidates.append((xx, yy, z_top))
        random.shuffle(rooftop_candidates)

        # 2) 依次挑选，满足与起点/终点/既有充电点的最小间距
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

        def far_enough(p, others):
            for q in others:
                if manhattan(p, q) < self.D_MIN:
                    return False
            return True

        # 计算起点集（全部 UAV 初始位置）与终点
        starts = [(u.x, u.y, u.z) for u in self.uavs]
        goal = (self.target[0].x, self.target[0].y, self.target[0].z)

        for cand in rooftop_candidates:
            if len(self.chargers) >= self.M_CHARGERS:
                break
            # 与起点/终点/已选充电点保持距离
            if not far_enough(cand, starts):
                continue
            if manhattan(cand, goal) < self.D_MIN:
                continue
            if not far_enough(cand, self.chargers):
                continue
            self.chargers.append(cand)

        # ---- 若不足数量：逐步放宽约束回填，直到恰好 M_CHARGERS 个 ----
        # 第一轮已经是：同时满足 与所有起点/终点/彼此均 >= D_MIN

        # 第二轮：放宽与【起点/终点】的距离限制，只要求与“已选充电点”保持 D_MIN，优先从屋顶候选补齐
        if len(self.chargers) < self.M_CHARGERS:
            for cand in rooftop_candidates:
                if len(self.chargers) >= self.M_CHARGERS:
                    break
                if cand in self.chargers:
                    continue
                # 仅保留与已选充电点之间的最小间距
                if not far_enough(cand, self.chargers):
                    continue
                self.chargers.append(cand)

        # 第三轮：再放宽——不再要求与已选充电点保持 D_MIN，只要是屋顶候选且不重复就收，确保一定凑满
        if len(self.chargers) < self.M_CHARGERS:
            for cand in rooftop_candidates:
                if len(self.chargers) >= self.M_CHARGERS:
                    break
                if cand in self.chargers:
                    continue
                self.chargers.append(cand)

        # 兜底：如果连屋顶候选都没有（极端：没有建筑），从自由空间随机补齐到 M_CHARGERS
        if len(self.chargers) < self.M_CHARGERS:
            rng = np.random.default_rng()
            tries = 2000
            while len(self.chargers) < self.M_CHARGERS and tries > 0:
                tries -= 1
                cx = int(rng.integers(1, self.len - 1))
                cy = int(rng.integers(1, self.width - 1))
                # 高度尽量不要太低，也不要超过地图上限；这里取 1~min(h-1, 13)
                cz = int(rng.integers(1, min(self.h - 1, 13)))
                if self.map[cx, cy, cz] == 0:
                    coord = (cx, cy, cz)
                    if coord not in self.chargers:
                        self.chargers.append(coord)

        # 最终截断到恰好 M_CHARGERS（理论上此时一定 >=M）
        self.chargers = self.chargers[: self.M_CHARGERS]

        # 3) 写入地图标记（3）
        for (cx, cy, cz) in self.chargers:
            self.map[cx, cy, cz] = self.CHARGER_MAP_VAL

        # 4) 可达性检查（以第 0 架 UAV 为代表；需要更精细可在模块 B 做 per-UAV mask）
        self.charger_mask = []
        if len(self.uavs) > 0:
            start = (self.uavs[0].x, self.uavs[0].y, self.uavs[0].z)
        else:
            start = None

        for (cx, cy, cz) in self.chargers:
            ok1 = self._bfs_reachable(start, (cx, cy, cz)) if start is not None else True
            ok2 = self._bfs_reachable((cx, cy, cz), goal)
            self.charger_mask.append(1 if (ok1 and ok2) else 0)

        # 5) Morton 稳定编号与重排（动作 0/1/2 的稳定映射）
        self.charger_ids = []
        for (cx, cy, cz) in self.chargers:
            self.charger_ids.append(morton3D(cx, cy, cz))
        # 依据 Morton 序对 chargers/mask 同步排序
        order = np.argsort(self.charger_ids).tolist()
        self.chargers = [self.chargers[i] for i in order]
        self.charger_mask = [self.charger_mask[i] for i in order]
        self.charger_ids = [self.charger_ids[i] for i in order]

    def _bfs_reachable(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> bool:
        """26 邻域 BFS，可穿越 0/2/3（空气/终点/充电点），不可穿越 1（建筑）。"""
        if start is None or goal is None:
            return False
        if not (0 <= start[0] < self.len and 0 <= start[1] < self.width and 0 <= start[2] < self.h):
            return False
        if not (0 <= goal[0] < self.len and 0 <= goal[1] < self.width and 0 <= goal[2] < self.h):
            return False

        if self.map[goal] == 1:
            return False

        from collections import deque

        visited = np.zeros((self.len, self.width, self.h), dtype=np.uint8)
        q = deque()
        q.append(start)
        visited[start] = 1

        # 26 邻域（与下层动作一致）
        neigh = [(dx, dy, dz) for dx in (-1, 0, 1)
                           for dy in (-1, 0, 1)
                           for dz in (-1, 0, 1) if not (dx == 0 and dy == 0 and dz == 0)]

        while q:
            x, y, z = q.popleft()
            if (x, y, z) == goal:
                return True
            for dx, dy, dz in neigh:
                nx, ny, nz = x + dx, y + dy, z + dz
                if nx < 0 or nx >= self.len or ny < 0 or ny >= self.width or nz < 0 or nz >= self.h:
                    continue
                if visited[nx, ny, nz]:
                    continue
                cell = self.map[nx, ny, nz]
                if cell == 1:  # 建筑不可通行
                    continue
                visited[nx, ny, nz] = 1
                q.append((nx, ny, nz))
        return False

    def _corridor_density(self, start_xyz, goal_xyz, samples: int = 20, radius: int = 1) -> float:
        """
        沿 start->goal 的直线方向采样，统计“走廊”里障碍(=1)占比，返回 [0,1].
        采样简单高效，不做 BFS/最短路，保持通用性。
        """
        import numpy as np
        (x0, y0, z0) = start_xyz
        (x1, y1, z1) = goal_xyz
        x0 = int(round(x0));
        y0 = int(round(y0));
        z0 = int(round(z0))
        x1 = int(round(x1));
        y1 = int(round(y1));
        z1 = int(round(z1))

        # 线性插值 samples 个点
        ts = np.linspace(0.0, 1.0, num=max(2, samples))
        hits = 0
        total = 0
        for t in ts:
            xt = int(round(x0 + t * (x1 - x0)))
            yt = int(round(y0 + t * (y1 - y0)))
            zt = int(round(z0 + t * (z1 - z0)))
            # 在每个采样点附近的 r 立方邻域里统计
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        xx, yy, zz = xt + dx, yt + dy, zt + dz
                        if 0 <= xx < self.len and 0 <= yy < self.width and 0 <= zz < self.h:
                            total += 1
                            if self.map[xx, yy, zz] == 1:  # 建筑
                                hits += 1
        if total == 0:
            return 0.0
        return float(hits) / float(total)

    def _edge_features(self, uav_xyz, tgt_xyz) -> list:
        """
        从 UAV->目标（充电点/终点）提取 6 维直观边特征：
          [ dx_norm, dy_norm, dz_norm, r_edge_norm, corridor_density, wind_align ]
        规范：
          - dx/dy/dz: 用各轴尺寸归一化（[−1,1] 量级）
          - r_edge_norm: 用对角线归一化的欧氏距离
          - corridor_density: 走廊障碍密度（[0,1]）
          - wind_align: 边方向与风向的对齐度（顺风≈+1，逆风≈−1，侧风≈0）
        """
        import math
        (ux, uy, uz) = uav_xyz
        (tx, ty, tz) = tgt_xyz

        dx = float(tx - ux)
        dy = float(ty - uy)
        dz = float(tz - uz)

        # 坐标差归一化（按轴长）
        dx_n = dx / float(max(1, self.len))
        dy_n = dy / float(max(1, self.width))
        dz_n = dz / float(max(1, self.h))

        # 欧氏距离归一化（按对角线）
        diag = math.sqrt(self.len ** 2 + self.width ** 2 + self.h ** 2)
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        r_n = float(r) / float(max(1e-6, diag))

        # 走廊障碍密度
        cd = self._corridor_density((ux, uy, uz), (tx, ty, tz), samples=20, radius=1)

        # 风对齐度（用水平面上的夹角余弦）
        # 环境里风用极角表示：WindField[1] ∈ [0, 2π)   （参考 env.py 的 WindField 用法）
        theta_w = float(self.WindField[1])
        wx, wy = math.cos(theta_w), math.sin(theta_w)  # 风的单位向量（水平）
        # 边的水平单位向量
        denom = math.sqrt(dx * dx + dy * dy)
        if denom < 1e-6:
            wind_align = 0.0
        else:
            ex, ey = dx / denom, dy / denom
            wind_align = float(ex * wx + ey * wy)  # ∈[-1,1]

        return [dx_n, dy_n, dz_n, r_n, float(cd), wind_align]


    # ====== 轻量版：走廊“最小净距”近似（可替换成更精细算法） ======
    def _corridor_min_clearance_proxy(self, start_xyz, goal_xyz) -> float:
        """
        轻量近似：用 (1 - corridor_density) 作为最小净距的单调代理。
        - self._corridor_density ∈ [0,1] 越大表示越“挤/危险”
        - 返回值越大表示越“宽松/安全”
        若后期需要，可替换为沿线采样+局部距离变换的真实最小净距。
        """
        cd = self._corridor_density(start_xyz, goal_xyz, samples=20, radius=1)
        return float(max(0.0, min(1.0, 1.0 - cd)))

    def build_taskvec_25(self, uav_idx: int, choice_xyz: tuple) -> np.ndarray:
        """
        Task Novelty Estimation (TNE) 输入向量（31 维）：
          A. 情景 14 维：
             [start_xyz(3), goal_xyz(3), bt(1), wind_speed_norm(1), wind_dir_norm(1),
              cd_g(1), clr_g(1), cd_c0(1), cd_c1(1), cd_c2(1)]
          B. 新增 6 维（本次加的）：
             [curr_abs_n(3), tgt_abs_n(3)]  # 0-1 归一
          C. 被选边 6 维：
             _edge_features(start->choice) = [dx_n, dy_n, dz_n, r1_n, density1, wind_align1]
          D. 上下文 5 维：
             [final_hop_flag, r2_n, density2, wind_align2, two_leg_cost_proxy]
        说明：
          - 本实现将原来的 battery_norm 替换为 bt（原始预算电量，不归一）。
          - two_leg_cost_proxy 仍为最后一维，兼容现有统计逻辑。
        """
        import math
        import numpy as np

        u = self.uavs[uav_idx]

        # --- 起点/终点 ---
        start_xyz = getattr(u, "start", (int(round(u.x)), int(round(u.y)), int(round(u.z))))
        goal_xyz = (self.target[0].x, self.target[0].y, self.target[0].z)
        sx, sy, sz = start_xyz
        gx, gy, gz = goal_xyz

        # --- 电量：使用 bt 原值（不归一） ---
        bt = float(getattr(u, "bt", 0.0))  # 初始/预算电量

        # --- 风 ---
        wind_speed = float(self.WindField[0]) if len(self.WindField) > 0 else 0.0
        wind_dir = float(self.WindField[1]) if len(self.WindField) > 1 else 0.0
        wind_speed_norm = wind_speed / 100.0  # 你的环境里常用 80/100 皆可，保持现有口径
        wind_dir_norm = wind_dir / math.pi  # 映射到大致 [-1,1]

        # === 情景（14 维，battery_norm -> bt）===
        scenario = [sx, sy, sz, gx, gy, gz, bt, wind_speed_norm, wind_dir_norm]

        chargers = list(getattr(self, "chargers", []))
        cd_g = self._corridor_density(start_xyz, goal_xyz, samples=20, radius=1)
        clr_g = self._corridor_min_clearance_proxy(start_xyz, goal_xyz)

        cd_list = []
        for k in range(3):
            if k < len(chargers):
                ck = chargers[k]
                cd_k = self._corridor_density(start_xyz, ck, samples=20, radius=1)
            else:
                cd_k = 0.0
            cd_list.append(cd_k)

        # 保持你原本“Goal 放 (密度,净距) + C0/C1/C2 仅密度”的构成
        scenario.extend([cd_g, clr_g])  # +2
        scenario.extend(cd_list[:3])  # +3
        # 目前 scenario 共 14 维

        # === 新增 6 维：当前绝对坐标 + 被选目标绝对坐标（0–1 归一）===
        ux, uy, uz = int(round(u.x)), int(round(u.y)), int(round(u.z))

        def _norm_xyz(x, y, z):
            return [
                x / float(max(1, self.len)),
                y / float(max(1, self.width)),
                z / float(max(1, self.h)),
            ]

        curr_abs_n = _norm_xyz(ux, uy, uz)

        tx, ty, tz = int(round(choice_xyz[0])), int(round(choice_xyz[1])), int(round(choice_xyz[2]))
        tgt_abs_n = _norm_xyz(tx, ty, tz)

        scenario.extend(curr_abs_n + tgt_abs_n)  # +6 → 情景块现在共 20 维

        # === 被选边 6 维（start -> choice）===
        dx1, dy1, dz1, r1_n, density1, wind_align1 = self._edge_features(start_xyz, (tx, ty, tz))

        # === 上下文 5 维（choice -> goal）===
        final_hop_flag = 1.0 if (tx == gx and ty == gy and tz == gz) else 0.0
        if final_hop_flag > 0.5:
            r2_n, density2, wind_align2 = 0.0, 0.0, 0.0
            two_leg_cost_proxy = float(r1_n)
        else:
            edge2 = self._edge_features((tx, ty, tz), goal_xyz)
            r2_n, density2, wind_align2 = float(edge2[3]), float(edge2[4]), float(edge2[5])
            # 维持你现有口径的组合代理
            two_leg_cost_proxy = float(r1_n + r2_n + density1 + density2 - (wind_align1 + wind_align2))

        feat = scenario + \
               [dx1, dy1, dz1, r1_n, density1, wind_align1] + \
               [final_hop_flag, r2_n, density2, wind_align2, two_leg_cost_proxy]

        # 维度应为 31
        feat = np.asarray(feat, dtype=np.float32)
        assert feat.shape[0] == 31, f"TNE taskvec must be 31-D (got {feat.shape[0]})"
        return feat

    # ---------------- 充电语义（供上层使用；与 UAV.recharge 对齐） ----------------
    def is_on_charger(self, uav_idx: int) -> bool:
        """UAV 是否处于充电点格（按四舍五入取格索引）。"""
        u = self.uavs[uav_idx]
        ix = int(round(u.x)); iy = int(round(u.y)); iz = int(round(u.z))
        if ix < 0 or ix >= self.len or iy < 0 or iy >= self.width or iz < 0 or iz >= self.h:
            return False
        return self.map[ix, iy, iz] == self.CHARGER_MAP_VAL

    def recharge(self, uav_idx: int) -> bool:
        """若在充电点上则执行充电；语义与 UAV.recharge() 对齐。"""
        if not self.is_on_charger(uav_idx):
            return False
        # 调用 UAV 内部的充电逻辑（通常是清零 cost / 恢复可用电量）
        if hasattr(self.uavs[uav_idx], "recharge"):
            self.uavs[uav_idx].recharge()
            return True
        return False

    # ---------------- 上层“高层状态摘要”接口（供模块 B 使用） ----------------
    def get_high_state(self, uav_idx: int):
        """
        高层 RL 输入（总 45 维）：
        - 全局 6 维：起点(3) + bt(1) + 风速(1) + 风向角(1, 归一到[-1,1])
        - 每条候选边 6 维 × 4 条（C0/C1/C2/Goal）：
            [dx_n, dy_n, dz_n, r_edge_n, corridor_density, wind_align]
        - 追加绝对坐标 15 维：
            [当前UAV绝对坐标(3)] + [C0/C1/C2 绝对坐标(3×3)] + [Goal 绝对坐标(3)]
          注：绝对坐标做了 [0,1] 归一化，作为相对量的补充。
        """
        import math
        u = self.uavs[uav_idx]

        # ---- 全局 6 维 ----
        # 1) 起点坐标（需要在 reset() 里给 u.start 赋值）
        if hasattr(u, "start"):
            xs, ys, zs = u.start
        else:
            # 若历史模型未赋值过 start，则兜底为当前位置信息
            xs, ys, zs = int(round(u.x)), int(round(u.y)), int(round(u.z))

        # 2) 电量上限（不规范化也可；如需规范化，可除以经验上限）
        bt_val = float(getattr(u, "bt", 5000.0))

        # 3) 风速（简单按 80 归一）
        wind_speed = float(self.WindField[0]) if hasattr(self, "WindField") else 0.0
        wind_speed_n = wind_speed / 80.0

        # 4) 风向角（规一到 [-1,1]）
        theta = float(self.WindField[1]) if hasattr(self, "WindField") else 0.0
        wind_dir_norm = (theta / math.pi)  # θ∈[0,2π) → wind_dir_norm ∈ [0,2]
        if wind_dir_norm > 1.0:
            wind_dir_norm = wind_dir_norm - 2.0  # [0,2] → [-1,1]

        global_feats = [
            float(xs), float(ys), float(zs),  # 3
            bt_val,  # 1
            wind_speed_n,  # 1
            float(wind_dir_norm),  # 1
        ]
        # ← 合计 6 维（保持不变）

        # ---- 候选边特征：C0/C1/C2/Goal，每条 6 维 ----
        ux, uy, uz = int(round(u.x)), int(round(u.y)), int(round(u.z))
        u_xyz = (ux, uy, uz)

        # 充电点列表：若不足 3 个，用当前位置补齐（保持定长）
        chargers = list(getattr(self, "chargers", []))
        while len(chargers) < 3:
            chargers.append((ux, uy, uz))  # 占位：与自身重合 → 该边特征为零向量

        # 终点
        gx, gy, gz = self.target[0].x, self.target[0].y, self.target[0].z
        goal_xyz = (gx, gy, gz)

        edge_feats = []
        # C0, C1, C2
        for k in range(3):
            edge_feats += self._edge_features(u_xyz, chargers[k])
        # Goal
        edge_feats += self._edge_features(u_xyz, goal_xyz)

        # ---- NEW: 追加绝对坐标（归一化到 [0,1]）----
        def _norm_xyz(x, y, z):
            # 防止除零：len/width/h 至少为 1
            return [
                float(x) / float(max(1, self.len - 1)),
                float(y) / float(max(1, self.width - 1)),
                float(z) / float(max(1, self.h - 1)),
            ]

        # 当前 UAV 绝对坐标 (3)
        abs_curr = _norm_xyz(ux, uy, uz)

        # 三个充电点绝对坐标 (3×3)
        abs_targets = []
        for k in range(3):
            cx, cy, cz = chargers[k]
            abs_targets += _norm_xyz(cx, cy, cz)

        # 终点绝对坐标 (3)
        abs_targets += _norm_xyz(*goal_xyz)

        # ---- 拼接特征并返回 ----
        feats = np.array(global_feats + edge_feats + abs_curr + abs_targets, dtype=np.float32)

        # 断言维度：30（原有）+ 3（当前绝对坐标）+ 9（三充电点）+ 3（终点） = 45
        assert feats.shape[0] == 45, f"High-state must be 45-D after adding abs coords, got {feats.shape[0]}"
        return feats

    # ---------------- reset_test（保持原样，可用于可视化） ----------------
    def reset_test(self):
        self.uavs = []
        self.bds = []
        self.map = np.zeros((self.len, self.width, self.h))
        self.WindField = []
        self.WindField.append(np.random.normal(40, 5))
        self.WindField.append(2 * math.pi * random.random())

        for i in range(random.randint(self.level, 2 * self.level)):
            self.bds.append(
                building(
                    random.randint(10, self.len - 10),
                    random.randint(10, self.width - 10),
                    random.randint(1, 10),
                    random.randint(1, 10),
                    random.randint(9, 13),
                )
            )
            self.map[
                self.bds[i].x - self.bds[i].l : self.bds[i].x + self.bds[i].l,
                self.bds[i].y - self.bds[i].w : self.bds[i].y + self.bds[i].w,
                0 : self.bds[i].h,
            ] = 1

        while True:
            x = random.randint(60, 90)
            y = random.randint(10, 90)
            z = random.randint(3, 15)
            if self.map[x, y, z] == 0:
                break
        self.target = [sn(x, y, z)]
        self.map[x, y, z] = 2

        u = UAV(20, 20, 3, self)
        u.done_flag = False
        u.last_info = None
        self.uavs.append(u)

        # 放置充电点
        self._place_chargers_on_rooftops()

        self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])
        return self.state


if __name__ == "__main__":
    # 示例：只做冒烟测试（真实训练走 train.py / train_highlevel.py）
    env = Env(n_states=42, n_actions=27, LEARNING_RATE=1e-3)
    env.reset()
    # env.render(flag=1)  # 可选：可视化
    # plt.pause(5)
