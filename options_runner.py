# options_runner.py
# ---------------------------------------------------------------------
# Call-and-return 执行器：让低层 DQN 执行到“上层选择的子目标”
# 复用现有 env.get_action / env.step / env.learn，不改低层逻辑
# 子目标“到达”判定：曼哈顿距离 <= 5
# 到达“充电点”子目标时，立即 uav.recharge()（回满电）
# 纯 DQN 不做“路过就充”，不改 env.step
# ---------------------------------------------------------------------

from typing import Tuple, Dict, Any
import torch
from replay_buffer import Transition

def manhattan(a, b):
    """三维曼哈顿距离 |dx|+|dy|+|dz|，a/b 为 (x,y,z)"""
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) + abs(int(a[2]) - int(b[2]))

def run_option(
    env,
    uav_idx: int,
    subgoal_xyz: Tuple[int, int, int],
    k_max: int,
    eps_low: float,
    gamma_low: float,
    batch_size_low: int,
    learn_every: int = 1,
    subgoal_radius: int = 5,   # 与终点一致：L1 <= 5 视为到达
) -> Dict[str, Any]:

    uav = env.uavs[uav_idx]

    # 设置子目标；重置该段起点与步数统计
    uav.set_target(subgoal_xyz[0], subgoal_xyz[1], subgoal_xyz[2],
                   reset_origin=True, reset_steps=True)

    # 记录执行前后到“全局终点”的距离（用于上层 Δd_goal）
    gx, gy, gz = env.target[0].x, env.target[0].y, env.target[0].z
    d_goal_before = manhattan(
        (int(round(uav.x)), int(round(uav.y)), int(round(uav.z))),
        (gx, gy, gz)
    )

    # 子目标是否为“充电点”（按坐标完全匹配判断，因为上层就是从 env.chargers 取的点）
    chargers = list(getattr(env, "chargers", []))
    is_charger_subgoal = any(
        (subgoal_xyz[0] == cx and subgoal_xyz[1] == cy and subgoal_xyz[2] == cz)
        for (cx, cy, cz) in chargers
    )

    succ = False
    fail_type = None
    total_reward_low = 0.0
    steps_low = 0

    # 初始状态张量
    state = uav.state()
    try:
        device = next(env.q_local.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_t = torch.tensor([state], dtype=torch.float32, device=device)

    # 低层循环
    while steps_low < k_max:
        # 选择动作（复用你的低层策略）
        action_t = env.get_action(state_t, eps_low, check_eps=True)

        # 与训练口径一致：无梯度进入 env.step
        next_state, reward, env_done, env_info = env.step(action_t.detach(), uav_idx)

        total_reward_low += float(reward)
        steps_low += 1

        # 判定“到达子目标”：L1 <= subgoal_radius
        curr_xyz = (int(round(uav.x)), int(round(uav.y)), int(round(uav.z)))
        reached_subgoal = (manhattan(curr_xyz, subgoal_xyz) <= subgoal_radius)

        # 写入回放（子目标达到或环境真终止，都作为 done_for_low=True，利于低层学习）
        next_state_t = torch.tensor([next_state], dtype=torch.float32, device=device)
        done_for_low = bool(env_done or reached_subgoal)
        reward_t = torch.tensor([[reward]], dtype=torch.float32, device=device)
        done_t   = torch.tensor([[1.0 if done_for_low else 0.0]], dtype=torch.float32, device=device)

        env.replay_memory.push(Transition(state_t, action_t, reward_t, next_state_t, done_t))

        # 低层学习
        if (steps_low % learn_every) == 0:
            env.learn(gamma_low, batch_size_low)

        state_t = next_state_t

        # 先处理“子目标完成”，再处理环境真正终止
        if reached_subgoal:
            succ = True
            # 若子目标是充电点：到达（L1<=5）即充电
            if is_charger_subgoal:
                try:
                    uav.recharge()
                except AttributeError:
                    try:
                        env.recharge(uav_idx)
                    except Exception:
                        pass
            break

        if env_done:
            # 环境自己终止：1=到达全局终点；2/3/5=失败类型
            if env_info == 1:
                succ = True
            else:
                succ = False
                fail_type = env_info  # 2/3/5 碰撞/没电/超步
            break

    # 若未 done 则视为超时失败
    if steps_low >= k_max and not succ and fail_type is None:
        fail_type = "timeout"

    # 计算 d_goal_after
    d_goal_after = manhattan(
        (int(round(uav.x)), int(round(uav.y)), int(round(uav.z))),
        (gx, gy, gz)
    )

    return {
        "succ": succ,
        "fail_type": fail_type,           # None / 2 / 3 / 5 / "timeout"
        "steps_low": steps_low,
        "total_reward_low": total_reward_low,
        "d_goal_before": d_goal_before,
        "d_goal_after": d_goal_after,
    }
