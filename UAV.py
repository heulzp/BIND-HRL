######################################################################
# UAV Class
# ---------------------------------------------------------------
# author by younghow
# email: younghowkg@gmail.com
# --------------------------------------------------------------
# UAV 类描述,对无人机的状态参数进行初始化,
# 包括坐标、目标队列、环境、电量、方向、基础能耗、当前能耗、已经消耗能量、
# 侦测半径、周围障碍情况、坠毁概率、距离目标距离、已走步长等.
# 成员函数能返回自身状态,并根据决策行为对自身状态进行更新.
# ----------------------------------------------------------------
# UAV class description, initialize the state parameters of the UAV,
# including coordinates, target queue, environment, power, direction,
# basic energy consumption, current energy consumption, consumed energy,
# detection radius, surrounding obstacles, crash probability, distance Target distance, steps taken, etc.
# Member functions can return their own state and update their own state according to the decision-making behavior.
#################################################################
import math
import random


class UAV:
    def __init__(self, x, y, z, ev):
        # 初始化无人机坐标位置
        self.x = x
        self.y = y
        self.z = z
        # 初始化无人机目标坐标（默认指向环境终点）
        self.target = [ev.target[0].x, ev.target[0].y, ev.target[0].z]
        self.ev = ev  # 无人机所处环境

        # ------------ 能量与运动学 ------------
        # 注意：本代码以 self.cost（累计能耗）与 self.bt（电量上限）比较来判断“是否没电”
        self.bt = 5000          # 电量上限/预算（不随充电改变）
        self.dir = 0            # 无人机水平运动方向(弧度)
        self.p_bt = 10          # 基础能耗(每步)
        self.now_bt = 4         # 当前状态能耗(每步，会随风向等变化)
        self.cost = 0           # 累计能耗（充电时应把它清零）
        self.detect_r = 5       # 探测范围(格)

        # ------------ 感知与风险 ------------
        self.ob_space = [0]*26  # 近邻 26 栅格占用情况
        self.nearest_distance = 10
        self.dir_ob = None
        self.p_crash = 0

        # ------------ 几何距离与计步 ------------
        self.done = False
        self.distance = abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(self.z - self.target[2])
        self.d_origin = abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(self.z - self.target[2])
        self.step = 0

    # =============== HRL 适配：新增两个小接口 ===============

    def set_target(self, x, y, z, reset_origin: bool = True, reset_steps: bool = True):
        """
        切换子目标（充电点/终点）。为了让低层奖励成形更稳定，默认：
        - reset_origin=True: 以新目标的距离重置 d_origin（避免上一段残余影响当前子任务奖励）
        - reset_steps=True : 重置步数计数
        """
        self.target = [int(x), int(y), int(z)]
        # 重算当前到新目标的距离
        self.distance = abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(self.z - self.target[2])
        if reset_origin:
            self.d_origin = self.distance
        if reset_steps:
            self.step = 0
        # 邻域/风险等在下一次 state()/update() 时自然更新

    def recharge(self):
        """
        充电：将累计能耗 cost 清零（表示补满电）。
        注意：判定没电的条件是 self.cost > self.bt，所以补能应“清空 cost”，而不是修改 bt。
        """
        self.cost = 0

    # =============== 原有逻辑保持不变 ===============

    def cal(self, num):
        # 利用动作值计算运动改变量
        if num == 0:
            return -1
        elif num == 1:
            return 0
        elif num == 2:
            return 1
        else:
            raise NotImplementedError

    def state(self):
        dx = self.target[0] - self.x
        dy = self.target[1] - self.y
        dz = self.target[2] - self.z
        state_grid = [
            self.x, self.y, self.z,
            dx, dy, dz,
            self.target[0], self.target[1], self.target[2],
            self.d_origin, self.step, self.distance,
            self.dir, self.p_crash,
            self.now_bt, self.cost,
        ]
        # 更新临近栅格状态（26 邻域）
        self.ob_space = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    if (
                        self.x + i < 0 or self.x + i >= self.ev.len
                        or self.y + j < 0 or self.y + j >= self.ev.width
                        or self.z + k < 0 or self.z + k >= self.ev.h
                    ):
                        self.ob_space.append(1)
                        state_grid.append(1)
                    else:
                        self.ob_space.append(self.ev.map[self.x + i, self.y + j, self.z + k])
                        state_grid.append(self.ev.map[self.x + i, self.y + j, self.z + k])
        return state_grid

    def update(self, action):
        # 更新无人机状态
        dx, dy, dz = [0, 0, 0]
        temp = action

        # 相关参数（保留原值）
        b = 3       # 撞毁参数
        wt = 0.005  # 目标参数
        wc = 0.07   # 爬升参数
        we = 0      # 能量损耗参数
        c = 0.05    # 风阻能耗参数
        crash = 0   # 坠毁概率惩罚增益
        Ddistance = 0  # 距离变化

        # 计算位移
        dx = self.cal(temp % 3)
        temp = int(temp / 3)
        dy = self.cal(temp % 3)
        temp = int(temp / 3)
        dz = self.cal(temp)

        # 静止惩罚
        if dx == 0 and dy == 0 and dz == 0:
            return -1000, False, False

        # 位置更新
        self.x += dx
        self.y += dy
        self.z += dz

        # 距离变化（接近为正）
        Ddistance = self.distance - (abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(self.z - self.target[2]))
        self.distance = abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(self.z - self.target[2])
        self.step += abs(dx) + abs(dy) + abs(dz)

        # 速度方向（水平面）
        flag = 1
        if abs(dy) != dy:
            flag = -1
        if dx * dx + dy * dy != 0:
            self.dir = math.acos(dx / math.sqrt(dx * dx + dy * dy)) * flag

        # 能耗与相关奖励
        self.cost = self.cost + self.now_bt  # 累计能耗
        a = abs(self.dir - self.ev.WindField[1])
        self.now_bt = self.p_bt + c * self.ev.WindField[0] * (math.sin(a) - math.cos(a))
        # r_e = -we * math.exp((self.cost + self.now_bt) / self.bt)
        r_e = we * (self.p_bt - self.now_bt)

        # 碰撞概率估计（寻找最近障碍）
        r_ob = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue
                if (
                    self.x + i < 0 or self.x + i >= self.ev.len
                    or self.y + j < 0 or self.y + j >= self.ev.width
                    or self.z < 0 or self.z >= self.ev.h
                ):
                    continue
                if self.ev.map[self.x + i, self.y + j, self.z] == 1 and abs(i) + abs(j) < self.nearest_distance:
                    self.nearest_distance = abs(i) + abs(j)
                    flag = 1
                    if abs(j) == -j:
                        flag = -1
                    self.dir_ob = math.acos(i / (i * i + j * j)) * flag

        # 坠毁概率
        if self.nearest_distance >= 4 or self.ev.WindField[0] <= self.ev.v0:
            self.p_crash = 0
        else:
            self.p_crash = math.exp(
                -b * self.nearest_distance * self.ev.v0 * self.ev.v0
                / (0.5 * math.pow(self.ev.WindField[0] * math.cos(abs(self.ev.WindField[1] - self.dir_ob) - self.ev.v0), 2))
            )
            # self.p_crash = 0

        # 奖励分量
        r_climb = -wc * (abs(self.z - self.target[2]))
        # r_target = -wt * (abs(self.x - self.target[0]) + abs(self.y - self.target[1]))   # 奖励函数1
        # r_target = Ddistance                                                               # 奖励函数2
        if self.distance > 1:
            r_target = 2 * (self.d_origin / self.distance) * Ddistance   # 奖励函数3
        else:
            r_target = 2 * (self.d_origin) * Ddistance

        # 总奖励
        r = r_climb + r_target + r_e - crash * self.p_crash

        # 终止判定（保留原语义）
        if (
            self.x <= 0 or self.x >= self.ev.len - 1
            or self.y <= 0 or self.y >= self.ev.width - 1
            or self.z <= 0 or self.z >= self.ev.h - 1
            or self.ev.map[self.x, self.y, self.z] == 1
            or random.random() < self.p_crash
        ):
            return r - 200, True, 2  # 碰撞
        if self.distance <= 5:
            return r + 200, True, 1  # 达成目标（无论是充电点还是终点）
        if self.step >= self.d_origin + 2 * self.ev.h:
            return r - 20, True, 5   # 超步
        if self.cost > self.bt:
            return r - 20, True, 3   # 没电
        return r, False, 4
