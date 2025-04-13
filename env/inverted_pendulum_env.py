import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class InvertedPendulumEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(InvertedPendulumEnv, self).__init__()

        # 环境参数
        self.dt = 0.005  # 时间步长
        self.max_steps = 1000  # 每回合最大步数

        # 物理参数
        self.m = 0.055   # kg，摆杆质量
        self.g = 9.81    # m/s²，重力加速度
        self.l = 0.042   # m，质心到转轴的距离
        self.J = 1.91e-4 # kg·m²，转动惯量
        self.b = 3e-6    # Nm·s/rad，摩擦系数
        self.K = 0.0536  # Nm/A，电机转矩常数
        self.R = 9.5     # Ω，电阻

        # 状态空间 [角度, 角速度]
        high = np.array([np.pi, 15*np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # 动作空间：电压输入 [-3V, 3V]
        self.action_space = spaces.Box(low=np.array([-3.0]), high=np.array([3.0]), dtype=np.float32)

        self.reset()

    def reset(self):
        # 随机初始状态：小角度偏移
       
        angle =np.random.uniform(low=np.pi - 0.01, high=np.pi + 0.01)
        angular_velocity = np.random.uniform(low=-0.05, high=0.05)
        self.state = np.array([angle, angular_velocity], dtype=np.float32)
        self.steps = 0
        self.reward_scale = 0.001 + 0.009 * (self.steps / self.max_steps)
        # 前期 reward 乘 0.001，后期逐步恢复或乘更小因子，让策略能继续优化细节。
        return self.state

    def step(self, action):
        alpha, alpha_dot = self.state
        u = float(np.clip(action, self.action_space.low[0], self.action_space.high[0]))
        # u = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])

        # 计算加速度 
        torque_gravity = self.m * self.g * self.l * np.sin(alpha)
        torque_damping = self.b * alpha_dot
        torque_back_emf = (self.K ** 2 / self.R) * alpha_dot
        torque_input = (self.K / self.R) * u

        alpha_ddot = (1 / self.J) * (torque_gravity - torque_damping - torque_back_emf + torque_input)

        # 更新状态（欧拉积分）
       
        alpha_dot += self.dt * alpha_ddot 
         # 限制角速度在 [-15pi, 15pi]，避免角速度爆炸
        # max_speed = 15 * np.pi
        # alpha_dot = np.clip(alpha_dot, -max_speed, max_speed)
        alpha += self.dt * alpha_dot

        # 限制角度在 [-pi, pi]，避免角度爆炸
        alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([alpha, alpha_dot], dtype=np.float32)
        self.steps += 1

        # 奖励函数：惩罚偏离直立位置（alpha=0），控制输入也惩罚
        Q = np.array([[5.0, 0.0],
              [0.0, 0.1]])
        R = 1.0

        s = np.array([alpha, alpha_dot])
        reward = - (s.T @ Q @ s + R * u ** 2)
        reward = reward * self.reward_scale


        # 终止条件：角度过大或步数超过限制
        done = self.steps >= self.max_steps or not np.isfinite(alpha)  # 防止 NaN

        return self.state, reward, done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')
            self.ax.set_xlim(-0.08, 0.08)
            self.ax.set_ylim(-0.08, 0.08)
            self.pendulum_line, = self.ax.plot([], [], lw=4, color='royalblue')
            self.ax.set_title("Rotary Inverted Pendulum")

        alpha, _ = self.state

        # 以转轴为圆心，绘制摆杆位置
        origin = np.array([0.0, 0.0])
        end = origin + np.array([
            self.l * np.sin(alpha),
            self.l * np.cos(alpha)
        ])

        self.pendulum_line.set_data([origin[0], end[0]], [origin[1], end[1]])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.pause(0.001)

    def close(self):
        pass
