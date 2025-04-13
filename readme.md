# 基于PPO的倒立摆swing-up控制项目( 2024-2025春强化学习课程作业)
## 作者
[Long Hu](https://github.com/Hu-l)

本项目实现了一个使用**强化学习 PPO 算法**训练的倒立摆控制器，能够实现从倒挂状态起摆至平衡点的全过程控制。

---

## 🚀 项目简介

- **环境**：自定义 Gym 环境，模拟一个由电机驱动的旋转倒立摆系统
- **算法**：Proximal Policy Optimization (PPO)
- **框架**：Python + Gym + PyTorch + Stable-Baselines3
- **可视化**：训练过程曲线、起摆动画（GIF）、轨迹图

---

## 📁 文件结构

```
inverted-pendulum-ppo
├── env.py                # 自定义 Gym 环境定义
├── train.py              # PPO 训练脚本
├── test.py               # 策略测试与轨迹绘图
├── render_to_gif.py      # 保存起摆过程为 GIF
├── plot_tensorboard_logs.py # 绘制 reward/loss 曲线
├── gifs/                 # 保存的 gif 动画
├── ppo_tensorboard/      # TensorBoard 日志
├── models/               # 保存的 PPO 模型
└── README.md             # 项目说明文件
```

---

##  环境说明

- **状态空间**：\[角度（rad）, 角速度（rad/s）\]
- **动作空间**：电压输入 ∈ [−3V, 3V]
- **奖励函数**：惩罚偏离平衡位置、角速度和能耗
- **初始状态**：从下垂位置 \( \pi \) 附近开始

---

##  开始训练

```bash
python train.py
```

- 日志保存在 `ppo_tensorboard/`
- 模型保存为 `ppo_inverted_pendulum_<时间戳>.zip`

启动 TensorBoard 查看训练曲线：
```bash
tensorboard --logdir ppo_tensorboard/
```

---

##  策略可视化

### 1. 保存 swing-up 动画（GIF）：
```bash
python test.py
```
> 生成 `gifs/swingup_ep1.gif`、`trajectory_ep1.png` 等图像

### 2. 绘制训练曲线（reward/loss）：
```bash
python plot_tensorboard_logs.py
```
> 输出 `training_curves.png`

---

## 实验结果示例

- ✅ 最终平均奖励：约 -28.1
- ✅ 每次运行满步长（1000步 5秒），无失败
- ✅ explained variance ≈ 0.92，value 网络拟合良好
- ✅ gif 动画清晰展示起摆和稳定过程

---

## 🔄 改进方向

- 状态空间加入 `sin(α), cos(α)` 增强可学习性
- 尝试使用 TD3/SAC 等离策略算法
- 引入 reward shaping 或 LSTM 策略网络

---

## 环境依赖
env==0.1.0
gym==0.26.2
imageio==2.36.1
matplotlib==3.10.1
numpy==1.23.5
stable_baselines3==2.5.0

```bash
pip install -r requirements.txt
```

---

## 致谢与引用

- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- Chatgpt: https://chatgpt.com/
（chatgpt救我狗命）
