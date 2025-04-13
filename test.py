import gym
import env  # 注册环境
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import imageio
import os

def test_model(model_path="ppo_inverted_pendulum_20250413_173442.zip", episodes=5, output_dir="gifs"):
    # 加载环境 & 模型
    os.makedirs(output_dir, exist_ok=True)
    env = gym.make("InvertedPendulum-v5")
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        angles = []
        angular_velocities = []
        actions = []
        frames = []
        # env.render()  # ✅ 每一步显示摆动动画
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            angles.append(obs[0])
            angular_velocities.append(obs[1])
            actions.append(action[0])
            # env.render()  # ✅ 每一步显示摆动动画
            
            if len(angles) % 4 == 0:
                env.l=0.042  # m，质心到转轴的距离
                alpha, _ = env.state
                origin = np.array([0.0, 0.0])
                end = origin + np.array([
                    env.l * np.sin(alpha),
                    env.l * np.cos(alpha)
                ])
                

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_aspect('equal')
                ax.set_xlim(-0.08, 0.08)
                ax.set_ylim(-0.08, 0.08)
                ax.plot([origin[0], end[0]], [origin[1], end[1]], lw=4, color='royalblue')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Frame {len(frames)}")
                plt.tight_layout()

                # 抓图并存为 numpy RGB 帧
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)
                plt.close(fig)

        print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}")
        # 保存为 gif
        gif_path = os.path.join(output_dir, f"swingup_ep{ep+1}.gif")
        imageio.mimsave(gif_path, frames, duration=0.02)
        print(f"✅ Episode {ep + 1} 动画已保存为：{gif_path}")


        # 绘图（轨迹可视化）
        t = np.arange(len(angles)) *  0.005  # 时间轴
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(t, angles)
        plt.ylabel("Angle (rad)")
        plt.title(f"Episode {ep + 1} Trajectory")

        plt.subplot(3, 1, 2)
        plt.plot(t, angular_velocities)
        plt.ylabel("Angular Velocity (rad/s)")

        plt.subplot(3, 1, 3)
        plt.plot(t, actions)
        plt.ylabel("Action (V)")
        plt.xlabel("Time (s)")

        plt.tight_layout()
        plt.savefig(f"trajectory_ep{ep + 1}.png", dpi=300)
        print("yes")
        # plt.show()

    env.close()


if __name__ == "__main__":
    test_model()
