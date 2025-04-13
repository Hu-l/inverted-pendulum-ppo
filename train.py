import os
import gym
import env  # 注册环境（确保 __init__.py 中注册了 InvertedPendulum-v0）
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime

def main():
    # 创建向量化环境（可并行加速）
    env_id = 'InvertedPendulum-v5'
    train_env = gym.make("InvertedPendulum-v5")

    # 创建模型（可换成 MlpPolicy / CnnPolicy）
    model = PPO("MlpPolicy", train_env, verbose=1,
                learning_rate=1e-4,
                n_steps=2048,
                batch_size=64,
                gamma=0.98,
                gae_lambda=0.95,
                clip_range=0.3,
                ent_coef=0.05,
                tensorboard_log="./ppo_tensorboard/")
    
    # 回调函数：定期评估并保存最优模型
    eval_env = gym.make(env_id)
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path="./models/",
                                 log_path="./logs/",
                                 eval_freq=5000,
                                 deterministic=True,
                                 render=False)

    # 训练模型
    model.learn(total_timesteps=200_000, callback=eval_callback)

    # 保存最终模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"ppo_inverted_pendulum_{timestamp}")

if __name__ == "__main__":
    main()
