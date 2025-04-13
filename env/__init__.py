from gym.envs.registration import register

register(
    id='InvertedPendulum-v5',
    entry_point='env.inverted_pendulum_env:InvertedPendulumEnv',
)
