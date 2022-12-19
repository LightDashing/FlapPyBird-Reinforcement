from flappygym import FlappyEnv
from stable_baselines3 import PPO

env = FlappyEnv()
model = PPO.load("model.zip", env)
vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    if done:
        vec_env.close()
        break
