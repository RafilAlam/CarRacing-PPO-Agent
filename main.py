import gymnasium as gym
from stable_baselines3 import PPO
import itertools
import os

###### FILES ######

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

###### MAIN ######

mode = input("Train ? (y/n): ")

env = gym.make('CarRacing-v2', render_mode="human" if mode == 'n' else None)

N = 20
n_steps = 0
n_episodes = 0

t_steps = 10000

if mode == 'y':
    Agent = PPO('CnnPolicy', env, verbose=1, tensorboard_log=logdir)
    for i in range(n_episodes) if n_episodes > 0 else itertools.count():
        observation, _ = env.reset()
        done = False
        while not done:
            Agent.learn(t_steps, reset_num_timesteps=False, tb_log_name="PPO")
            Agent.save(f"{models_dir}/PPO")
else:
    Agent = PPO.load(f"{models_dir}/PPO")
    for i in range(n_episodes) if n_episodes > 0 else itertools.count():
        observation, _ = env.reset()
        done = False
        score_history = []
        while not done:
            action, _ = Agent.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            score_history.append(reward)
        avg_score = sum(score_history) / len(score_history)
        print(f"Episode: {i}, Reward: {avg_score}")