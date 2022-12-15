from env import HadamardMlpEnv, HadamardLstmEnv

from stable_baselines3 import A2C, PPO, DQN, HER 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

tmp_path = "./tmp/sb3_log/"
new_logger = configure(tmp_path, ["tensorboard"])

E = HadamardMlpEnv(20)
model = PPO('MlpPolicy', E, verbose=1)
#model = DQN('MlpPolicy', E, verbose=1)
#model = A2C('MlpPolicy', E, verbose=1)
model.set_logger(new_logger)
model.learn(5000000)
mean_reward, std_reward = evaluate_policy(model, E, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")