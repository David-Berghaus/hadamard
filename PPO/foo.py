from stable_baselines3 import A2C, PPO, DQN 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
import torch

from env import HadamardMlpEnv, HadamardMlpFlippingEnv

tmp_path = "./tmp/sb3_log/"
new_logger = configure(tmp_path, ["tensorboard"])

N = 20
lr = 5e-5
policy = "MlpPolicy"
algorithm = "A2C"
iteration_training_steps = 1000000
# Custom actor (pi) and value function (vf) networks
# of specified layer dimensionalities each with Relu activation function
# Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[128, 64, 4], vf=[128, 64, 4])])

E = HadamardMlpEnv(N)
#E = HadamardMlpFlippingEnv(N)
if algorithm == "PPO":
    model = PPO(policy, E, learning_rate=lr, verbose=1, policy_kwargs=policy_kwargs)
elif algorithm == "A2C":
    model = A2C(policy, E, learning_rate=lr, verbose=1, policy_kwargs=policy_kwargs)
elif algorithm == "DQN":
    model = DQN(policy, E, learning_rate=lr, verbose=1, policy_kwargs=policy_kwargs)
model.set_logger(new_logger)

while True:
    model.learn(iteration_training_steps)