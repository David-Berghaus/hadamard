from datetime import datetime
from stable_baselines3 import A2C, PPO, DQN 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
import torch
import os

from env import HadamardMlpEnv, HadamardMlpFlippingEnv

if __name__ == "__main__":
    N = 35
    lr = 5e-3
    policy = "MlpPolicy"
    algorithm = "DQN"
    flipping_env = True
    num_envs = 56
    torch_num_threads = 6
    iteration_training_steps = 500000

    #base_dir = ""
    base_dir = "/cephfs/user/s6ddberg/Hadamard/"
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S") #If you want to load a previous model, you have to enter the time stamp here
    base_path = base_dir + "data/" + "/" + str(N) + "/" + algorithm + "/" + time_stamp + "/"
    os.makedirs(base_path, exist_ok=True)
    log_path = base_path + "log/"
    os.makedirs(log_path, exist_ok=True)
    new_logger = configure(log_path, ["tensorboard", "csv"])

    if not flipping_env:
        if num_envs > 1:
            E = SubprocVecEnv([lambda: HadamardMlpEnv(N, dir=base_path) for i in range(num_envs)])
            E = VecMonitor(E)
        else:
            E = HadamardMlpEnv(N, dir=base_path)
            E = Monitor(E)
    else:
        if num_envs > 1:
            E = SubprocVecEnv([lambda: HadamardMlpFlippingEnv(N, dir=base_path) for i in range(num_envs)])
            E = VecMonitor(E)
        else:
            E = HadamardMlpFlippingEnv(N, dir=base_path)
            E = Monitor(E)
    if algorithm == "PPO":
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[256, 256])]) #Custom actor (pi) and value function (vf)
        if os.path.exists(base_path + "model.zip"):
            model = PPO.load(base_path + "model.zip", env=E, learning_rate=lr, verbose=1, policy_kwargs=policy_kwargs)
        else:
            model = PPO(policy, E, learning_rate=lr, verbose=1, policy_kwargs=policy_kwargs)
    elif algorithm == "A2C":
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[256, 256])]) #Custom actor (pi) and value function (vf)
        if os.path.exists(base_path + "model.zip"):
            model = A2C.load(base_path + "model.zip", env=E, learning_rate=lr, verbose=1, policy_kwargs=policy_kwargs)
        else:
            model = A2C(policy, E, learning_rate=lr, verbose=1, policy_kwargs=policy_kwargs)
    elif algorithm == "DQN":
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256]) #No policy for DQN
        if os.path.exists(base_path + "model.zip"):
            print("Loaded model")
            model = DQN.load(base_path + "model.zip", env=E, learning_rate=lr, verbose=1, learning_starts=0, policy_kwargs=policy_kwargs)
        else:
            model = DQN(policy, E, learning_rate=lr, verbose=1, learning_starts=0, policy_kwargs=policy_kwargs)
    model.set_logger(new_logger)
    torch.set_num_threads(torch_num_threads)

    while True:
        model.learn(iteration_training_steps)
        model.save(base_path + "model")
        obs = E.reset()
        #Play one game with the greedy policy
        for i in range(4*N):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = E.step(action)