import gym
from gym import spaces
import numpy as np

class HadamardMlpEnv(gym.Env):
    """Custom Environment for generating binary vectors. This environment is used for the MLP model because it contains a one-hot encoded step count."""
    def __init__(self, N, dir=""):
        super(HadamardMlpEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(2) # 0 or 1
        self.observation_space = spaces.MultiBinary(n=2*4*N) #Flattened v vectors and one-hot encoded step
        self.observation_space_np = np.zeros(2*4*N, dtype=np.int8)
        self.N = N
        self.dir = dir
        self.step_count = 0
        self.lowest_recorded_epsilon = 100000000000
        self.reward_factor = None

    def step(self, action):
        self.observation_space_np[self.step_count] = action
        if self.step_count != 0:
            self.observation_space_np[4*self.N+self.step_count-1] = 0
        self.observation_space_np[4*self.N+self.step_count] = 1
        if self.step_count != 4*self.N-1:
            done = False
            reward = 0 #To Do: check if this is a good reward
        else:
            done = True
            score = -score_state(self.observation_space_np, self.N)
            if self.reward_factor is None:
                self.reward_factor = int(score/10.0) #To Do: check if this is a good reward factor
            reward = self.reward_factor/(self.reward_factor+score)
            if score < self.lowest_recorded_epsilon:
                self.lowest_recorded_epsilon = score
                print(f"New lowest epsilon: {score}")
                with open(self.dir + "/best_scores.txt", "a") as f:
                    f.write(str(score) + "\n")
                if score < 0.1:
                    print(self.observation_space_np)
                    with open(self.dir + "/best_states.txt", "a") as f:
                        f.write(str(self.observation_space_np) + "\n")
        self.step_count += 1
        info = {}
        return self.observation_space_np, reward, done, info
        
    def reset(self):
        self.observation_space_np = np.zeros(2*4*self.N, dtype=np.int8)
        self.step_count = 0
        return self.observation_space_np  # reward, done, info can't be included

class HadamardMlpFlippingEnv(gym.Env):
    """Custom Environment for flipping binary vectors. The agent receives a binary vector and suggests a bit to flip."""
    def __init__(self, N, dir=""):
        super(HadamardMlpFlippingEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(4*N)
        self.observation_space = spaces.MultiBinary(n=4*N) #Flattened v vectors
        self.observation_space_np = np.random.randint(2, size=4*N)
        self.observation_space_np_copy = np.copy(self.observation_space_np)
        self.best_observation_space = np.copy(self.observation_space_np)
        self.N = N
        self.dir = dir
        self.step_count = 0
        self.max_steps = 4*N #To Do: check if this is a good max_steps
        self.lowest_recorded_epsilon = 100000000000
        self.reward_factor = None

    def step(self, action):
        self.observation_space_np[action] = 1-self.observation_space_np[action]
        score = -score_state(self.observation_space_np, self.N)
        done = self.step_count >= self.max_steps
        if self.reward_factor is None:
                self.reward_factor = int(score/10.0) #To Do: check if this is a good reward factor
        reward = self.reward_factor/(self.reward_factor+score)
        # Provide disccounted reward if the game is not done
        if not done:
            reward = reward/self.max_steps
        if score < self.lowest_recorded_epsilon:
            self.lowest_recorded_epsilon = score
            self.best_observation_space = np.copy(self.observation_space_np)
            print(f"New lowest epsilon: {score}")
            with open(self.dir + "/best_scores.txt", "a") as f:
                f.write(str(score) + "\n")
            if score < 0.1:
                print(self.observation_space_np)
                with open(self.dir + "/best_states.txt", "a") as f:
                    f.write(str(self.observation_space_np) + "\n")
        self.step_count += 1
        info = {}
        return self.observation_space_np, reward, done, info
    
    def reset(self):
        # self.observation_space_np = np.random.randint(2, size=4*self.N)
        # self.observation_space_np = np.copy(self.observation_space_np_copy)
        # self.observation_space_np = np.copy(self.observation_space_np)
        self.observation_space_np = np.copy(self.best_observation_space) #This is dangerous because the agent might just flip the same bit over and over again
        self.step_count = 0
        return self.observation_space_np  # reward, done, info can't be included

def score_state(state, N):
    v1, v2, v3, v4 = get_vecs(state, N)
    w1, w2, w3, w4 = np.fft.fft(v1), np.fft.fft(v2), np.fft.fft(v3), np.fft.fft(v4) #To Do: check if this uses the correct normalization
    res = np.square(np.abs(w1)) + np.square(np.abs(w2)) + np.square(np.abs(w3)) + np.square(np.abs(w4))
    return -np.sum(np.abs(res-4*N))

def transform_zeros_to_neg_ones(vec):
    return np.where(vec == 0, -1, vec)

def get_vecs(state, N):
    v1, v2, v3, v4 = transform_zeros_to_neg_ones(state[0:N]), transform_zeros_to_neg_ones(state[N:2*N]), transform_zeros_to_neg_ones(state[2*N:3*N]), transform_zeros_to_neg_ones(state[3*N:4*N])
    return v1, v2, v3, v4