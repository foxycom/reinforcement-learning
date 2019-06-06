import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from impl.env import BeamNGenv
from simulation import Simulation


simulation = Simulation()
#env = gym.make('MountainCarContinuous-v0')
env = DummyVecEnv([lambda: BeamNGenv(simulation=simulation)])

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(CnnPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=400000)
model.save("ddpg_beamng")

#del model # remove to demonstrate saving and loading

#model = DDPG.load("ddpg_beamng")

#obs = env.reset()
#while True:
    #action, _states = model.predict(obs)
    #obs, rewards, dones, info = env.step(action)
    #env.render()