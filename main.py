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
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.4), theta=float(0.6))

model = DDPG(CnnPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, gamma=0.9, clip_norm=0.005,
             nb_train_steps=250, batch_size=64, nb_rollout_steps=100, tensorboard_log="./beamng_tensorboard/",
             actor_lr=1e-6, critic_lr=1e-4)
model.learn(total_timesteps=40000)
model.save("ddpg_beamng")

#del model # remove to demonstrate saving and loading

#model = DDPG.load("ddpg_beamng")

#obs = env.reset()
#while True:
    #action, _states = model.predict(obs)
    #obs, rewards, dones, info = env.step(action)
    #env.render()