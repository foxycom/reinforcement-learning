import gym
from gym import spaces
import numpy as np


class BeamNGenv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ep_length=100, simulation=None) -> None:
        super(BeamNGenv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,))
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(128, 128, 3), dtype=np.uint8)
        self.ep_length = ep_length
        self.current_step = 0
        self.simulation = simulation

    def step(self, action):
        """

        :param action:
        :return: Observation, reward, done, info
        """
        reward = self._get_reward(action)
        self._get_next_state(action)
        self.current_step += 1
        done = self.current_step > self.ep_length or self.simulation.done()
        return self.state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self._get_next_state()
        return self.state

    def _get_next_state(self, action):
        # Gets the next observation from the simulation
        self.state = self.simulation.step(action)

    def _get_reward(self, action):
        return self.current_step

    def render(self, mode='human', close=False):
        pass