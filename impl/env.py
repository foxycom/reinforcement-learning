import gym
from gym import spaces
import numpy as np


class BeamNGenv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ep_length=1000, simulation=None) -> None:
        super(BeamNGenv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(64, 64, 3), dtype=np.uint8)
        self.ep_length = ep_length
        self.current_step = 0
        self.simulation = simulation
        self.prev_position = self.simulation.position()
        self.prev_direction = self.simulation.direction()
        self.direction = self.simulation.direction()
        self.total_reward = 0
        self.reset()

    def step(self, action):
        """

        :param action:
        :return: Observation, reward, done, info
        """
        self._get_next_state(action)
        reward = self._get_reward()
        self.current_step += 1

        velocity = np.linalg.norm(np.array(self.simulation.velocity()))
        done = self.current_step > self.ep_length or self.simulation.done() or velocity > 20.78
        if done:
            print("Episode done. Total reward: ", self.total_reward)
        return self.state, reward, done, {}

    def reset(self):
        self.simulation.reset()
        self.current_step = 0
        self.total_reward = 0
        self._get_next_state()
        self.prev_position = self.simulation.position()
        self.prev_direction = self.simulation.direction()
        self._get_next_state()
        return self.state

    def _get_next_state(self, action=None):
        # Gets the next observation from the simulation
        if action is None:
            action = (np.float32(0.0), np.float32(0.0), np.float32(0.0))
        self.prev_position = self.simulation.position()
        self.state = self.simulation.step(action)

    def _get_reward(self):
        prev_position = self.prev_position[:2]
        current_position = self.simulation.position()[:2]
        dist = np.linalg.norm(np.array(current_position) - np.array(prev_position))
        dir = np.linalg.norm(np.array(self.simulation.direction()) - np.array(self.prev_direction))

        velocity = np.linalg.norm(np.array(self.simulation.velocity()))
        velocity = velocity * 3.6

        delta_dir = np.linalg.norm(np.array(self.simulation.direction()) - self.direction)
        reward = 0

        self.total_reward += reward
        return 1

    def render(self, mode='human', close=False):
        pass