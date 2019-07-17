# Original author: Roma Sokolkov
# Edited by Antonin Raffin
import os
import warnings

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from config import INPUT_DIM, MIN_STEERING, MAX_STEERING, JERK_REWARD_WEIGHT, MAX_STEERING_DIFF
from donkey_gym.core.donkey_proc import DonkeyUnityProcess
from .donkey_sim import DonkeyUnitySimContoller


class DonkeyVAEEnv(gym.Env):
    """
    Gym interface for DonkeyCar with support for using
    a VAE encoded observation instead of raw pixels if needed.

    :param level: (int) DonkeyEnv level
    :param frame_skip: (int) frame skip, also called action repeat
    :param vae: (VAEController object)
    :param const_throttle: (float) If set, the car only controls steering
    :param min_throttle: (float)
    :param max_throttle: (float)
    :param max_cte_error: (float) Max cross track error before ending an episode
    :param n_command_history: (int) number of previous commmands to keep
        it will be concatenated with the vae latent vector
    :param n_stack: (int) Number of frames to stack (used in teleop mode only)
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self, level=0, frame_skip=2, vae=None, const_throttle=None,
                 min_throttle=0.2, max_throttle=0.5,
                 max_cte_error=3.0, n_command_history=0,
                 n_stack=1):
        self.vae = vae
        self.z_size = None
        if vae is not None:
            self.z_size = vae.z_size

        self.const_throttle = const_throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.np_random = None

        # Save last n commands (throttle + steering)
        self.n_commands = 2
        self.command_history = np.zeros((1, self.n_commands * n_command_history))
        self.n_command_history = n_command_history
        # Custom frame-stack
        self.n_stack = n_stack
        self.stacked_obs = None

        # Check for env variable
        exe_path = os.environ.get('DONKEY_SIM_PATH')
        if exe_path is None:
            # You must start the executable on your own
            print("Missing DONKEY_SIM_PATH environment var. We assume the unity env is already started")

        # TCP port for communicating with simulation
        port = int(os.environ.get('DONKEY_SIM_PORT', 9091))

        self.unity_process = None
        if exe_path is not None:
            print("Starting DonkeyGym env")
            # Start Unity simulation subprocess if needed
            self.unity_process = DonkeyUnityProcess()
            headless = os.environ.get('DONKEY_SIM_HEADLESS', False) == '1'
            self.unity_process.start(exe_path, headless=headless, port=port)

        # start simulation com
        self.viewer = DonkeyUnitySimContoller(level=level, port=port, max_cte_error=max_cte_error)

        if const_throttle is not None:
            # steering only
            self.action_space = spaces.Box(low=np.array([-MAX_STEERING]),
                                           high=np.array([MAX_STEERING]),
                                           dtype=np.float32)
        else:
            # steering + throttle, action space must be symmetric
            self.action_space = spaces.Box(low=np.array([-MAX_STEERING, -1]),
                                           high=np.array([MAX_STEERING, 1]), dtype=np.float32)

        if vae is None:
            # Using pixels as input
            if n_command_history > 0:
                warnings.warn("n_command_history not supported for images"
                              "(it will not be concatenated with the input)")
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=INPUT_DIM, dtype=np.uint8)
        else:
            # z latent vector from the VAE (encoded input image)
            self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, self.z_size + self.n_commands * n_command_history),
                                                dtype=np.float32)

        # Frame-stacking with teleoperation
        if n_stack > 1:
            obs_space = self.observation_space
            low = np.repeat(obs_space.low, self.n_stack, axis=-1)
            high = np.repeat(obs_space.high, self.n_stack, axis=-1)
            self.stacked_obs = np.zeros(low.shape, low.dtype)
            self.observation_space = spaces.Box(low=low, high=high, dtype=obs_space.dtype)

        self.seed()
        # Frame Skipping
        self.frame_skip = frame_skip
        # wait until loaded
        self.viewer.wait_until_loaded()

    def close_connection(self):
        return self.viewer.close_connection()

    def exit_scene(self):
        self.viewer.handler.send_exit_scene()

    def jerk_penalty(self):
        """
        Add a continuity penalty to limit jerk.
        :return: (float)
        """
        jerk_penalty = 0
        if self.n_command_history > 1:
            # Take only last command into account
            for i in range(1):
                steering = self.command_history[0, -2 * (i + 1)]
                prev_steering = self.command_history[0, -2 * (i + 2)]
                steering_diff = (prev_steering - steering) / (MAX_STEERING - MIN_STEERING)

                if abs(steering_diff) > MAX_STEERING_DIFF:
                    error = abs(steering_diff) - MAX_STEERING_DIFF
                    jerk_penalty += JERK_REWARD_WEIGHT * (error ** 2)
                else:
                    jerk_penalty += 0
        return jerk_penalty

    def postprocessing_step(self, action, observation, reward, done, info):
        """
        Update the reward (add jerk_penalty if needed), the command history
        and stack new observation (when using frame-stacking).

        :param action: ([float])
        :param observation: (np.ndarray)
        :param reward: (float)
        :param done: (bool)
        :param info: (dict)
        :return: (np.ndarray, float, bool, dict)
        """
        # Update command history
        if self.n_command_history > 0:
            self.command_history = np.roll(self.command_history, shift=-self.n_commands, axis=-1)
            self.command_history[..., -self.n_commands:] = action
            observation = np.concatenate((observation, self.command_history), axis=-1)

        jerk_penalty = self.jerk_penalty()
        # Cancel reward if the continuity constrain is violated
        if jerk_penalty > 0 and reward > 0:
            reward = 0
        reward -= jerk_penalty

        if self.n_stack > 1:
            self.stacked_obs = np.roll(self.stacked_obs, shift=-observation.shape[-1], axis=-1)
            if done:
                self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1]:] = observation
            return self.stacked_obs, reward, done, info

        return observation, reward, done, info

    def step(self, action):
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # action[0] is the steering angle
        # action[1] is the throttle
        if self.const_throttle is not None:
            action = np.concatenate([action, [self.const_throttle]])
        else:
            # Convert from [-1, 1] to [0, 1]
            t = (action[1] + 1) / 2
            # Convert from [0, 1] to [min, max]
            action[1] = (1 - t) * self.min_throttle + self.max_throttle * t

        # Clip steering angle rate to enforce continuity
        if self.n_command_history > 0:
            prev_steering = self.command_history[0, -2]
            max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
            diff = np.clip(action[0] - prev_steering, -max_diff, max_diff)
            action[0] = prev_steering + diff

        # Repeat action if using frame_skip
        for _ in range(self.frame_skip):
            self.viewer.take_action(action)
            observation, reward, done, info = self.observe()

        return self.postprocessing_step(action, observation, reward, done, info)

    def reset(self):
        self.viewer.reset()
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))
        observation, reward, done, info = self.observe()

        if self.n_command_history > 0:
            observation = np.concatenate((observation, self.command_history), axis=-1)

        if self.n_stack > 1:
            self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1]:] = observation
            return self.stacked_obs

        return observation

    def render(self, mode='human'):
        """
        :param mode: (str)
        """
        if mode == 'rgb_array':
            return self.viewer.handler.original_image
        return None

    def observe(self):
        """
        Encode the observation using VAE if needed.

        :return: (np.ndarray, float, bool, dict)
        """
        observation, reward, done, info = self.viewer.observe()
        # Learn from Pixels
        if self.vae is None:
            return observation, reward, done, info
        # Encode the image
        return self.vae.encode(observation), reward, done, info

    def close(self):
        if self.unity_process is not None:
            self.unity_process.quit()
        self.viewer.quit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_vae(self, vae):
        """
        :param vae: (VAEController object)
        """
        self.vae = vae
