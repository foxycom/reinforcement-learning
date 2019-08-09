# Original author: Roma Sokolkov
# Edited by Antonin Raffin

import numpy as np
from gym import spaces
from gym.utils import seeding

from config import INPUT_DIM, MIN_STEERING, MAX_STEERING, MAX_STEERING_DIFF, JERK_REWARD_WEIGHT
from client.AIExchangeService import get_service
from client.aiExchangeMessages_pb2 import SimStateResponse, DataRequest, Control, TestResult


class DriveBuildEnv(object):
    """
    Gym interface for DonkeyCar with support for using
    a VAE encoded observation instead of raw pixels if needed.

    :param frame_skip: (int) frame skip, also called action repeat
    :param vae: (VAEController object)
    :param const_throttle: (float) If set, the car only controls steering
    :param min_throttle: (float)
    :param max_throttle: (float)
    :param n_command_history: (int) number of previous commmands to keep
        it will be concatenated with the vae latent vector
    :param n_stack: (int) Number of frames to stack (used in teleop mode only)
    """

    def render(self, mode='human'):
        pass

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self, frame_skip=2, vae=None, const_throttle=None,
                 min_throttle=0.2, max_throttle=0.5, n_command_history=1, n_stack=1):
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

        self.unity_process = None
        self.service = get_service()
        self.sid = None
        self.vid = None

        # steering + throttle, action space must be symmetric
        self.action_space = spaces.Box(low=np.array([-MAX_STEERING, -1]),
                                       high=np.array([MAX_STEERING, 1]), dtype=np.float32)


        # z latent vector from the VAE (encoded input image)
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(1, self.z_size + self.n_commands * n_command_history),
                                            dtype=np.float32)

        self.seed()
        # Frame Skipping
        self.frame_skip = frame_skip
        # wait until loaded
        # self.viewer.wait_until_loaded()

    def close_connection(self):
        #return self.service.()
        pass

    def exit_scene(self):
        return

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

    def step(self, action):
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # action[0] is the steering angle
        # action[1] is the throttle
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
            control = Control()
            control.avCommand.steer = action[0]
            control.avCommand.accelerate = action[1]
            # control.avCommand.brake = <Some value between 0.0 and 1.0>
            self.service.control(self.sid, self.vid, control)

    def reset(self):
        self.service.reset()
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))
        observation, reward, done, info = self.observe()

        if self.n_command_history > 0:
            observation = np.concatenate((observation, self.command_history), axis=-1)

        if self.n_stack > 1:
            self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1]:] = observation
            return self.stacked_obs

        return observation

    def observe(self, ids):
        """
        Encode the observation using VAE if needed.

        :return: (np.ndarray, float, bool, dict)
        """
        request = DataRequest()
        request.request_ids.extend(ids)
        data = self.service.request_data(self.sid, self.vid, request)
        observation = data.data['egoFrontCamera']   # FIXME whatever structure
        # FIXME convert byte string to array

        # Encode the image
        return self.vae.encode(observation)

    def wait(self):
        return self.service.wait_for_simulator_request(self.sid, self.vid)

    def set_vae(self, vae):
        """
        :param vae: (VAEController object)
        """
        self.vae = vae

    def result(self):
        return self.service.get_status(self.sid)

    def status(self):
        return self.service.get_status(self.sid)

    def finish_sim(self):
        if self.wait() is SimStateResponse.SimState.RUNNING:
            result = TestResult()
            result.result = TestResult.Result.FAILED
            self.service.control_sim(self.sid, result)
