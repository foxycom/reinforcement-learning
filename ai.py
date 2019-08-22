from client.aiExchangeMessages_pb2 import SimulationID, VehicleID
from training_gym.envs.drivebuild_sim import Simulation
from algos import DDPG
import gym
import numpy as np
from stable_baselines.common import set_global_seeds
from utils.utils import create_test_env, get_saved_hyperparams
from config import LEVEL_NAME

STATS_PATH = "logs\\ddpg\\BeamNG-0_1\\BeamNG-0"


class DDPGAI(object):

    def __init__(self):
        set_global_seeds(0)

        hyperparams, stats_path = get_saved_hyperparams(STATS_PATH, norm_reward=False)
        hyperparams['vae_path'] = LEVEL_NAME.vae()
        self.simulation = Simulation()
        self.env = create_test_env(stats_path=stats_path, seed=0, log_dir=None, hyperparams=hyperparams,
                                   simulation=self.simulation)
        self.model = DDPG.load(LEVEL_NAME.model())

    def start(self, sid: SimulationID, vid: VehicleID) -> None:
        from client.aiExchangeMessages_pb2 import SimStateResponse
        from client.AIExchangeService import get_service

        service = get_service()
        self.simulation.sid = sid
        self.simulation.vid = vid

        running_reward = 0.0
        ep_len = 0

        sim_state = self.simulation.wait()
        if sim_state is SimStateResponse.SimState.RUNNING:
            obs = self.env.reset()

        while True:
            print(sid.sid + ": Test status: " + service.get_status(sid))
            print(vid + ": Wait")

            sim_state = self.simulation.wait()
            if sim_state is SimStateResponse.SimState.RUNNING:  # Check whether simulation is still running
                action, _ = self.model.predict(obs, deterministic=True)
                # Clip Action to avoid out of bound errors
                if isinstance(self.env.action_space, gym.spaces.Box):
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                obs, reward, _, _ = self.env.step(action)

                running_reward += reward[0]
                ep_len += 1

            else:
                # clean up
                self.env.reset()
                break

