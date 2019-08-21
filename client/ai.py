import os
from client.aiExchangeMessages_pb2 import SimulationID, VehicleID
from training_gym.envs.beamng_vae import BeamNGenv
from training_gym.envs.drivebuild_sim import Simulation
from algos import DDPG
import gym
import numpy as np
from stable_baselines.common import set_global_seeds
from utils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from config import ENV_ID

from utils.utils import load_vae
from config import MIN_THROTTLE, MAX_THROTTLE, N_COMMAND_HISTORY, FRAME_SKIP

MODEL_PATH = os.path.join(os.getcwd(), "models\\lines-agent.pkl")
VAE_PATH = os.path.join(os.getcwd(), "models\\lines-vae.pkl")


class DDPGAI(object):

    def __init__(self):
        set_global_seeds(0)

        log_path = os.path.join(os.getcwd(), '..', 'logs', 'ddpg', '{}_{}'.format(ENV_ID, '1'))
        #stats_path = os.path.join(log_path, ENV_ID)
        stats_path = "C:\\Users\\Tim\\PycharmProjects\\reinforcement-learning\\logs\\ddpg\\DonkeyVae-v0-level-0_1\\DonkeyVae-v0-level-0"
        hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False)
        hyperparams['vae_path'] = VAE_PATH
        #simulation = Simulation()
        self.env = create_test_env(stats_path=stats_path, seed=0, log_dir=None, hyperparams=hyperparams)
                                   #simulation=simulation)
        self.model = DDPG.load(MODEL_PATH)

    def start(self, sid: SimulationID, vid: VehicleID) -> None:
        from client.aiExchangeMessages_pb2 import SimStateResponse
        from client.AIExchangeService import get_service

        #service = get_service()
        #self.env.set_sid(sid)
        #self.env.set_vid(vid)

        running_reward = 0.0
        ep_len = 0

        #sim_state = self.env.wait()
        #if sim_state is SimStateResponse.SimState.RUNNING:
        if True:
            obs = self.env.reset()

        for _ in range(0, 10000):

            #print(sid.sid + ": Test status: " + service.get_status(sid))
            #print(vid + ": Wait")

            #sim_state = self.env.wait()
            #if sim_state is SimStateResponse.SimState.RUNNING:  # Check whether simulation is still running
            if True:

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


if __name__ == "__main__":
    ai = DDPGAI()
    sid = SimulationID()
    sid.sid = "snid_0_sim_2636"
    vid = VehicleID()
    vid.vid = "ego"
    ai.start(sid, vid)
