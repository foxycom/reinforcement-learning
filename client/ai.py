import os
from client.aiExchangeMessages_pb2 import SimulationID, VehicleID
from training_gym.envs.drivebuild_vae import DriveBuildEnv
from algos import DDPG
from gym import spaces
import numpy as np

from utils.utils import load_vae
from config import MIN_THROTTLE, MAX_THROTTLE, N_COMMAND_HISTORY, FRAME_SKIP

MODEL_PATH = os.path.join(os.getcwd(), "models/agent.pkl")
VAE_PATH = os.path.join(os.getcwd(), "models/vae.pkl")


class DDPGAI(object):

    def __init__(self):
        vae = load_vae(VAE_PATH)
        assert vae is not None

        self.env = DriveBuildEnv(vae=vae, frame_skip=FRAME_SKIP, max_throttle=MAX_THROTTLE, min_throttle=MIN_THROTTLE,
                                 n_command_history=N_COMMAND_HISTORY)
        self.model = DDPG.load(MODEL_PATH)

    def start(self, sid: SimulationID, vid: VehicleID) -> None:
        from client.aiExchangeMessages_pb2 import SimStateResponse

        self.env.sid = sid
        self.env.vid = vid

        i = 0
        while i < 10:
            i += 1

            print(sid.sid + ": Test status: " + self.env.status())
            # Wait for the simulation to request this AI
            state = self.env.wait()
            if state is SimStateResponse.SimState.RUNNING:  # Check whether simulation is still running
                # Request data this AI needs
                obs = self.env.observe(["egoFrontCamera", "egoLaneDist"])
                action, _ = self.model.predict(obs, deterministic=True)
                if isinstance(self.env.action_space, spaces.Box):
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                self.env.step(action)
            else:
                print(sid.sid + ": The simulation is not running anymore (Final state: %%).")
                print(sid.sid + ": Final test result: " + self.env.result())
                # Clean up everything you have to
                break

        self.env.finish_sim()


if __name__ == "__main__":
    ai = DDPGAI()
    sid = SimulationID()
    sid.sid = "snid_0_sim_2636"
    vid = VehicleID()
    vid.vid = "ego"
    ai.start(sid, vid)
