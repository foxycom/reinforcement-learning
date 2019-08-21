import io
import numpy as np

from config import THROTTLE_REWARD_WEIGHT, CRASH_SPEED_WEIGHT, REWARD_CRASH, ROI, REWARD_STEP, MAX_DIST
from client.aiExchangeMessages_pb2 import SimStateResponse, Control, SimulationID, VehicleID, DataRequest
from client.AIExchangeService import get_service
from PIL import Image


class Simulation(object):

    def __init__(self) -> None:
        super().__init__()

        self.last_action = (0, 0)
        self.service = get_service()
        self.sid = None
        self.vid = None

    def take_action(self, action):
        steering, throttle = action
        steering = steering.item()
        throttle = throttle.item()
        self.last_action = action

        control = Control()
        control.avCommand.accelerate = throttle
        control.avCommand.steer = steering
        control.avCommand.brake = 0
        self.service.control(self.sid, self.vid, control)

    def _reward(self):
        throttle = self.last_action[1]
        reward = REWARD_CRASH - CRASH_SPEED_WEIGHT * throttle
        return reward

    def observe(self):
        request = DataRequest()
        request.request_ids.extend(["egoFrontCamera"])
        data = self.service.request_data(self.sid, self.vid, request)
        byte_im = data.data['egoFrontCamera'].camera.color
        image = Image.open(io.BytesIO(byte_im))
        image = image.convert("RGB")
        image = np.array(image)
        r = ROI

        # Cut to the relevant region
        image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # Convert to BGR
        state = image[:, :, ::-1]

        reward = self._reward()
        return state, reward, False, {}

    def close_connection(self):
        pass

    def reset(self):
        pass

    def wait(self):
        return self.service.wait_for_simulator_request(self.sid, self.vid)
