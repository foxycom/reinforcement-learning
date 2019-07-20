import fileinput
import sys

from beamngpy import BeamNGpy, Scenario, Vehicle, Road
from beamngpy.sensors import Camera
import numpy as np
import math
from threading import Thread
from config import THROTTLE_REWARD_WEIGHT, CRASH_SPEED_WEIGHT, REWARD_CRASH, \
    BEAMNG_HOME, ROI, CAMERA_WIDTH, CAMERA_HEIGHT, STEP_CRASH_WEIGHT
from shapely.geometry.polygon import Polygon


def update_prefab(prefab_path):
    for i, line in enumerate(fileinput.input(prefab_path, inplace=1)):
        sys.stdout.write(line.replace('overObjects = "0"', 'overObjects = "1"'))  # replace 'sit' and write


class Simulation(object):

    def __init__(self) -> None:
        super().__init__()

        thread = Thread(target=self._intervene)
        thread.start()

        self.step = 0
        self.done = False
        self.last_action = [0.0, 0.0]
        self.bng = BeamNGpy('localhost', 64256, home=BEAMNG_HOME)
        self.scenario = Scenario('smallgrid', 'DDPG', authors='Vsevolod Tymofyeyev',
                                 description='Reinforcement learning')

        nodes = [
            (-1.5, 4.3, 0, 5),
            (-1.63, -28.3, 0, 6),
            (24, -73, 0, 5),
            #(67, -70, 0, 5),
            (105, -35, 0, 5),
            (62, 20, 0, 5),
            (32, 5, 0, 5),
            (73, -21, 0, 5),
            (127, -20, 0, 5),
            (114, -116, 0, 5),
            (58, -116, 0, 5),
            (28, -145, 0, 5),
            (-40, -136, 0, 5),
            (-51, -62, 0, 5),
            (-105, -47, 0, 5),
            (-76, 35, 0, 5),
            (-25, 82, 0, 5),
            (5.6, 66, 0, 5)
        ]
        road = Road(material='a_asphalt_01_a', rid='road', looped=True)
        road.nodes.extend(nodes)
        self.scenario.add_road(road)

        self.vehicle = Vehicle('ego_vehicle', model='etk800', licence='RL FTW', color='Red')
        front_camera = Camera(pos=(0, 1.3, 1.6), direction=(0, 1, -0.1), fov=90, resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
                              colour=True, depth=False, annotation=False)
        self.vehicle.attach_sensor("front_camera", front_camera)
        self.scenario.add_vehicle(self.vehicle, pos=(-1.5, 4.3, 0), rot=(0, 0, -165))

        self.scenario.make(self.bng)
        prefab_path = self.scenario.get_prefab_path()
        update_prefab(prefab_path)

        self.bng.open()
        # self.bng.set_steps_per_second(60)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()

        # self.bng.hide_hud()
        self.bng.pause()

    def _intervene(self):
        while True:
            a = input()
            self.done = not self.done

    @staticmethod
    def _distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def take_action(self, action):
        steering, throttle = action
        self.last_action = action
        self.step += 1

        self.vehicle.control(steering=steering.item(), throttle=throttle.item(), brake=0.0)
        self.bng.step(6)

    def _is_in_lane(self):
        return True

    def _reward(self):
        if not self.done:
            reward = 1 + THROTTLE_REWARD_WEIGHT * self.last_action[1]
            return reward
        else:
            reward = REWARD_CRASH - CRASH_SPEED_WEIGHT * self.last_action[1] - STEP_CRASH_WEIGHT * self.step
            return reward

    def observe(self):
        sensors = self.bng.poll_sensors(self.vehicle)
        image = sensors['front_camera']['colour'].convert("RGB")
        image = np.array(image)
        r = ROI
        image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        state = image[:, :, ::-1]

        reward = self._reward()
        done = self.done

        return state, reward, done, {}

    def velocity(self):
        state = self.vehicle.state
        velocity = np.linalg.norm(state["vel"])
        return velocity * 3.6

    def position(self):
        return self.vehicle.state["pos"]

    def close_connection(self):
        self.bng.close()

    def reset(self):
        self.vehicle.control(throttle=0, brake=0, steering=0)

        while self.done:
            self.bng.step(1)

        self.step = 0
        self.done = False
        #self.bng.restart_scenario()
        self.bng.pause()
