import fileinput
import sys
import csv

from beamngpy import BeamNGpy, Scenario, Vehicle, Road
from beamngpy.sensors import Camera
import numpy as np
import math
from threading import Thread
from config import THROTTLE_REWARD_WEIGHT, CRASH_SPEED_WEIGHT, REWARD_CRASH, \
    BEAMNG_HOME, ROI, CAMERA_WIDTH, CAMERA_HEIGHT, STEP_CRASH_WEIGHT
from shapely.geometry import Point
from .roadnodes import RoadNodes

rn = RoadNodes().get('test')

def update_prefab(prefab_path):
    for i, line in enumerate(fileinput.input(prefab_path, inplace=1)):
        sys.stdout.write(line.replace('overObjects = "0"', 'overObjects = "1"'))  # replace 'sit' and write


def get_road():
    r = SimulationRoad()
    with open(rn['csv'], newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            number_row = list()
            for num in row:
                number_row.append(float(num))
            r.append(RoadPoint(number_row[:3], dir=number_row[3:]))

    print("Road loaded")
    return r


class Simulation(object):

    def __init__(self) -> None:
        super().__init__()

        thread = Thread(target=self._intervene)
        thread.start()

        self.step = 0
        self.done = False
        self.road = get_road()
        self.last_action = [0.0, 0.0]
        self.bng = BeamNGpy('localhost', 64256, home=BEAMNG_HOME)
        self.scenario = Scenario('smallgrid', 'DDPG', authors='Vsevolod Tymofyeyev',
                                 description='Reinforcement learning')

        road = Road(material='a_asphalt_01_a', rid='road', looped=True)
        road.nodes.extend(rn['nodes'])
        self.scenario.add_road(road)

        self.vehicle = Vehicle('ego_vehicle', model='etk800', licence='RL FTW', color='Red')
        front_camera = Camera(pos=(0, 1.3, 1.8), direction=(0, 1, -0.3), fov=90, resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
                              colour=True, depth=False, annotation=False)
        self.vehicle.attach_sensor("front_camera", front_camera)
        self.scenario.add_vehicle(self.vehicle, pos=rn['pos'], rot=rn['rot'])

        self.scenario.make(self.bng)
        prefab_path = self.scenario.get_prefab_path()
        update_prefab(prefab_path)

        self.bng.open()
        self.bng.set_steps_per_second(60)
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
        self.bng.step(3)

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

        # Cut to the relevant region
        image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # Convert to BGR
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

        #while self.done:
         #   self.bng.step(1)
        closest = self.road.closest_point(self.vehicle.state["pos"])
        self.step = 0
        self.done = False
        self.bng.teleport_vehicle(vehicle=self.vehicle, pos=closest.pos(), rot=closest.rot())
        #self.bng.restart_scenario()
        self.bng.pause()


class SimulationRoad(object):

    def __init__(self):
        self._road_points = list()

    def append(self, point):
        self._road_points.append(point)

    def closest_point(self, pos):
        min = sys.maxsize
        nearest_point = None
        current_point = RoadPoint(pos)
        index = 0
        for i, checkpoint in enumerate(self._road_points):
            if current_point.distance(checkpoint) < min:
                min = current_point.distance(checkpoint)
                index = i

        index += 1
        index = index % len(self._road_points)
        nearest_point = self._road_points[index]

        return nearest_point

    def list(self):
        r = [point.pos() for point in self._road_points]
        return r


class RoadPoint(Point):

    def __init__(self, *args, dir=None, rot=None):
        super().__init__(*args)
        self._dir = dir
        self._rot = rot

    def pos(self):
        return self.x, self.y, self.z

    def dir(self):
        return self._dir

    def rot(self):
        if self._rot is None:
            self._rot = self._compute_rot()

        return self._rot

    def _rad_to_degree(self, x):
        phi = (x * 180) / np.pi
        return phi

    def _compute_rot(self):
        x = self._dir[0]
        y = self._dir[1]
        hypotenuse = np.linalg.norm(self._dir)
        if x < 0 and y < 0:
            z_rot = np.absolute(np.arcsin(x / hypotenuse))
            z_rot = self._rad_to_degree(z_rot)
        elif x < 0 < y:
            z_rot = np.pi - np.absolute(np.arcsin(x / hypotenuse))
            z_rot = self._rad_to_degree(z_rot)
        elif x > 0 > y:
            z_rot = 2 * np.pi - np.arcsin(x / hypotenuse)
            z_rot = self._rad_to_degree(z_rot)
        else:
            z_rot = np.pi + np.arcsin(x / hypotenuse)
            z_rot = self._rad_to_degree(z_rot)

        return 0, 0, z_rot
