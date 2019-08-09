import sys
import csv
from enum import Enum

import re
from beamngpy import BeamNGpy, Scenario, Vehicle, Road
from beamngpy.sensors import Camera
import numpy as np
from threading import Thread

from config import THROTTLE_REWARD_WEIGHT, CRASH_SPEED_WEIGHT, REWARD_CRASH, \
    BEAMNG_HOME, ROI, CAMERA_WIDTH, CAMERA_HEIGHT, SPS, STEPS_INTERVAL, FOV, \
    REWARD_STEP, MAX_DIST
from shapely.geometry import Point, LineString
from .roadnodes import RoadNodes


class WayPoint(object):
    def __init__(self, name=None, pos=None):
        self.name = name
        self.pos = pos

    def has_pos(self):
        if self.pos is not None:
            return True
        else:
            return False


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


ASFAULT_PREFAB = 'C:/Users/Tim/Documents/BeamNG.research/levels/asfault/scenarios/asfault.prefab'
DEFAULT_START = RoadPoint(1041.628662109375, 171.52037048339844, 0.20556232333183289,
                          dir=(-0.9885048270225525, -0.15105968713760376, -0.006256206426769495))

rn = RoadNodes().get('test')


class Mode(Enum):
    NONE = -1,
    ASPHALT = 0,
    MID_LINE = 1,
    LEFT_LINE = 2,
    RIGHT_LINE = 3,
    WAYPOINT = 4


def extract_node(line):
    try:
        node = re.search('"(.+?)"', line).group(1)
        node = node.split()
        numbers = [float(n) for n in node]
        return tuple(numbers)

    except AttributeError:
        return None


def read_prefab(prefab):
    mode = Mode.NONE
    asphalt_nodes = list()
    mid_line_nodes = list()
    left_line_nodes = list()
    right_line_nodes = list()

    with open(prefab, 'r') as file:
        for line in file:
            if "new DecalRoad(street_1) {" in line:
                mode = Mode.ASPHALT
                continue
            elif "new DecalRoad(divider_1_1) {" in line:
                mode = Mode.MID_LINE
                continue
            elif "new DecalRoad(boundary_1_l1) {" in line:
                mode = Mode.LEFT_LINE
                continue
            elif "new DecalRoad(boundary_1_r1) {" in line:
                mode = Mode.RIGHT_LINE
                continue
            elif "new BeamNGWaypoint" in line:
                mode = Mode.WAYPOINT

                continue

            if mode is Mode.ASPHALT and "Node =" in line:
                asphalt_nodes.append(extract_node(line))
            elif mode is Mode.MID_LINE and "Node =" in line:
                mid_line_nodes.append(extract_node(line))
            elif mode is Mode.LEFT_LINE and "Node =" in line:
                left_line_nodes.append(extract_node(line))
            elif mode is Mode.RIGHT_LINE and "Node =" in line:
                right_line_nodes.append(extract_node(line))

    return {'asphalt': asphalt_nodes, 'mid': mid_line_nodes, 'left': left_line_nodes, 'right': right_line_nodes}


def update_prefab(prefab_path):
    mode = Mode.NONE
    lines = list()
    with open(prefab_path, mode='r') as file:
        for line in file:
            if "overObjects" in line:
                line = line.replace("0", "1")
            elif "improvedSpline" in line:
                line = line.replace("1", "0")
            elif "new DecalRoad(asphalt)" in line:
                mode = Mode.ASPHALT
            elif "new DecalRoad(left_line)" in line:
                mode = Mode.LEFT_LINE
            elif "new DecalRoad(right_line)" in line:
                mode = Mode.RIGHT_LINE
            elif "new DecalRoad(mid_line)" in line:
                mode = Mode.MID_LINE

            if "renderPriority" in line:
                if mode == Mode.ASPHALT:
                    line = line.replace("0", "10")
                elif mode == Mode.MID_LINE:
                    line = line.replace("1", "9")
                elif mode == Mode.LEFT_LINE:
                    line = line.replace("2", "9")
                elif mode == Mode.RIGHT_LINE:
                    line = line.replace("3", "9")
            lines.append(line)

    with open(prefab_path, mode='w') as file:
        file.writelines(lines)


class TrainingRoad(object):

    def __init__(self, prefab):
        training_road = read_prefab(prefab)
        self.road_line = None

        self.asphalt = Road(material='a_asphalt_01_a', rid='asphalt', texture_length=2.5, break_angle=180)
        self.asphalt.nodes.extend(training_road['asphalt'])
        self.mid_nodes = [RoadPoint(node[:3]) for node in self.asphalt.nodes]

        self.mid_line = Road(material='line_yellow', rid='mid_lane', texture_length=16, break_angle=180, drivability=-1)
        self.mid_line.nodes.extend(training_road['mid'])
        self.mid_line_nodes = [RoadPoint(node[:3]) for node in self.mid_line.nodes]

        self.left_line = Road(material='line_white', rid='left_line', texture_length=16, break_angle=180,
                              drivability=-1)
        self.left_line.nodes.extend(training_road['left'])
        self.left_nodes = [RoadPoint(node[:3]) for node in self.left_line.nodes]

        self.right_line = Road(material='line_white', rid='right_line', texture_length=16, break_angle=180,
                               drivability=-1)
        self.right_line.nodes.extend(training_road['right'])
        self.right_nodes = [RoadPoint(node[:3]) for node in self.right_line.nodes]

    def calculate_road_line(self, back=False):
        road_line = list()
        for l_node, r_node in zip(self.mid_line_nodes, self.right_nodes):
            pos, dir = self.calculate_waypoint(l_node.pos(), r_node.pos())
            road_line.append(RoadPoint(pos, dir=dir))
           # print("Left: {} | Right: {} | Waypoint: {}".format(l_node.pos(), r_node.pos(), pos))

        if back:
            back_line = list()
            for l_node, r_node in zip(self.mid_line_nodes, self.left_nodes):
                pos, dir = self.calculate_waypoint(l_node.pos(), r_node.pos())
                back_line.append(RoadPoint(pos, dir=dir))
            back_line.reverse()
            road_line.extend(back_line)

        self.road_line = road_line

    def dist_to_center(self, pos):
        pos = RoadPoint(pos)
        line = LineString(self.road_line)
        return pos.distance(line)

    def calculate_waypoint(self, left_v, right_v):
        left_v = np.array(left_v[:2])
        right_v = np.array(right_v[:2])
        #mid = (left_v + right_v) / 2
        cross_road_v = right_v - left_v
        mid = left_v + cross_road_v / 2 + cross_road_v * 0.1
        along_road_v = np.array([cross_road_v[1], -cross_road_v[0]])

        dir = -along_road_v
        dir /= np.linalg.norm(dir)
        return np.append(mid, [0.2]), np.append(dir, [0])

    def closest_waypoint(self, pos):
        min = sys.maxsize
        current_point = RoadPoint(pos)
        for i, waypoint in enumerate(self.road_line):
            if current_point.distance(waypoint) < min:
                min = current_point.distance(waypoint)
                index = i

        index += 1
        if index > len(self.road_line) * 0.9:
            index = 0

        closest_waypoint = self.road_line[index]
        return closest_waypoint

    def random_waypoint(self):
        r = np.random.randint(low=0, high=int(len(self.road_line) * 0.8))
        return self.road_line[r]

    def spawn_point(self):
        return self.road_line[0]


# TODO fix / remove
def get_road():
    # r = SimulationRoad()
    r = list()
    with open(rn['csv'], newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            number_row = list()
            for num in row:
                number_row.append(float(num))
            r.append(RoadPoint(number_row[:3], dir=number_row[3:]))
    return r


class Simulation(object):

    def __init__(self) -> None:
        super().__init__()

        thread = Thread(target=self._intervene)
        thread.start()

        self.step = 0
        self.dist_driven = 0
        self.done = False
        self.last_action = (0.0, 0.0)
        self.bng = BeamNGpy('localhost', 64257, home=BEAMNG_HOME)
        self.scenario = Scenario('train', 'train', authors='Vsevolod Tymofyeyev',
                                 description='Reinforcement learning')

        self.road = TrainingRoad(ASFAULT_PREFAB)
        self.road.calculate_road_line()

        spawn_point = self.road.spawn_point()
        self.last_pos = spawn_point.pos()
        self.scenario.add_road(self.road.asphalt)
        self.scenario.add_road(self.road.mid_line)
        self.scenario.add_road(self.road.left_line)
        self.scenario.add_road(self.road.right_line)

        self.vehicle = Vehicle('ego_vehicle', model='etk800', licence='RL FTW', color='Red')
        front_camera = Camera(pos=(0, 1.4, 1.8), direction=(0, 1, -0.23), fov=FOV,
                              resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
                              colour=True, depth=False, annotation=False)
        self.vehicle.attach_sensor("front_camera", front_camera)

        self.scenario.add_vehicle(self.vehicle, pos=spawn_point.pos(), rot=spawn_point.rot())

        self.scenario.make(self.bng)
        prefab_path = self.scenario.get_prefab_path()
        update_prefab(prefab_path)

        self.bng.open()
        self.bng.set_deterministic()
        self.bng.set_steps_per_second(SPS)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()

        # self.bng.hide_hud()
        self.bng.pause()

    def _intervene(self):
        while True:
            a = input()
            self.done = not self.done

    def take_action(self, action):
        steering, throttle = action
        steering = steering.item()
        throttle = throttle.item()
        self.last_action = action
        self.step += 1

        self.vehicle.control(steering=steering, throttle=throttle, brake=0.0)
        self.bng.step(STEPS_INTERVAL)

    def _reward(self, done, dist):
        steering = self.last_action[0]
        throttle = self.last_action[1]
        velocity = self.velocity()  # km/h
        if not done:
            reward = REWARD_STEP - 1.2 * dist + THROTTLE_REWARD_WEIGHT * throttle
            #print("Line reward: {}".format(REWARD_STEP - 0.5 * dist))
        else:
            reward = REWARD_CRASH - CRASH_SPEED_WEIGHT * throttle

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

        #done = self.done
        pos = self.vehicle.state['pos']
        self.refresh_dist(pos)
        self.last_pos = pos
        dist = self.road.dist_to_center(self.last_pos)
        done = dist > MAX_DIST
        reward = self._reward(done, dist)

        return state, reward, done, {}

    def velocity(self):
        state = self.vehicle.state
        velocity = np.linalg.norm(state["vel"])
        return velocity * 3.6

    def position(self):
        return self.vehicle.state["pos"]

    def refresh_dist(self, pos):
        pos = np.array(pos)
        last_pos = np.array(self.last_pos)
        dist = np.linalg.norm(pos - last_pos)
        self.dist_driven += dist

    def close_connection(self):
        self.bng.close()

    def reset(self):
        self.vehicle.control(throttle=0, brake=0, steering=0)
        self.bng.poll_sensors(self.vehicle)

        self.dist_driven = 0
        self.step = 0
        self.done = False

        current_pos = self.vehicle.state['pos']
        closest_point = self.road.closest_waypoint(current_pos)
        #closest_point = self.road.random_waypoint()
        self.bng.teleport_vehicle(vehicle=self.vehicle, pos=closest_point.pos(), rot=closest_point.rot())
        self.bng.pause()
