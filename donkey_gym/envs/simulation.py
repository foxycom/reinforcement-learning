from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from threading import Thread
from time import sleep
import numpy as np
import math
import csv
from config import THROTTLE_REWARD_WEIGHT, CRASH_SPEED_WEIGHT, REWARD_CRASH, VELOCITY_PENALTY, TARGET_VELOCITY, \
    BEAMNG_HOME, CHECKPOINTS
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from utils.state import Road, Checkpoint, FOREST

def wait():
    sleep(1000000)


class Simulation(object):

    def __init__(self) -> None:
        super().__init__()
        self._checkpoints = CHECKPOINTS

        self.done = False
        t = Thread(target=self._intervene)
        t.start()
        self.lane = self._get_lane()
        self.last_action = [0.0, 0.0]
        self.road = Road()
        self.bng = BeamNGpy('localhost', 64256, home=BEAMNG_HOME)

        self.scenario = Scenario('west_coast_usa', 'Reinforcement learning with DDPG')
        self.vehicle = Vehicle('ego_vehicle', model='etk800', licence='RL FTW', color='Red')
        self.front_camera = Camera(pos=(0, 2.2, 1.5), direction=(0, 1, -0.2), fov=120, resolution=(160, 80),
                                   colour=True, depth=False, annotation=False)
        self.vehicle.attach_sensor("front_camera", self.front_camera)

        self.scenario.add_vehicle(self.vehicle, pos=FOREST["pos"], rot=FOREST["rot"])

        self.scenario.make(self.bng)
        self.run()

    def run(self):
        # Launch BeamNG.research
        self.bng.open()

        #self.bng.set_deterministic()
        self.bng.set_steps_per_second(60)

        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()

        self.vehicle.ai_set_mode('disabled')
        #self.vehicle.ai_drive_in_lane(True)
        #self.vehicle.ai_set_speed(13.8)  # 50 km/h

        #self.bng.hide_hud()
        self.bng.pause()

    def _intervene(self):
        while True:
            a = input()
            self.done = True

    def _get_lane(self):
        left_lane = self._csv(path="C:\\Users\\Tim\\PycharmProjects\\reinforcement-learning\\legacy\\left.csv")
        right_lane = self._csv(path="C:\\Users\\Tim\\PycharmProjects\\reinforcement-learning\\legacy\\right.csv")
        lane = left_lane + right_lane
        return Polygon(lane)

    def _csv(self, path, low=0, high=2):
        v = []
        with open(path, mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                float_row = []
                for i in range(low, high):
                    float_row.append(float(row[i]))

                v.append(tuple(float_row))
        return v

    def _distance(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _closest_dist(self, pos):
        pos = pos[:2]
        distances = [self._distance(pos, point[:2]) for point in self.lane]
        return np.min(distances)

    def _is_between(self, a, b, pos):
        return math.isclose(self._distance(a, pos) + self._distance(pos, b), self._distance(a, b), abs_tol=0.02)

    def _is_in_lane(self):
        #pos = self.position()[:2]
        #pos = Point(np.array(pos))
        #return self.lane.contains(pos)
        return not self.done

    def take_action(self, action):
        steering, throttle = action
        self.last_action = action

        self.vehicle.control(steering=steering.item(), throttle=throttle.item(), brake=0.0)
        self.bng.step(6)

    def _velocity(self):
        state = self.vehicle.state
        velocity = np.linalg.norm(state["vel"])
        return velocity * 3.6

    def _reward(self):
        if self._is_in_lane():
            reward = 1 + THROTTLE_REWARD_WEIGHT * self.last_action[1]
            #if self._velocity() > TARGET_VELOCITY:
            #    reward -= VELOCITY_PENALTY * (self._velocity() - TARGET_VELOCITY)
            return reward
        else:
            reward = REWARD_CRASH - CRASH_SPEED_WEIGHT * self.last_action[1]
            #velocity = self._velocity()
            #if velocity > TARGET_VELOCITY:
            #    reward -= VELOCITY_PENALTY * (velocity - TARGET_VELOCITY)
            return reward

    def observe(self):
        sensors = self.bng.poll_sensors(self.vehicle)

        camera_image = sensors['front_camera']['colour'].convert("RGB")
        state = np.asarray(camera_image, dtype=np.uint8)

        reward = self._reward()
        done = not self._is_in_lane()

        return state, reward, done, {}

    def _wait(self):
        thread = Thread(target=wait)
        thread.start()

    def done(self):
        return False

    def position(self):
        return self.vehicle.state["pos"]

    def close_connection(self):
        self.bng.close()

    def next_spawn(self, state):
        checkpoint = self.road.nearest_checkpoint(state=state)

        return checkpoint.pos(), checkpoint.rot()

    def reset(self):
        self.done = False
        self.vehicle.control(throttle=0, brake=0, steering=0)

        if self._checkpoints:
            self._reset_with_checkpoints()
        else:
            self.bng.restart_scenario()

        self.bng.pause()

    def _reset_with_checkpoints(self):
        state = self.vehicle.state
        self.bng.restart_scenario()
        pos, rot = self.next_spawn(state)
        self.bng.teleport_vehicle(self.vehicle, pos=pos, rot=rot)

    def quit(self):
        self.bng.close()
