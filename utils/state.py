import numpy as np
import csv
import sys
from shapely.geometry import Point

REVERSE_STATES_WEIGHT = 0.4
DEFAULT_POS = (-829.6900634765625, -2.620422840118408, 117.47492980957031)
DEFAULT_ROT = (0, 0, 45)

CITY = {"pos": DEFAULT_POS, "rot": DEFAULT_ROT}
TUNNEL = {"pos": (-392.8107604980469, -315.23541259765625, 97.71546936035156)}
FOREST = {"pos": (947.3220825195312, -948.9266967773438, 167.6211395263672), "rot": (0, 0, 115)}

class Road(object):
    def __init__(self):
        self._checkpoints = self._csv()

    def _csv(self):
        v = []
        with open("C:\\Users\\Tim\\PycharmProjects\\reinforcement-learning\\legacy\\road.csv", mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                float_row = []
                for x in row:
                    float_row.append(float(x))

                v.append(Checkpoint(float_row[:3], dir=float_row[3:]))
        return v

    def nearest_checkpoint(self, state):
        min = sys.maxsize
        current_point = Point(state["pos"])
        nearest_point = None
        checkpoint_idx = -1
        for i, checkpoint in enumerate(self._checkpoints):

            if current_point.distance(checkpoint) < min:
                checkpoint_idx = i
                min = current_point.distance(checkpoint)
                nearest_point = checkpoint

        limit = int(round(9 * len(self._checkpoints) / 10))
        print("Checkpoint index: {}".format(checkpoint_idx))
        if checkpoint_idx > limit:
            return Checkpoint(DEFAULT_POS, rot=DEFAULT_ROT)
        else:
            return nearest_point


class Checkpoint(Point):
    def __init__(self, *args, dir=(0, 0, 0), rot=None):
        super().__init__(*args)
        self._dir = dir
        self._rot = rot

    def pos(self):
        return self.x, self.y, self.z

    def dir(self):
        return self._dir

    def _rad_to_degree(self, x):
        phi = (x * 180) / np.pi
        return phi

    def rot(self):
        if self._rot is None:
            return self._compute_rot()
        else:
            return self._rot

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
