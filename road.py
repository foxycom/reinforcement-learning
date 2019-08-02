import fileinput
import sys
import csv

import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, Road, ProceduralCone, ProceduralCube
from beamngpy.sensors import Camera

from donkey_gym.envs.beamng_sim import TrainingRoad, ASFAULT_PREFAB, RoadPoint, update_prefab
from donkey_gym.envs.roadnodes import RoadNodes

CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160
BEAMNG_HOME = 'C:\\Users\\Tim\\Documents\\BeamNG.research'

road = TrainingRoad(ASFAULT_PREFAB)


def main() -> None:
    from beamngpy import BeamNGpy, Scenario, Vehicle
    bng = BeamNGpy("localhost", 64256, home=BEAMNG_HOME)
    scenario = Scenario("train", "train")
    scenario.add_road(road.asphalt)
    scenario.add_road(road.mid_line)
    scenario.add_road(road.left_line)
    scenario.add_road(road.right_line)

    vehicle = Vehicle('ego_vehicle', model='etk800', licence='EGO', color='Red')

    start_point = RoadPoint(1043.915283203125, 1048.5626220703125, 0.20800182223320007,
                            dir=(-0.7083801031112671, -0.7057840824127197, -0.00814660545438528))
    scenario.add_vehicle(vehicle, pos=start_point.pos(), rot=start_point.rot())

    scenario.make(bng)
    prefab_path = scenario.get_prefab_path()
    update_prefab(prefab_path)
    bng.open()

    bng.set_steps_per_second(60)
    bng.set_deterministic()
    bng.load_scenario(scenario)
    bng.start_scenario()
    bng.pause()

    total_dist = 0
    prev_position = (1043.915283203125, 1048.5626220703125, 0.20800182223320007)

    while True:
        bng.step(30)
        bng.poll_sensors(vehicle)
        current_position = vehicle.state["pos"]
        current_direction = vehicle.state["dir"]
        print(current_position, current_direction)
        dist = np.linalg.norm(np.array(current_position) - np.array(prev_position))
        prev_position = current_position
        total_dist += dist

        line = current_position + current_direction
        with open(".\\road.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(line)

        print(total_dist)

    input("press when done")


if __name__ == "__main__":
    main()
