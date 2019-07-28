import fileinput
import sys
import csv

import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, Road, ProceduralCone, ProceduralCube
from beamngpy.sensors import Camera
from donkey_gym.envs.roadnodes import RoadNodes

CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160
BEAMNG_HOME = 'C:\\Users\\Tim\\Documents\\research_new\\trunk'


def update_prefab():
    for i, line in enumerate(fileinput.input('C:\\Users\\Tim\\Documents\\research_new\\trunk\\levels\\smallgrid\\scenarios\\DamageSensorTest.prefab', inplace=1)):
        sys.stdout.write(line.replace('overObjects = "0"', 'overObjects = "1"'))  # replace 'sit' and write


def main() -> None:
    from beamngpy import BeamNGpy, Scenario, Vehicle, ProceduralCube
    bng = BeamNGpy("localhost", 64256, home=BEAMNG_HOME)
    scenario = Scenario("smallgrid", "DamageSensorTest", authors="Stefan Huber",
                        description="Test usage and check output of the damage sensor")

    rn = RoadNodes().get('test')
    road = Road(material='a_asphalt_01_a', rid='road', looped=True)
    road.nodes.extend(rn['nodes'])
    scenario.add_road(road)

    vehicle = Vehicle('ego_vehicle', model='etk800', licence='EGO', color='Red')

    scenario.add_vehicle(vehicle, pos=rn['pos'], rot=rn['rot'])

    scenario.make(bng)
    update_prefab()
    bng.open()

    bng.set_steps_per_second(120)
    bng.load_scenario(scenario)
    bng.start_scenario()
    bng.pause()

    total_dist = 0
    prev_position = (-1.5, 4.3, 0)

    while True:
        bng.step(30)
        bng.poll_sensors(vehicle)
        current_position = vehicle.state["pos"]
        current_direction = vehicle.state["dir"]
        dist = np.linalg.norm(np.array(current_position) - np.array(prev_position))
        prev_position = current_position
        total_dist += dist

        line = current_position + current_direction
        with open(".\\testroad.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(line)

        print(total_dist)

    input("press when done")


if __name__ == "__main__":
    main()
