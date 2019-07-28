import csv
import fileinput
import sys

from beamngpy import BeamNGpy, Scenario, Vehicle, Road
from beamngpy.sensors import Camera
from config import CAMERA_HEIGHT, CAMERA_WIDTH, ROI
import numpy as np
from PIL import Image
import cv2
from donkey_gym.envs.roadnodes import RoadNodes

from donkey_gym.envs.beamng_sim import SimulationRoad, RoadPoint


def get_road():
    r = SimulationRoad()
    with open('C:\\Users\\Tim\\PycharmProjects\\reinforcement-learning\\road.csv', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            number_row = list()
            for num in row:
                number_row.append(float(num))
            r.append(RoadPoint(number_row[:3], dir=number_row[3:]))

    print("Road loaded")
    return r

def update_prefab(prefab_path):
    for i, line in enumerate(fileinput.input(prefab_path, inplace=1)):
        sys.stdout.write(line.replace('overObjects = "0"', 'overObjects = "1"'))  # replace 'sit' and write

# Instantiate BeamNGpy instance running the simulator from the given path,
# communicating over localhost:64256
bng = BeamNGpy('localhost', 64256, home='C:\\Users\\Tim\\Documents\\research\\trunk')
# Create a scenario in west_coast_usa called 'example'
scenario = Scenario('smallgrid', 'generate_data')

rn = RoadNodes().get('test')
road = Road(material='a_asphalt_01_a', rid='road', looped=True)
road.nodes.extend(rn['nodes'])
scenario.add_road(road)

vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')
front_camera = Camera(pos=(0, 1.3, 1.8), direction=(0, 1, -0.3), fov=90, resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
                                   colour=True, depth=False, annotation=False)
vehicle.attach_sensor("front_camera", front_camera)

# Add it to our scenario at this position and rotation
scenario.add_vehicle(vehicle, pos=rn['pos'], rot=rn['rot'])
# Place files defining our scenario for the simulator to read
scenario.make(bng)

prefab_path = scenario.get_prefab_path()
update_prefab(prefab_path)
bng.open()

#bng.set_nondeterministic()
bng.set_steps_per_second(60)
# Load and start our scenario
bng.load_scenario(scenario)

bng.start_scenario()
# Make the vehicle's AI span the map

number_of_images = 0
while number_of_images < 10000:
    number_of_images += 1
    print(number_of_images)
    bng.step(1)
    sensors = bng.poll_sensors(vehicle)
    image = sensors['front_camera']['colour'].convert('RGB')
    image = np.array(image)
    image = image[:, :, ::-1]
    cv2.imwrite('C:\\Users\\Tim\\PycharmProjects\\reinforcement-learning\\datasets\\{}.jpg'.format(number_of_images), image)

bng.close()