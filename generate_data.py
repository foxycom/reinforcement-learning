import fileinput
import sys

from beamngpy import BeamNGpy, Scenario, Vehicle, Road
from beamngpy.sensors import Camera
from config import CAMERA_HEIGHT, CAMERA_WIDTH, ROI
import numpy as np
from PIL import Image


def update_prefab(prefab_path):
    for i, line in enumerate(fileinput.input(prefab_path, inplace=1)):
        sys.stdout.write(line.replace('overObjects = "0"', 'overObjects = "1"'))  # replace 'sit' and write

# Instantiate BeamNGpy instance running the simulator from the given path,
# communicating over localhost:64256
bng = BeamNGpy('localhost', 64256, home='C:\\Users\\Tim\\Documents\\research\\trunk')
# Create a scenario in west_coast_usa called 'example'
scenario = Scenario('smallgrid', 'generate_data')

nodes = [
            (0, 0, 0, 6),
            (50, 50, 0, 6),
            (150, 90, 0, 6),
            (200, 200, 0, 6),
            (240, 270, 0, 6),
            (200, 350, 0, 6),
            (280, 400, 0, 6),
            (360, 400, 0, 6),
            (400, 280, 0, 6),
            (400, 180, 0, 6),
            (350, 150, 0, 6)
        ]
road = Road(material='a_asphalt_01_a', rid='road', looped=False)
road.nodes.extend(nodes)
scenario.add_road(road)

nodes = [
    (100, 500, 0, 6),
    (180, 350, 0, 6),
    (280, 300, 0, 6),
    (300, 270, 0, 6),
    (330, 400, 0, 6),
    (360, 450, 0, 6),
    (430, 410, 0, 6),
    (430, 300, 0, 6),
    (350, 250, 0, 6),
    (250, 180, 0, 6),
    (150, 100, 0, 6)
]
road = Road(material='a_asphalt_01_a', rid='road2', looped=False)
road.nodes.extend(nodes)
scenario.add_road(road)

vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')
front_camera = Camera(pos=(0, 2.2, 1.5), direction=(0, 1, -0.15), fov=135, resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
                                   colour=True, depth=False, annotation=False)
vehicle.attach_sensor("front_camera", front_camera)

# Add it to our scenario at this position and rotation
scenario.add_vehicle(vehicle, pos=(2, 2, 0), rot=(0, 0, 220))
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
#bng.hide_hud()
#bng.pause()

#vehicle.ai_set_mode('span')
#vehicle.ai_set_speed(13.5)
#vehicle.ai_drive_in_lane(True)


number_of_images = 2306
while number_of_images < 10000:
    number_of_images += 1
    print(number_of_images)
    bng.step(1)
    sensors = bng.poll_sensors(vehicle)
    image = sensors['front_camera']['colour'].convert('RGB')
    image = np.array(image)

    r = ROI
    image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    image = Image.fromarray(image, 'RGB')

    image.save('C:\\Users\\Tim\\PycharmProjects\\reinforcement-learning\\datasets\\' + str(number_of_images) + '.jpg',
               format='JPEG')

bng.close()