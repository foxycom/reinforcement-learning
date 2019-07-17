from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import numpy as np
import time
from threading import Thread
import csv

def wait():
    time.sleep(1000000)

t = Thread(target=wait)
t.start()

# Instantiate BeamNGpy instance running the simulator from the given path,
# communicating over localhost:64256
bng = BeamNGpy('localhost', 64256, home='C:\\Users\\Tim\\Documents\\research\\trunk')
# Create a scenario in west_coast_usa called 'example'
scenario = Scenario('west_coast_usa', 'example')
# Create an ETK800 with the licence plate 'PYTHON'
vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')
front_camera = Camera(pos=(0, 2.2, 2.5), direction=(0, 1, -0.55), fov=120, resolution=(160, 80),
                                   colour=True, depth=False, annotation=False)
vehicle.attach_sensor("front_camera", front_camera)
# Add it to our scenario at this position and rotation
current_position = (947.3220825195312, -948.9266967773438, 167.6211395263672)

scenario.add_vehicle(vehicle, pos=current_position, rot=(0, 0, 115))

# Place files defining our scenario for the simulator to read
scenario.make(bng)

# Launch BeamNG.research
bng.open()
# Load and start our scenario
bng.load_scenario(scenario)
bng.start_scenario()
# Make the vehicle's AI span the map

bng.set_steps_per_second(120)
bng.step(2000)
bng.pause()
prev_position = current_position
total_dist = 0

br = False
def st():
    global br
    inp = input()
    br = True

stop = Thread(target=st)
stop.start()

while not br:
    bng.step(30)
    bng.poll_sensors(vehicle)
    current_position = vehicle.state["pos"]
    current_direction = vehicle.state["dir"]
    dist = np.linalg.norm(np.array(current_position) - np.array(prev_position))
    prev_position = current_position
    total_dist += dist

    line = current_position + current_direction
    with open(".\\road.csv", "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(line)

    print(total_dist)