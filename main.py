import random

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from threading import Thread
from time import sleep
from PIL import Image

random.seed(1700)


def wait():
    sleep(1000000)


thread = Thread(target=wait)
thread.start()

# Instantiate BeamNGpy instance running the simulator from the given path,
# communicating over localhost:64256
bng = BeamNGpy('localhost', 64256, home='C:\\Users\\Tim\\Documents\\research\\trunk')

# Create a scenario in west_coast_usa called 'example'
scenario = Scenario('west_coast_usa', 'example')
# Create an ETK800 with the licence plate 'PYTHON'
vehicle = Vehicle('ego_vehicle', model='etk800', licence='RL FTW', color='Red')
front_camera = Camera(pos=(-0.3, 1, 1.0), direction=(0, 1, 0), fov=120, resolution=(1024, 1024), colour=True)
vehicle.attach_sensor("monocular camera", front_camera)

# Add it to our scenario at this position and rotation
scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot=(0, 0, 45))
# Place files defining our scenario for the simulator to read
scenario.make(bng)

# Launch BeamNG.research
bng.open()

bng.set_deterministic()
bng.set_steps_per_second(60)
# Load and start our scenario
bng.load_scenario(scenario)
bng.start_scenario()
bng.pause()
bng.hide_hud()


throttle = random.uniform(0.0, 1.0)
steering = random.uniform(-1.0, 1.0)
brake = random.choice([0, 0, 0, 1])
vehicle.control(throttle=throttle, steering=steering, brake=brake)

bng.step(20)
sensors = bng.poll_sensors(vehicle)
print(sensors)
sensors['monocular camera']['colour'].show()
print(type(sensors['monocular camera']['colour']))

