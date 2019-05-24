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
front_camera = Camera(pos=(-0.3, 2, 1.0), direction=(0, 1, 0), fov=120, resolution=(128, 128), colour=True,
                      annotation=True, depth=True)
vehicle.attach_sensor("front_camera", front_camera)

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

vehicle.ai_set_mode('span')
vehicle.ai_drive_in_lane(True)
vehicle.ai_set_speed(13.8)


bng.hide_hud()
bng.pause()
i = 0

while i < 60000:
    sensors = bng.poll_sensors(vehicle)
    camera_image = sensors['front_camera']['colour']
    annotated_image = sensors['front_camera']['annotation']

    #camera_image.save('datasets\\camera\\' + str(i) + '.bmp', format='BMP')
    #annotated_image.save('datasets\\target\\' + str(i) + '.bmp', format='BMP')
    i += 1
    bng.step(5)

bng.close()


