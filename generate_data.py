from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import numpy as np

# Instantiate BeamNGpy instance running the simulator from the given path,
# communicating over localhost:64256
bng = BeamNGpy('localhost', 64256, home='C:\\Users\\Tim\\Documents\\research\\trunk')
# Create a scenario in west_coast_usa called 'example'
scenario = Scenario('west_coast_usa', 'example')
# Create an ETK800 with the licence plate 'PYTHON'
vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')
front_camera = Camera(pos=(0, 2.2, 1.5), direction=(0, 1, -0.2), fov=120, resolution=(160, 80),
                                   colour=True, depth=False, annotation=False)
vehicle.attach_sensor("front_camera", front_camera)

# Add it to our scenario at this position and rotation
scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot=(0, 0, 45))
# Place files defining our scenario for the simulator to read
scenario.make(bng)

# Launch BeamNG.research
bng.open()

bng.set_nondeterministic()
bng.set_steps_per_second(60)
# Load and start our scenario
bng.load_scenario(scenario)

bng.start_scenario()
# Make the vehicle's AI span the map
bng.hide_hud()
bng.pause()

vehicle.ai_set_mode('span')
vehicle.ai_set_speed(13.5)
vehicle.ai_drive_in_lane(True)


number_of_images = 0
while number_of_images < 10000:
    number_of_images += 1
    bng.step(4)
    sensors = bng.poll_sensors(vehicle)
    camera_image = sensors['front_camera']['colour']
    camera_image.convert("RGB").save('C:\\Users\\Tim\\PycharmProjects\\reinforcement-learning\\datasets\\' + str(number_of_images) + '.jpg', format='JPEG')

bng.close()