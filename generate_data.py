import numpy as np
import os
import cv2

from beamngpy import BeamNGpy, Scenario, Vehicle
from config import CAMERA_HEIGHT, CAMERA_WIDTH, FOV

from beamngpy.sensors import Camera
from training_gym.envs.beamng_sim import TrainingRoad, ASFAULT_PREFAB, update_prefab

NUMBER_OF_IMAGES = 9000


def main():
    """
    Generate a bunch of images by driving along a predefined sequence of points, while capturing camera images
    to JPG files.

    :return: (None)
    """
    bng_home = os.environ['BEAMNG_HOME']
    road = TrainingRoad(ASFAULT_PREFAB)
    road.calculate_road_line(back=True)

    bng = BeamNGpy('localhost', 64256, home=bng_home)
    scenario = Scenario('smallgrid', 'train')

    # Asphalt and lines are actually considered as differently colored roads by beamngpy.
    scenario.add_road(road.asphalt)
    scenario.add_road(road.mid_line)
    scenario.add_road(road.left_line)
    scenario.add_road(road.right_line)


    vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')
    # Create a dash cam that is somewhat down-sloped.
    front_camera = Camera(pos=(0, 1.4, 1.8), direction=(0, 1, -0.23), fov=FOV, resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
                                       colour=True, depth=False, annotation=False)
    vehicle.attach_sensor("front_camera", front_camera)

    # Get a spawn point and initial rotation of the vehicle.
    spawn_point = road.spawn_point()
    scenario.add_vehicle(vehicle, pos=spawn_point.pos(), rot=spawn_point.rot())
    # Place files defining our scenario for the simulator to read.
    scenario.make(bng)

    prefab_path = scenario.get_prefab_path()
    update_prefab(prefab_path)
    bng.open()

    bng.set_nondeterministic()
    bng.set_steps_per_second(60)
    # Load and start our scenario
    bng.load_scenario(scenario)

    bng.start_scenario()
    vehicle.ai_set_mode('span')
    vehicle.ai_set_speed(5)
    vehicle.ai_set_line([{'pos': node.pos(), 'speed': 10} for node in road.road_line])

    number_of_images = 0
    while number_of_images < NUMBER_OF_IMAGES:
        bng.poll_sensors(vehicle)
        number_of_images += 1
        bng.step(1)
        sensors = bng.poll_sensors(vehicle)
        image = sensors['front_camera']['colour'].convert('RGB')
        image = np.array(image)
        image = image[:, :, ::-1]
        dist = road.dist_to_center(vehicle.state['pos'])
        cv2.imwrite('datasets\\{}.jpg'.format(number_of_images), image)

    bng.close()


if __name__ == '__main__':
    main()
