from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from threading import Thread
from time import sleep
from matplotlib import image, pyplot
from PIL import Image


def wait():
    sleep(1000000)


class Simulation(object):

    def __init__(self) -> None:
        super().__init__()

        self._wait()

        # Instantiate BeamNGpy instance running the simulator from the given path,
        # communicating over localhost:64256
        self.bng = BeamNGpy('localhost', 64256, home='C:\\Users\\Tim\\Documents\\research\\trunk')

        self.scenario = Scenario('west_coast_usa', 'example')
        self.vehicle = Vehicle('ego_vehicle', model='etk800', licence='RL FTW', color='Red')
        self.front_camera = Camera(pos=(-0.3, 2, 1.0), direction=(0, 1, 0), fov=120, resolution=(128, 128), colour=True,
                                   annotation=False, depth=False)
        self.vehicle.attach_sensor("front camera", self.front_camera)

        self.scenario.add_vehicle(self.vehicle, pos=(-717, 101, 118), rot=(0, 0, 45))

        self.scenario.make(self.bng)
        self.run()

    def run(self):
        # Launch BeamNG.research
        self.bng.open()

        self.bng.set_deterministic()
        self.bng.set_steps_per_second(60)

        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()

        self.vehicle.ai_set_mode('span')
        self.vehicle.ai_drive_in_lane(True)
        self.vehicle.ai_set_speed(13.8)  # 50 km/h

        self.bng.hide_hud()
        self.bng.pause()

    def step(self, action):
        steering = action[0]
        throttle = action[1]
        brake = action[2]

        self.vehicle.control(steering=steering, throttle=throttle, brake=brake)
        self.bng.step(20)
        sensors = self.bng.poll_sensors(self.vehicle)
        camera_image = sensors['front_camera']['colour']
        state = image.imread(camera_image)
        print(state.shape)
        return state

        # camera_image.save('datasets\\camera\\' + str(i) + '.bmp', format='BMP')
        # annotated_image.save('datasets\\target\\' + str(i) + '.bmp', format='BMP')

    def _wait(self):
        thread = Thread(target=wait)
        thread.start()

    def done(self):
        return False

    def close(self):
        self.bng.close()
