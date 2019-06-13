from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from threading import Thread
from time import sleep
from matplotlib import image, pyplot
from PIL import Image
import numpy as np


def wait():
    sleep(1000000)


class Simulation(object):

    def __init__(self) -> None:
        super().__init__()

        self._wait()
        self.intervene = False
        self._start_monitoring_interventions()

        # Instantiate BeamNGpy instance running the simulator from the given path,
        # communicating over localhost:64256
        self.bng = BeamNGpy('localhost', 64256, home='C:\\Users\\Tim\\Documents\\research\\trunk')

        self.scenario = Scenario('west_coast_usa', 'Reinforcement learning')
        self.vehicle = Vehicle('ego_vehicle', model='etk800', licence='RL FTW', color='Red')
        self.front_camera = Camera(pos=(-0.3, 2, 1.0), direction=(0, 1, 0), fov=120, resolution=(64, 64),
                                   colour=True, depth=False, annotation=False)
        self.vehicle.attach_sensor("front_camera", self.front_camera)

        self.scenario.add_vehicle(self.vehicle, pos=(-829.6900634765625, -2.620422840118408, 117.47492980957031),
                                  rot=(0, 0, 45))

        self.scenario.make(self.bng)
        self.run()

    def run(self):
        # Launch BeamNG.research
        self.bng.open()

        self.bng.set_deterministic()
        self.bng.set_steps_per_second(120)

        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()

        self.vehicle.ai_set_mode('disabled')
        #self.vehicle.ai_drive_in_lane(True)
        #self.vehicle.ai_set_speed(13.8)  # 50 km/h

        self.bng.hide_hud()
        self.bng.pause()

    def step(self, action):
        steering = action[0].item()
        throttle = action[1].item()
        brake = -1.0

        # Translate [-1.0, 1.0] range into [0.0, 1.0] range
        throttle = (((throttle - (-1.0)) * 1.0) / 2.0)
        brake = (((brake - (-1.0)) * 1.0) / 2.0)

        print("Steering: ", steering, " Throttle: ", throttle, " Break: ", brake)

        self.vehicle.control(steering=steering, throttle=throttle, brake=brake)
        self.bng.step(6)
        sensors = self.bng.poll_sensors(self.vehicle)
        camera_image = sensors['front_camera']['colour']
        state = np.array(camera_image)[:, :, 0:3]
        return state

        # camera_image.save('datasets\\camera\\' + str(i) + '.bmp', format='BMP')
        # annotated_image.save('datasets\\target\\' + str(i) + '.bmp', format='BMP')

    def _wait(self):
        thread = Thread(target=wait)
        thread.start()

    def _start_monitoring_interventions(self):
        thread = Thread(target=self._monitor)
        thread.start()

    def _monitor(self):
        while True:
            stop = input()
            self.intervene = True

    def done(self):
        done = self.intervene
        self.intervene = False
        return done

    def reset(self):
        #self.bng.teleport_vehicle(self.vehicle, pos=(-717, 101, 119), rot=(0, 0, 45))
        self.vehicle.control(throttle=0, brake=0, steering=0)
        self.bng.restart_scenario()
        self.bng.step(30)
        self.bng.pause()

    def position(self):
        return self.vehicle.state["pos"]

    def direction(self):
        return self.vehicle.state["dir"]

    def velocity(self):
        return self.vehicle.state["vel"]

    def close(self):
        self.bng.close()
