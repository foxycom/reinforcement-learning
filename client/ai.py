import os
from client.aiExchangeMessages_pb2 import SimulationID, VehicleID
from algos import DDPG

MODEL_PATH = os.path.join(os.getcwd(), "models/agent.pkl")
VAE_PATH = os.path.join(os.getcwd(), "models/vae.pkl")


class DDPGAI(object):

    def __init__(self):
        self.model = DDPG.load(MODEL_PATH)

    def start(self, sid: SimulationID, vid: VehicleID) -> None:
        from client.AIExchangeService import get_service
        from client.aiExchangeMessages_pb2 import SimStateResponse, DataRequest, Control
        service = get_service()
        i = 0
        while i < 10:
            i += 1
            print(sid.sid + ": Test status: " + service.get_status(sid))
            # Wait for the simulation to request this AI
            sim_state = service.wait_for_simulator_request(sid, vid)
            if sim_state is SimStateResponse.SimState.RUNNING:  # Check whether simulation is still running
                # Request data this AI needs
                request = DataRequest()
                request.request_ids.extend(["egoFrontCamera", "egoLaneDist"])  # Add all IDs of data to be requested
                data = service.request_data(sid, vid, request)  # Request the actual data
                print(data)
                # Calculate commands controlling the car
                control = Control()
                # Define a control command like
                # control.avCommand.accelerate = <Some value between 0.0 and 1.0>
                # control.avCommand.steer = <Some value between -1.0 (left) and 1.0 (right)
                # control.avCommand.brake = <Some value between 0.0 and 1.0>
                service.control(sid, vid, control)
            else:
                print(sid.sid + ": The simulation is not running anymore (Final state: "
                      + SimStateResponse.SimState.Name(sim_state) + ").")
                print(sid.sid + ": Final test result: " + service.get_result(sid))
                # Clean up everything you have to
                break

        control = Control()
        control.simCommand.command = Control.SimCommand.Command.FAIL
        service.control(sid, vid, control)


if __name__ == "__main__":
    ai = DDPGAI()
    sid = SimulationID()
    sid.sid = "snid_1_sim_4445"
    vid = VehicleID()
    vid.vid = "ego"
    ai.start(sid, vid)
