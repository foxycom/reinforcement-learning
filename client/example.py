import os
from threading import Thread
from typing import List

from client.aiExchangeMessages_pb2 import SimulationID


def _handle_vehicle(sid: SimulationID, vid: str, requests: List[str]) -> None:
    vid_obj = VehicleID()
    vid_obj.vid = vid

    i = 0
    while i < 10:
        i += 1
        print(sid.sid + ": Test status: " + service.get_status(sid))
        print(vid + ": Wait")
        sim_state = service.wait_for_simulator_request(sid, vid_obj)  # wait()
        if sim_state is SimStateResponse.SimState.RUNNING:
            print(vid + ": Request data")
            request = DataRequest()
            request.request_ids.extend(requests)
            data = service.request_data(sid, vid_obj, request)  # request()
            print(data)
            print(vid + ": Wait for control")
            control = Control()
            while not is_pressed("space"):  # Wait for the user to trigger manual drive
                pass
            print(vid + ": Control")
            if is_pressed("s"):
                control.simCommand.command = Control.SimCommand.Command.SUCCEED
            elif is_pressed("f"):
                control.simCommand.command = Control.SimCommand.Command.FAIL
            elif is_pressed("c"):
                control.simCommand.command = Control.SimCommand.Command.CANCEL
            else:
                accelerate = 0
                steer = 0
                brake = 0
                if is_pressed("up"):
                    accelerate = 1
                if is_pressed("down"):
                    brake = 1
                if is_pressed("right"):
                    steer = steer + 1
                if is_pressed("left"):
                    steer = steer - 1
                control.avCommand.accelerate = accelerate
                control.avCommand.steer = steer
                control.avCommand.brake = brake
            service.control(sid, vid_obj, control)  # control()
        else:
            print(sid.sid + ": The simulation is not running anymore (State: "
                  + SimStateResponse.SimState.Name(sim_state) + ").")
            print(sid.sid + ": Final result: " + service.get_result(sid))
            break

    control = Control()
    control.simCommand.command = Control.SimCommand.Command.FAIL
    service.control(sid, vid_obj, control)


if __name__ == "__main__":
    from client.AIExchangeService import get_service
    from client.aiExchangeMessages_pb2 import SimStateResponse, Control, SimulationID, VehicleID, DataRequest
    from keyboard import is_pressed

    service = get_service()

    try:
        username = os.environ['DRIVEBUILD_USER']
        password = os.environ['DRIVEBUILD_PASSWORD']
    except KeyError:
        print("No user data found")
        exit(1)

    # Send tests
    sids = service.run_tests(username, password, "envs/criteriaA.dbc.xml", "envs/environmentA.dbe.xml")

    # Interact with a simulation
    if not sids:
        exit(1)
    sid = SimulationID()
    sid.sid = sids.sids[0]
    ego_requests = ["egoSpeed"]
    ego_vehicle = Thread(target=_handle_vehicle, args=(sid, "ego", ego_requests))
    ego_vehicle.start()
    ego_vehicle.join()
