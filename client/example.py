import os
from threading import Thread
from typing import List

from client.aiExchangeMessages_pb2 import SimulationID, TestResult


def _handle_vehicle(sid: SimulationID, vid: str, requests: List[str]) -> None:
    vid_obj = VehicleID()
    vid_obj.vid = vid

    i = 0
    while i < 3:
        i += 1
        print(sid.sid + ": Test status: " + service.get_status(sid))
        print(vid + ": Wait")
        sim_state = service.wait_for_simulator_request(sid, vid_obj)  # wait()
        if sim_state is SimStateResponse.SimState.RUNNING:
            print(vid + ": Request data")
            request = DataRequest()
            request.request_ids.extend(requests)
            data = service.request_data(sid, vid_obj, request)  # request()
            print(data.data['egoFrontCamera'])
            print(vid + ": Wait for control")
            control = Control()
            control.avCommand.accelerate = 1
            service.control(sid, vid_obj, control)
        else:
            print(sid.sid + ": The simulation is not running anymore (State: "
                  + SimStateResponse.SimState.Name(sim_state) + ").")
            print(sid.sid + ": Final result: " + service.get_result(sid))
            break

    sim_state = service.wait_for_simulator_request(sid, vid_obj)  # wait()
    if sim_state is SimStateResponse.SimState.RUNNING:
        result = TestResult()
        result.result = TestResult.Result.FAILED
        service.control_sim(sid, result)


if __name__ == "__main__":
    from client.AIExchangeService import get_service
    from client.aiExchangeMessages_pb2 import SimStateResponse, Control, SimulationID, VehicleID, DataRequest

    service = get_service()

    try:
        username = os.environ['DRIVEBUILD_USER']
        password = os.environ['DRIVEBUILD_PASSWORD']
    except KeyError:
        print("No user data found")
        exit(1)

    # Send tests
    sids = service.run_tests(username, password, "xmls/criteriaA.dbc.xml", "xmls/environmentA.dbe.xml")
    # -> Response status: 500

    print("Tests sent")

    # Interact with a simulation
    if not sids:
        exit(1)

    sid = SimulationID()
    sid.sid = sids.sids[0]
    ego_requests = ["egoSpeed", "egoFrontCamera"]
    ego_vehicle = Thread(target=_handle_vehicle, args=(sid, "ego", ego_requests))
    ego_vehicle.start()
    ego_vehicle.join()
