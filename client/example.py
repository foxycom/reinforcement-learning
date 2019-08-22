import os
import io

from threading import Thread
from typing import List
from PIL import Image

from client.aiExchangeMessages_pb2 import SimulationID, TestResult, VehicleID


def _handle_vehicle(sid: SimulationID, vid: VehicleID, requests: List[str]) -> None:

    i = 0
    while i < 3:
        i += 1
        print(sid.sid + ": Test status: " + service.get_status(sid))
        print(vid.vid + ": Wait")
        #sim_state = service.wait_for_simulator_request(sid, vid)  # wait()
        #if sim_state is SimStateResponse.SimState.RUNNING:
        print(vid.vid + ": Request data")
        request = DataRequest()
        request.request_ids.extend(requests)
        data = service.request_data(sid, vid, request)  # request()
        byte_im = data.data['egoFrontCamera'].camera.color
        image = Image.open(io.BytesIO(byte_im))
        image.convert("RGB")
        #image.save("imgs/{}.png".format(i), "PNG")
        print(vid.vid + ": Wait for control")
        control = Control()
        control.avCommand.accelerate = 1
        service.control(sid, vid, control)
        #else:
        #    print(sid.sid + ": The simulation is not running anymore (State: "
        #          + SimStateResponse.SimState.Name(sim_state) + ").")
        #    print(sid.sid + ": Final result: " + service.get_result(sid))
        #    break

    sim_state = service.wait_for_simulator_request(sid, vid)  # wait()
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

    vid = VehicleID()
    vid.vid = "ego"

    sid = SimulationID()
    sid.sid = sids.sids[0]

    ego_requests = ["egoFrontCamera"]
    ego_vehicle = Thread(target=_handle_vehicle, args=(sid, vid, ego_requests))
    ego_vehicle.start()
    ego_vehicle.join()
