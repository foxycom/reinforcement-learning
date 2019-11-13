import os

from ai import DDPGAI

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
sids = service.run_tests(username, password, "client/xmls/criteriaA.dbc.xml", "client/xmls/environmentA.dbe.xml")

# Interact with a simulation
if not sids:
    exit(1)

vid = VehicleID()
vid.vid = "ego"

sid = SimulationID()
sid.sid = sids.sids[0]


ai = DDPGAI(service)

ai.start(sid, vid)
