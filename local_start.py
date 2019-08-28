from ai import DDPGAILocal
from client.aiExchangeMessages_pb2 import SimulationID, VehicleID

ai = DDPGAILocal()

sid = SimulationID()
sid.sid = "some simulation id"

vid = VehicleID()
vid.vid = "ego"

ai.start(sid, vid)
