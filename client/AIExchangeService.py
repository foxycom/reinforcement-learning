from http.client import HTTPResponse
from typing import Optional

from client.aiExchangeMessages_pb2 import VehicleID, SimulationID, SimulationIDs, TestResult, MaySimulationIDs


# FIXME Missing functions: Check status of simulations


class AIExchangeService:
    from client.aiExchangeMessages_pb2 import SimStateResponse, DataRequest, DataResponse, Control, Void

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    @staticmethod
    def _print_error(response: HTTPResponse) -> None:
        from client.common import eprint
        eprint("Response status: " + str(response.status))
        eprint("Reason: " + response.reason)
        eprint("Messsage:")
        eprint(b"\n".join(response.readlines()))

    def wait_for_simulator_request(self, sid: SimulationID, vid: VehicleID) -> SimStateResponse.SimState:
        """
        Waits for the simulation with ID sid to request the car with ID vid. This call blocks until the simulation
        requests the appropriate car in the given simulation.
        :param sid: The ID of the simulation the vehicle is included in.
        :param vid: The ID of the vehicle in the simulation to wait for.
        :return: The current state of the simulation at the point when the call to this function returns. The return
        value should be used to check whether the simulation is still running. Another vehicle or the even user may have
        stopped the simulation.
        """
        from client.aiExchangeMessages_pb2 import SimStateResponse
        from client.httpUtil import do_get_request
        response = do_get_request(self.host, self.port, "/ai/waitForSimulatorRequest", {
            "sid": sid.SerializeToString(),
            "vid": vid.SerializeToString()
        })
        if response.status == 200:
            result = b"".join(response.readlines())
            sim_state = SimStateResponse()
            sim_state.ParseFromString(result)
            return sim_state.state
        else:
            AIExchangeService._print_error(response)

    def request_data(self, sid: SimulationID, vid: VehicleID, request: DataRequest) -> DataResponse:
        """
        Request data of a certain vehicle contained by a certain simulation.
        :param sid: The ID of the simulation the vehicle to request data about is part of.
        :param vid: The ID of the vehicle to get collected data from.
        :param request: The types of data to be requested about the given vehicle. A DataRequest object is build like
        the following:
        request = DataRequest()
        request.request_ids.extend(["id_1", "id_2",..., "id_n"])
        NOTE: You have to use extend(...)! An assignment like request.request_ids = [...] will not work due to the
        implementation of Googles protobuffer.
        :return: The data the simulation collected about the given vehicle. The way of accessing the data is dependant
        on the type of data you requested. To find out how to access the data properly you should set a break point and
        checkout the content of the returned value using a debugger.
        """
        from client.aiExchangeMessages_pb2 import DataResponse
        from client.httpUtil import do_get_request
        response = do_get_request(self.host, self.port, "/ai/requestData", {
            "request": request.SerializeToString(),
            "sid": sid.SerializeToString(),
            "vid": vid.SerializeToString()
        })
        if response.status == 200:
            result = b"".join(response.readlines())
            data_response = DataResponse()
            data_response.ParseFromString(result)
            return data_response
        else:
            AIExchangeService._print_error(response)

    def control(self, sid: SimulationID, vid: VehicleID, commands: Control) -> Void:
        """
        Control the simulation or a certain vehicle in the simulation.
        :param sid: The ID the simulation either to control or containing th vehicle to control.
        :param vid: The ID of the vehicle to possibly control.
        :param commands: The command either controlling a simulation or a vehicle in a simualtion. To define a command
        controlling a simulation you can use commands like:
        control = Control()
        control.simCommand.command = Control.SimCommand.Command.SUCCEED  # Force simulation to succeed
        control.simCommand.command = Control.SimCommand.Command.FAIL  # Force simulation to fail
        control.simCommand.command = Control.SimCommand.Command.CANCEL  # Force simulation to be cancelled/skipped
        For controlling a vehicle you have to define steering, acceleration and brake values:
        control = Control()
        control.avCommand.accelerate = <Acceleration intensity having a value between 0.0 and 1.0>
        control.avCommand.steer = <A steering value between -1.0 and 1.0 (Negative value steers left; a positive one
        steers right)>
        control.avCommand.brake = <Brake intensity having a value between 0.0 and 1.0>
        :return: A Void object possibly containing a info message.
        """
        from client.httpUtil import do_mixed_request
        response = do_mixed_request(self.host, self.port, "/ai/control", {
            "sid": sid.SerializeToString(),
            "vid": vid.SerializeToString()
        }, commands.SerializeToString())
        if response.status == 200:
            print("Controlled car")  # FIXME What to do here?
        else:
            AIExchangeService._print_error(response)

    def control_sim(self, sid: SimulationID, result: TestResult) -> Void:
        """
        Force a simulation to end having the given result.
        :param sid: The simulation to control.
        :param result: The test result to be set to the simulation.
        :return: A Void object possibly containing a info message.
        """
        from client.httpUtil import do_get_request
        response = do_get_request(self.host, self.port, "/sim/stop", {
            "sid": sid.SerializeToString(),
            "result": result.SerializeToString()
        })
        if response.status == 200:
            pass  # FIXME What to do here?
        else:
            AIExchangeService._print_error(response)

    def get_status(self, sid: SimulationID) -> str:
        """
        Check the status of the given simulation.
        :param sid: The simulation to get the status of.
        :return: A string representing the status of the simulation like RUNNING, FINISHED or ERRORED.
        """
        from client.httpUtil import do_get_request
        response = do_get_request(self.host, self.port, "/stats/status", {
            "sid": sid.SerializeToString()
        })
        if response.status == 200:
            return b"".join(response.readlines()).decode()
        else:
            AIExchangeService._print_error(response)
            return "Status could not be determined."

    def get_result(self, sid: SimulationID) -> str:
        """
        Get the test result of the given simulation.
        :param sid: The simulation to get the test result of.
        :return: The current test result of the given simulation like UNKNOWN, SUCCEEDED, FAILED or CANCELLED.
        """
        from client.httpUtil import do_get_request
        response = do_get_request(self.host, self.port, "/stats/result", {
            "sid": sid.SerializeToString()
        })
        if response.status == 200:
            return b"".join(response.readlines()).decode()
        else:
            AIExchangeService._print_error(response)
            return "Result could not be determined."

    def run_tests(self, username: str, password: str, *paths: str) -> Optional[SimulationIDs]:
        """
        Upload the sequence of given files to DriveBuild, execute them and get their associated simulation IDs.
        :param username: The username for login.
        :param password: The password for login.
        :param paths: The sequence of file paths of files or folders containing files to be uploaded.
        :return: A sequence containing simulation IDs for all *valid* test cases uploaded. Returns None iff the upload
        of tests failed or the tests could not be run. Returns an empty list of none of the given test cases was valid.
        """
        from client.httpUtil import do_mixed_request
        from client.aiExchangeMessages_pb2 import User
        from tempfile import NamedTemporaryFile
        from zipfile import ZipFile
        from os import remove, listdir
        from os.path import abspath, basename, isfile, isdir, join
        from client.common import eprint
        temp_file = NamedTemporaryFile(mode="w", suffix=".zip", delete=False)
        temp_file.close()

        with ZipFile(temp_file.name, "w") as write_zip_file:
            def _add_all_files(path: str) -> None:
                if isfile(path):
                    abs_file_path = abspath(path)
                    filename = basename(abs_file_path)
                    write_zip_file.write(abs_file_path, filename)
                elif isdir(path):
                    for sub_path in listdir(path):
                        _add_all_files(join(path, sub_path))
                else:
                    eprint("Can not handle path \"" + path + "\".")
            for path in paths:
                _add_all_files(path)
        user = User()
        user.username = username
        user.password = password
        with open(temp_file.name, "rb") as read_zip_file:
            response = do_mixed_request(self.host, self.port, "/runTests", {
                "user": user.SerializeToString()
            }, b"".join(read_zip_file.readlines()))
        remove(temp_file.name)
        sids = MaySimulationIDs()
        sids.ParseFromString(b"".join(response.readlines()))
        self._print_error(response)
        if response.status == 200:
            return sids.sids
        else:
            eprint("Running tests errored:")
            eprint(sids.message.message)
            return None


def get_service() -> AIExchangeService:
    return AIExchangeService("defender.fim.uni-passau.de", 8383)
