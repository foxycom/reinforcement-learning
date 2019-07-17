# Original author: Felix Yu

import subprocess
import os


class DonkeyUnityProcess(object):
    """
    Utility class to start unity process if needed.
    """
    def __init__(self):
        self.process = None

    def start(self, sim_path, headless=False, port=9090):
        """
        :param sim_path: (str) Path to the executable
        :param headless: (bool)
        :param port: (int)
        """
        if not os.path.exists(sim_path):
            print(sim_path, "does not exist. not starting sim.")
            return

        port_args = ["--port", str(port), '-logFile', 'unitylog.txt']

        # Launch Unity environment
        if headless:
            self.process = subprocess.Popen(
                [sim_path, '-nographics', '-batchmode'] + port_args)
        else:
            self.process = subprocess.Popen(
                [sim_path] + port_args)

        print("Donkey subprocess started")

    def quit(self):
        """
        Shutdown unity environment
        """
        if self.process is not None:
            print("Closing donkey sim subprocess")
            self.process.kill()
            self.process = None
