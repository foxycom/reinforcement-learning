# Original author: Tawn Kramer
import time


class FPSTimer(object):
    """
    Helper function to monitor the speed of the control.
    :param verbose: (int)
    """
    def __init__(self, verbose=0):
        self.start_time = time.time()
        self.iter = 0
        self.verbose = verbose

    def reset(self):
        self.start_time = time.time()
        self.iter = 0

    def on_frame(self):
        self.iter += 1
        if self.iter == 100:
            end_time = time.time()
            if self.verbose >= 1:
                print('{:.2f} fps'.format(100.0 / (end_time - self.start_time)))
            self.start_time = time.time()
            self.iter = 0
