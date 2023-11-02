import time


class Timer():
    def __init__(self):
        # following v is to calculate computing time;
        self.time_sum = 0.0

        self.time_start = 0.0

    def write_time(self, info):
        with open('time_info.txt', 'a') as f:
            f.write("Time of " + info + ":")
            f.write(str(self.time_sum))

    def start(self):
        self.time_start = time.time()

    def end(self):
        self.time_sum += time.time() - self.time_start




