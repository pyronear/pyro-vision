# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.


from gpiozero import CPUTemperature
from time import sleep, strftime
import psutil


class MonitorPi:
    """This class aims to monitor some metrics from Raspberry Pi system.

    Example
    --------
    recordlogs = MonitorPi("/home/pi/Desktop/")
    recordlogs.record(5) # record metrics every 5 seconds
    
    """

    def __init__(self, monitoringFolder, logFile="pi_perf.csv"):

        # cpu temperature
        self.cpu_temp = CPUTemperature()

        # path
        self.monitoringFolder = monitoringFolder
        self.logFile = logFile

    def record(self, timeStep):
            
        with open(self.monitoringFolder + self.logFile, "a") as log:
            log.write("datetime;cpu_temperature_C;mem_available_GB;cpu_usage_percent\n")
                        
            while True:
                l = "{date};{t:.2f};{mem:.2f};{cpu_usage}\n".format(
                        date = strftime("%Y-%m-%d %H:%M:%S"),
                        t = self.cpu_temp.temperature,
                        mem = psutil.virtual_memory().available / 1024**3,
                        cpu_usage = psutil.cpu_percent())
                    
                log.write(l)
                
                sleep(timeStep)


if __name__ == "__main__":


    recordlogs = MonitorPi("/home/pi/Desktop/")
    recordlogs.record(30)