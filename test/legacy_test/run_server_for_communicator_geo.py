import sys
import os

import time
import threading
import subprocess
import unittest
import numpy

import paddle
import paddle.fluid as fluid

from paddle.distributed.communicator import Communicator
import paddle.incubate.distributed.fleet.role_maker as role_maker
from paddle.incubate.distributed.fleet.parameter_server.mode import DistributedMode
import paddle.distributed.fleet as fleet

from test_communicator_geo import TestCommunicatorGeoEnd2End

paddle.enable_static()

pipe_name = os.getenv("PIPE_FILE")

class RunServer(TestCommunicatorGeoEnd2End):
    def runTest(self):
        pass

os.environ["TRAINING_ROLE"] = "PSERVER"

half_run_server = RunServer()
with open(pipe_name, 'w') as pipe:
    pipe.write('done')
    
half_run_server.run_ut()
