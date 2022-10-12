# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import argparse
import os
import sys
import signal
import time
import socket
from contextlib import closing
from six import string_types
import math
import paddle
import paddle.distributed as dist
import paddle.fluid as fluid
import test_collective_api_base as test_base


class TestCollectiveSendRecvAPI(test_base.TestCollectiveAPIRunnerBase):

    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        with fluid.program_guard(main_prog, startup_program):
            # NOTE: this is a hack relying on an undocumented behavior that `to_tensor` uses uint16 to replace bfloat16
            if indata.dtype == "bfloat16":
                tindata = paddle.to_tensor(indata, "float32").cast("uint16")
                if rank == 0:
                    dist.send(tindata, dst=1)
                else:
                    dist.recv(tindata, src=0)
                return [tindata.cast("float32").numpy()]
            else:
                tindata = paddle.to_tensor(indata)
                if rank == 0:
                    dist.send(tindata, dst=1)
                else:
                    dist.recv(tindata, src=0)
                return [tindata.numpy()]


if __name__ == "__main__":
    test_base.runtime_main(TestCollectiveSendRecvAPI, "sendrecv")
