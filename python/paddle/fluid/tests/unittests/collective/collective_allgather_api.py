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

<<<<<<< HEAD
import os
import pickle
import sys

import test_collective_api_base as test_base

import paddle
import paddle.fluid as fluid
=======
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
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core
import unittest
import pickle
from multiprocessing import Process
import paddle.fluid.layers as layers
from functools import reduce
import test_collective_api_base as test_base
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class TestCollectiveAllgatherAPI(test_base.TestCollectiveAPIRunnerBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, dtype=None):
        dtype = "float32" if dtype is None else dtype
        with fluid.program_guard(main_prog, startup_program):
            tensor_list = []
<<<<<<< HEAD
            tindata = paddle.static.data(
                name="tindata", shape=[-1, 10, 1000], dtype=dtype
            )
=======
            tindata = layers.data(name="tindata", shape=[10, 1000], dtype=dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            paddle.distributed.all_gather(tensor_list, tindata)
            return tensor_list

    def run_trainer(self, args):
        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        nranks = 2
        paddle.distributed.init_parallel_env()
        if args['backend'] == 'nccl':
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = fluid.CUDAPlace(
<<<<<<< HEAD
                device_id
            )  # if args.use_gpu else fluid.CPUPlace()
=======
                device_id)  #if args.use_gpu else fluid.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        elif args['backend'] == 'bkcl':
            device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
            place = fluid.XPUPlace(device_id)
        else:
            place = fluid.CPUPlace()
<<<<<<< HEAD
        indata = test_base.create_test_data(
            shape=(10, 1000), dtype=args["dtype"], seed=os.getpid()
        )
        assert (
            args['static_mode'] == 1
        ), "collective_allgather_api only support static graph mode"
        result = self.get_model(
            train_prog, startup_prog, rank, dtype=args["dtype"]
        )
=======
        indata = test_base.create_test_data(shape=(10, 1000),
                                            dtype=args["dtype"],
                                            seed=os.getpid())
        assert args[
            'static_mode'] == 1, "collective_allgather_api only support static mode"
        result = self.get_model(train_prog,
                                startup_prog,
                                rank,
                                dtype=args["dtype"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        fetch_list = []
        for elem in result:
            fetch_list.append(elem.name)
<<<<<<< HEAD
        out = exe.run(
            train_prog, feed={'tindata': indata}, fetch_list=fetch_list
        )
=======
        out = exe.run(train_prog,
                      feed={'tindata': indata},
                      fetch_list=fetch_list)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        sys.stdout.buffer.write(pickle.dumps(out))


if __name__ == "__main__":
    test_base.runtime_main(TestCollectiveAllgatherAPI, "allgather")
