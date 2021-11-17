# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from contextlib import closing
from six import string_types
import math
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core
import unittest
from multiprocessing import Process
import paddle.fluid.layers as layers
from functools import reduce
from test_collective_base_npu import TestCollectiveRunnerBase, runtime_main

Initializer = fluid.initializer.NumpyArrayInitializer

paddle.enable_static()


class TestCollectiveConcat(TestCollectiveRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def run_model(self, main_prog, startup_program, dtype):
        nranks = 2
        return self.test_c_concat(4, 8, nranks, self.rank, dtype, main_prog,
                                  startup_program)

    def test_c_concat(self, rows, cols, nranks, rank, dtype, main_program,
                      startup_program):
        assert cols % nranks == 0, f"cols({cols}) must be divided by nanks({nranks})"
        block = main_program.global_block()

        total_init = np.random.random((rows, cols)).astype(dtype)
        init = total_init[:, rank * cols // nranks:(rank + 1) * cols // nranks]

        total_init_grads = np.random.random((rows, cols)).astype(dtype)
        init_grads = total_init_grads[:, rank * cols // nranks:(rank + 1) * cols
                                      // nranks]

        with fluid.program_guard(main_program, startup_program):
            data = paddle.nn.Linear(
                rows,
                cols // nranks,
                weight_attr=fluid.ParamAttr(
                    initializer=Initializer(init))).weight
            out = paddle.nn.Linear(
                rows, cols,
                weight_attr=fluid.ParamAttr(initializer=None)).weight
            out_grads = paddle.nn.Linear(
                rows,
                cols,
                weight_attr=fluid.ParamAttr(
                    initializer=Initializer(total_init_grads))).weight
            data = paddle.cast(data, dtype=dtype)
            out = paddle.cast(out, dtype=dtype)
            out_grads = paddle.cast(out_grads, dtype=dtype)

            block.append_op(
                type="c_concat",
                inputs={"X": [data]},
                outputs={"Out": [out]},
                attrs={"nranks": nranks,
                       "rank": rank,
                       "ring_id": 0})

        return (total_init, init_grads, data, out, out_grads)

    def run_trainer(self, args):
        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        nranks = 2
        self.rank = rank
        np.random.seed(2021)
        paddle.seed(os.getpid())
        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        (numetric_out, numetric_grads, in_data, actual_out,
         outgrads) = self.run_model(train_prog, startup_prog, "float32")
        (numetric_out_fp16, numetric_grads_fp16, in_data_fp16, actual_out_fp16,
         outgrads_fp16) = self.run_model(train_prog, startup_prog, "float32")

        input_grads = paddle.static.gradients([actual_out, actual_out_fp16],
                                              [in_data, in_data_fp16],
                                              [outgrads, outgrads_fp16])
        actual_grads = input_grads[0]
        actual_grads_fp16 = input_grads[1]

        self.initCommunicator(startup_prog, rank, nranks, True,
                              current_endpoint, endpoints)
        place = fluid.NPUPlace(self.rank)
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        output = exe.run(fetch_list=[
            actual_out, actual_grads, actual_out_fp16, actual_grads_fp16
        ],
                         program=train_prog)

        diff_out = np.abs(output[0] - numetric_out).mean()
        diff_grads = np.abs(output[1] - numetric_grads).mean()
        print(f"diff outs: {diff_out} diff grads: {diff_grads}")
        thresh = 1e-7
        assert diff_out < thresh, f"{diff_out} vs {thresh}"
        assert diff_grads < thresh, f"{diff_grads} vs {thresh}"

        diff_out_fp16 = np.abs(output[2] - numetric_out_fp16).mean()
        diff_grads_fp16 = np.abs(output[3] - numetric_grads_fp16).mean()
        print(f"diff outs fp16: {diff_out} diff grads fp16: {diff_grads}")
        thresh = 1e-3
        assert diff_out < thresh, f"{diff_out_fp16} vs {thresh}"
        assert diff_grads < thresh, f"{diff_grads_fp16} vs {thresh}"


if __name__ == "__main__":
    runtime_main(TestCollectiveConcat, "concat", 0)
