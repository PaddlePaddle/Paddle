# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core
import paddle.distributed.fleet as fleet
from paddle.fluid.incubate.fleet.base import role_maker
import unittest
from multiprocessing import Process
import paddle.fluid.layers as layers
from functools import reduce
from test_collective_multi_nodes import TestCollectiveAPIRunnerBase, runtime_main
from paddle import nn
import numpy as np


def weight_init(mp, shape, col=True, seed=1024):
    np.random.seed(seed)
    w = np.random.normal(0, 0.02, size=shape)
    if mp is None:
        _w = w
    else:
        if col:
            step = shape[1] // mp.nranks
            _w = w[:, mp.rank * step:mp.rank * step + step]
        else:
            step = shape[0] // mp.nranks
            _w = w[mp.rank * step:mp.rank * step + step, :]
    return paddle.fluid.initializer.NumpyArrayInitializer(_w)


class Model(fleet.meta_parallel.PipelineLayer):

    def __init__(self, hcg):
        # super(Model, self).__init__()
        paddle.seed(1024)
        dp_linear = nn.Linear(32, 128)
        self.layers_pp = []
        self.topology = hcg.topology()
        self.layers_pp.append(dp_linear)
        mp = hcg.get_model_parallel_group()
        for i in range(6):
            if mp is not None and mp.nranks > 1:
                mp_linear_1 = fleet.meta_parallel.ColumnParallelLinear(
                    128,
                    512,
                    weight_attr=weight_init(mp, (128, 512), True, 1204 + i))
                mp_linear_2 = fleet.meta_parallel.RowParallelLinear(
                    512,
                    128,
                    weight_attr=weight_init(mp, (512, 128), False, 2012 + i))
            else:
                mp_linear_1 = nn.Linear(128,
                                        512,
                                        weight_attr=weight_init(
                                            None, (128, 512), True, 1204 + i))
                mp_linear_2 = nn.Linear(512,
                                        128,
                                        weight_attr=weight_init(
                                            None, (512, 128), True, 2012 + i))
            act = nn.ReLU6()
            layer_seq = nn.Sequential(mp_linear_1, mp_linear_2, act)
            self.layers_pp.append(layer_seq)

        out = nn.Linear(128, 32)
        self.layers_pp.append(out)
        # self.layers = fleet.meta_parallel.parallel_layers.PipelineLayer(layers_pp, num_stages=2)
        super(Model, self).__init__(layers=self.layers_pp,
                                    topology=self.topology)


class TestDygrapgHybridDPPPMP(TestCollectiveAPIRunnerBase):

    def __init__(self):
        pass

    def check_pass(self, *args, **kwargs):
        from common import init_parallel_env
        import paddle
        from paddle.distributed import fleet
        hcg = init_parallel_env("DP2-MP2-PP2-SH1-O1", 32)
        import numpy as np

        model1 = Model(hcg)
        # model3 = Model()
        model2 = Model(hcg)
        print("======= model 1 =========", file=sys.stderr)
        print(model1, file=sys.stderr)
        print("======= model 2 =========")
        print(model2)
        x = paddle.to_tensor(np.random.random((16, 32))).astype("float32")

        # model2 = fleet.meta_parallel.parallel_layers.PipelineLayer(model2, num_stages=2)

        # model1 = fleet.distributed_model(model1)
        model2 = fleet.distributed_model(model2)

        out1 = model1(x)
        out2 = model2(x)
        assert np.allclose(out1.numpy(), out2.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    runtime_main(TestDygrapgHybridDPPPMP, "dpppmp")
