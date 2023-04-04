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

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import fluid, nn
from paddle.fluid import core
from paddle.nn import BatchNorm

np.random.seed(2023)
place = (
    fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
)


class PrimeNet(paddle.nn.Layer):
    def __init__(self, data_layout='NCHW'):
        super().__init__()
        self.conv = nn.Conv2D(2, 4, (3, 3), bias_attr=False)
        self.bn = BatchNorm(4, act="relu", data_layout=data_layout)

    def forward(self, x):
        y = self.conv(x)
        out = self.bn(y)
        res = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        return res


def apply_to_static(net):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = False
    return paddle.jit.to_static(net, build_strategy=False)


# This test case for checking dy2st + eval + AMP(O1) under prim
class TestCompositeDropout_AMP_O1(unittest.TestCase):
    with fluid.dygraph.guard(place):
        x = paddle.randn([4, 2, 6, 6], dtype="float32")
        x.stop_gradient = False
        net = PrimeNet(data_layout="NCHW")
        core._set_prim_all_enabled(False)
        net.eval()
        static_net = apply_to_static(net)
        res = static_net(x)

        # set prim all enabled
        core._set_prim_all_enabled(True)
        net.eval()
        static_net = apply_to_static(net)
        with paddle.amp.auto_cast(level='O1'):
            res_amp = static_net(x)

    np.testing.assert_allclose(
        res,
        res_amp,
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == '__main__':
    unittest.main()
