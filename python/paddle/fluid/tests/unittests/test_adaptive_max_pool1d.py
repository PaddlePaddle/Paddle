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

import numpy as np
import unittest
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard
import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


def max_pool1D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False,
                             exclusive=False,
                             adaptive=False,
                             data_type=np.float64):
    N, C, L = x.shape
    if global_pool == 1:
        ksize = [L]
    if adaptive:
        L_out = ksize[0]
    else:
        L_out = (L - ksize[0] + 2 * paddings[0] + strides[0] - 1
                 ) // strides[0] + 1 if ceil_mode else (
                     L - ksize[0] + 2 * paddings[0]) // strides[0] + 1

    out = np.zeros((N, C, L_out))
    for i in range(L_out):
        if adaptive:
            r_start = adaptive_start_index(i, L, ksize[0])
            r_end = adaptive_end_index(i, L, ksize[0])
        else:
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], L))
        x_masked = x[:, :, r_start:r_end]

        out[:, :, i] = np.max(x_masked, axis=(2))
    return out


class TestPool1D_API(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_adaptive_max_dygraph_results(self, place):
        with fluid.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = fluid.dygraph.to_variable(input_np)
            result = F.adaptive_max_pool1d(input, output_size=16)

            result_np = max_pool1D_forward_naive(
                input_np, ksize=[16], strides=[0], paddings=[0], adaptive=True)
            self.assertTrue(np.allclose(result.numpy(), result_np))

            ada_max_pool1d_dg = paddle.nn.layer.AdaptiveMaxPool1D(
                output_size=16)
            result = ada_max_pool1d_dg(input)
            self.assertTrue(np.allclose(result.numpy(), result_np))

    def check_adaptive_max_static_results(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[2, 3, 32], dtype="float32")
            result = F.adaptive_max_pool1d(input, output_size=16)

            input_np = np.random.random([2, 3, 32]).astype("float32")
            result_np = max_pool1D_forward_naive(
                input_np, ksize=[16], strides=[2], paddings=[0], adaptive=True)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input": input_np},
                              fetch_list=[result])
            self.assertTrue(np.allclose(fetches[0], result_np))

    def test_adaptive_max_pool1d(self):
        for place in self.places:
            self.check_adaptive_max_dygraph_results(place)
            self.check_adaptive_max_static_results(place)


if __name__ == '__main__':
    unittest.main()
