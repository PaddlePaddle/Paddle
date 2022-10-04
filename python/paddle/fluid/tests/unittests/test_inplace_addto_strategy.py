#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.backward import calc_gradient
import numpy as np


class ConvBNLayer(fluid.Layer):

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 data_format="NCHW"):
        super(ConvBNLayer, self).__init__()

        self._conv = paddle.nn.Conv2D(in_channels=num_channels,
                                      out_channels=num_filters,
                                      kernel_size=filter_size,
                                      stride=stride,
                                      padding=(filter_size - 1) // 2,
                                      groups=groups,
                                      bias_attr=False,
                                      data_format=data_format)

        self._batch_norm = paddle.nn.BatchNorm(num_filters,
                                               data_layout=data_format)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


def create_program(data_format="NCHW"):
    main = fluid.Program()
    startup = fluid.Program()
    with fluid.program_guard(main, startup):
        x = fluid.data(name='img', shape=[-1, 3, 224, 224])
        x.stop_gradient = False
        if data_format == "NHWC":
            x = paddle.transpose(x, [0, 2, 3, 1])
        x = fluid.layers.prelu(x, mode="channel")
        conv = ConvBNLayer(num_channels=3,
                           num_filters=3,
                           filter_size=1,
                           data_format=data_format)
        y = conv(x) + x

        loss = fluid.layers.reduce_sum(y)

        sgd = fluid.optimizer.SGD(learning_rate=0.01)
        sgd.minimize(loss)

    return loss, main, startup, conv._conv.weight


class TestInplaceAddto(unittest.TestCase):

    def check_result(self, data_format="NCHW"):

        def run_program(enable_addto):
            np.random.seed(10)
            paddle.seed(10)
            paddle.framework.random._manual_program_seed(10)
            if fluid.core.is_compiled_with_cuda():
                fluid.set_flags({"FLAGS_cudnn_deterministic": True})
            fluid.set_flags({"FLAGS_max_inplace_grad_add": 2})
            loss, main, startup, w = create_program(data_format=data_format)
            place = fluid.CUDAPlace(
                0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
            exe = fluid.Executor(place)

            strategy = fluid.BuildStrategy()
            strategy.enable_addto = enable_addto
            compiled = fluid.CompiledProgram(main).with_data_parallel(
                loss_name=loss.name, build_strategy=strategy)

            exe.run(startup)
            img = np.random.uniform(-128, 128,
                                    [8, 3, 224, 224]).astype(np.float32)
            for i in range(10):
                res = exe.run(compiled,
                              feed={'img': img},
                              fetch_list=[loss.name, w.name])
            return res

        res1, w1 = run_program(True)
        res2, w2 = run_program(False)

        np.testing.assert_array_equal(res1, res2)

    def test_nchw(self):
        self.check_result()

    def test_nhwc(self):
        self.check_result("NHWC")


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
