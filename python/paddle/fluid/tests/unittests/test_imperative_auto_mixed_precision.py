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
import paddle.fluid as fluid
import numpy as np


class SimpleConv(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(SimpleConv, self).__init__()
        self._conv = fluid.dygraph.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=None,
            use_cudnn=True)

    def forward(self, inputs):
        return self._conv(inputs)


class TestAutoCast(unittest.TestCase):
    def reader_decorator(self, reader):
        def _reader_imple():
            for item in reader():
                doc = np.array(item[0]).reshape(3, 224, 224)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield doc, label

        return _reader_imple

    def test_amp_guard_white_op(self):
        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with fluid.dygraph.guard():
            conv2d = fluid.dygraph.Conv2D(3, 2, 3, bias_attr=False, act=None)
            data = fluid.dygraph.to_variable(data)
            with fluid.dygraph.amp_guard(True):
                out_fp16 = conv2d(data)

            with fluid.dygraph.amp_guard(False):
                out_fp32 = conv2d(data)

        self.assertTrue(data.dtype == fluid.core.VarDesc.VarType.FP32)
        self.assertTrue(out_fp16.dtype == fluid.core.VarDesc.VarType.FP16)
        self.assertTrue(out_fp32.dtype == fluid.core.VarDesc.VarType.FP32)

    def test_amp_guard_black_op(self):
        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with fluid.dygraph.guard():
            data = fluid.dygraph.to_variable(data)
            with fluid.dygraph.amp_guard(True):
                out_fp32 = fluid.layers.mean(data)

        self.assertTrue(data.dtype == fluid.core.VarDesc.VarType.FP32)
        self.assertTrue(out_fp32.dtype == fluid.core.VarDesc.VarType.FP32)

    def test_custom_op_list(self):
        with fluid.dygraph.guard():
            tracer = fluid.framework._dygraph_tracer()
            base_white_list = fluid.dygraph.amp.auto_cast.WHITE_LIST
            base_black_list = fluid.dygraph.amp.auto_cast.BLACK_LIST
            print(tracer._get_amp_op_list())
            with fluid.dygraph.amp_guard(
                    custom_white_list=["log"], custom_black_list=["conv2d"]):
                white_list, black_list = tracer._get_amp_op_list()
                self.assertTrue(
                    set(white_list) ==
                    (set(base_white_list) | {"log"}) - {"conv2d"})

                self.assertTrue(
                    set(black_list) ==
                    (set(base_black_list) - {"log"}) | {"conv2d"})

    def test_custom_op_list_exception(self):
        inp_np = np.random.random(size=[1, 3, 128, 128]).astype(np.float32)

        def func():
            with fluid.dygraph.guard():
                model = SimpleConv(
                    num_channels=3,
                    num_filters=64,
                    filter_size=7,
                    stride=2,
                    act='relu')

                with fluid.dygraph.amp_guard(
                        custom_white_list=["conv2d"],
                        custom_black_list=["conv2d"]):
                    inp = fluid.dygraph.to_variable(inp_np)
                    out = model(inp)

        self.assertRaises(ValueError, func)


if __name__ == '__main__':
    unittest.main()
