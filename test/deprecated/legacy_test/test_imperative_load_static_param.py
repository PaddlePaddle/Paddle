# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import tempfile
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.nn import BatchNorm, Linear
from paddle.pir_utils import IrGuard

paddle.enable_static()


class TestDygraphLoadStatic(unittest.TestCase):
    def testLoadStaticModel(self):
        with IrGuard():
            # static graph in pir mode
            temp_dir = tempfile.TemporaryDirectory()
            a = paddle.static.data(name="a", shape=[10, 10])
            conv_in = paddle.static.data(
                name="conv_in", shape=[None, 10, 10, 10]
            )

            fc_out1 = paddle.static.nn.fc(a, 10)
            fc_out2 = paddle.static.nn.fc(a, 20)

            conv1 = paddle.nn.Conv2D(
                in_channels=10, out_channels=10, kernel_size=5
            )
            conv_out_1 = conv1(conv_in)
            conv2 = paddle.nn.Conv2D(
                in_channels=10, out_channels=10, kernel_size=5
            )
            conv_out_2 = conv2(conv_in)

            conv3d_in = paddle.static.data(
                name='conv3d_in', shape=[None, 3, 12, 32, 32], dtype='float32'
            )
            conv3d_1 = paddle.nn.Conv3D(
                in_channels=3, out_channels=2, kernel_size=3
            )
            conv3d_out_1 = conv3d_1(conv3d_in)
            conv3d_2 = paddle.nn.Conv3D(
                in_channels=3, out_channels=2, kernel_size=3
            )
            conv3d_out_2 = conv3d_2(conv3d_in)

            batchnorm_in = paddle.static.data(
                name="batchnorm_in", shape=[None, 10], dtype='float32'
            )
            batchnorm_out_1 = paddle.nn.BatchNorm(10)(batchnorm_in)
            batchnorm_out_2 = paddle.nn.BatchNorm(10)(batchnorm_in)

            emb_in = paddle.static.data(
                name='emb_in', shape=[None, 10], dtype='int64'
            )
            emb1 = paddle.nn.Embedding(1000, 100)
            emb_out_1 = emb1(emb_in)
            emb2 = paddle.nn.Embedding(2000, 200)
            emb_out_2 = emb2(emb_in)

            layernorm = paddle.static.data(
                name="ln", shape=[None, 10], dtype='float32'
            )
            layernorm_1 = paddle.nn.LayerNorm([10])(layernorm)
            layernorm_2 = paddle.nn.LayerNorm(10)(layernorm)

            groupnorm_in = paddle.static.data(
                name='groupnorm_in', shape=[None, 8, 32, 32], dtype='float32'
            )
            groupnorm_out1 = paddle.nn.GroupNorm(4, 8)(groupnorm_in)
            groupnorm_out2 = paddle.nn.GroupNorm(4, 8)(groupnorm_in)

            para1 = paddle.create_parameter(
                [100, 100], 'float32', name="weight_test_1"
            )
            para2 = paddle.create_parameter(
                [20, 200], 'float32', name="weight_test_2"
            )

            exe = base.Executor(
                base.CPUPlace()
                if not base.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            exe.run(paddle.static.default_startup_program())

            paddle.static.save(
                paddle.static.default_main_program(),
                os.path.join(temp_dir.name, "test_1"),
            )

            para_dict = paddle.static.load_program_state(
                os.path.join(temp_dir.name, "test_1")
            )

            new_dict = {}
            for k, v in para_dict.items():
                if k.startswith("fc"):
                    name = k.replace("fc", "linear", 1)
                    new_dict[name] = v
                else:
                    new_dict[k] = v

        with base.dygraph.guard():

            class MyTest(paddle.nn.Layer):
                def __init__(self):
                    super().__init__()

                    self.linear1 = Linear(10, 10)
                    self.lienar2 = Linear(10, 20)

                    self.conv2d_1 = paddle.nn.Conv2D(
                        in_channels=10, out_channels=10, kernel_size=5
                    )
                    self.conv2d_2 = paddle.nn.Conv2D(
                        in_channels=10, out_channels=10, kernel_size=5
                    )

                    self.conv3d_1 = paddle.nn.Conv3D(
                        in_channels=3, out_channels=2, kernel_size=3
                    )
                    self.conv3d_2 = paddle.nn.Conv3D(
                        in_channels=3, out_channels=2, kernel_size=3
                    )

                    self.batch_norm_1 = BatchNorm(10)
                    self.batch_norm_2 = BatchNorm(10)

                    self.emb1 = paddle.nn.Embedding(1000, 100)
                    self.emb2 = paddle.nn.Embedding(2000, 200)

                    self.layer_norm_1 = paddle.nn.LayerNorm([10])
                    self.layer_norm_2 = paddle.nn.LayerNorm(10)

                    self.group_norm1 = paddle.nn.GroupNorm(4, 8)
                    self.gourp_norm2 = paddle.nn.GroupNorm(4, 8)

                    self.w_1 = self.create_parameter(
                        [100, 100], dtype='float32', attr="weight_test_1"
                    )
                    self.w_2 = self.create_parameter(
                        [20, 200], dtype='float32', attr="weight_test_2"
                    )

            my_test = MyTest()
            my_test.set_dict(new_dict, use_structured_name=False)
            for k, v in my_test.state_dict().items():
                np.testing.assert_array_equal(v.numpy(), new_dict[v.name])
        temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
