#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from pass_test import PassTest

import paddle
from paddle import base
from paddle.base import core

paddle.enable_static()


class Conv2dBnFusePassTest(PassTest):
    def setUp(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        # main_program, start_program = (
        #     paddle.static.Program(),
        #     paddle.static.Program(),
        # )
        # with paddle.static.program_guard(main_program, start_program):
        #     x = paddle.static.data(
        #         name='x', shape=[3, 1, 28, 28], dtype='float32'
        #     )
        #     conv2d = paddle.nn.Conv2D(
        #         in_channels=1,
        #         out_channels=32,
        #         kernel_size=3,
        #         padding=1,
        #         data_format='NCHW',
        #         bias_attr=False,
        #     )
        #     bn = paddle.nn.BatchNorm2D(
        #         num_features=32, data_format='NCHW'
        #     )
        #     result1 = conv2d(x)
        #     result2 = bn(result1)

        with paddle.pir_utils.IrGuard():
            x = paddle.static.data(
                name='x', shape=[3, 1, 28, 28], dtype='float32'
            )
            conv2d = paddle.nn.Conv2D(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
                data_format='NCHW',
                bias_attr=False,
            )
            bn = paddle.nn.BatchNorm2D(num_features=32, data_format='NCHW')
            result1 = conv2d(x)
            result2 = bn(result1)

            executor = base.Executor(place)
            out = executor.run(
                feed={"x": np.random.random((3, 1, 28, 28)).astype("float32")},
                fetch_list=[result2],
            )
            # exit(0)
            breakpoint()

        self.feeds = {"x": np.random.random((3, 1, 28, 28)).astype("float32")}
        self.fetch_list = [result2]
        self.pass_names = "conv2d_bn_fuse"
        self.fused_op_type = "conv2d"
        # self.num_fused_ops = 2

    def test_check_output(self):
        use_gpu_set = [False]
        if core.is_compiled_with_cuda():
            use_gpu_set.append(True)
        for use_gpu in use_gpu_set:
            self.pass_attrs = {"conv2d_bn_fuse": {"use_gpu": use_gpu}}
            place = base.CUDAPlace(0) if use_gpu else base.CPUPlace()
            self.check_output_with_place(place, startup_on_cpu=True)


if __name__ == "__main__":
    unittest.main()
