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
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(
                self.main_program, self.startup_program
            ):
                x = paddle.static.data(
                    name='x', shape=[3, 1, 28, 28], dtype='float32'
                )
                bn = x
                # conv1_1 = paddle.static.nn.conv2d(
                #     input=x,
                #     filter_size=3,
                #     num_filters=32,
                #     stride=1,
                #     padding=1,
                #     act=None,
                #     bias_attr=False,
                #     data_format='NHWC',
                # )
                # bn = paddle.static.nn.batch_norm(
                #     input=conv1_1, act=None, data_layout='NHWC'
                # )

        self.feeds = {"x": np.random.random((3, 1, 28, 28)).astype("float32")}
        self.fetch_list = [bn]
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
