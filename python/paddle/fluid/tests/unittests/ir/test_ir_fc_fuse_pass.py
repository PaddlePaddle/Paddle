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
import paddle.fluid as fluid
import paddle.fluid.core as core


class FCFusePassTest(PassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[32, 128], dtype="float32", lod_level=0
            )
            tmp_0 = fluid.layers.fc(
                input=data, size=128, num_flatten_dims=1, act="relu"
            )
            tmp_1 = fluid.layers.fc(input=tmp_0, size=32, num_flatten_dims=1)
            tmp_2 = paddle.nn.functional.softmax(tmp_1)

        self.feeds = {"data": np.random.random((32, 128)).astype("float32")}
        self.fetch_list = [tmp_0, tmp_1, tmp_2]
        self.pass_names = "fc_fuse_pass"
        self.fused_op_type = "fc"
        self.num_fused_ops = 2

    def test_check_output(self):
        use_gpu_set = [False]
        if core.is_compiled_with_cuda():
            use_gpu_set.append(True)
        for use_gpu in use_gpu_set:
            self.pass_attrs = {"fc_fuse_pass": {"use_gpu": use_gpu}}
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            self.check_output_with_place(place, startup_on_cpu=True)


if __name__ == "__main__":
    unittest.main()
