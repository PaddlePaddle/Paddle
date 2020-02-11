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

import numpy as np
from pass_test import PassTest
import paddle.fluid as fluid
import paddle.fluid.core as core


class FusionGroupPassTest(PassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data1 = fluid.data(name="data1", shape=[32, 128], dtype="float32")
            data2 = fluid.data(name="data2", shape=[32, 128], dtype="float32")
            data3 = fluid.data(name="data3", shape=[32, 128], dtype="float32")
            tmp_1 = fluid.layers.elementwise_add(data1, data2)
            tmp_2 = fluid.layers.elementwise_mul(data3, tmp_1)

        self.feeds = {
            "data1": np.random.random((32, 128)).astype("float32"),
            "data2": np.random.random((32, 128)).astype("float32"),
            "data3": np.random.random((32, 128)).astype("float32")
        }
        self.fetch_list = [tmp_1, tmp_2]
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"
        self.num_fused_ops = 1

    def test_check_output(self):
        use_gpu_set = [True]
        if core.is_compiled_with_cuda():
            use_gpu_set.append(True)
        for use_gpu in use_gpu_set:
            self.pass_attrs = {"fusion_group_pass": {"use_gpu": use_gpu}}
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            self.check_output_with_place(place, startup_on_cpu=True)


class FusionGroupPassTest1(PassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data1 = fluid.data(name="data1", shape=[32, 128], dtype="float32")
            data2 = fluid.data(name="data2", shape=[32, 128], dtype="float32")
            data3 = fluid.data(name="data3", shape=[32, 128], dtype="float32")
            tmp_1 = fluid.layers.elementwise_add(data1, data2)
            tmp_2 = fluid.layers.elementwise_sub(data3, tmp_1)

        self.feeds = {
            "data1": np.random.random((32, 128)).astype("float32"),
            "data2": np.random.random((32, 128)).astype("float32"),
            "data3": np.random.random((32, 128)).astype("float32")
        }
        self.fetch_list = [tmp_1, tmp_2]
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"
        self.num_fused_ops = 1

    def test_check_output(self):
        use_gpu_set = [True]
        if core.is_compiled_with_cuda():
            use_gpu_set.append(True)
        for use_gpu in use_gpu_set:
            self.pass_attrs = {"fusion_group_pass": {"use_gpu": use_gpu}}
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            self.check_output_with_place(place, startup_on_cpu=True)


class FusionGroupPassTest2(PassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data1 = fluid.data(name="data1", shape=[32, 128], dtype="float32")
            data2 = fluid.data(name="data2", shape=[32, 128], dtype="float32")
            data3 = fluid.data(name="data3", shape=[32, 128], dtype="float32")
            tmp_1 = fluid.layers.elementwise_sub(data1, data2)
            tmp_2 = fluid.layers.elementwise_mul(data3, tmp_1)
            tmp_3 = fluid.layers.softmax(tmp_2)

        self.feeds = {
            "data1": np.random.random((32, 128)).astype("float32"),
            "data2": np.random.random((32, 128)).astype("float32"),
            "data3": np.random.random((32, 128)).astype("float32")
        }
        self.fetch_list = [tmp_1, tmp_2, tmp_3]
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"
        self.num_fused_ops = 1

    def test_check_output(self):
        use_gpu_set = [True]
        if core.is_compiled_with_cuda():
            use_gpu_set.append(True)
        for use_gpu in use_gpu_set:
            self.pass_attrs = {"fusion_group_pass": {"use_gpu": use_gpu}}
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            self.check_output_with_place(place, startup_on_cpu=True)


if __name__ == "__main__":
    unittest.main()
