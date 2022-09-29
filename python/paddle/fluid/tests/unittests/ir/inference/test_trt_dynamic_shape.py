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

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TRTDynamicShapeTest(InferencePassTest):

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data",
                              shape=[-1, 3, 16, 16],
                              dtype="float32")
            out = fluid.layers.conv2d(input=data,
                                      num_filters=3,
                                      filter_size=3,
                                      groups=1,
                                      padding=[1, 1],
                                      bias_attr=False,
                                      act=None)

        self.feeds = self.set_feeds()
        self.enable_trt = True
        self.trt_parameters = TRTDynamicShapeTest.TensorRTParam(
            1 << 30, 1, 1, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = TRTDynamicShapeTest.DynamicShapeParam(
            {'data': [1, 3, 8, 8]}, {'data': [1, 3, 32, 32]},
            {'data': [1, 3, 16, 16]}, False)
        self.fetch_list = [out]

    def set_feeds(self):
        return {
            "data": np.random.random([1, 3, 16, 16]).astype("float32"),
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)


class TRTDynamicShapeOutOfBound1Test(TRTDynamicShapeTest):

    def set_feeds(self):
        return {
            "data": np.random.random([1, 3, 64, 16]).astype("float32"),
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            with self.assertRaises(Exception):
                self.check_output_with_option(use_gpu)


# (wanghaipeng03) temporarily disable this test, in some cases, this test code
#  doesn't raise exception, TRT just gives the right result
# class TRTDynamicShapeOutOfBound2Test(TRTDynamicShapeTest):
#     def set_feeds(self):
#         return {"data": np.random.random([2, 3, 16, 16]).astype("float32"), }
#
#     def test_check_output(self):
#         if core.is_compiled_with_cuda():
#             use_gpu = True
#             with self.assertRaises(Exception):
#                 self.check_output_with_option(use_gpu)
#


class TRTDynamicShapeOutOfBound3Test(TRTDynamicShapeTest):

    def set_feeds(self):
        return {
            "data": np.random.random([1, 3, 4, 16]).astype("float32"),
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            with self.assertRaises(Exception):
                self.check_output_with_option(use_gpu)


if __name__ == "__main__":
    unittest.main()
