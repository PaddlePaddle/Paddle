#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

from op_mapper_test import OpMapperTest


class TestGaussianRandomOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {}
        self.shape = [2, 3]
        self.mean = 0.0
        self.std = 1.0
        self.seed = 10
        self.dtype = "float32"

    def set_op_type(self):
        return "gaussian_random"

    def set_op_inputs(self):
        return {}

    def set_op_attrs(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "seed": self.seed,
            "shape": self.shape,
            "dtype": self.nptype2paddledtype(self.dtype),
        }

    def set_op_outputs(self):
        return {'Out': [self.dtype]}

    def test_check_results(self):
        # Due to the different random number generation numbers implemented
        # in the specific implementation, the random number results generated
        # by CINN and Paddle are not the same, but they all conform to the
        # Gaussian distribution.
        self.check_outputs_and_grads(max_relative_error=10000)


class TestGaussianRandomCase1(TestGaussianRandomOp):
    def init_input_data(self):
        self.feed_data = {}
        self.shape = [2, 3, 4]
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10
        self.dtype = "float32"


class TestGaussianRandomCase2(TestGaussianRandomOp):
    def init_input_data(self):
        self.feed_data = {}
        self.shape = [2, 3, 4]
        self.mean = 2.0
        self.std = 3.0
        self.seed = 10
        self.dtype = "float64"


if __name__ == "__main__":
    unittest.main()
