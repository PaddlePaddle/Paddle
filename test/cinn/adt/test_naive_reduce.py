#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
from cinn.common import DefaultNVGPUTarget, Float
from cinn.frontend import NetBuilder


class TestMapExprNaiveReduce(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "x": np.random.uniform(-1.0, 1.0, [2, 1024, 1024]).astype("float32")
        }

    def test_naive_reduce(self):
        builder = NetBuilder("TestMapExprNaiveReduce")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")

        out = builder.reduce_sum(x, [0], False)
        prog = builder.build()
        target = DefaultNVGPUTarget()

        result = prog.build_and_get_output(
            target, [x], [self.inputs["x"]], [out], passes=[], scope=None
        )

        np.testing.assert_allclose(
            result[0].numpy(target),
            np.sum(self.inputs["x"], axis=0),
            err_msg="TestMapExprNaiveReduce failed!",
        )


if __name__ == "__main__":
    unittest.main()
