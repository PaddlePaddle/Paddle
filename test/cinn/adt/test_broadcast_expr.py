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


class TestMapExprBroadcast(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "x1": np.random.uniform(-1.0, 1.0, [4, 16]).astype("float32"),
            "x2": np.random.uniform(-1.0, 1.0, [16]).astype("float32"),
        }

    def test_broadcast(self):
        builder = NetBuilder("TestMapExprBroadcast")
        x1 = builder.create_input(Float(32), self.inputs["x1"].shape, "x1")
        x2 = builder.create_input(Float(32), self.inputs["x2"].shape, "x2")
        z = builder.elementwise_add(x1, x2)
        out = builder.relu(z)
        prog = builder.build()

        target = DefaultNVGPUTarget()

        result = prog.build_and_get_output(
            target,
            [x1, x2],
            [self.inputs["x1"], self.inputs["x2"]],
            [out],
            passes=[],
            scope=None,
        )

        np.testing.assert_allclose(
            result[0].numpy(target),
            self.inputs["x"] + self.inputs["y"],
            err_msg="TestMapExprBroadcast failed!",
        )
        print("Finish Test")


if __name__ == "__main__":
    unittest.main()
