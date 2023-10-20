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


from cinn.common import DefaultNVGPUTarget, Float
from cinn.frontend import NetBuilder
from op_test import OpTest

inputs = {
    "x": OpTest.random([32, 2048], "float32", -1.0, 1.0),
    "y": OpTest.random([32, 2048], "float32", -1.0, 1.0),
}

builder = NetBuilder("ReduceMapExprTest")
x = builder.create_input(Float(32), inputs["x"].shape, "x")
y = builder.create_input(Float(32), inputs["y"].shape, "y")

t = builder.elementwise_add(x, y)
out = builder.reduce_sum(t, [0], False)

prog = builder.build()

target = DefaultNVGPUTarget()
# target = DefaultHostTarget()

result = prog.build_and_get_output(
    target, [x, y], [inputs["x"], inputs["y"]], [out], passes=[], scope=None
)

import numpy as np

import paddle

pd_x = paddle.to_tensor(inputs["x"])
pd_y = paddle.to_tensor(inputs["y"])
pd_out = paddle.sum(pd_x + pd_y, axis=0)

np.testing.assert_allclose(
    result[0].numpy(target),
    pd_out.numpy(),
    err_msg="PrecisionTest failed!",
)

print("Test Success")
