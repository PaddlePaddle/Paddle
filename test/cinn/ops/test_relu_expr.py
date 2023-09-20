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

inputs = {"x": OpTest.random([8, 8], "float32", -1.0, 1.0)}

builder = NetBuilder("MapExprTest")
x = builder.create_input(Float(32), inputs["x"].shape, "x")
y = builder.sin(x)
out = builder.relu(y)
prog = builder.build()

target = DefaultNVGPUTarget()

result = prog.build_and_get_output(
    target, [x], [inputs["x"]], [out], passes=[], scope=None
)
