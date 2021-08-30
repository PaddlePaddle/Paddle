# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
from paddle.fluid import core
from paddle.fluid.core import InterpreterCore

import numpy as np

paddle.enable_static()


class LinearTestCase(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()

    def test_interp_base(self):
        a = paddle.static.data(name="a", shape=[2, 2], dtype='float32')
        b = paddle.ones([2, 2]) * 2
        t = paddle.static.nn.fc(a, 2)
        c = t + b

        main_program = paddle.fluid.default_main_program()
        startup_program = paddle.fluid.default_startup_program()
        p = core.Place()
        p.set_place(self.place)
        inter_core = InterpreterCore(p, main_program.desc, startup_program.desc,
                                     core.Scope())

        out = inter_core.run({
            "a": np.ones(
                [2, 2], dtype="float32") * 2
        }, [c.name])
        for i in range(10):
            out = inter_core.run({
                "a": np.ones(
                    [2, 2], dtype="float32") * i
            }, [c.name])


if __name__ == "__main__":
    unittest.main()
