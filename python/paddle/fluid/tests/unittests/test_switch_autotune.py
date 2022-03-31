# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import unittest


class TestSwitchAutoTuneDyGraph(unittest.TestCase):
    def dygraph_program(self):
        x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)
        conv = paddle.nn.Conv2D(4, 6, (3, 3))
        out = conv(x_var)
        loss = paddle.mean(out)
        adam = paddle.optimizer.Adam(
            learning_rate=0.1, parameters=conv.parameters())
        out.backward()
        adam.step()
        adam.clear_grad()

    def test_enable_autotune(self):
        paddle.fluid.core.enable_autotune()
        for i in range(2):
            self.dygraph_program()
            status = paddle.fluid.core.auto_tune_status()
            self.assertEqual(status["step_id"], i)
            self.assertEqual(status["use_autotune"], True)
            self.assertEqual(status["cache_size"], 0)
            self.assertEqual(status["cache_hit_rate"], 0)

    def test_disable_autotune(self):
        paddle.fluid.core.disable_autotune()
        status = paddle.fluid.core.auto_tune_status()
        self.assertEqual(status["use_autotune"], False)


if __name__ == '__main__':
    unittest.main()
