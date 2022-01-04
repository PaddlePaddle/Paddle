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
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle


class TestIdentityAPI(unittest.TestCase):
    def setUp(self):
        self.shape = [4, 4]
        self.x = np.random.random((4, 4)).astype(np.float32)
        self.place = paddle.CPUPlace()

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.shape)
            id_layer = paddle.nn.Identity()
            out = id_layer(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])

        out_ref = self.x
        for out in res:
            self.assertEqual(np.allclose(out, out_ref, rtol=1e-08), True)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        id_layer = paddle.nn.Identity()
        out = id_layer(x_tensor)

        out_ref = self.x
        self.assertEqual(np.allclose(out.numpy(), out_ref, rtol=1e-08), True)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
