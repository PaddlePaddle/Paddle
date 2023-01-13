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

import unittest

import numpy as np

import paddle


class TestNPUIdentityOp(unittest.TestCase):
    def setUp(self):
        self.op_type = "npu_identity"
        self.shape = [64, 6, 28, 28]
        self.x = np.random.random(self.shape).astype(np.float32)
        self.format = 3  # ACL_FORMAT_NC1HWC0 = 3
        self.place = paddle.CPUPlace()

    def test_api_static(self):
        paddle.enable_static()

        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        with paddle.static.program_guard(main_program, startup_program):
            x_data = paddle.static.data(
                shape=self.shape, name="data", dtype='float32'
            )
            output = paddle.incubate._npu_identity(x=x_data, format=self.format)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        result = exe.run(
            main_program, feed={x_data.name: self.x}, fetch_list=[output]
        )

        np.testing.assert_allclose(result[0], self.x, rtol=1e-08)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)

        x = paddle.to_tensor(self.x)
        out = paddle.incubate._npu_identity(x, self.format)

        np.testing.assert_allclose(out.numpy(), self.x, rtol=1e-08)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
