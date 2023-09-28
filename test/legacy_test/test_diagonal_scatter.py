#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import paddle

class TestDiagnalScatterAPI(unittest.TestCase):
    def setUp(self):
        
        self.x_dtype = 'float32'
        self.x_shape = [3, 3]
        self.x = np.zeros([3, 3]).astype('float32')
        self.src_dtype = 'float32'
        self.src_shape = [3]
        self.src  = np.ones(3).astype('float32')
        self.offset  = 0
        self.init_input()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_input(self):
        pass

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x_shape, dtype=self.x_dtype)
            src = paddle.static.data('src', self.src_shape, dtype=self.x_dtype)
            out = paddle.diagonal_scatter(x, src, self.offset)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x,
                    'src': self.src,
                },
                fetch_list=[out],
            )
            expect_output = [[1.0,0.0,0.0],
                [0.0,1.0,0.0],
                [0.0,0.0,1.0]]
            np.testing.assert_allclose(expect_output, res[0], atol=1e-07)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        src = paddle.to_tensor(self.src)
        res = paddle.diagonal_scatter(x, src, self.offset)

        expect_output = np.array([[1.0,0.0,0.0],
                [0.0,1.0,0.0],
                [0.0,0.0,1.0]])
        np.testing.assert_allclose(expect_output, res, atol=1e-07)
        paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()