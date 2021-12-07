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

import paddle
import unittest
import numpy as np
paddle.set_device('cpu')


class TestTopk(unittest.TestCase):
    def setUp(self):
        paddle.seed(200)
        self.x = paddle.randn([10])
        self.k = 2

    def test_compile_infer(self):
        k = paddle.full([1], fill_value=self.k, dtype='int32')
        k.desc.set_desc_value([self.k])
        out, _ = paddle.topk(self.x, k)
        print(out.shape)
        self.assertEqual(out.shape[0], self.k)

        train_program = paddle.static.default_main_program()
        print(train_program)
        exe = paddle.static.Executor(paddle.CPUPlace())
        train_res = exe.run(train_program, fetch_list=[out])[0]
        print(train_res)

        paddle.save(train_program, './test_desc')
        paddle.seed(200)
        prog = paddle.load('./test_desc')
        print(prog)
        infer_res = exe.run(train_program, fetch_list=[out.name])[0]
        print(infer_res)

        self.assertTrue(np.array_equal(train_res, infer_res))


class TestReshape(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 3, 4)
        self.x = paddle.randn(self.shape)

    # case 1: shape is a Tensor
    def test_shape_tensor(self):
        shape = paddle.shape(self.x)
        out = paddle.reshape(self.x, shape)
        self.assertEqual(shape.desc.get_desc_value(), list(self.shape))
        print(out.shape)
        self.assertEqual(out.shape, self.shape)

        train_program = paddle.static.default_main_program()
        print(train_program)
        exe = paddle.static.Executor(paddle.CPUPlace())
        train_res = exe.run(train_program, fetch_list=[out])[0]
        print(train_res)

    # case 2: shape is list[Tensor]


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
