# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestSetItem(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def init_data(self):
        paddle.seed(2023)
        x = paddle.randn([4, 8])
        x.stop_gradient = False
        return x

    def test_case1(self):
        def foo(x):
            y = x + 1
            y[:, 2] = x[:, 2] + 1
            # y = paddle.fluid.framework._setitem_impl_(y, (slice(None,None,None), 2) , x[:, 2]+1)
            return y

        # dy_res = self.run_dygrah(foo)
        st_res = self.run_to_static(foo)

        # for dy_out, st_out in zip(dy_res, st_res):
        #     np.testing.assert_allclose(dy_out.numpy, st_out.numpy())

    def run_dygrah(self, func):
        x = self.init_data()
        y = func(x)
        x_grad = paddle.grad(y, x)[0]
        print(y, x_grad)
        return y, x_grad

    def run_to_static(self, func):
        func = paddle.jit.to_static(func)
        return self.run_dygrah(func)


if __name__ == '__main__':
    unittest.main()
