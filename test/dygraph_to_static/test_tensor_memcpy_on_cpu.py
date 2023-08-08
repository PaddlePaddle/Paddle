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

import numpy as np

import paddle


@paddle.jit.to_static
def tensor_copy_to_cpu(x):
    x = paddle.to_tensor(x)
    y = x.cpu()
    return y


class TestTensorCopyToCpuOnDefaultCPU(unittest.TestCase):
    def _run(self, to_static):
        paddle.jit.enable_to_static(to_static)
        x1 = paddle.ones([1, 2, 3])
        x2 = tensor_copy_to_cpu(x1)
        return x1.place, x2.place, x2.numpy()

    def test_tensor_cpu_on_default_cpu(self):
        paddle.fluid.framework._set_expected_place(paddle.CPUPlace())
        dygraph_x1_place, dygraph_place, dygraph_res = self._run(
            to_static=False
        )
        static_x1_place, static_place, static_res = self._run(to_static=True)
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)
        self.assertTrue(dygraph_x1_place.is_cpu_place())
        self.assertTrue(static_x1_place.is_cpu_place())
        self.assertTrue(dygraph_place.is_cpu_place())
        self.assertTrue(static_place.is_cpu_place())


if __name__ == '__main__':
    unittest.main()
