#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid


class TestStdLayer(unittest.TestCase):
    def setUp(self):
        self._dtype = "float64"
        self._input = np.random.random([2, 3, 4, 5]).astype(self._dtype)

    def static(self, axis=None, keepdim=False, unbiased=True):
        prog = fluid.Program()
        with fluid.program_guard(prog):
            data = fluid.data(
                name="data", dtype=self._dtype, shape=[None, 3, 4, 5])
            out = prog.current_block().create_var(
                dtype=self._dtype, shape=[2, 3, 4, 5])
            paddle.std(input=data,
                       axis=axis,
                       keepdim=keepdim,
                       unbiased=unbiased,
                       out=out)

        exe = fluid.Executor(self._place)
        return exe.run(feed={"data": self._input},
                       program=prog,
                       fetch_list=[out])[0]

    def dynamic(self, axis=None, keepdim=False, unbiased=True):
        with fluid.dygraph.guard(self._place):
            data = fluid.dygraph.to_variable(self._input)
            out = paddle.std(input=data,
                             axis=axis,
                             keepdim=keepdim,
                             unbiased=unbiased)
            return out.numpy()

    def numpy(self, axis=None, keepdim=False, unbiased=True):
        ddof = 1 if unbiased else 0
        axis = tuple(axis) if isinstance(axis, list) else axis
        return np.std(self._input, axis=axis, keepdims=keepdim, ddof=ddof)

    def test_equal(self):
        places = []
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self._place = place
            self.assertTrue(np.allclose(self.numpy(), self.static()))
            self.assertTrue(
                np.allclose(
                    self.numpy(axis=[0, 2]), self.dynamic(axis=[0, 2])))
            self.assertTrue(
                np.allclose(
                    self.numpy(
                        axis=[1, 3], keepdim=True),
                    self.dynamic(
                        axis=[1, 3], keepdim=True)))
            self.assertTrue(
                np.allclose(
                    self.numpy(unbiased=False), self.dynamic(unbiased=False)))


if __name__ == '__main__':
    unittest.main()
