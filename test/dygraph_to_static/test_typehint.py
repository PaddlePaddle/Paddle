#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils_new import (
    Dy2StTestBase,
    test_legacy_and_pir_exe_and_pir_api,
)

import paddle

SEED = 2020
np.random.seed(SEED)


class A:
    pass


def function(x: A) -> A:
    t: A = A()
    return 2 * x


class TestTypeHint(Dy2StTestBase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.x = np.zeros(shape=(1), dtype=np.int32)
        self._init_dyfunc()

    def _init_dyfunc(self):
        self.dyfunc = function

    def _run_static(self):
        return self._run(to_static=True)

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run(self, to_static):
        # Set the input of dyfunc to Tensor
        tensor_x = paddle.to_tensor(self.x)
        if to_static:
            ret = paddle.jit.to_static(self.dyfunc)(tensor_x)
        else:
            ret = self.dyfunc(tensor_x)
        if hasattr(ret, "numpy"):
            return ret.numpy()
        else:
            return ret

    @test_legacy_and_pir_exe_and_pir_api
    def test_ast_to_func(self):
        static_numpy = self._run_static()
        dygraph_numpy = self._run_dygraph()
        print(static_numpy, dygraph_numpy)
        np.testing.assert_allclose(dygraph_numpy, static_numpy, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
