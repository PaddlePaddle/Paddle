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

<<<<<<< HEAD
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
=======
import numpy as np
import paddle.fluid as fluid
import unittest

from paddle.fluid.dygraph.jit import declarative
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

SEED = 2020
np.random.seed(SEED)


class A:
    pass


def function(x: A) -> A:
    t: A = A()
    return 2 * x


class TestTransformWhileLoop(unittest.TestCase):
<<<<<<< HEAD
    def setUp(self):
        self.place = (
            fluid.CUDAPlace(0)
            if fluid.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
=======

    def setUp(self):
        self.place = fluid.CUDAPlace(
            0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.x = np.zeros(shape=(1), dtype=np.int32)
        self._init_dyfunc()

    def _init_dyfunc(self):
        self.dyfunc = function

    def _run_static(self):
        return self._run(to_static=True)

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run(self, to_static):
        with fluid.dygraph.guard(self.place):
            # Set the input of dyfunc to VarBase
            tensor_x = fluid.dygraph.to_variable(self.x, zero_copy=False)
            if to_static:
<<<<<<< HEAD
                ret = paddle.jit.to_static(self.dyfunc)(tensor_x)
=======
                ret = declarative(self.dyfunc)(tensor_x)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            else:
                ret = self.dyfunc(tensor_x)
            if hasattr(ret, "numpy"):
                return ret.numpy()
            else:
                return ret

    def test_ast_to_func(self):
        static_numpy = self._run_static()
        dygraph_numpy = self._run_dygraph()
        print(static_numpy, dygraph_numpy)
        np.testing.assert_allclose(dygraph_numpy, static_numpy, rtol=1e-05)


class TestTypeHint(TestTransformWhileLoop):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _init_dyfunc(self):
        self.dyfunc = function


if __name__ == '__main__':
<<<<<<< HEAD
    unittest.main()
=======
    with fluid.framework._test_eager_guard():
        unittest.main()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
