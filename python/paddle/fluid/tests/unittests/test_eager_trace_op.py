#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import _test_eager_guard


class TestEagerTraceOp(unittest.TestCase):

    def test_branches(self):
        with _test_eager_guard():
            data = np.random.random([1, 1]).astype(np.float32)
            x = paddle.to_tensor(data)

            paddle.fluid.framework._dygraph_tracer().trace_op(
<<<<<<< HEAD
                'broadcast_tensors', {
                    'X': [x, x],
                    'Out': [x, x]
                }, {'Out': [x, x]}, {})
=======
                'broadcast_tensors',
                {'X': [x, x], 'Out': [x, x]},
                {'Out': [x, x]},
                {},
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            paddle.fluid.framework._dygraph_tracer().trace_op(
                'scale', {'X': x}, {'Out': x}, {'scale': 0.5}
            )

            scale = paddle.to_tensor(np.random.random([1]).astype(np.float32))
            paddle.fluid.framework._dygraph_tracer().trace_op(
<<<<<<< HEAD
                'instance_norm', {
                    'Scale': [scale],
                    'X': [x]
                }, {'Y': [x]}, {})
=======
                'instance_norm', {'Scale': [scale], 'X': [x]}, {'Y': [x]}, {}
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f


if __name__ == "__main__":
    unittest.main()
