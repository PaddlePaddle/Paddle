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


class TestEagerTraceOp(unittest.TestCase):
    def test_branches(self):
        data = np.random.random([1, 1]).astype(np.float32)
        x = paddle.to_tensor(data)

        paddle.fluid.framework._dygraph_tracer().trace_op(
            'broadcast_tensors',
            {'X': [x, x], 'Out': [x, x]},
            {'Out': [x, x]},
            {},
        )
        paddle.fluid.framework._dygraph_tracer().trace_op(
            'scale', {'X': x}, {'Out': x}, {'scale': 0.5}
        )

        scale = paddle.to_tensor(np.random.random([1]).astype(np.float32))
        paddle.fluid.framework._dygraph_tracer().trace_op(
            'instance_norm', {'Scale': [scale], 'X': [x]}, {'Y': [x]}, {}
        )


if __name__ == "__main__":
    unittest.main()
