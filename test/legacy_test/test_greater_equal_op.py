#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import static
from paddle.base import core


class Test_Greater_Equal_Op_Fp16(unittest.TestCase):
    def test_api_fp16(self):
        paddle.enable_static()
        with static.program_guard(static.Program(), static.Program()):
            label = paddle.to_tensor([3, 3], dtype="float16")
            limit = paddle.to_tensor([3, 2], dtype="float16")
            out = paddle.greater_equal(x=label, y=limit)
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = static.Executor(place)
                (res,) = exe.run(fetch_list=[out])
                self.assertEqual((res == np.array([True, True])).all(), True)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
