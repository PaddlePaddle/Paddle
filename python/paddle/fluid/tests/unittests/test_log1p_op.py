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

import paddle
import paddle.fluid.core as core
import paddle.static as static


class Test_Log1p_Op_Fp16(unittest.TestCase):
    def test_api_fp16(self):
        paddle.enable_static()
        with static.program_guard(static.Program(), static.Program()):
            x = [[2, 3, 4], [7, 8, 9]]
            x = paddle.to_tensor(x, dtype='float16')
            out = paddle.log1p(x)
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = static.Executor(place)
                (res,) = exe.run(fetch_list=[out])


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
