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


import collections
import sys
import unittest

import paddle


class TestEvalFrame(unittest.TestCase):
    def setUp(self):
        self.x = paddle.to_tensor(2).astype('int64')

    def tearDown(self):
        pass

    def test_eval_frame(self):
        if not (sys.version_info >= (3, 8) and sys.version_info < (3, 14)):
            # skip test_eval_frame, current only support 3.8 - 3.13
            return

        CustomCode = collections.namedtuple(
            "CustomCode", ["code", "disable_eval_frame"]
        )

        def mul(a, b):
            return a * b

        code = CustomCode(mul.__code__, True)

        def callback(frame_obj):
            # Do your callback function here and return a object with `.code`
            if frame_obj.f_code.co_name == "add":
                return code
            return CustomCode(
                code=frame_obj.f_code, disable_eval_frame=True
            )  # do nothing.

        def add(a, b):
            return a + b

        x = 1
        y = 2

        paddle.base.core.set_eval_frame(callback)
        assert add(x, y) == 2, "should be 2"
        paddle.base.core.set_eval_frame(None)
        assert add(x, y) == 3, "should be 3"


if __name__ == "__main__":
    unittest.main()
