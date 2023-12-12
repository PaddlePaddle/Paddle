# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from dygraph_to_static_utils import (
    Dy2StTestBase,
    static_guard,
    test_ast_only,
    test_pir_only,
)

import paddle
from paddle.pir.core import _convert_into_value, static_op_arg_cast_guard


class TestSetStaticOpArgPreCastHook(Dy2StTestBase):
    @test_ast_only
    @test_pir_only
    def test_set_static_op_arg_pre_cast_hook(self):
        eager_tensor = paddle.rand((10, 10), 'float32')

        with static_guard():
            with self.assertRaisesRegex(
                ValueError,
                r"abs\(\): argument \(position 1\) must be OpResult, but got Tensor",
            ):
                paddle.abs(eager_tensor)

            with static_op_arg_cast_guard(_convert_into_value):
                paddle.abs(eager_tensor)


if __name__ == '__main__':
    unittest.main()
