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

import paddle
from paddle.fluid.layer_helper import LayerHelper


def feed_with_place():
    helper = LayerHelper('feed_with_place', **locals())

    out = helper.create_variable_for_type_inference('float32')
    helper.append_op(
        type='feed_with_place',
        inputs={},
        outputs={'out': out},
        attrs={
            'index': 0,
            'dtype': 0,
            'place': 0,
        },
    )
    return out


class TestNewIr(unittest.TestCase):
    def test_with_new_ir(self):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        main_program = paddle.static.Program()
        new_scope = paddle.static.Scope()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                out = feed_with_place()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
