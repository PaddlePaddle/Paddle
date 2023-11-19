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

# NOTE(SigureMo): This unittest does NOT need to run in PIR mode. Don't import Dy2StTestBase.

import unittest

import paddle
from paddle.jit.dy2static.program_translator import ProgramTranslator
from paddle.static.amp.fp16_utils import (
    DEFAULT_AMP_OPTIONS,
    AmpOptions,
    prepare_op_amp_options,
)

GLOBAL_ENABLE_AMP_OPTIONS = DEFAULT_AMP_OPTIONS
GLOBAL_DISABLE_AMP_OPTIONS = AmpOptions(
    enable=False,
    custom_black_list=DEFAULT_AMP_OPTIONS.custom_black_list,
    custom_white_list=DEFAULT_AMP_OPTIONS.custom_white_list,
    level=DEFAULT_AMP_OPTIONS.level,
    dtype=DEFAULT_AMP_OPTIONS.dtype,
    use_promote=DEFAULT_AMP_OPTIONS.use_promote,
)


class LocalAutoCastLayer1(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._fc = paddle.nn.Linear(10, 10)

    @paddle.jit.to_static(full_graph=True)
    def forward(self, x):
        x = self._fc(x)
        y = self._fc(x) * 2
        with paddle.amp.auto_cast(False):
            x = x.astype("float32")
            y = y.astype("float32")
            if x[0][0] > 1:
                x = x + y
            else:
                x = x - y
                x = x * 2

        return x + 1


class LocalAutoCastLayer2(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._fc = paddle.nn.Linear(10, 10)

    @paddle.jit.to_static(full_graph=True)
    def forward(self, x):
        with paddle.amp.auto_cast(False):
            x = x.astype("float32")
            x = self._fc(x)
            y = self._fc(x) * 2
        if x[0][0] > 1:
            x = x + y
        else:
            x = x - y
            x = x * 2

        return x + 1


class LocalAutoCastLayer3(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._fc = paddle.nn.Linear(10, 10)

    @paddle.jit.to_static(full_graph=True)
    def forward(self, x):
        with paddle.amp.auto_cast(True):
            x = x.astype("float32")
            x = self._fc(x)
            y = self._fc(x) * 2
        if x[0][0] > 1:
            x = x + y
        else:
            x = x - y
            x = x * 2

        return x + 1


class TestLocalCast(unittest.TestCase):
    def get_auto_cast_ops_info_from_program(self, program):
        auto_cast_ops_info = []
        for block in program.blocks:
            current_block_should_auto_cast = []
            auto_cast_ops_info.append(current_block_should_auto_cast)
            for op in block.ops:
                current_block_should_auto_cast.append(op.amp_options.enable)
        return auto_cast_ops_info

    def should_auto_cast_for_each_ops(self, layer, input, global_amp_options):
        concrete_program, _ = layer.forward.get_concrete_program(input)
        program = concrete_program.main_program
        prepare_op_amp_options(
            program,
            ProgramTranslator.get_instance()._amp_records,
            global_amp_options,
        )
        auto_cast_ops_info = self.get_auto_cast_ops_info_from_program(program)
        paddle.enable_static()
        # Ensure the cloned program has the same auto_cast ops info
        cloned_program = program.clone()
        paddle.disable_static()
        cloned_auto_cast_ops_info = self.get_auto_cast_ops_info_from_program(
            cloned_program
        )
        self.assertEqual(auto_cast_ops_info, cloned_auto_cast_ops_info)
        return auto_cast_ops_info

    def test_should_auto_cast_1(self):
        layer = LocalAutoCastLayer1()
        input = paddle.randn([10, 10])
        expected = [
            # There are part of ops in auto_cast(False) block
            [
                True, True, True, True, True,
                False, False, False, False, False, False, False, False, False, False, False,
                True,
            ],
            # All if branch in auto_cast(False) block
            [False, False],
            # All else branch in auto_cast(False) block
            [False, False, False],
        ]  # fmt: skip
        actual = self.should_auto_cast_for_each_ops(
            layer, input, GLOBAL_ENABLE_AMP_OPTIONS
        )
        self.assertEqual(expected, actual)

    def test_should_auto_cast_2(self):
        layer = LocalAutoCastLayer2()
        input = paddle.randn([10, 10])
        expected = [
            # There are part of ops in auto_cast(False) block
            [
                False, False, False, False, False, False,
                True, True, True, True, True, True, True, True, True, True,
            ],
            # All if branch out of auto_cast(False) block
            [True, True],
            # All else branch out of auto_cast(False) block
            [True, True, True],
        ]  # fmt: skip
        actual = self.should_auto_cast_for_each_ops(
            layer, input, GLOBAL_ENABLE_AMP_OPTIONS
        )
        self.assertEqual(expected, actual)

    def test_should_auto_cast_3(self):
        layer = LocalAutoCastLayer3()
        input = paddle.randn([10, 10])
        expected = [
            # There are part of ops in auto_cast(True) block
            [
                True, True, True, True, True, True,
                False, False, False, False, False, False, False, False, False, False,
            ],
            # All if branch out of auto_cast(True) block
            [False, False],
            # All else branch out of auto_cast(True) block
            [False, False, False],
        ]  # fmt: skip
        actual = self.should_auto_cast_for_each_ops(
            layer, input, GLOBAL_DISABLE_AMP_OPTIONS
        )

        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
