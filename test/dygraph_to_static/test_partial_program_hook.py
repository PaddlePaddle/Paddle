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

import os
import unittest

import paddle
from paddle.base import core
from paddle.jit.dy2static import partial_program, program_translator


class TestPartiaProgramLayerHook(unittest.TestCase):
    def setUp(self):
        os.environ["ENABLE_FALL_BACK"] = "False"
        self._hook = partial_program.PartialProgramLayerHook()

    def test_before_append_backward(self):
        self.assertIsNone(self._hook.before_append_backward(None))

    def test_after_append_backward(self):
        self.assertIsNone(self._hook.after_append_backward(None, 0))

    def test_after_infer(self):
        self.assertIsNone(self._hook.after_infer(None))


class TestPrimHook(unittest.TestCase):
    def setUp(self):
        os.environ["ENABLE_FALL_BACK"] = "False"
        core._set_prim_all_enabled(False)

        def f():
            return paddle.nn.functional.dropout(paddle.rand((1,)))

        concrete_program, partial_program = paddle.jit.to_static(
            f
        ).get_concrete_program()
        self._hook = program_translator.PrimHooker(
            concrete_program.main_program, None
        )
        self._forward = partial_program.forward_program
        self._whole = partial_program._train_program

        core._set_prim_all_enabled(True)

    def tearDown(self):
        core._set_prim_all_enabled(False)

    def test_before_append_backward(self):
        self._hook.before_append_backward(self._forward)
        self.assertNotIn(
            'dropout', tuple(op.type for op in self._forward.blocks[0].ops)
        )

    def test_after_append_backward(self):
        self._hook.after_append_backward(self._whole, 0)
        self.assertNotIn(
            'dropout_grad', tuple(op.type for op in self._whole.blocks[0].ops)
        )


if __name__ == '__main__':
    unittest.main()
