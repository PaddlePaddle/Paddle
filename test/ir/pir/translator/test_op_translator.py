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
from paddle import pir
from paddle.base import core
from paddle.base.backward import append_backward

paddle.enable_static()

paddle.pir_utils._switch_to_old_ir_()


class TestOpTranslator(unittest.TestCase):
    def setUp(self):
        self.place = core.Place()
        self.place.set_place(paddle.CPUPlace())
        self.new_scope = paddle.static.Scope()
        self.main_program = paddle.static.Program()

    def append_op(self):
        raise NotImplementedError("Define the op to be tested here!")

    def build_model(self):
        with paddle.static.scope_guard(self.new_scope):
            with paddle.static.program_guard(self.main_program):
                self.append_op()

    def check(self):
        self.build_model()
        pir_program = pir.translate_to_pir(self.main_program.desc)
        assert hasattr(self, "op_type"), "Op_type should be specified!"
        assert self.op_type in str(pir_program), (
            self.op_type
            + " should be translated to pd_op."
            + self.op_type
            + '!'
        )


class TestOpWithBackwardTranslator(unittest.TestCase):
    def setUp(self):
        self.place = core.Place()
        self.place.set_place(paddle.CPUPlace())
        self.new_scope = paddle.static.Scope()
        self.main_program = paddle.static.Program()

    def append_op(self):
        raise NotImplementedError("Define the op to be tested here!")

    def build_model(self):
        with paddle.static.scope_guard(self.new_scope):
            with paddle.static.program_guard(self.main_program):
                out = self.append_op()
                out.stop_gradient = False
                append_backward(out)

    def check(self):
        self.build_model()
        pir_program = pir.translate_to_pir(self.main_program.desc)
        assert hasattr(
            self, "forward_op_type"
        ), "forward_op_type should be specified!"
        assert hasattr(
            self, "backward_op_type"
        ), "backward_op_type should be specified!"
        serialized_pir_program = str(pir_program)
        assert self.forward_op_type in serialized_pir_program, (
            self.forward_op_type
            + " should be translated to pd_op."
            + self.forward_op_type
            + '!'
        )
        assert self.backward_op_type in serialized_pir_program, (
            self.backward_op_type
            + " should be translated to pd_op."
            + self.backward_op_type
            + '!'
        )
