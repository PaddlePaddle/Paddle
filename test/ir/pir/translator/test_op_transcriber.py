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


class TestOpTranscriber(unittest.TestCase):
    def setUp(self):
        self.place = core.Place()
        self.place.set_place(paddle.CPUPlace())
        self.new_scope = paddle.static.Scope()
        self.main_program = paddle.static.Program()

    def build_model(self):
        raise Exception('Build a new program for the op to be tested!')

    def check(self):
        assert hasattr(self, "op_name"), "Op_name should be specified!"
        self.build_model()
        l = pir.translate_to_pir(self.main_program.desc)
        assert self.op_name in str(l), (
            self.op_name
            + " should be translated to pd_op."
            + self.op_name
            + '!'
        )
