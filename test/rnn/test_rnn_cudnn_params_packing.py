# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from unittest import TestCase

import paddle


def create_model():
    hidden_size = 32
    bilstm = paddle.nn.LSTM(
        hidden_size, hidden_size, num_layers=1, direction='bidirectional'
    )
    return bilstm


class TestRNNProgramClone(TestCase):
    def setUp(self):
        paddle.enable_static()

    def test_rnn_with_cudnn_clone(self):
        train_program = paddle.static.Program()
        test_program = paddle.static.Program()
        startup_prog = paddle.static.Program()

        # test a typical case in static graph usage: create two nearly
        # identical program with a shared startup program to share their
        # parameters
        #
        # when creating a parameter, the name is checked. If there is already
        # a parameter with the same name, which is the output of a operator
        # (i.e. its creator), its re-creation is skipped.
        #
        # but if that parameter has been the output of more than one operator,
        # an exception is raised. For special cases, white list is added.
        # flattening rnn's parameters for the need to call cudnn kernel is such
        # a case.
        with paddle.static.program_guard(train_program, startup_prog):
            with paddle.base.unique_name.guard():
                bilstm = create_model()

        with paddle.base.program_guard(test_program, startup_prog):
            with paddle.base.unique_name.guard():
                bilstm = create_model()


if __name__ == "__main__":
    unittest.main()
