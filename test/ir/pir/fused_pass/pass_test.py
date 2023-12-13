#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import abc
import unittest

import paddle
from paddle import pir


class PassTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.feeds = None
        self.fetch_list = None
        self.valid_op_map = {}
        self.pass_list = []
        self.pir_program = None
        self.place_runtime = "cpu"

    def run_pir_pass(self, program):
        if not isinstance(self.pass_list, list):
            self.pass_list = [self.pass_list]

        pm = pir.PassManager(opt_level=4)
        for pass_name in self.pass_list:
            pm.add_pass(pass_name)

        pm.run(program)
        return program

    def check_fused_ops(self, program):
        self.assertTrue(
            len(self.valid_op_map) != 0,
            "self.fuse_op_map cannot  be empty!",
        )
        op_names = [op.name() for op in program.global_block().ops]
        for valid_op_name, valid_op_count in self.valid_op_map.items():
            acctual_valid_op_count = op_names.count(valid_op_name)
            self.assertTrue(
                valid_op_count == acctual_valid_op_count,
                "Checking of the number of fused operator < {} > failed. "
                "Expected: {}, Received: {}".format(
                    valid_op_name, valid_op_count, acctual_valid_op_count
                ),
            )

    @abc.abstractmethod
    def is_program_valid(self, program=None):
        """
        judge the effectiveness of the pir program
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_program(self):
        """
        Generate all pir grogram
        """
        raise NotImplementedError

    def check_pass_correct(self, atol=1e-5):
        self.assertTrue(
            self.place_runtime == "cpu" or self.place_runtime == "gpu",
            "The place param must be either GPU or CPU ",
        )
        if self.place_runtime == "cpu":
            executor = paddle.static.Executor(paddle.base.CPUPlace())
        elif self.place_runtime == "gpu":
            executor = paddle.static.Executor(paddle.base.CUDAPlace(0))

        for program, need_translate_to_pir in self.sample_program():
            if need_translate_to_pir:
                program = pir.translate_to_pir(program.desc)
            if not self.is_program_valid(program):
                continue
            program = self.run_pir_pass(program)
            self.check_fused_ops(program)
