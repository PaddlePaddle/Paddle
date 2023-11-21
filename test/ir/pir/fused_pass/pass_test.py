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

import unittest

import paddle
from paddle import pir


class PassTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.main_program = paddle.static.Program()
        self.feeds = None
        self.fetch_list = None
        self.valid_op_map = {}
        self.pass_list = []
        self.pir_program = None
        self.place_runtime = "cpu"

    def run_pir_pass(self):
        if not isinstance(self.pass_list, list):
            self.pass_list = [self.pass_list]

        pm = pir.PassManager()
        for pass_name in self.pass_list:
            pm.add_pass(pass_name)

        pm.run(self.pir_program)

    def check_fused_ops(self):
        self.assertTrue(
            len(self.valid_op_map) != 0,
            "self.fuse_op_map cannot  be empty!",
        )
        op_names = [op.name() for op in self.pir_program.global_block().ops]
        for valid_op_name, valid_op_count in self.valid_op_map.items():
            acctual_valid_op_count = op_names.count(valid_op_name)
            self.assertTrue(
                valid_op_count == acctual_valid_op_count,
                "Checking of the number of fused operator < {} > failed. "
                "Expected: {}, Received: {}".format(
                    valid_op_name, valid_op_count, acctual_valid_op_count
                ),
            )

    def check_pass_correct(self, need_translate_to_pir=False, atol=1e-5):
        self.assertTrue(
            self.place_runtime == "cpu" or self.place_runtime == "gpu",
            "The place param must be either GPU or CPU ",
        )
        if self.place_runtime == "cpu":
            executor = paddle.static.Executor(paddle.base.CPUPlace())
        elif self.place_runtime == "gpu":
            executor = paddle.static.Executor(paddle.base.CUDAPlace(0))
        self.assertTrue(
            need_translate_to_pir is False and self.pir_program is not None,
            "using old ir need_translate_to_pir Cannot be fasle.\n \
             using new ir program Cannot be None. \n",
        )
        if need_translate_to_pir and self.pir_program is None:
            self.pir_program = pir.translate_to_pir(self.main_program.desc)

        self.run_pir_pass()
        self.check_fused_ops()
