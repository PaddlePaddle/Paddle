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

import abc
import unittest

import numpy as np

import paddle
from paddle import pir


class PassTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.feeds = None
        self.fetch_list = None
        self.valid_op_map = {}
        self.pir_program = None
        self.places = []
        self.skip_accuracy_verification = False
        self.pass_attr_list = []  # pass_name:pass_attr(default:None)

    def run_pir_pass(self, program):
        pm = pir.PassManager(opt_level=4)
        pm.enable_print_statistics()
        pm.enable_ir_printing()
        for pass_item in self.pass_attr_list:
            for pass_name, pass_attr in pass_item.items():
                pm.add_pass(pass_name, pass_attr)
        pm.run(program)
        return program

    def check_fused_ops(self, program):
        self.assertTrue(
            len(self.valid_op_map) != 0,
            "self.fuse_op_map cannot  be empty!",
        )
        op_names = [op.name() for op in program.global_block().ops]
        for valid_op_name, valid_op_count in self.valid_op_map.items():
            actual_valid_op_count = op_names.count(valid_op_name)
            self.assertTrue(
                valid_op_count == actual_valid_op_count,
                f"Checking of the number of fused operator < {valid_op_name} > failed. "
                f"Expected: {valid_op_count}, Received: {actual_valid_op_count}",
            )

    @abc.abstractmethod
    def sample_program(self):
        """
        Generate all pir grogram
        """
        raise NotImplementedError

    def run_program(self, executor, startup_program, main_program):
        with paddle.pir_utils.IrGuard():
            with paddle.static.program_guard(startup_program, main_program):
                fetches = executor.run(main_program, feed=self.feeds)
                return fetches

    def compare_accuracy(
        self, baseline_data, actual_data, atol=1e-5, rtol=1e-5
    ):
        self.assertTrue(
            len(baseline_data) == len(actual_data),
            f"The output baseline_data are not equal, the baseline output_data is {len(baseline_data)}, but got {len(actual_data)}",
        )
        for i in range(len(baseline_data)):
            self.assertEqual(
                baseline_data[i].shape,
                actual_data[i].shape,
                f"The output shapes are not equal, the baseline shape is {baseline_data[i].shape}, but got {actual_data[i].shape}",
            )
            np.testing.assert_allclose(
                baseline_data[i], actual_data[i], atol=atol, rtol=rtol
            )

    def check_pass_correct(self, atol=1e-5, rtol=1e-5):
        for place in self.places:
            for program, need_translate_to_pir in self.sample_program():
                main_program = program[0]
                startup_program = program[1]
                if need_translate_to_pir:
                    main_program = pir.translate_to_pir(main_program.desc)
                with paddle.pir_utils.IrGuard():
                    with paddle.static.program_guard(
                        main_program, startup_program
                    ):
                        executor = paddle.static.Executor(place)
                        executor.run(startup_program)
                    with paddle.static.program_guard(main_program):
                        out = paddle._pir_ops.fetch(
                            main_program.list_vars()[-1], "fetch0", 0
                        )
                        out.persistable = True
                baseline_fetch = self.run_program(
                    executor, startup_program, main_program
                )
                main_program = self.run_pir_pass(main_program)
                self.check_fused_ops(main_program)
                actual_fetch = self.run_program(
                    executor, startup_program, main_program
                )
                if self.skip_accuracy_verification is False:
                    self.compare_accuracy(
                        baseline_fetch, actual_fetch, atol, rtol
                    )
