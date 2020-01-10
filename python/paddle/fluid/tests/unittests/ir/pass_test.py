#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import os
import random
import unittest
import warnings
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.backward import append_backward


class PassTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.feeds = None
        self.fetch_list = None
        self.pass_names = None
        self.pass_attrs = {}
        self.fused_op_type = None
        self.num_fused_ops = -1

        np.random.seed(123)
        random.seed(124)

    def _get_places(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        return places

    def check_output(self, startup_on_cpu=False, atol=1e-5):
        places = self._get_places()
        for place in places:
            self.check_output_with_place(place, startup_on_cpu, atol)

    def _run_program(self, executor, program):
        outs = executor.run(program=program,
                            feed=self.feeds,
                            fetch_list=self.fetch_list,
                            return_numpy=False)
        outs_np = []
        outs_lod = []
        for out in outs:
            outs_np.append(np.array(out))
            outs_lod.append(out.lod())
        return outs_np, outs_lod

    def _apply_ir_passes(self):
        graph = core.Graph(self.main_program.desc)
        graph.set_not_owned("__param_scope__", fluid.global_scope())

        if not isinstance(self.pass_names, list):
            self.pass_names = [self.pass_names]

        pass_builder = core.PassBuilder()
        for name in self.pass_names:
            ir_pass = pass_builder.append_pass(name)
            # Set attr for pass
            if self.pass_attrs.get(name, None) is not None:
                attrs = self.pass_attrs[name]
                for key in attrs:
                    ir_pass.set(key, attrs[key])

        trans_pass = pass_builder.append_pass("graph_to_program_pass")
        opt_program = fluid.Program()
        trans_pass.set_not_owned("program", opt_program.desc)
        for p in pass_builder.all_passes():
            p.apply(graph)
        return opt_program

    def check_output_with_place(self, place, startup_on_cpu=False, atol=1e-5):
        executor = fluid.Executor(place)
        if startup_on_cpu:
            # Initialize parameters on CPU
            cpu_executor = fluid.Executor(fluid.CPUPlace())
            cpu_executor.run(self.startup_program)
            outs, lods = self._run_program(cpu_executor, self.main_program)
        else:
            executor.run(self.startup_program)
            outs, lods = self._run_program(executor, self.main_program)
        self.assertTrue(
            len(self.fetch_list) == len(outs),
            "Checking the number of fetchs failed. Expected: {}, Received: {}".
            format(len(self.fetch_list), len(outs)))

        # Parameters may be changed in ir passes.
        opt_program = self._apply_ir_passes()
        if self.fused_op_type is not None and self.num_fused_ops >= 0:
            self.check_fused_ops(opt_program)

        if startup_on_cpu and not isinstance(place, fluid.CPUPlace):
            warnings.warn(
                "Parameters are on CPU, and will be transfered to GPU automatically by data transform."
            )

        outs_opt, lods_opt = self._run_program(executor, opt_program)
        self.assertTrue(
            len(self.fetch_list) == len(outs_opt),
            "Checking the number of fetchs failed. Expected: {}, Received: {}".
            format(len(self.fetch_list), len(outs_opt)))
        for i in xrange(len(self.fetch_list)):
            self.assertTrue(
                np.allclose(
                    outs_opt[i], outs[i], atol=atol),
                "Output < {} > has diff at {}".format(self.fetch_list[i].name,
                                                      str(place)))

    def check_fused_ops(self, program=None):
        if self.fused_op_type is None or self.num_fused_ops < 0:
            return

        if program is None or program == self.main_program:
            program = self._apply_ir_passes()

        acctual_num_fused_ops = 0
        for op in program.desc.block(0).all_ops():
            if op.type() == self.fused_op_type:
                acctual_num_fused_ops += 1
        self.assertTrue(
            self.num_fused_ops == acctual_num_fused_ops,
            "Checking of the number of fused operator < {} > failed. Expected: {}, Received: {}".
            format(self.fused_op_type, self.num_fused_ops,
                   acctual_num_fused_ops))
