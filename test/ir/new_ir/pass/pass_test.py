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

import random
import unittest
import warnings

import numpy as np

import paddle
from paddle import pir
from paddle.base import core


class PassTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.main_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()
        self.feeds = None
        self.fetch_list = None
        self.pass_names = None
        self.pass_attrs = {}
        self.graph_attrs = {}
        self.fused_op_type = None
        self.num_fused_ops = -1

        self.newir_program = None

        np.random.seed(123)
        random.seed(124)

    def _get_places(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        return places

    # No use for this time
    # def grad(self, var):
    #     grad_name = var.name + "@GRAD"
    #     return self.main_program.global_block().var(grad_name)

    # def append_gradients(self, outs):
    #     with base.program_guard(self.main_program, self.startup_program):
    #         loss = paddle.mean(outs)
    #         base.backward.append_backward(loss)

    def check_output(self, startup_on_cpu=False, atol=1e-5):
        '''
        Check whether the fetched outputs of the origin program and the
        optimized program are the same.

        For inference model, the parameters are loaded to CPUPlace first,
        after apply all specified passes, then copy the parameters to GPUPlace.
        We can set startup_on_cpu to True to test inference pass.
        '''
        places = self._get_places()
        for place in places:
            self.check_output_with_place(place, startup_on_cpu, atol)

    def _run_program(self, executor, program):
        outs = executor.run(
            program=program,
            feed=self.feeds,
            fetch_list=self.fetch_list,
            return_numpy=False,
        )
        outs_np = []
        outs_lod = []
        for out in outs:
            outs_np.append(np.array(out))
            outs_lod.append(out.lod())
        return outs_np, outs_lod

    def _apply_ir_passes(self):
        # graph = core.Graph(self.main_program.desc)
        # graph.set_not_owned("__param_scope__", base.global_scope())
        # for attr_name, attr_value in self.graph_attrs.items():
        #     graph.set(attr_name, attr_value)

        newir_program = pir.translate_to_new_ir(self.main_program.desc)
        if not isinstance(self.pass_names, list):
            self.pass_names = [self.pass_names]

        # TODO: here maybe use self.pass_attrs to pass the attrs
        # {"conv2d_bn_fuse": {"use_gpu": use_gpu}}

        pm = pir.PassManager()
        for name in self.pass_names:
            pm.add_pass(name)

        # here I didn't know how to make pir.program run by the executor
        pm.run(newir_program)

        return newir_program

    def check_output_with_place(self, place, startup_on_cpu=False, atol=1e-5):
        '''
        Check whether the fetched outputs of the origin program and the
        optimized program are the same.

        For inference model, the parameters are loaded to CPUPlace first,
        after apply all specified passes, then copy the parameters to GPUPlace.
        We can set startup_on_cpu to True to test inference pass.
        '''
        executor = paddle.static.Executor(place)
        if startup_on_cpu:
            # Initialize parameters on CPU
            cpu_executor = paddle.static.Executor(paddle.CPUPlace())
            cpu_executor.run(self.startup_program)
            outs, lods = self._run_program(cpu_executor, self.main_program)
        else:
            executor.run(self.startup_program)
            outs, lods = self._run_program(executor, self.main_program)
        self.assertTrue(
            len(self.fetch_list) == len(outs),
            "Checking the number of fetchs failed. Expected: {}, Received: {}".format(
                len(self.fetch_list), len(outs)
            ),
        )

        # Parameters may be changed in ir passes.
        opt_program = self._apply_ir_passes()

        # TODO: check program, unfinished
        # self.check_program(opt_program)

        if startup_on_cpu and not isinstance(place, paddle.CPUPlace):
            warnings.warn(
                "Parameters are on CPU, and will be transferred to GPU "
                "automatically by data transform."
            )

        outs_opt, lods_opt = self._run_program(executor, opt_program)
        self.assertTrue(
            len(self.fetch_list) == len(outs_opt),
            "Checking the number of fetchs failed. Expected: {}, Received: {}".format(
                len(self.fetch_list), len(outs_opt)
            ),
        )
        for i in range(len(self.fetch_list)):
            is_allclose = np.allclose(outs_opt[i], outs[i], atol=atol)
            if not is_allclose:
                a = outs_opt[i]
                b = outs[i]
                diff_mat = np.abs(a - b) / np.abs(a)
                max_diff = np.max(diff_mat)
                offset = np.argmax(diff_mat > atol)
                self.assertTrue(
                    is_allclose,
                    "Output (name: %s, shape: %s, dtype: %s) has diff at %s. The maximum diff is %e, first error element is %d, expected %e, but got %e"
                    % (
                        self.fetch_list[i].name,
                        str(self.fetch_list[i].shape),
                        self.fetch_list[i].dtype,
                        str(place),
                        max_diff,
                        offset,
                        a.flatten()[offset],
                        b.flatten()[offset],
                    ),
                )
