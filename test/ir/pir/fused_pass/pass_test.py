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

import random
import unittest

import numpy as np

import paddle
from paddle import pir


class PassTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.main_program = paddle.static.Program()
        self.feeds = None
        self.fetch_list = None
        self.fused_op_type = None
        self.pass_list = []
        self.pir_program = None
        self.fused_ops = []

        np.random.seed(123)
        random.seed(124)

    # def run_program(self, executor, program):
    #     outs = executor.run(
    #         program=program,
    #         feed=self.feeds,
    #         fetch_list=self.fetch_list,
    #         return_numpy=False,
    #     )
    #     outs_np = []
    #     outs_lod = []
    #     for out in outs:
    #         outs_np.append(np.array(out))
    #         outs_lod.append(out.lod())
    #     return outs_np, outs_lod
    def run_pir_pass(self):
        if not isinstance(self.pass_list, list):
            self.pass_list = [self.pass_list]

        pm = pir.PassManager()
        for pass_name in self.pass_list:
            pm.add_pass(pass_name)

        pm.run(self.pir_program)

    def check_fused_ops(self):
        '''
        确认op是否正确融合
        Check whether the fused ops are correct.
        '''
        if self.fused_op_type is None or len(self.fused_ops) < 0:
            return
        op_names = [op.name() for op in self.pir_program.global_block().ops]
        if self.fused_op_type:
            self.assertTrue(self.fused_op_type in op_names, "error!")
        for fused_op in self.fused_ops:
            self.assertTrue(fused_op not in op_names, "error!")

    def check_output_with_place(
        self, place, need_translate_to_pir=False, atol=1e-5
    ):
        '''
        Check whether the fetched outputs of the origin program and the
        optimized program are the same.

        For inference model, the parameters are loaded to CPUPlace first,
        after apply all specified passes, then copy the parameters to GPUPlace.
        We can set startup_on_cpu to True to test inference pass.
        '''
        executor = paddle.static.Executor(place)
        # 转成新ir的 program
        import pdb

        pdb.set_trace()
        self.assertTrue(
            need_translate_to_pir is False and self.pir_program is not None,
            "error!",
        )
        if need_translate_to_pir and self.pir_program is None:
            self.pir_program = pir.translate_to_pir(self.main_program.desc)

        # 获取baseline的输出
        # baseline_outs, lods = self.run_program(executor, self.pir_program)
        # self.assertTrue(
        #     len(self.fetch_list) == len(baseline_outs),
        #     "Checking the number of fetchs failed. Expected: {}, Received: {}".format(
        #         len(self.fetch_list), len(baseline_outs)
        #     ),
        # )

        # 跑pass
        self.run_pir_pass()
        # 验证是否融合 1:op被融合某个op/被删除 2:融成新的op
        self.check_fused_ops()

        # 跑经过pass以后的结果
        # pass_outs, lods_opt = self.run_program(executor, self.pir_program)
        # self.assertTrue(
        #     len(self.fetch_list) == len(pass_outs),
        #     "Checking the number of fetchs failed. Expected: {}, Received: {}".format(
        #         len(self.fetch_list), len(pass_outs)
        #     ),
        # )
        # for i in range(len(self.fetch_list)):
        #     is_allclose = np.allclose(pass_outs[i], baseline_outs[i], atol=atol)
        #     if not is_allclose:
        #         a = pass_outs[i]
        #         b = baseline_outs[i]
        #         diff_mat = np.abs(a - b) / np.abs(a)
        #         max_diff = np.max(diff_mat)
        #         offset = np.argmax(diff_mat > atol)
        #         self.assertTrue(
        #             is_allclose,
        #             "Output (name: %s, shape: %s, dtype: %s) has diff at %s. The maximum diff is %e, first error element is %d, expected %e, but got %e"
        #             % (
        #                 self.fetch_list[i].name,
        #                 str(self.fetch_list[i].shape),
        #                 self.fetch_list[i].dtype,
        #                 str(place),
        #                 max_diff,
        #                 offset,
        #                 a.flatten()[offset],
        #                 b.flatten()[offset],
        #             ),
        #         )
