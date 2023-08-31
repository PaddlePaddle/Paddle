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
from paddle.distributed.passes.pass_utils import get_skip_gc_vars, split_program
from paddle.fluid import core
from paddle.fluid.executor import _add_feed_fetch_ops, _StandaloneExecutor

paddle.enable_static()


def build_program():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        # data -> [matmul] -> out ->[add] -> add_out
        with paddle.static.device_guard('gpu'):
            data = paddle.ones([1024, 2048], dtype='float32', name='data')
            weight = paddle.randn([2048, 2048], name='weight')  # gpu
            matmul_out = paddle.matmul(data, weight, name='matmul_out')  # gpus
            bias = paddle.ones([1024, 2048], dtype='float32', name='bias')
            add_out = paddle.add(matmul_out, bias, name='add_out')
            # add_out -> [sub] -> sub_out -> [tanh] -> tanh_out
            sub_out = paddle.subtract(add_out, data, name='sub_out')
            tanh_out = paddle.tanh(sub_out, name='tanh_out')
            bias_1 = paddle.add(bias, sub_out, name='bias_1')
            out_before = paddle.tanh(bias_1, name='out_before')
            out_last = paddle.subtract(tanh_out, data, name='out_last')
            out_last2 = paddle.matmul(out_last, weight, name="matmul_2_out")

            out = paddle.add(out_before, out_last2, name='out')
            mean = paddle.mean(out, name='mean_out')

    return main_program, startup_program, [mean]


class TestMannulEvent(unittest.TestCase):
    """
    fill_constant(def)     gaussian_random(def)
      |     |        |        |
      |     |        matmul_v2(s1) fill_constant(def)
      |     |        |        |     |  |
      |     |     elementwise_add(s1)  |
      |     |           |              |
      |  elementwise_sub(s1)           |
      |     |           |              |
      |  tanh(s1)     elementwise_add(s1)
      |     |                  |
    elementwise_sub(s1)      tanh(s1)
            |                  |
        matmul_v2(s1)          |
                 |             |    ---split prog----
                elementwise_add(s2)
                        |
                  reduce_mean(s2)
    """

    def setUp(self):
        self.steps = 3
        self.place_desc = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.place = core.Place()
        self.place.set_place(self.place_desc)

    def set_custom_stream(self, prog):
        op_index_for_stream1 = [2, 4, 5, 6, 7, 8, 9, 10]
        op_index_for_stream2 = [11, 12]
        ops = prog.global_block().ops
        for op_index in op_index_for_stream1:
            ops[op_index].dist_attr.execution_stream = "s1"
            ops[op_index].dist_attr.stream_priority = 0
        for op_index in op_index_for_stream2:
            ops[op_index].dist_attr.execution_stream = "s2"
            ops[op_index].dist_attr.stream_priority = -1

    def split_program(self, prog, apply_mannual_event=False):
        # split two subprograms
        segment_0_events = {8: "e8", 10: "e10"}
        segment_1_wait_events = {11: ["e8", "e10"]}
        prog_block = prog.global_block()
        ops = prog_block.ops
        for idx, op in enumerate(ops):
            if idx in segment_0_events and apply_mannual_event:
                op.desc.dist_attr.has_mannual_event = True
                op.desc.dist_attr.event_name = segment_0_events[idx]
            if idx in segment_1_wait_events and apply_mannual_event:
                op.desc.dist_attr.mannual_wait_events = segment_1_wait_events[
                    idx
                ]
        main_progs, _, _ = split_program(prog, [11])
        return main_progs

    def create_standalone_exe(self, main_progs, startup_progs, fetch_list):
        micro_batch_num = 1
        micro_batch_id = 0
        job_list = []
        prog_num = len(main_progs)
        fetch_op_num = len(fetch_list)
        skip_gc_vars = get_skip_gc_vars(main_progs)

        if prog_num == 1:  # single prog
            main_progs[0] = _add_feed_fetch_ops(
                main_progs[0],
                [],
                fetch_list,
                "feed",
                "fetch",
                use_fetch_v2=True,
            )
            op_num = len(main_progs[0].block(0).ops)
            fetch_op_indics = list(range(op_num - fetch_op_num, op_num))
        else:
            main_progs[-1] = _add_feed_fetch_ops(
                main_progs[-1],
                [],
                fetch_list,
                "feed",
                "fetch",
                use_fetch_v2=True,
            )
            op_num = len(main_progs[-1].block(0).ops)
            fetch_op_indics = list(range(op_num - fetch_op_num, op_num))

        # create jobs
        for program_id in range(prog_num):
            job = core.Job(f"prog_{program_id}")
            job.set_skip_gc_vars(skip_gc_vars[program_id])
            # Set col_attr info for fetch_op to fetch the correct data after running multiple micro batch
            if program_id == prog_num - 1:
                for i in range(fetch_op_num):
                    job.set_col_attr_for_fetch_op(
                        fetch_op_indics[i],
                        i * micro_batch_num + micro_batch_id,
                    )
            job_list.append(job)

        type_to_program = {}
        for program_id in range(prog_num):
            type_to_program[f"prog_{program_id}"] = main_progs[program_id].desc

        plan = core.Plan(job_list, type_to_program)
        scope = core.Scope()
        main_exe = _StandaloneExecutor(self.place, plan, scope)
        return main_exe

    def run_program(
        self,
        apply_custom_stream=False,
        split_prog=False,
        apply_mannual_event=False,
    ):
        paddle.seed(2022)
        main_program, startup_program, fetch_list = build_program()
        self.assertEqual(len(startup_program.global_block().ops), 0)

        if apply_custom_stream:
            self.set_custom_stream(main_program)
        main_progs = [main_program]
        startup_progs = [startup_program]
        if apply_custom_stream and split_prog:
            main_progs = self.split_program(main_program, apply_mannual_event)
        outs = []
        exe = self.create_standalone_exe(main_progs, startup_progs, fetch_list)
        for i in range(self.steps):
            outs.append(exe.run(feed_names=[]))
        return outs

    def test_result(self):
        if not core.is_compiled_with_cuda():
            return

        baselines = self.run_program()
        stream_outs = self.run_program(apply_custom_stream=True)
        split_outs = self.run_program(apply_custom_stream=True, split_prog=True)
        mannual_outs = self.run_program(
            apply_custom_stream=True, split_prog=True, apply_mannual_event=True
        )
        for bl, out0, out1, out2 in zip(
            baselines, stream_outs, split_outs, mannual_outs
        ):
            self.assertEqual(bl[0], out0[0])
            self.assertEqual(bl[0], out2[0])
            # self.assertNotEqual(bl[0], out1[0])


if __name__ == "__main__":
    unittest.main()
