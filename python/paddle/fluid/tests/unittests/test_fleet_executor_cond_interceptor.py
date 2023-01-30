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
import paddle.fluid.core as core
from paddle.distributed.fleet.fleet_executor_utils import TaskNode

paddle.enable_static()


def cond(i, ten):
    return i < ten


def body(i, ten):
    i = i + 1
    return [i, ten]


class TestFleetExecutor(unittest.TestCase):
    def test_cond_interceptor(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.full(
                shape=[1], fill_value=0, dtype='int64'
            )  # loop counter
            ten = paddle.full(
                shape=[1], fill_value=10, dtype='int64'
            )  # loop length
            i, ten = paddle.static.nn.while_loop(cond, body, [i, ten])

        program_a = paddle.static.Program()
        program_b = paddle.static.Program()

        for var_name in main_program.block(0).vars:
            if var_name != "_generated_var_0":
                var = main_program.block(0).var(var_name)
                program_a.block(0).create_var(
                    name=var_name,
                    shape=var.shape,
                    dtype=var.dtype,
                    stop_gradient=var.stop_gradient,
                )
                program_b.block(0).create_var(
                    name=var_name,
                    shape=var.shape,
                    dtype=var.dtype,
                    stop_gradient=var.stop_gradient,
                )

        for op in main_program.block(0).ops:
            if op.type != "while":
                program_a.block(0).append_op(
                    type=op.type,
                    inputs=op.desc.inputs(),
                    outputs=op.desc.outputs(),
                    attrs=op.all_attrs(),
                )

        for var_name in main_program.block(1).vars:
            var = main_program.block(1).var(var_name)
            program_b.block(0).create_var(
                name=var_name,
                shape=var.shape,
                dtype=var.dtype,
                stop_gradient=var.stop_gradient,
            )

        for op in main_program.block(1).ops:
            program_b.block(0).append_op(
                type=op.type,
                inputs=op.desc.inputs(),
                outputs=op.desc.outputs(),
                attrs=op.all_attrs(),
            )

        cond_var_name = "tmp_0"
        num_micro_batches = 3

        task_a = TaskNode(
            0,
            num_micro_batches,
            node_type="Compute",
            task_id=0,
            program=program_a,
            lazy_initialize=True,
        )
        task_b = TaskNode(
            0,
            num_micro_batches,
            node_type="Cond",
            task_id=1,
            program=paddle.static.Program(),
            cond_var_name=cond_var_name,
            lazy_initialize=True,
        )
        task_c = TaskNode(
            0,
            num_micro_batches,
            node_type="Compute",
            task_id=2,
            program=program_b,
            lazy_initialize=True,
        )
        task_d = TaskNode(
            0,
            num_micro_batches,
            node_type="Compute",
            task_id=3,
            program=paddle.static.Program(),
            lazy_initialize=True,
        )
        task_e = TaskNode(
            0,
            num_micro_batches,
            node_type="Compute",
            task_id=4,
            program=paddle.static.Program(),
            lazy_initialize=True,
        )

        task_a.add_downstream_task(task_b.task_id(), 2)
        task_b.add_upstream_task(task_a.task_id(), 2)
        task_b.add_downstream_task(task_c.task_id(), 100)
        task_c.add_upstream_task(task_b.task_id(), 100)
        task_c.add_downstream_task(task_d.task_id(), 2)
        task_d.add_upstream_task(task_c.task_id(), 2)
        task_d.add_downstream_task(task_b.task_id(), 100, core.DependType.LOOP)
        task_b.add_upstream_task(task_d.task_id(), 100, core.DependType.LOOP)
        task_b.add_downstream_task(
            task_e.task_id(), 100, core.DependType.STOP_LOOP
        )
        task_e.add_upstream_task(
            task_b.task_id(), 100, core.DependType.STOP_LOOP
        )

        main_program._pipeline_opt = {
            "fleet_opt": {
                'tasks': [task_a, task_b, task_c, task_d, task_e],
                'task_id_to_rank': {
                    task_a.task_id(): 0,
                    task_b.task_id(): 0,
                    task_c.task_id(): 0,
                    task_d.task_id(): 0,
                    task_e.task_id(): 0,
                },
                'num_micro_batches': num_micro_batches,
            },
        }

        place = paddle.fluid.CUDAPlace(0)
        exe = paddle.fluid.Executor(place)
        exe.run(main_program)


if __name__ == "__main__":
    unittest.main()
