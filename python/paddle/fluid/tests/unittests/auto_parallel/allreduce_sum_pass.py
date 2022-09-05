# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import random
import numpy as np
import paddle

import paddle.distributed.fleet as fleet
import paddle.distributed.auto_parallel as auto

from paddle.fluid import program_guard
from paddle.distributed.auto_parallel.engine import Engine
from get_gpt_model import generate_model, create_data_holder, FakeDataset
from paddle.distributed.auto_parallel.process_group import get_all_process_groups, _g_process_group_map, ProcessGroup
from paddle.distributed.auto_parallel.dist_context import DistributedContext, set_default_distributed_context
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr

paddle.enable_static()


def print_param(program):
    from paddle.fluid.framework import Parameter

    def is_parameter(var):
        return isinstance(var, Parameter)

    def get_tensor(var):
        t = paddle.fluid.global_scope().find_var(var.name).get_tensor()
        return np.array(t)

    def get_name(var):
        return var.name

    parameter_list = list(filter(is_parameter, program.list_vars()))
    for p in sorted(parameter_list, key=get_name):
        print(p.name)
        print(get_tensor(p))


def reset():
    _g_process_group_map.clear()
    _g_process_group_map[0] = ProcessGroup(0, [])
    set_default_distributed_context(DistributedContext())
    paddle.fluid.framework.switch_main_program(paddle.static.Program())
    paddle.fluid.framework.switch_startup_program(paddle.static.Program())


class TestAllreduceSum(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.batch_num = 5
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)
        # fleet.init(is_collective=True)
        engine.mode = "train"
        engine._executor.run(engine.startup_program)

    def run_program(self, engine):

        dataloader = engine._create_dataloader(self.dataset, self.batch_size)
        loss = engine.fetch_vars['loss'][0]

        group = None
        all_process_groups = get_all_process_groups()
        for process_group in all_process_groups:
            if paddle.distributed.get_rank() in process_group.ranks:
                group = process_group
                break

        dataloader._inner_dataloader.start()
        while True:
            step = 0
            try:
                while step < self.batch_num:
                    out = engine._executor.run(engine.main_program,
                                               fetch_list=[loss.name])
                    print("****out:", out)
                    paddle.disable_static()
                    tensor = paddle.to_tensor(out[0])
                    allreduce_out = paddle.distributed.all_reduce(tensor,
                                                                  group=group)
                    paddle.enable_static()
                    step += 1
            except paddle.framework.core.EOFException:
                dataloader._inner_dataloader.reset()
                break
        return allreduce_out

    def get_engine(self, mode):
        reset()

        opt = paddle.optimizer.AdamW(learning_rate=0.00001)
        model, loss = generate_model(mode)
        inputs_spec, labels_spec = create_data_holder(self.batch_size)

        engine = Engine(model, inputs_spec, labels_spec, strategy=None)
        engine.prepare(optimizer=opt, loss=loss)
        self.init(engine)
        return engine

    def check_program(self, program):
        start_idx = None
        for idx, op in enumerate(program.global_block().ops):
            if op.type == "sum" and op.output_arg_names[
                    0] == "word_embeddings@GRAD":
                start_idx = idx
            if start_idx:
                if op.type == "c_allreduce_sum" and op.output_arg_names[
                        0] == "word_embeddings@GRAD":
                    return

        raise ValueError("program is not changed")

    def test_allreduce_result(self):

        # serial training
        serial_engine = self.get_engine("serial")
        print("param" + "******" * 30)
        print_param(serial_engine.main_program)
        serial_out = self.run_program(serial_engine)
        print("out" + "******" * 30)
        print(serial_out)
        # print("program" + "******"*30)
        # print_program_with_dist_attr(serial_engine.main_program, serial_engine.dist_context)

        # dp2 training
        dp_engine = self.get_engine("dp")
        print("param" + "******" * 30)
        print_param(dp_engine.main_program)
        dp_out = self.run_program(dp_engine)
        print("out" + "******" * 30)
        print(dp_out)
        # print("program" + "******"*30)
        # print_program_with_dist_attr(dp_engine.main_program, dp_engine.dist_context)

        self.check_program(dp_engine.main_program)
        np.testing.assert_allclose(np.array(serial_out) / 2,
                                   np.array(dp_out) / 2,
                                   rtol=1e-05,
                                   atol=1e-08)


if __name__ == "__main__":
    unittest.main()
