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

import copy
import unittest

import paddle
from paddle.distributed.fleet import auto
from paddle.distributed.passes import new_pass

paddle.enable_static()


def make_program():
    main_program = paddle.base.Program()
    start_program = paddle.base.Program()
    with paddle.static.program_guard(main_program, start_program):
        x = paddle.static.data(name='x', shape=[4, 6, 8], dtype='float32')
        y = paddle.static.data(name='y', shape=[4, 6, 6], dtype='float32')
        z = paddle.static.data(name='y', shape=[4, 6, 6], dtype='float32')

        auto.shard_tensor(x, auto.ProcessMesh([0], ['d0']), [None, None, None])

        out0 = paddle.static.nn.fc(
            x,
            size=6,
            num_flatten_dims=2,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            ),
        )
        where_0 = paddle.where(y > 1, y, out0)

        out1 = paddle.static.nn.fc(
            out0,
            size=6,
            num_flatten_dims=2,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            ),
        )
        where_1 = paddle.where(y > 1, y, out1)

        paddle.assign(where_1, where_0)

    return main_program, start_program


def parallelizer(program_func, rank):
    from paddle.distributed.auto_parallel.static.completion import Completer
    from paddle.distributed.auto_parallel.static.dist_context import (
        DistributedContext,
    )
    from paddle.distributed.auto_parallel.static.partitioner import Partitioner

    main_program, start_program = program_func()

    dist_context = DistributedContext()
    completer = Completer(dist_context)
    completer.complete_forward_annotation(main_program)
    dist_context.block_state.parse_forward_blocks(main_program)

    strategy = auto.Strategy()
    amp = strategy.amp
    amp.enable = True
    amp.dtype = "float16"
    amp.level = "o2"
    amp.init_loss_scaling = 32768
    amp.use_fp16_guard = False
    amp.custom_black_list = ['where']

    config = copy.deepcopy(strategy.amp.to_dict())
    config["dist_context"] = dist_context
    config["params_grads"] = []
    config["loss"] = None
    config["base_opt"] = None
    auto_parallel_fp16_pass = new_pass("auto_parallel_fp16", config)
    auto_parallel_fp16_pass.apply([main_program], [start_program], None)

    partitioner = Partitioner(dist_context, rank)
    dist_main_prog, _, _ = partitioner.partition(
        main_program, start_program, []
    )

    return dist_main_prog, dist_context


class TestFp16Assign(unittest.TestCase):
    def assert_fp32_dtype(self, block, op):
        for slot in op.input_names:
            for name in op.input(slot):
                if block.vars[name].dtype == paddle.bool:
                    continue
                assert block.vars[name].dtype == paddle.float32
        for slot in op.output_names:
            for name in op.output(slot):
                if block.vars[name].dtype == paddle.bool:
                    continue
                assert block.vars[name].dtype == paddle.float32

    def assert_fp16_dtype(self, block, op):
        for slot in op.input_names:
            if slot == "Condition":
                continue
            for name in op.input(slot):
                if block.vars[name].dtype == paddle.bool:
                    continue
                assert block.vars[name].dtype == paddle.float16
        for slot in op.output_names:
            for name in op.output(slot):
                if block.vars[name].dtype == paddle.bool:
                    continue
                assert block.vars[name].dtype == paddle.float16

    def test_fp16_assign(self):
        dist_main_prog, dist_context = parallelizer(make_program, 0)
        block = dist_main_prog.global_block()
        for op in block.ops:
            if op.type == "cast":
                continue
            if op.type == "where":
                self.assert_fp32_dtype(block, op)
            elif op.type == "assign":
                self.assert_fp32_dtype(block, op)
            else:
                self.assert_fp16_dtype(block, op)


if __name__ == "__main__":
    unittest.main()
