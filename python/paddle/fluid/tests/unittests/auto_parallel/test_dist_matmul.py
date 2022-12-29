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

import paddle
from paddle.distributed.fleet import auto
from paddle.fluid import program_guard
from paddle.fluid.backward import append_backward

paddle.enable_static()

mesh = auto.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])


def init_x_row(trans_x):
    if trans_x:
        x = paddle.static.data(name='x', shape=[10, 6, 8], dtype='float32')
        auto.shard_tensor(x, mesh, ["x", "y", None])

        return x
    else:
        x = paddle.static.data(name='x', shape=[10, 8, 6], dtype='float32')
        auto.shard_tensor(x, mesh, ["x", None, "y"])

        return x


def init_x_col(trans_x):
    if trans_x:
        x = paddle.static.data(name='x', shape=[6, 8], dtype='float32')
        auto.shard_tensor(x, mesh, [None, "x"])

        return x
    else:
        x = paddle.static.data(name='x', shape=[8, 6], dtype='float32')
        auto.shard_tensor(x, mesh, ["x", None])

        return x


def init_y_row(trans_y):
    if trans_y:
        y = paddle.static.data(name='y', shape=[4, 6], dtype='float32')
        auto.shard_tensor(y, mesh, [None, "y"])

        return y
    else:
        y = paddle.static.data(name='y', shape=[6, 4], dtype='float32')
        auto.shard_tensor(y, mesh, ["y", None])

        return y


def init_y_col(trans_y):
    if trans_y:
        y = paddle.static.data(name='y', shape=[4, 6], dtype='float32')
        auto.shard_tensor(y, mesh, ["y", None])

        return y
    else:
        y = paddle.static.data(name='y', shape=[6, 4], dtype='float32')
        auto.shard_tensor(y, mesh, [None, "y"])

        return y


def matmul_dp2mp2(init_x, init_y, trans_x, trans_y):
    main_program = paddle.fluid.Program()
    start_program = paddle.fluid.Program()
    with paddle.static.program_guard(main_program, start_program):
        x = init_x(trans_x)
        y = init_y(trans_y)
        x.stop_gradient = False
        y.stop_gradient = False
        out = paddle.matmul(x, y, transpose_x=trans_x, transpose_y=trans_y)
        loss = paddle.mean(out)
    return main_program, start_program, loss


def matmulv2_dp2mp2(init_x, init_y, trans_x, trans_y):
    main_program = paddle.fluid.Program()
    start_program = paddle.fluid.Program()
    with paddle.static.program_guard(main_program, start_program):
        x = init_x(trans_x)
        y = init_y(trans_y)
        x.stop_gradient = False
        y.stop_gradient = False
        out = paddle.matmul(x, y, transpose_x=trans_x, transpose_y=trans_y)
        loss = paddle.mean(out)
    return main_program, start_program, loss


def parallelizer(program_func, *args, **kwargs):
    from paddle.distributed.auto_parallel.completion import Completer
    from paddle.distributed.auto_parallel.dist_context import DistributedContext
    from paddle.distributed.auto_parallel.partitioner import Partitioner

    main_program, start_program, loss = program_func(*args, **kwargs)

    dist_context = DistributedContext()
    completer = Completer(dist_context)
    completer.complete_forward_annotation(main_program)
    dist_context.block_state.parse_forward_blocks(main_program)

    with program_guard(main_program, start_program):
        append_backward(loss, distop_context=dist_context.dist_op_context)
    completer.complete_backward_annotation(main_program)
    dist_context.block_state.parse_backward_blocks(main_program)

    partitioner = Partitioner(dist_context, 0)
    dist_main_prog, _, _ = partitioner.partition(
        main_program, start_program, []
    )

    return dist_main_prog, dist_context


class TestDistMatmul(unittest.TestCase):
    def check_col_program(self, main_program, dist_ctx):
        # [0, -1] * [-1, 1] --> [0, 1]
        ref_ops = [
            "c_identity",
            "matmul_v2",
            "reduce_mean",
            "fill_constant",
            "reduce_mean_grad",
            "matmul_v2_grad",
        ]
        ops = []
        block = main_program.global_block()
        for op in block.ops:
            ops.append(op.type)
            if op.type == "matmul_v2":
                out_name = op.output('Out')[0]
                out_var = block.vars[out_name]
                op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
                assert op_dist_attr.impl_idx == 0
                assert op_dist_attr.impl_type == "matmul_v2"
                out_dims_mapping = op_dist_attr.get_output_dims_mapping(
                    out_name
                )
                assert out_dims_mapping == [0, 1]
                tensor_dist_attr = dist_ctx.get_tensor_dist_attr_for_program(
                    out_var
                )
                assert tensor_dist_attr.dims_mapping == [0, 1]
            if op.type == "matmul_v2_grad":
                op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
                assert op_dist_attr.impl_idx == 0
                assert op_dist_attr.impl_type == "matmul_v2"

        assert ops == ref_ops

    def check_row_program(self, main_program, dist_ctx):
        # [0, -1, 1] * [1, -1] --> [0, -1, -1]
        ref_ops = [
            "matmul_v2",
            "c_allreduce_sum",
            "reduce_mean",
            "fill_constant",
            "reduce_mean_grad",
            "matmul_v2_grad",
        ]
        ops = []
        block = main_program.global_block()
        for op in block.ops:
            ops.append(op.type)
            if op.type == "matmul_v2":
                out_name = op.output('Out')[0]
                out_var = block.vars[out_name]
                op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
                assert op_dist_attr.impl_idx == 1
                assert op_dist_attr.impl_type == "matmul_v2"
                out_dims_mapping = op_dist_attr.get_output_dims_mapping(
                    out_name
                )
                assert out_dims_mapping == [0, -1, -1]
                tensor_dist_attr = dist_ctx.get_tensor_dist_attr_for_program(
                    out_var
                )
                assert tensor_dist_attr.dims_mapping == [0, -1, -1]
            if op.type == "matmul_v2_grad":
                op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
                assert op_dist_attr.impl_idx == 1
                assert op_dist_attr.impl_type == "matmul_v2"
        assert ops == ref_ops


class TestDistMatmulCol(TestDistMatmul):
    def init(self, trans_x, trans_y):
        dist_main_prog, dist_ctx = parallelizer(
            matmul_dp2mp2, init_x_col, init_y_col, trans_x, trans_y
        )
        return dist_main_prog, dist_ctx

    def test_matmul_col(self):
        dist_main_prog, dist_ctx = self.init(False, False)
        self.check_col_program(dist_main_prog, dist_ctx)

    def test_trans_x(self):
        dist_main_prog, dist_ctx = self.init(True, False)
        self.check_col_program(dist_main_prog, dist_ctx)

    def test_trans_y(self):
        dist_main_prog, dist_ctx = self.init(False, True)
        self.check_col_program(dist_main_prog, dist_ctx)

    def test_trans_x_trans_y(self):
        dist_main_prog, dist_ctx = self.init(True, True)
        self.check_col_program(dist_main_prog, dist_ctx)


class TestDistMatmulRow(TestDistMatmul):
    def init(self, trans_x, trans_y):
        dist_main_prog, dist_ctx = parallelizer(
            matmul_dp2mp2, init_x_row, init_y_row, trans_x, trans_y
        )
        return dist_main_prog, dist_ctx

    def test_matmul_row(self):
        dist_main_prog, dist_ctx = self.init(False, False)
        self.check_row_program(dist_main_prog, dist_ctx)

    def test_trans_x(self):
        dist_main_prog, dist_ctx = self.init(True, False)
        self.check_row_program(dist_main_prog, dist_ctx)

    def test_trans_y(self):
        dist_main_prog, dist_ctx = self.init(False, True)
        self.check_row_program(dist_main_prog, dist_ctx)

    def test_trans_x_trans_y(self):
        dist_main_prog, dist_ctx = self.init(True, True)
        self.check_row_program(dist_main_prog, dist_ctx)


class TestDistMatmulV2(unittest.TestCase):
    def check_col_program(self, main_program, dist_ctx):
        # [0, -1] * [-1, 1] --> [0, 1]
        ref_ops = [
            "c_identity",
            "matmul_v2",
            "reduce_mean",
            "fill_constant",
            "reduce_mean_grad",
            "matmul_v2_grad",
        ]
        ops = []
        block = main_program.global_block()
        for op in block.ops:
            ops.append(op.type)
            if op.type == "matmul_v2":
                out_name = op.output('Out')[0]
                out_var = block.vars[out_name]
                op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
                assert op_dist_attr.impl_idx == 0
                assert op_dist_attr.impl_type == "matmul_v2"
                out_dims_mapping = op_dist_attr.get_output_dims_mapping(
                    out_name
                )
                assert out_dims_mapping == [0, 1]
                tensor_dist_attr = dist_ctx.get_tensor_dist_attr_for_program(
                    out_var
                )
                assert tensor_dist_attr.dims_mapping == [0, 1]
            if op.type == "matmul_v2_grad":
                op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
                assert op_dist_attr.impl_idx == 0
                assert op_dist_attr.impl_type == "matmul_v2"

        assert ops == ref_ops

    def check_row_program(self, main_program, dist_ctx):
        # [0, -1, 1] * [1, -1] --> [0, -1, -1]
        ref_ops = [
            "matmul_v2",
            "c_allreduce_sum",
            "reduce_mean",
            "fill_constant",
            "reduce_mean_grad",
            "matmul_v2_grad",
        ]
        ops = []
        block = main_program.global_block()
        for op in block.ops:
            ops.append(op.type)
            if op.type == "matmul_v2":
                out_name = op.output('Out')[0]
                out_var = block.vars[out_name]
                op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
                assert op_dist_attr.impl_idx == 1
                assert op_dist_attr.impl_type == "matmul_v2"
                out_dims_mapping = op_dist_attr.get_output_dims_mapping(
                    out_name
                )
                assert out_dims_mapping == [0, -1, -1]
                tensor_dist_attr = dist_ctx.get_tensor_dist_attr_for_program(
                    out_var
                )
                assert tensor_dist_attr.dims_mapping == [0, -1, -1]
            if op.type == "matmul_v2_grad":
                op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
                assert op_dist_attr.impl_idx == 1
                assert op_dist_attr.impl_type == "matmul_v2"
        assert ops == ref_ops


class TestDistMatmulV2Col(TestDistMatmulV2):
    def init(self, trans_x, trans_y):
        dist_main_prog, dist_ctx = parallelizer(
            matmulv2_dp2mp2, init_x_col, init_y_col, trans_x, trans_y
        )
        return dist_main_prog, dist_ctx

    def test_matmul_col(self):
        dist_main_prog, dist_ctx = self.init(False, False)
        self.check_col_program(dist_main_prog, dist_ctx)

    def test_trans_x(self):
        dist_main_prog, dist_ctx = self.init(True, False)
        self.check_col_program(dist_main_prog, dist_ctx)

    def test_trans_y(self):
        dist_main_prog, dist_ctx = self.init(False, True)
        self.check_col_program(dist_main_prog, dist_ctx)

    def test_trans_x_trans_y(self):
        dist_main_prog, dist_ctx = self.init(True, True)
        self.check_col_program(dist_main_prog, dist_ctx)


class TestDistMatmulV2Row(TestDistMatmulV2):
    def init(self, trans_x, trans_y):
        dist_main_prog, dist_ctx = parallelizer(
            matmulv2_dp2mp2, init_x_row, init_y_row, trans_x, trans_y
        )
        return dist_main_prog, dist_ctx

    def test_matmul_row(self):
        dist_main_prog, dist_ctx = self.init(False, False)
        self.check_row_program(dist_main_prog, dist_ctx)

    def test_trans_x(self):
        dist_main_prog, dist_ctx = self.init(True, False)
        self.check_row_program(dist_main_prog, dist_ctx)

    def test_trans_y(self):
        dist_main_prog, dist_ctx = self.init(False, True)
        self.check_row_program(dist_main_prog, dist_ctx)

    def test_trans_x_trans_y(self):
        dist_main_prog, dist_ctx = self.init(True, True)
        self.check_row_program(dist_main_prog, dist_ctx)


if __name__ == "__main__":
    unittest.main()
