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

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.pir_pass import (
    ReshardPasses,
)
from paddle.distributed.auto_parallel.static.utils import set_all_ops_op_role
from paddle.distributed.fleet.meta_optimizers.common import OpRole


class TestReshardRToS:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._shard = eval(os.getenv("shard"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")

        a = paddle.ones(self._shape)

        in_placements = [dist.Replicate()]
        input_tensor = dist.shard_tensor(a, self._mesh, in_placements)

        out_placements = [dist.Shard(self._shard)]

        out = dist.reshard(input_tensor, self._mesh, out_placements)

        out_shape = list(self._shape)

        if out_shape[self._shard] % 2 == 0:
            out_shape[self._shard] = out_shape[self._shard] // 2
            np.testing.assert_equal(out.numpy(), input_tensor.numpy())
        else:
            out_shape[self._shard] = (
                out_shape[self._shard] // 2
                if dist.get_rank() == 1
                else out_shape[self._shard] // 2 + 1
            )

        assert np.equal(out.shape, input_tensor.shape).all()
        assert np.equal(out._local_shape, out_shape).all()

    def run_pir_test_case(self):
        paddle.enable_static()
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        BATCH_SIZE = 2
        SEQ_LEN = 4
        HIDDEN_SIZE = 8
        MP_SIZE = 2

        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
                input = paddle.static.data(
                    name='input', shape=[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
                )
                w0 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE, HIDDEN_SIZE],
                    name="w0",
                    initializer=paddle.nn.initializer.Uniform(),
                )

                input_tensor = dist.shard_tensor(
                    w0, self._mesh, [dist.Replicate()]
                )
                paddle._C_ops.reshard(
                    input_tensor, self._mesh, [dist.Shard(self._shard)]
                )
            dist_program = main_program.clone()
            set_all_ops_op_role(dist_program.global_block(), OpRole.Forward)
            ReshardPasses.apply_reshard_pass(dist_program)
            np.testing.assert_equal(dist_program.num_ops(), 6)
            old_ops = [op.name() for op in main_program.global_block().ops]
            new_ops = [op.name() for op in dist_program.global_block().ops]

            assert 'pd_op.slice' in new_ops
            assert 'dist_op.reshard' not in new_ops
            assert 'dist_op.reshard' in old_ops


if __name__ == '__main__':
    TestReshardRToS().run_test_case()
    TestReshardRToS().run_pir_test_case()
