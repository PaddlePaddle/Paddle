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
from test_utils import (
    BATCH_SIZE,
    CLASS_NUM,
    IMAGE_SIZE,
    DemoNet,
    create_data_loader,
)

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.auto_parallel.static.pir_pass import (
    ReshardPasses,
)
from paddle.distributed.auto_parallel.static.utils import set_all_ops_op_role
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.framework import core


class TestReshardPToR:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        dev_ctx = core.DeviceContext.create(place)
        a = paddle.ones(self._shape)

        input_tensor = dist.shard_tensor(a, self._mesh, [dist.Partial()])
        out = dist.reshard(input_tensor, self._mesh, [dist.Replicate()])

        assert np.equal(out.shape, input_tensor.shape).all()
        np.testing.assert_equal(out._local_value().numpy(), a.numpy())

    def run_pir_static_test_case(self):
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
                    w0, self._mesh, [dist.Partial()]
                )
                reshard_tensor = paddle._C_ops.reshard(
                    input_tensor, self._mesh, [dist.Replicate()]
                )
            set_all_ops_op_role(main_program.global_block(), OpRole.Forward)
            ReshardPasses.apply_reshard_pass(main_program)
        np.testing.assert_equal(main_program.num_ops(), 4)
        ops = main_program.global_block().ops
        np.testing.assert_equal(
            [op.name() for op in ops],
            [
                'builtin.parameter',
                'pd_op.data',
                'dist_op.shard_tensor',
                'pd_op.all_reduce',
            ],
        )

        for op in ops:
            if (
                op.name() == 'pd_op.all_reduce'
                and op.int_attr('reduce_type') == dist.ReduceOp.SUM
            ):
                # check op dist_attr
                assert op.dist_attr.num_operands() == 1
                assert op.dist_attr.num_results() == 1

                op_operand_dist_attr = op.dist_attr.operand(
                    0
                ).as_tensor_dist_attr()
                op_result_dist_attr = op.dist_attr.result(
                    0
                ).as_tensor_dist_attr()

                assert op.dist_attr.process_mesh == self._mesh
                assert op_operand_dist_attr.process_mesh == self._mesh
                assert op_operand_dist_attr.dims_mapping == [-1, -1]
                assert op_operand_dist_attr.partial_status == {
                    0: paddle.distributed.ReduceType.kRedSum
                }

                assert op_result_dist_attr.process_mesh == self._mesh
                assert op_result_dist_attr.dims_mapping == [-1, -1]
                assert op_result_dist_attr.partial_status == {}

                # check op_value dist_attr
                assert op.num_results() == 1
                op_value = op.result(0)
                assert op_value.is_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.dist_attr().process_mesh == self._mesh
                assert op_value.dist_attr().dims_mapping == [-1, -1]
                assert op_value.dist_attr().partial_status == {}

    def run_pir_to_static_test_case(self):
        paddle.disable_static()
        in_dygraph_mode = paddle.in_dynamic_mode()
        with paddle.pir_utils.IrGuard():
            if in_dygraph_mode:
                paddle.disable_static()

            mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
            layer = DemoNet(mesh)
            opt = paddle.optimizer.SGD(
                learning_rate=0.1, parameters=layer.parameters()
            )
            loss_fn = nn.MSELoss()
            loader = create_data_loader()
            dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
            dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)

            mode = "train"
            dist_model.train()
            main_program = dist_model._engine._pir_dist_main_progs["train"]

        relu_idx = 0
        matmul_idx = 0
        data_idx = 0
        matmul_grad_idx = 0
        sgd_idx = 0
        ops = main_program.global_block().ops

        backward_op_list = [
            "pd_op.sgd_",
            "pd_op.sgd_",
            "pd_op.relu_grad",
            "pd_op.all_reduce",
            "pd_op.matmul_grad",
            "pd_op.relu_grad",
            "pd_op.matmul_grad",
            "pd_op.relu_grad",
            "pd_op.subtract_grad",
            "pd_op.square_grad",
            "pd_op.mean_grad",
        ]
        index = -1
        for op_name in backward_op_list:
            assert ops[index].name() == op_name
            index = index - 1

        for op in ops:
            # skip shadow_output
            if op.num_results() == 0:
                continue
            tensor = op.result(0)
            # while tensor's stop_gradient is true, the corresponding grad tensor is initialized.
            if not tensor.initialized():
                continue
            assert tensor.is_dist_dense_tensor_type()
            assert tensor.dist_attr().process_mesh.shape == [2]
            assert tensor.dist_attr().process_mesh.process_ids == [0, 1]

            if op.name() == 'pd_op.data':
                if data_idx != 0:
                    assert tensor.dist_attr().dims_mapping == [-1, -1]
                    assert tensor.dist_attr().partial_dims == set()
                data_idx += 1
            elif op.name() == 'builtin.parameter':
                assert tensor.is_dense_tensor_type()
                assert tensor.is_dist_dense_tensor_type()
                assert tensor.is_dist_dense_tensor_type()
                assert tensor.dist_attr().process_mesh.shape == [2]
                assert tensor.dist_attr().process_mesh.process_ids == [0, 1]
                if tensor.shape == [IMAGE_SIZE, IMAGE_SIZE]:
                    assert tensor.dist_attr().dims_mapping == [-1, 0]
                elif tensor.shape == [IMAGE_SIZE, CLASS_NUM]:
                    assert tensor.dist_attr().dims_mapping == [0, -1]
                assert tensor.dist_attr().partial_dims == set()
            if op.name() == 'pd_op.relu':
                if relu_idx == 0:
                    assert tensor.dist_attr().dims_mapping == [-1, -1]
                    assert tensor.dist_attr().partial_dims == set()
                    assert tensor._local_shape == [BATCH_SIZE, IMAGE_SIZE]
                elif relu_idx == 1:
                    assert tensor.dist_attr().dims_mapping == [-1, 0]
                    assert tensor.dist_attr().partial_dims == set()
                    assert tensor._local_shape == [BATCH_SIZE, IMAGE_SIZE // 2]
                elif relu_idx == 2:
                    assert tensor.dist_attr().dims_mapping == [-1, -1]
                    assert tensor.dist_attr().partial_dims == set()
                    assert tensor._local_shape == [BATCH_SIZE, CLASS_NUM]
                relu_idx += 1
            if op.name() == 'pd_op.matmul':
                if matmul_idx == 0:
                    assert tensor.dist_attr().dims_mapping == [-1, 0]
                    assert tensor.dist_attr().partial_dims == set()
                    assert tensor._local_shape == [BATCH_SIZE, IMAGE_SIZE // 2]
                elif matmul_idx == 1:
                    assert tensor.dist_attr().dims_mapping == [-1, -1]
                    assert tensor.dist_attr().partial_dims == {0}
                    assert tensor._local_shape == [BATCH_SIZE, CLASS_NUM]
                matmul_idx += 1
            if op.name() == 'pd_op.matmul_grad':
                if matmul_grad_idx == 0:
                    assert tensor.dist_attr().dims_mapping == [-1, 0]
                    assert tensor.dist_attr().partial_dims == set()
                    assert tensor._local_shape == [BATCH_SIZE, CLASS_NUM]
                elif matmul_grad_idx == 1:
                    assert tensor.dist_attr().dims_mapping == [-1, -1]
                    assert tensor.dist_attr().partial_dims == {0}
                    assert tensor._local_shape == [BATCH_SIZE, IMAGE_SIZE]
                matmul_grad_idx += 1
            if op.name() == 'pd_op.sgd_':
                if sgd_idx == 0:
                    assert tensor.dist_attr().dims_mapping == [0, -1]
                    assert tensor.dist_attr().partial_dims == set()
                    assert tensor._local_shape == [IMAGE_SIZE // 2, CLASS_NUM]
                elif sgd_idx == 1:
                    assert tensor.dist_attr().dims_mapping == [-1, 0]
                    assert tensor.dist_attr().partial_dims == set()
                    assert tensor._local_shape == [IMAGE_SIZE, IMAGE_SIZE // 2]
                sgd_idx += 1


if __name__ == '__main__':
    TestReshardPToR().run_test_case()
    TestReshardPToR().run_pir_to_static_test_case()
    TestReshardPToR().run_pir_static_test_case()
