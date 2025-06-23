# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.pipelining._backward import (
    stage_backward,
    stage_backward_input,
    stage_backward_weight,
)
from paddle.distributed.auto_parallel.pipelining.stage import (
    PipelineStage,
    _RecvInfo,
)
from paddle.distributed.auto_parallel.pipelining.utils import (
    PipeliningShapeError,
    TensorMeta,
    _detach_and_keep_grad,
    _friendly_debug_info,
    _get_stage_mesh,
    _validate_tensor_metadata,
    _validate_tensors_metadata,
    _zero_initialize_with_meta,
)
from paddle.io import Dataset

if TYPE_CHECKING:  # 添加类型检查块
    from paddle.distributed.communication.group import Group
logger = logging.getLogger(__name__)


def fix_seeds(seed=2025):
    """Fix random seeds to ensure reproducibility"""
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _batch_p2p(p2p_ops, desc=None):
    # TODO(zhengtianyu): 等合入Scheduler后，删除该函数
    """Execute batch point-to-point communication operations"""
    if len(p2p_ops) == 0:
        return None
    desc_str = f"{desc}, " if desc else ""
    logger.debug("batch_p2p %s%s", desc_str, p2p_ops)
    return dist.batch_isend_irecv(p2p_ops).pop()


def _sorted_batch_p2p(p2p_ops, desc=None):
    # TODO(zhengtianyu): 等合入Scheduler后，删除该函数
    """Sort and execute batch point-to-point communication by peer rank"""
    ops_by_peer: dict[int, list[dist.P2POp]] = defaultdict(list)
    work_by_peer: dict[int, dist.Work] = {}
    if len(p2p_ops) == 0:
        return work_by_peer

    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = _batch_p2p(ops, desc=desc)

    return work_by_peer


class MyModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 8, bias_attr=False)
        self.linear2 = nn.Linear(8, 8, bias_attr=False)
        self.linear3 = nn.Linear(8, 8)
        self.linear4 = nn.Linear(8, 8)

    def forward(self, x, debug_str=None):
        if hasattr(self, 'linear1'):
            if debug_str:
                logger.debug(f"{debug_str} linear1")
            x = self.linear1(x)
            x = self.linear2(x)
        if hasattr(self, 'linear3'):
            x = self.linear3(x)
            x = self.linear4(x)
        return x


class PPMyModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.mesh = paddle.distributed.ProcessMesh([0, 1], dim_names=["pp"])
        self.num_layers = 4
        self.num_layers_per_card = self.num_layers // 2

        # Create layers same as MyModel
        self.linears = nn.LayerList()
        for i in range(self.num_layers):
            if i // 2 == 0:
                linear = nn.Linear(8, 8, bias_attr=False)
            else:
                linear = nn.Linear(8, 8)

            # Mark network parameters
            linear.weight = dist.shard_tensor(
                linear.weight,
                self.get_pp_mesh(i),
                [dist.Replicate()],
            )

            self.linears.append(linear)

    def get_pp_mesh(self, layer_index):
        # layer_index=0-3 corresponds to mesh_idx 0,0,1,1 respectively
        mesh_idx = int(layer_index / (self.num_layers / 2))
        return self.mesh[mesh_idx]

    def forward(self, x):
        x.stop_gradient = False
        out = x

        for i in range(self.num_layers):
            # Mark intermediate variables, reshard when device switching is needed
            if i % self.num_layers_per_card == 0 and i > 0:
                out = dist.reshard(out, self.get_pp_mesh(i), [dist.Replicate()])

            out = self.linears[i](out)

        return paddle.cast(out, 'float32')


class RandomDataset(Dataset):
    def __init__(self, image_size, num_samples=1):
        super().__init__()
        self.image_size = image_size
        self.num_samples = num_samples

    def __getitem__(self, index):
        # Keep dimension as [8]
        input = paddle.rand([self.image_size], dtype='float32')
        label = paddle.rand([8], dtype='float32')
        return input, label

    def __len__(self):
        return self.num_samples


def manual_model_split(
    model: MyModel, stage_idx: int, group: Group
) -> PipelineStage:
    """Manually split model into pipeline stages"""
    if stage_idx == 0:
        del model.linear3
        del model.linear4
    elif stage_idx == 1:
        del model.linear1
        del model.linear2
    else:
        raise ValueError("Invalid stage index.")

    return PipelineStage(model, stage_idx, 2, group=group)


class TestPipelineStage:
    @classmethod
    def setUpClass(cls):
        """Initialize test class setup"""
        paddle.distributed.init_parallel_env()
        cls.group = paddle.distributed.new_group([0, 1])
        cls.rank = dist.get_rank()
        cls.mesh = paddle.distributed.ProcessMesh([0, 1], dim_names=["pp"])
        fleet.auto.set_mesh(cls.mesh)

    def test_PipelineStage(self):
        """Test complete pipeline including forward, backward and model comparison"""
        fix_seeds()
        self.model = MyModel()
        self.micro_batches = 1  # The PipelineStage component is currently tested separately, so it is set to 1, and the micro_batches > 1 scenario will be overridden when the schedule component is tested in the future
        self.stage = manual_model_split(self.model, self.rank, self.group)
        self.stage.has_backward = True
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=self.model.parameters()
        )
        loss_fn = nn.MSELoss()
        dataset = RandomDataset(image_size=8, num_samples=100)
        losses = []
        num_iterations = 20
        data0 = paddle.zeros([8], dtype='float32')
        label0 = paddle.zeros([8], dtype='float32')
        data0 = paddle.to_tensor(data0).unsqueeze(0)
        label0 = paddle.to_tensor(label0).unsqueeze(0)
        #  Prepare infrastructure
        if self.rank == 0:
            output = self.stage._prepare_forward_infra(
                self.micro_batches,
                (data0,),
                {
                    "debug_str": "test debug_str",
                },
            )
        else:
            output = self.stage._prepare_forward_infra(
                self.micro_batches,
                (),
                {
                    "debug_str": "test debug_str",
                },
            )
        loss = None
        if self.stage.is_last:
            loss = loss_fn(output[0], label0)
        self.stage._prepare_backward_infra(self.micro_batches, loss)
        for iter_idx in range(num_iterations):
            data, label = dataset[iter_idx]
            data = paddle.to_tensor(data).unsqueeze(0)
            label = paddle.to_tensor(label).unsqueeze(0)
            # Forward pass
            fwd_sends_to_wait = []

            # Receive operations
            ops = self.stage.get_fwd_recv_ops(0)
            works = _sorted_batch_p2p(ops, desc="fwd_recv")
            for work in works.values():
                work.wait()

            # Forward computation
            output = self.stage.forward_one_chunk(
                0,
                (data,),
                {
                    "debug_str": "test debug_str",
                },
            )
            # Send operations
            ops = self.stage.get_fwd_send_ops(0)
            works = _sorted_batch_p2p(ops, desc="fwd_send")
            fwd_sends_to_wait.extend(works.values())

            # Wait for all send operations to complete
            for work in fwd_sends_to_wait:
                work.wait()

            # Calculate loss if last stage
            loss = None
            if self.stage.is_last:
                loss = loss_fn(output, label)
                assert loss is not None
                losses.append(loss.item())
            # Backward pass
            bwd_sends_to_wait = []

            # Receive gradients
            ops = self.stage.get_bwd_recv_ops(0)
            works = _sorted_batch_p2p(ops, desc="bwd_recv")
            for work in works.values():
                work.wait()

            # Backward computation
            grads = self.stage.backward_one_chunk(
                0, loss=loss, last_backward=True
            )
            assert grads is not None

            # Send gradients
            ops = self.stage.get_bwd_send_ops(0)
            works = _sorted_batch_p2p(ops, desc="bwd_send")
            bwd_sends_to_wait.extend(works.values())

            # Wait for all send operations to complete
            for work in bwd_sends_to_wait:
                work.wait()
            self.stage.clear_runtime_states()
            opt.step()
            opt.clear_grad()

        return losses

    def test_pp_model(self):
        """Test pipeline parallel model using MyModel"""
        fix_seeds()

        pp_model = PPMyModel()
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=pp_model.parameters()
        )
        loss_fn = nn.MSELoss()

        dataset = RandomDataset(image_size=8, num_samples=100)

        pp_losses = []
        num_iterations = 20

        for iter_idx in range(num_iterations):
            data, label = dataset[iter_idx]
            data = paddle.to_tensor(data).unsqueeze(0)
            label = paddle.to_tensor(label).unsqueeze(0)

            output = pp_model(data)

            loss = loss_fn(output, label)
            pp_losses.append(loss.item())

            loss.backward()
            opt.step()
            opt.clear_grad()

        return pp_losses

    def test_single_gpu(self):
        """Test single GPU training with the complete model"""
        # Only run single GPU training on rank 1
        if self.rank == 1:
            fix_seeds()
            single_model = MyModel()
            opt = paddle.optimizer.AdamW(
                learning_rate=0.001, parameters=single_model.parameters()
            )
            loss_fn = nn.MSELoss()

            dataset = RandomDataset(image_size=8, num_samples=100)

            losses = []
            num_iterations = 20

            for iter_idx in range(num_iterations):
                data, label = dataset[iter_idx]
                output = single_model(data)

                loss = loss_fn(output, label)
                losses.append(loss.item())
                loss.backward()

                opt.step()
                opt.clear_grad()

            return losses
        return None

    def test_simple_func_about_schedules(self):
        """Test local data transfer functions between stages on the same rank"""
        if self.rank == 0:
            # 1. Test set_local_fwd_input
            tensor = paddle.to_tensor([1.0, 2.0, 3.0])
            stage = PipelineStage(nn.Linear(3, 3), 1, 2, group=self.group)
            stage.args_recv_info[0] = (_RecvInfo("test", 0, paddle.empty([3])),)
            stage.set_local_fwd_input(tensor, 0)
            assert stage.args_recv_info[0][0].buffer is not None

            # 2. Test get_local_bwd_output
            stage.has_backward = True
            grad_tensor = paddle.to_tensor([4.0, 5.0, 6.0])
            stage.bwd_cache[0] = (grad_tensor,)
            stage.chunks = 2
            bwd_output = stage.get_local_bwd_output(0)
            assert bwd_output[0].equal_all(grad_tensor)

            # 3. Test set_local_bwd_input
            prev_stage = PipelineStage(nn.Linear(3, 3), 0, 2, group=self.group)
            prev_stage.has_backward = True
            prev_stage.grad_recv_info[0] = (
                _RecvInfo("test", 1, paddle.empty([3])),
            )
            grad_input = (paddle.to_tensor([7.0, 8.0, 9.0]),)
            prev_stage.set_local_bwd_input(grad_input, 0)
            assert prev_stage.grad_recv_info[0][0].buffer.equal_all(
                grad_input[0]
            )

    def test_backward_some_simple_examples(self):
        """Test simple examples in backward"""
        if self.rank == 0:
            # 1. Test backward propagation with dictionary and tuple outputs
            input_tensor = paddle.to_tensor([1.0, 2.0], stop_gradient=False)

            output_dict = {
                "out": input_tensor * 2.0,
                "out_tensor_is_dict_grad_is_None": {"out": input_tensor * 2.0},
                "out_tensor_is_tuple_grad_is_None": (input_tensor * 2.0,),
            }
            grad_dict = {
                "out": paddle.to_tensor([0.1, 0.2]),
                "out_tensor_is_dict_grad_is_None": None,
                "out_tensor_is_tuple_grad_is_None": None,
            }

            input_grads = stage_backward(output_dict, grad_dict, [input_tensor])
            expected_grad = paddle.to_tensor([2 * 0.1, 2 * 0.2])

            np.testing.assert_allclose(
                input_grads[0].numpy(), expected_grad.numpy(), rtol=1e-5
            )
            # 2. Test not yet implemented stage_backward_input and stage_backward_weight
            try:
                stage_backward_input(
                    [input_tensor * 2.0],
                    [paddle.to_tensor([0.1, 0.2])],
                    [input_tensor],
                    iter([paddle.to_tensor([1.0, 1.0])]),
                )
                raise AssertionError("Should raise Error")
            except NotImplementedError as e:
                pass
            try:
                stage_backward_weight(
                    iter([paddle.to_tensor([1.0, 1.0])]),
                    [{"params": [paddle.to_tensor([1.0, 1.0])]}],
                )
                raise AssertionError("Should raise Error")
            except NotImplementedError as e:
                pass

    def test_utils_some_simple_examples(self):
        """Test simple examples in utils"""
        if self.rank == 0:
            # 1. Test exceptions in _get_stage_mesh
            try:
                _get_stage_mesh(0, 2, style="v")
                raise AssertionError("Should raise Error")
            except NotImplementedError as e:
                pass
            try:
                _get_stage_mesh(0, 2, style="unknown")
                raise AssertionError("Should raise Error")
            except ValueError as e:
                pass

            # 2. Test exceptions in _validate_tensors_metadata
            try:
                # Length mismatch
                expected = [paddle.to_tensor([1.0, 2.0])]
                actual = [paddle.to_tensor([1.0]), paddle.to_tensor([2.0])]
                _validate_tensors_metadata("test", expected, actual)
                raise AssertionError("Should raise Error")
            except PipeliningShapeError as e:
                pass

            # 3. Test exceptions in _validate_tensor_metadata
            try:
                # Shape mismatch
                expected = paddle.to_tensor([1.0, 2.0])
                actual = paddle.to_tensor([1.0])
                _validate_tensor_metadata("test", expected, actual)
                raise AssertionError("Should raise Error")
            except PipeliningShapeError as e:
                pass

            try:
                # Dtype mismatch
                expected = paddle.to_tensor([1.0, 2.0], dtype='float32')
                actual = paddle.to_tensor([1, 2], dtype='int32')
                _validate_tensor_metadata("test", expected, actual)
                raise AssertionError("Should raise Error")
            except PipeliningShapeError as e:
                pass

            # 4. Test _detach_and_keep_grad
            a = paddle.to_tensor([2.0], stop_gradient=False)
            b = a * 2
            x = _detach_and_keep_grad(b)
            assert x is b
            assert x.stop_gradient == b.stop_gradient
            assert (x.numpy() == b.numpy()).all()
            x.stop_gradient = False
            z = x * 3
            z.backward()

            assert x.grad is not None
            assert a.grad is None

            # 5. Test TensorMeta and _zero_initialize_with_meta
            tensor = paddle.ones([4, 8])
            dist_tensor = dist.shard_tensor(tensor, self.mesh, [dist.Shard(0)])
            tensor_meta = TensorMeta(dist_tensor)
            assert tensor_meta.shape == [4, 8]
            assert tensor_meta._local_shape == [2, 8]

            zero_tensor = _zero_initialize_with_meta(tensor_meta, self.mesh)
            assert zero_tensor.shape == [4, 8]
            assert zero_tensor.is_dist()
            assert zero_tensor.process_mesh == self.mesh
            assert zero_tensor.placements == [dist.Shard(0)]

            # 6. Test _friendly_debug_info
            a = {"test the input is not a tensor": 1}
            assert _friendly_debug_info(a) == str(a)

    def run_test(self):
        """Compare losses between three training methods"""
        self.setUpClass()
        self.test_simple_func_about_schedules()
        self.test_backward_some_simple_examples()
        self.test_utils_some_simple_examples()
        # Run three training methods
        pipeline_losses = self.test_PipelineStage()
        pp_losses = self.test_pp_model()
        single_losses = self.test_single_gpu()

        if self.rank == 1:
            np.testing.assert_allclose(
                pipeline_losses,
                pp_losses,
                rtol=1e-5,
            )

            np.testing.assert_allclose(
                pipeline_losses,
                single_losses,
                rtol=1e-5,
            )


if __name__ == '__main__':
    TestPipelineStage().run_test()
