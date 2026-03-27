# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
    NoPipelineParallel,
)
from paddle.nn import Layer


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 8
length = 8
micro_batch_size = 2
vocab_size = 128
hidden_size = 16
d_model = hidden_size
dim_feedforward = 4 * d_model
micro_batch_number = batch_size / micro_batch_size


class SimpleNet(Layer):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.softmax_weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
        )
        self.softmax_bias = self.create_parameter(
            shape=[vocab_size],
            is_bias=False,
        )

    def forward(self, x1, x2, y1):
        x_emb = self.word_embeddings(x1)
        fc = paddle.matmul(x_emb, self.softmax_weight)
        fc = paddle.add(fc, self.softmax_bias)
        projection = paddle.reshape(fc, shape=[-1, vocab_size])
        loss_0 = (projection - y1).square().mean()
        loss_1 = (projection - y1).abs().mean()
        return loss_0, loss_1


class EmbeddingNet(Layer):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

    @property
    def embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, args):
        x1, x2 = args
        x_emb = self.word_embeddings(x1)
        x2.stop_gradient = True
        return x_emb, x2


class MatmulNet(Layer):
    def __init__(self):
        super().__init__()
        self.softmax_weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
        )

    def forward(self, args):
        x1, x2 = args
        fc = paddle.matmul(x1, self.softmax_weight)
        return fc, x2


class BiasNet(Layer):
    def __init__(self):
        super().__init__()
        self.softmax_bias = self.create_parameter(
            shape=[vocab_size],
        )

    def forward(self, args):
        fc, x2 = args
        fc = paddle.add(fc, self.softmax_bias)
        projection = paddle.reshape(fc, shape=[-1, vocab_size])
        return projection, x2


class MSEPipe(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, args, y1):
        projection, x2 = args
        loss = (projection - y1[0]).square().mean()
        return loss


class L1Pipe(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, args, y1):
        projection, x2 = args
        loss = (projection - y1[0]).abs().mean()
        return loss


class SimpleNetPipe(PipelineLayer):
    def __init__(self, **kwargs):
        self.descs = []
        self.descs.append(LayerDesc(EmbeddingNet))
        self.descs.append(LayerDesc(MatmulNet))
        self.descs.append(LayerDesc(BiasNet))

        super().__init__(
            layers=self.descs, loss_fn=[MSEPipe(), L1Pipe()], **kwargs
        )


class TestDistPPTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_model_backward(self):
        hcg = fleet.get_hybrid_communicate_group()
        world_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        topology = hcg.topology()
        set_random_seed(1024, dp_id, rank_id)

        # construct model a
        model_a = SimpleNet()
        scheduler_a = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        optimizer_a = paddle.optimizer.SGD(
            learning_rate=scheduler_a, parameters=model_a.parameters()
        )

        # construct model b
        model_b = SimpleNetPipe(topology=hcg.topology())

        scheduler_b = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        optimizer_b = paddle.optimizer.SGD(
            learning_rate=scheduler_b, parameters=model_b.parameters()
        )
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)

        param_len = len(model_a.parameters())

        parameters = []
        for param in model_a.parameters():
            parameters.append(param.numpy())

        model_b_params = model_b.parameters()

        if pp_id == 0:
            model_b_params[0].set_value(parameters[2])
        else:
            model_b_params[0].set_value(parameters[0])
            model_b_params[1].set_value(parameters[1])

        loss_fn_idx = 1
        for _ in range(5):
            x1_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
            x2_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
            y1_data = np.random.randint(0, hidden_size, size=[batch_size, 1])

            x1 = paddle.to_tensor(x1_data, dtype="int64")
            x2 = paddle.to_tensor(x2_data, dtype="float32")
            y1 = paddle.to_tensor(y1_data, dtype="float32")

            x1.stop_gradient = True
            x2.stop_gradient = True
            y1.stop_gradient = True

            loss_a = model_a(x1, x2, y1)
            loss_a[loss_fn_idx].backward()

            optimizer_a.step()
            optimizer_a.clear_grad()
            scheduler_a.step()

            loss_b = model_b.train_batch(
                [(x1, x2), (y1,)],
                optimizer_b,
                scheduler_b,
                loss_fn_idx=loss_fn_idx,
                return_micro_batch_loss=True,
            )

            for idx in range(2):
                loss_b_shape = loss_b[idx].shape[0]
                loss_b_idx = paddle.mean(loss_b[idx])
                np.testing.assert_equal(
                    int(loss_b_shape), int(micro_batch_number)
                )
                np.testing.assert_allclose(
                    loss_a[idx].numpy(),
                    loss_b_idx.numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                )


class TestDistPPTrainingDelayScaleLoss(TestDistPPTraining):
    """Test that explicitly setting delay_scale_loss=True in pp_configs
    (backward compat) still produces correct results after the unification."""

    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
            "pp_configs": {
                "delay_scale_loss": True,
            },
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_model_backward(self):
        super().test_pp_model_backward()


class TestDistPPTrainingMicroBatchLossValues(unittest.TestCase):
    """Verify that return_micro_batch_loss=True returns raw unscaled
    per-micro-batch loss values, and that each one matches the reference
    model's loss computed on the corresponding micro batch slice."""

    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_micro_batch_loss_unscaled(self):
        hcg = fleet.get_hybrid_communicate_group()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        set_random_seed(2048, dp_id, rank_id)

        accumulate_steps = batch_size // micro_batch_size

        # construct reference model
        model_a = SimpleNet()

        # construct pipeline model
        model_b = SimpleNetPipe(topology=hcg.topology())
        scheduler_b = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        optimizer_b = paddle.optimizer.SGD(
            learning_rate=scheduler_b, parameters=model_b.parameters()
        )
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)

        param_len = len(model_a.parameters())
        parameters = [p.numpy() for p in model_a.parameters()]

        model_b_params = model_b.parameters()
        if pp_id == 0:
            model_b_params[0].set_value(parameters[2])
        else:
            model_b_params[0].set_value(parameters[0])
            model_b_params[1].set_value(parameters[1])

        x1_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
        x2_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
        y1_data = np.random.randint(0, hidden_size, size=[batch_size, 1])

        x1 = paddle.to_tensor(x1_data, dtype="int64")
        x2 = paddle.to_tensor(x2_data, dtype="float32")
        y1 = paddle.to_tensor(y1_data, dtype="float32")
        x1.stop_gradient = True
        x2.stop_gradient = True
        y1.stop_gradient = True

        loss_fn_idx = 0
        loss_b = model_b.train_batch(
            [(x1, x2), (y1,)],
            optimizer_b,
            scheduler_b,
            loss_fn_idx=loss_fn_idx,
            return_micro_batch_loss=True,
        )

        # Compute per-micro-batch reference losses
        for mb_idx in range(accumulate_steps):
            start = mb_idx * micro_batch_size
            end = start + micro_batch_size
            mb_x1 = x1[start:end]
            mb_x2 = x2[start:end]
            mb_y1 = y1[start:end]
            ref_losses = model_a(mb_x1, mb_x2, mb_y1)

            for fn_idx in range(2):
                # Each micro batch loss should be unscaled and match reference
                np.testing.assert_allclose(
                    ref_losses[fn_idx].numpy(),
                    loss_b[fn_idx][mb_idx].numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=(
                        f"Micro batch {mb_idx} loss_fn {fn_idx} mismatch: "
                        f"expected raw unscaled loss"
                    ),
                )


class TestNoPipelineParallelTraining(unittest.TestCase):
    """Test NoPipelineParallel (single-stage pipeline) path.
    Verifies that train_batch and eval_batch return correct loss values
    after the delay_scale_loss unification."""

    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    @staticmethod
    def _sync_params_by_shape(src_model, dst_model):
        """Sync parameters from src_model to dst_model by matching shapes.

        SimpleNet and SimpleNetPipe have identical parameters but in different
        order.  Match them by shape to avoid index-order mismatches.
        """
        src_params = {tuple(p.shape): p.numpy() for p in src_model.parameters()}
        for p in dst_model.parameters():
            key = tuple(p.shape)
            assert key in src_params, (
                f"No matching param with shape {key} in source model"
            )
            p.set_value(src_params.pop(key))

    def test_no_pipeline_train_loss(self):
        """Verify NoPipelineParallel.train_batch returns correct average loss."""
        hcg = fleet.get_hybrid_communicate_group()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()
        set_random_seed(4096, dp_id, rank_id)

        strategy = fleet.fleet._user_defined_strategy

        # Build single-stage pipeline model (all layers on one device)
        pipe_model = SimpleNetPipe(num_stages=1)
        no_pp_model = NoPipelineParallel(pipe_model, strategy, hcg)

        # Build reference model with same parameters
        model_a = SimpleNet()

        # Sync parameters by shape: SimpleNet and SimpleNetPipe have the same
        # set of parameter shapes but enumerate them in different order.
        self._sync_params_by_shape(model_a, no_pp_model)

        scheduler_a = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        optimizer_a = paddle.optimizer.SGD(
            learning_rate=scheduler_a, parameters=model_a.parameters()
        )
        scheduler_b = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        optimizer_b = paddle.optimizer.SGD(
            learning_rate=scheduler_b, parameters=no_pp_model.parameters()
        )

        for _ in range(3):
            x1_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
            x2_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
            y1_data = np.random.randint(0, hidden_size, size=[batch_size, 1])

            x1 = paddle.to_tensor(x1_data, dtype="int64")
            x2 = paddle.to_tensor(x2_data, dtype="float32")
            y1 = paddle.to_tensor(y1_data, dtype="float32")
            x1.stop_gradient = True
            x2.stop_gradient = True
            y1.stop_gradient = True

            # Reference model: forward + backward + step
            loss_a = model_a(x1, x2, y1)
            loss_a[0].backward()
            optimizer_a.step()
            optimizer_a.clear_grad()
            scheduler_a.step()

            # NoPipelineParallel model: train_batch
            loss_b = no_pp_model.train_batch(
                [(x1, x2), (y1,)], optimizer_b, scheduler_b
            )

            # train_batch returns a list when there are multiple loss functions
            print("no_pp loss:", loss_a[0].numpy(), loss_b[0].numpy())
            np.testing.assert_allclose(
                loss_a[0].numpy(), loss_b[0].numpy(), rtol=1e-6, atol=1e-6
            )

    def test_no_pipeline_eval_loss(self):
        """Verify NoPipelineParallel.eval_batch returns correct average loss."""
        hcg = fleet.get_hybrid_communicate_group()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()
        set_random_seed(5120, dp_id, rank_id)

        strategy = fleet.fleet._user_defined_strategy

        pipe_model = SimpleNetPipe(num_stages=1)
        no_pp_model = NoPipelineParallel(pipe_model, strategy, hcg)

        model_a = SimpleNet()
        self._sync_params_by_shape(model_a, no_pp_model)

        optimizer_b = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=no_pp_model.parameters()
        )

        x1_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
        x2_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
        y1_data = np.random.randint(0, hidden_size, size=[batch_size, 1])

        x1 = paddle.to_tensor(x1_data, dtype="int64")
        x2 = paddle.to_tensor(x2_data, dtype="float32")
        y1 = paddle.to_tensor(y1_data, dtype="float32")
        x1.stop_gradient = True
        x2.stop_gradient = True
        y1.stop_gradient = True

        # Reference: compute full batch loss
        loss_a = model_a(x1, x2, y1)

        # NoPipelineParallel: eval_batch
        loss_b = no_pp_model.eval_batch([(x1, x2), (y1,)], compute_loss=True)

        # eval_batch returns a list when there are multiple loss functions
        print("no_pp eval loss:", loss_a[0].numpy(), loss_b[0].numpy())
        np.testing.assert_allclose(
            loss_a[0].numpy(), loss_b[0].numpy(), rtol=1e-6, atol=1e-6
        )

    def test_no_pipeline_return_micro_batch_loss(self):
        """Verify NoPipelineParallel.train_batch with return_micro_batch_loss
        returns raw unscaled per-micro-batch values."""
        hcg = fleet.get_hybrid_communicate_group()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()
        set_random_seed(6144, dp_id, rank_id)

        strategy = fleet.fleet._user_defined_strategy
        accumulate_steps = batch_size // micro_batch_size

        pipe_model = SimpleNetPipe(num_stages=1)
        no_pp_model = NoPipelineParallel(pipe_model, strategy, hcg)

        model_a = SimpleNet()
        self._sync_params_by_shape(model_a, no_pp_model)

        scheduler_b = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        optimizer_b = paddle.optimizer.SGD(
            learning_rate=scheduler_b, parameters=no_pp_model.parameters()
        )

        x1_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
        x2_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
        y1_data = np.random.randint(0, hidden_size, size=[batch_size, 1])

        x1 = paddle.to_tensor(x1_data, dtype="int64")
        x2 = paddle.to_tensor(x2_data, dtype="float32")
        y1 = paddle.to_tensor(y1_data, dtype="float32")
        x1.stop_gradient = True
        x2.stop_gradient = True
        y1.stop_gradient = True

        loss_b = no_pp_model.train_batch(
            [(x1, x2), (y1,)],
            optimizer_b,
            scheduler_b,
            return_micro_batch_loss=True,
        )

        # Verify each micro batch loss matches reference
        for mb_idx in range(accumulate_steps):
            start = mb_idx * micro_batch_size
            end = start + micro_batch_size
            ref_losses = model_a(x1[start:end], x2[start:end], y1[start:end])
            for fn_idx in range(2):
                np.testing.assert_allclose(
                    ref_losses[fn_idx].numpy(),
                    loss_b[fn_idx][mb_idx].numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                )


class TestDistPPEvalBatch(TestDistPPTraining):
    """Test PipelineParallel.eval_batch returns correct average loss."""

    def test_pp_eval_batch(self):
        hcg = fleet.get_hybrid_communicate_group()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        set_random_seed(7168, dp_id, rank_id)

        model_a = SimpleNet()
        model_b = SimpleNetPipe(topology=hcg.topology())
        optimizer_b = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model_b.parameters()
        )
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)

        params_a = [p.numpy() for p in model_a.parameters()]
        model_b_params = model_b.parameters()
        if pp_id == 0:
            model_b_params[0].set_value(params_a[2])
        else:
            model_b_params[0].set_value(params_a[0])
            model_b_params[1].set_value(params_a[1])

        x1_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
        x2_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
        y1_data = np.random.randint(0, hidden_size, size=[batch_size, 1])

        x1 = paddle.to_tensor(x1_data, dtype="int64")
        x2 = paddle.to_tensor(x2_data, dtype="float32")
        y1 = paddle.to_tensor(y1_data, dtype="float32")
        x1.stop_gradient = True
        x2.stop_gradient = True
        y1.stop_gradient = True

        loss_a = model_a(x1, x2, y1)
        loss_b = model_b.eval_batch([(x1, x2), (y1,)], compute_loss=True)

        # eval_batch returns a list when there are multiple loss functions
        print("eval loss:", loss_a[0].numpy(), loss_b[0].numpy())
        np.testing.assert_allclose(
            loss_a[0].numpy(), loss_b[0].numpy(), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
