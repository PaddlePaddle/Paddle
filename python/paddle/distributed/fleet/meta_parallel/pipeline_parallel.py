#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
import paddle.fluid as fluid
from .meta_parallel_base import MetaParallelBase
from .pp_utils.utils import is_float_tensor, get_tensor_dtype, paddle_2_number, number_2_dtype
from .pp_utils import utils
from .parallel_layers.pp_layers import PipelineLayer

from ..utils.hybrid_parallel_util import broadcast_mp_parameters
from ..utils.hybrid_parallel_util import broadcast_dp_parameters
from ..utils.log_util import logger
from ..meta_optimizers.dygraph_optimizer import HybridParallelOptimizer
from .pp_utils import p2p_communication as p2p

__all__ = []


class PipelineParallel(MetaParallelBase):
    def __init__(self, layers, hcg, strategy):
        if not isinstance(layers, PipelineLayer):
            raise TypeError(
                "The Layer should be a derived class of PipelineLayer.")
        super(PipelineParallel, self).__init__(layers, hcg, strategy)
        self.use_pipe_parallel = self._hcg.get_pipe_parallel_world_size() > 1
        self.use_data_parallel = self._hcg.get_data_parallel_world_size() > 1
        self.use_model_parallel = self._hcg.get_model_parallel_world_size() > 1

        self.is_pipe_partitioned = self.use_model_parallel

        self.num_caches = 0
        self.caches = {
            'inputs': [],
            'labels': [],
            'outputs': [],
        }

        self.recv_cache = None
        self.grad_tensors = None

        self.send_meta = True

        self.current_loss = paddle.to_tensor(0.0)
        self.total_loss = None

        self.micro_batch_size = self._strategy.pipeline_configs[
            'micro_batch_size']
        self.accumulate_steps = self._strategy.pipeline_configs[
            'accumulate_steps']

        self.num_stages = self._hcg.get_pipe_parallel_world_size()
        self.stage_id = self._hcg.get_stage_id()
        self.prev_stage_id = self.stage_id - 1
        self.next_stage_id = self.stage_id + 1
        self.pp_group = self._hcg.get_pipe_parallel_group()
        p2p.initialize_p2p_groups(hcg)

        self.is_first_stage = self.stage_id == 0
        self.is_last_stage = (self.stage_id == (self.num_stages - 1))
        self.global_rank = self._hcg.get_global_rank()

        self.mp_degree = self._hcg.get_model_parallel_world_size()
        self.mp_rank = self._hcg.get_model_parallel_rank()

        logger.info("Pipeline Info -- num_stages: {}, stage_id: {}".format(
            self.num_stages, self.stage_id))

        if self.use_model_parallel:
            logger.info("start broadcast mp parameters")
            broadcast_mp_parameters(self._layers, self._hcg)

        if self.use_data_parallel:
            logger.info("start broadcast dp parameters")
            broadcast_dp_parameters(self._layers, self._hcg)
        self.data_id = 0

    def train_batch(self, data, optimizer, lr_scheduler=None, scaler=None):
        assert isinstance(optimizer, HybridParallelOptimizer), (
            'optimizer should be HybridParallelOptimizer subclass.')
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        assert fluid.framework._dygraph_tracer()._has_grad, (
            'Please enable the generation of gradients.')

        if self.is_first_stage or self.is_last_stage:
            assert data is not None, (
                "For the first and the last stage, the data_iter must be set.")
        else:
            data = None

        self.data = data
        self._layers.train()

        # store total loss of entire batch
        self.total_loss = None

        # store data id for micro_batch
        self.data_id = 0

        self.micro_batch_size = self._strategy.pipeline_configs[
            'micro_batch_size']
        self.accumulate_steps = self._strategy.pipeline_configs[
            'accumulate_steps']

        self.num_stages = self._hcg.get_pipe_parallel_world_size()

        # Compute number of warmup microbatches.
        num_microbatches = self.accumulate_steps
        num_warmup_microbatches = (self.num_stages - self.stage_id - 1)
        num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches

        input_tensors = []
        output_tensors = []
        losses_reduced = []

        for step_id in range(num_warmup_microbatches):
            input_tensor = p2p.recv_forward()
            if input_tensor is not None:
                input_tensor.stop_gradient = False
            output_tensor = self._forward_step(input_tensor)
            p2p.send_forward(output_tensor)

            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

        if num_microbatches_remaining > 0:
            input_tensor = p2p.recv_forward()

        for i in range(num_microbatches_remaining):
            last_iteration = (i == (num_microbatches_remaining - 1))

            if input_tensor is not None:
                input_tensor.stop_gradient = False
            output_tensor = self._forward_step(input_tensor)

            output_tensor_grad = p2p.send_forward_recv_backward(output_tensor)

            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

            input_tensor, output_tensor = input_tensors.pop(
                0), output_tensors.pop(0)

            input_tensor_grad = \
                self._backward_step(input_tensor, output_tensor, output_tensor_grad)

            if last_iteration:
                input_tensor = None
                p2p.send_backward(input_tensor_grad)
            else:
                input_tensor = \
                    p2p.send_backward_recv_forward(input_tensor_grad)

        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p.recv_backward()

            input_tensor_grad = \
                self._backward_step(input_tensor, output_tensor, output_tensor_grad)
            p2p.send_backward(input_tensor_grad)

        self._layers.allreduce_shared_weight_gradients()

        # optimizer
        self.train_loss = self._reduce_final_loss()

        self._step()
        return self.train_loss

    def _forward_step(self, input_tensor):
        if self.stage_id == 0:
            input_tensor = self._load_micro_batch(self.data_id)

        output_tensor = self._layers.forward(input_tensor)

        if self.is_last_stage:
            labels = self._load_micro_batch(self.data_id)
            output_tensor = self._layers._loss_fn(output_tensor, labels)
            assert isinstance(
                output_tensor, paddle.
                Tensor), "Currently, loss_fn should obtain Paddle.Tensor dtype"

            if self.accumulate_steps > 1:
                output_tensor = output_tensor / self.accumulate_steps

            if self.total_loss is None:
                self.total_loss = paddle.zeros_like(output_tensor)
            self.total_loss += output_tensor.detach()

        self.data_id += 1
        return output_tensor

    def _backward_step(self, input_tensor, output_tensor, output_tensor_grad):
        paddle.autograd.backward(
            tensors=[output_tensor], grad_tensors=[output_tensor_grad])
        input_tensor_grad = None
        if input_tensor is not None:
            input_tensor_grad = input_tensor.grad

        return input_tensor_grad

    def _load_micro_batch(self, cache_id):
        inputs = self.data
        begin = cache_id * self.micro_batch_size
        end = begin + self.micro_batch_size

        if self.is_first_stage:
            assert len(inputs) == 2, "length of input should be 2"
            if isinstance(inputs[0], tuple):
                batch_size = inputs[0][0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size, (
                    "batch_size needs to be divisible by micro_batch_size. Currently, "
                    "batch_size = %d, micro_batch_size = %d, accumulate_steps = %d."
                    %
                    (batch_size, self.micro_batch_size, self.accumulate_steps))
                data = [
                    input[begin:end, :].clone().detach() for input in inputs[0]
                ]
                return tuple(data)
            else:
                batch_size = inputs[0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                return inputs[0][begin:end, :].clone().detach()
        elif self.is_last_stage:
            assert len(inputs) == 2, "length of input should be 2"
            if isinstance(inputs[1], tuple):
                batch_size = inputs[1][0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                data = [
                    input[begin:end, :].clone().detach() for input in inputs[1]
                ]
                return tuple(data)
            else:
                batch_size = inputs[1].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                return inputs[1][begin:end, :].clone().detach()
        else:
            # No data input is required for other stages
            inputs = None

    def _reduce_final_loss(self):
        if self.is_last_stage:
            assert self.total_loss is not None, "train_batch() in last stage should obtain vaild loss"
            loss = self.total_loss.clone()
            paddle.distributed.broadcast(
                loss,
                src=self.global_rank,
                use_calc_stream=True,
                group=self.pp_group)
        else:
            loss = paddle.to_tensor(0.0)
            paddle.distributed.broadcast(
                loss,
                src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                use_calc_stream=True,
                group=self.pp_group)
        return loss

    def _step(self):
        if self.scaler:
            self.scaler.minimize(self.optimizer, self.train_loss)
        else:
            self.optimizer.step()

        self.optimizer.clear_grad()
        if self.lr_scheduler:
            self.lr_scheduler.step()
