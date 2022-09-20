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

import paddle
import paddle.fluid as fluid
from .meta_parallel_base import MetaParallelBase
from .parallel_layers.pp_layers import PipelineLayer

from ..utils.hybrid_parallel_util import broadcast_mp_parameters
from ..utils.hybrid_parallel_util import broadcast_dp_parameters
from ..utils.hybrid_parallel_util import broadcast_sharding_parameters
from ..utils.log_util import logger
from ..meta_optimizers.dygraph_optimizer import HybridParallelOptimizer, HybridParallelGradScaler
import paddle.fluid.framework as framework
from .pp_utils import p2p_communication as p2p
import paddle.fluid.core as core

__all__ = []


class PipelineParallel(MetaParallelBase):

    def __init__(self, layers, hcg, strategy):
        if not isinstance(layers, PipelineLayer):
            raise TypeError(
                "The Layer should be a derived class of PipelineLayer.")
        super(PipelineParallel, self).__init__(layers, hcg, strategy)
        self.use_data_parallel = self._hcg.get_data_parallel_world_size() > 1
        self.use_model_parallel = self._hcg.get_model_parallel_world_size() > 1
        self.use_sharding_parallel = self._hcg.get_sharding_parallel_world_size(
        ) > 1

        self.total_loss = None

        self.micro_batch_size = self._strategy.pipeline_configs[
            'micro_batch_size']
        self.accumulate_steps = self._strategy.pipeline_configs[
            'accumulate_steps']

        self._using_cache = self._strategy.pipeline_configs['p2p_cache_shape']

        self.num_stages = self._hcg.get_pipe_parallel_world_size()
        self.stage_id = self._hcg.get_stage_id()
        self.pp_group = self._hcg.get_pipe_parallel_group()

        self._virtual_pp_world_size = None
        self._virtual_pp_rank = None
        self._real_pp_world_size = self.num_stages
        self._real_pp_rank = self.stage_id

        p2p.initialize_p2p_groups(hcg, self._using_cache)

        self.global_rank = self._hcg.get_global_rank()
        self.micro_batch_id = 0

        self._compute_loss = True

        logger.info("Pipeline Info -- num_stages: {}, stage_id: {}".format(
            self.num_stages, self.stage_id))

        if self.use_model_parallel:
            logger.info("start broadcast mp parameters")
            broadcast_mp_parameters(self._layers, self._hcg)

        if self.use_sharding_parallel:
            logger.info("start broadcast sharding parameters")
            broadcast_sharding_parameters(self._layers, self._hcg)

        if self.use_data_parallel:
            logger.info("start broadcast dp parameters")
            broadcast_dp_parameters(self._layers, self._hcg)

    def is_pipeline_first_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self._virtual_pp_world_size is not None:
                assert self._virtual_pp_rank is not None
                if self._virtual_pp_rank != 0:
                    return False
        assert self._real_pp_rank is not None
        return self._real_pp_rank == 0

    def is_pipeline_last_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self._virtual_pp_world_size is not None:
                assert self._virtual_pp_rank is not None
                if self._virtual_pp_rank != (self._virtual_pp_world_size - 1):
                    return False
        assert self._real_pp_rank is not None
        assert self._real_pp_world_size is not None
        return self._real_pp_rank == (self._real_pp_world_size - 1)

    def set_virtual_pipeline_rank(self, rank):
        self._virtual_pp_rank = rank

    def forward_backward_pipeline(self, data, scaler=None):
        # use the 1f1b scheduling strategy.
        # this strategy is inspired by:
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/schedules.py

        self.scaler = scaler

        # store data for train
        self.data = data

        # store total loss of entire batch
        self.total_loss = None

        # store data id for micro_batch
        self.micro_batch_id = 0

        startup_steps = (self.num_stages - self.stage_id - 1)
        startup_steps = min(startup_steps, self.accumulate_steps)
        steady_steps = self.accumulate_steps - startup_steps

        input_buffers = []
        output_buffers = []

        for step_id in range(startup_steps):
            input_tensor = p2p.recv_forward(self.is_pipeline_first_stage())

            output_tensor = self._forward_step(input_tensor)
            p2p.send_forward(output_tensor, self.is_pipeline_last_stage())

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

        if steady_steps > 0:
            input_tensor = p2p.recv_forward(self.is_pipeline_first_stage())

        for i in range(steady_steps):
            last_iter = (i == (steady_steps - 1))

            output_tensor = self._forward_step(input_tensor)

            output_tensor_grad = p2p.send_forward_recv_backward(
                output_tensor, self.is_pipeline_last_stage())

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            input_tensor, output_tensor = input_buffers.pop(
                0), output_buffers.pop(0)

            input_tensor_grad = self._backward_step(input_tensor, output_tensor,
                                                    output_tensor_grad)

            if last_iter:
                input_tensor = None
                p2p.send_backward(input_tensor_grad,
                                  self.is_pipeline_first_stage())
            else:
                input_tensor = p2p.send_backward_recv_forward(
                    input_tensor_grad, self.is_pipeline_first_stage())

        for i in range(startup_steps):
            input_tensor = input_buffers.pop(0)
            output_tensor = output_buffers.pop(0)

            output_tensor_grad = p2p.recv_backward(
                self.is_pipeline_last_stage())

            input_tensor_grad = self._backward_step(input_tensor, output_tensor,
                                                    output_tensor_grad)
            p2p.send_backward(input_tensor_grad, self.is_pipeline_first_stage())

        self._layers.allreduce_shared_weight_gradients()
        with paddle.amp.auto_cast(enable=False):
            train_loss = self._broadcast_final_loss()
        return train_loss

    def _prepare_training(self, data, optimizer, lr_scheduler):
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        assert isinstance(optimizer, HybridParallelOptimizer), (
            'optimizer should be HybridParallelOptimizer subclass.')

        assert fluid.framework._dygraph_tracer()._has_grad, (
            'Please enable the generation of gradients.')

        if self.is_pipeline_first_stage(
                ignore_virtual=True) or self.is_pipeline_last_stage(
                    ignore_virtual=True):
            assert data is not None, (
                "For the first and the last stage, the data must be set.")
        else:
            data = None

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self._layers.train()

        return data

    def train_batch(self, data, optimizer, lr_scheduler=None, scaler=None):
        data = self._prepare_training(data, optimizer, lr_scheduler)
        # 1f1b scheduler for pipeline parallel
        train_loss = self.forward_backward_pipeline(data, scaler)

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss

    def eval_batch(self, data, compute_loss=False):
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        self._layers.eval()
        self._compute_loss = compute_loss

        # save data for eval
        self.data = data
        # store data id for micro_batch
        self.micro_batch_id = 0

        # store total loss of entire batch
        self.total_loss = None

        startup_steps = (self.num_stages - self.stage_id - 1)
        startup_steps = min(startup_steps, self.accumulate_steps)
        steady_steps = self.accumulate_steps - startup_steps

        input_buffers = []
        output_buffers = []

        for step_id in range(startup_steps):
            input_tensor = p2p.recv_forward(self.is_pipeline_first_stage())

            output_tensor = self._forward_step(input_tensor)
            p2p.send_forward(output_tensor, self.is_pipeline_last_stage())

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

        if steady_steps > 0:
            input_tensor = p2p.recv_forward(self.is_pipeline_first_stage())

        for i in range(steady_steps):
            last_iter = (i == (steady_steps - 1))

            output_tensor = self._forward_step(input_tensor)
            p2p.send_forward(output_tensor, self.is_pipeline_last_stage())

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            if not last_iter:
                input_tensor = p2p.recv_forward(self.is_pipeline_first_stage())

        if self._compute_loss:
            self.train_loss = self._broadcast_final_loss()
        else:
            self.train_loss = output_buffers

        return self.train_loss

    def _forward_step(self, input_tensor, chunk_id=None):
        if self.is_pipeline_first_stage():
            input_tensor = self._load_micro_batch(self.micro_batch_id)

        assert chunk_id is None or isinstance(chunk_id, int)

        output_tensor = self._layers.forward(input_tensor, chunk_id=chunk_id)

        if self.is_pipeline_last_stage():
            # train calculate loss for train
            if self._compute_loss:
                assert self._layers._loss_fn is not None, "loss function should exist to compute loss"
                labels = self._load_micro_batch(self.micro_batch_id)
                output_tensor = self._layers._loss_fn(output_tensor, labels)
                assert isinstance(
                    output_tensor,
                    (paddle.Tensor, core.eager.Tensor
                     )), "Currently, loss_fn should obtain Paddle.Tensor dtype"

                with paddle.amp.auto_cast(enable=False):
                    if self.accumulate_steps > 1:
                        output_tensor = output_tensor / self.accumulate_steps

                    if self.total_loss is None:
                        self.total_loss = paddle.zeros_like(output_tensor)
                    self.total_loss += output_tensor.detach()

        if self.is_pipeline_first_stage() or self.is_pipeline_last_stage():
            # Only increase micro batch id at virtual first/last pp stage.
            # The micro batch id is used to load data, therefore, only increase it when load data.
            self.micro_batch_id += 1
        return output_tensor

    def _backward_step(self, input_tensor, output_tensor, output_tensor_grad):
        with paddle.amp.auto_cast(enable=False):
            if self.is_pipeline_last_stage():
                assert output_tensor_grad is None
                if self.scaler:
                    paddle.autograd.backward(self.scaler.scale(output_tensor))
                else:
                    paddle.autograd.backward(output_tensor)
            else:
                if isinstance(output_tensor, tuple):
                    outputs = [t for t in output_tensor if not t.stop_gradient]
                    assert len(outputs) == len(output_tensor_grad)
                    paddle.autograd.backward(
                        tensors=outputs,
                        grad_tensors=[t for t in output_tensor_grad])
                else:
                    paddle.autograd.backward(tensors=[output_tensor],
                                             grad_tensors=[output_tensor_grad])

            input_tensor_grad = None
            if input_tensor is not None:
                if isinstance(input_tensor, tuple):
                    input_tensor_grad = tuple(
                        [t.grad for t in input_tensor if not t.stop_gradient])
                else:
                    input_tensor_grad = input_tensor.grad
            return input_tensor_grad

    def _load_micro_batch(self, cache_id):
        inputs = self.data
        begin = cache_id * self.micro_batch_size
        end = begin + self.micro_batch_size

        # The virtual first and last pipeline stage need data, all others don't need.
        if self.is_pipeline_first_stage():
            assert len(inputs) == 2, "length of input should be 2"
            if isinstance(inputs[0], tuple):
                assert len(
                    inputs[0]
                ) > 1, "If you use tuple for input data, it should have at least two inputs."
                batch_size = inputs[0][0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size, (
                    "batch_size needs to be divisible by micro_batch_size. Currently, "
                    "batch_size = %d, micro_batch_size = %d, accumulate_steps = %d."
                    %
                    (batch_size, self.micro_batch_size, self.accumulate_steps))
                data = [input[begin:end, :].detach() for input in inputs[0]]
                return tuple(data)
            else:
                batch_size = inputs[0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                return inputs[0][begin:end, :].detach()
        elif self.is_pipeline_last_stage():
            assert len(inputs) == 2, "length of input should be 2"
            if isinstance(inputs[1], tuple):
                batch_size = inputs[1][0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                data = [input[begin:end, :].detach() for input in inputs[1]]
                return tuple(data)
            else:
                batch_size = inputs[1].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                return inputs[1][begin:end, :].detach()
        else:
            # No data input is required for other stages
            inputs = None

    def _broadcast_final_loss(self):
        # Since the last backward run in interleave will set the virtual rank to 0,
        # here we need to check last stage ignoring virtual stage.
        if self.is_pipeline_last_stage(ignore_virtual=True):
            assert self.total_loss is not None, "train_batch() in last stage should obtain vaild loss"
            loss = self.total_loss.detach()
            is_fp32 = paddle.to_tensor(
                1) if loss.dtype == paddle.float32 else paddle.to_tensor(0)
            paddle.distributed.broadcast(is_fp32,
                                         src=self.global_rank,
                                         use_calc_stream=True,
                                         group=self.pp_group)
            paddle.distributed.broadcast(loss,
                                         src=self.global_rank,
                                         use_calc_stream=True,
                                         group=self.pp_group)
        else:
            is_fp32 = paddle.to_tensor(1)
            paddle.distributed.broadcast(
                is_fp32,
                src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                use_calc_stream=True,
                group=self.pp_group)
            loss = paddle.zeros(shape=[
                1
            ], dtype="float32") if is_fp32.numpy()[0] else paddle.zeros(
                shape=[1], dtype="float16")
            paddle.distributed.broadcast(
                loss,
                src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                use_calc_stream=True,
                group=self.pp_group)
        return loss

    def _optimizer_step(self):
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.clear_grad()
        if self.lr_scheduler:
            self.lr_scheduler.step()


class PipelineParallelWithInterleave(PipelineParallel):
    # pipeline parallel with interleave scheduler

    def __init__(self, layers, hcg, strategy):
        super(PipelineParallelWithInterleave, self).__init__(layers=layers,
                                                             hcg=hcg,
                                                             strategy=strategy)
        assert layers.get_num_virtual_stages() > 1
        assert framework.in_dygraph_mode(
        ), "virtual pipeline stage with interleave only support eager dygraph mode"
        # setup for interleave scheduler
        self.num_model_chunks = layers.get_num_virtual_stages()
        self.model_chunks = layers.get_model_chunks()
        assert self.model_chunks is not None
        assert len(self.model_chunks) == self.num_model_chunks
        self._virtual_pp_world_size = self.num_model_chunks
        self._virtual_pp_rank = 0

    def _get_virtual_pp_rank(self, micro_step, forward):
        virtual_pp_stage = micro_step % (self.num_stages *
                                         self.num_model_chunks)
        virtual_pp_stage = virtual_pp_stage // self.num_stages
        if not forward:
            virtual_pp_stage = (self.num_model_chunks - virtual_pp_stage - 1)
        return virtual_pp_stage

    def _forward_step_helper(self, micro_step):
        virtual_pp_rank = self._get_virtual_pp_rank(micro_step, forward=True)
        self.set_virtual_pipeline_rank(virtual_pp_rank)

        # some checkers
        assert hasattr(self, 'input_tensors')
        assert hasattr(self, 'output_tensors')
        if not self._forward_only:
            assert hasattr(self, 'output_tensor_grads')

        if self.is_pipeline_first_stage():
            if len(self.input_tensors[virtual_pp_rank]) == len(
                    self.output_tensors[virtual_pp_rank]):
                self.input_tensors[virtual_pp_rank].append(None)
        input_tensor = self.input_tensors[virtual_pp_rank][-1]
        output_tensor = self._forward_step(input_tensor, virtual_pp_rank)
        self.output_tensors[virtual_pp_rank].append(output_tensor)

        if self._forward_only:
            # no need to store tensor for backward
            self.input_tensors[virtual_pp_rank].pop()
            self.output_tensors[virtual_pp_rank].pop()

        return output_tensor

    def _backward_step_helper(self, micro_step):
        virtual_pp_rank = self._get_virtual_pp_rank(micro_step, forward=False)
        self.set_virtual_pipeline_rank(virtual_pp_rank)

        # some checkers
        assert hasattr(self, 'input_tensors')
        assert hasattr(self, 'output_tensors')
        assert hasattr(self, 'output_tensor_grads')

        if self.is_pipeline_last_stage():
            if len(self.output_tensor_grads[virtual_pp_rank]) == 0:
                self.output_tensor_grads[virtual_pp_rank].append(None)

        input_tensor = self.input_tensors[virtual_pp_rank].pop(0)
        output_tensor = self.output_tensors[virtual_pp_rank].pop(0)
        output_tensor_grad = self.output_tensor_grads[virtual_pp_rank].pop(0)
        input_tensor_grad = self._backward_step(input_tensor, output_tensor,
                                                output_tensor_grad)

        return input_tensor_grad

    def interleave_pipeline(self,
                            data,
                            scaler,
                            forward_only=False,
                            compute_loss=True):
        # use interleave scheduling strategy.
        # this strategy is inspired by:
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/schedules.py
        if not compute_loss:
            assert not forward_only, "compute_loss can only be set to False when forward_only is set to True"

        # init some attributes for this batch run
        self.scaler = scaler
        self.data = data
        self.total_loss = None
        self.micro_batch_id = 0
        self._forward_only = forward_only

        # init some data buffers for interleave scheduler
        self.input_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensor_grads = [[] for _ in range(self.num_model_chunks)]

        num_steps = self.accumulate_steps * self.num_model_chunks
        all_startup_steps = False
        if forward_only:
            # If only forward, since there is no backward during running, all steps are startup steps
            startup_steps = num_steps
        else:
            if self.accumulate_steps == self.num_stages:
                startup_steps = num_steps
                all_startup_steps = True
            else:
                startup_steps = (self.num_stages - self.stage_id - 1) * 2
                startup_steps += (self.num_model_chunks - 1) * self.num_stages
                startup_steps = min(startup_steps, num_steps)

        steady_steps = num_steps - startup_steps

        self.set_virtual_pipeline_rank(0)
        self.input_tensors[0].append(
            p2p.recv_forward(self.is_pipeline_first_stage()))

        # run startup steps
        for micro_step in range(startup_steps):
            output_tensor = self._forward_step_helper(micro_step)

            # determine whether recv forward tensor or not
            next_virtual_pp_rank = self._get_virtual_pp_rank(micro_step + 1,
                                                             forward=True)
            recv_prev = True
            if self.is_pipeline_first_stage(ignore_virtual=True):
                if next_virtual_pp_rank == 0:
                    # next chunk is the first chunk, not need to pre recv an input tensor
                    recv_prev = False
            # last micro step, no next run
            if micro_step == (num_steps - 1):
                recv_prev = False

            # last stage shouldn't send tensor to downstream
            if self.is_pipeline_last_stage():
                output_tensor = None

            if micro_step == (startup_steps -
                              1) and not forward_only and not all_startup_steps:
                input_tensor_grad = None
                recv_next = True
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                # the last startup step needs on four direction comm to set up for steady 1f1b
                input_tensor, output_tensor_grad = p2p.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next)
                self.output_tensor_grads[self.num_model_chunks -
                                         1].append(output_tensor_grad)
            else:
                input_tensor = p2p.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev)
            self.input_tensors[next_virtual_pp_rank].append(input_tensor)

        # run 1f1b steady steps
        for micro_step in range(steady_steps):
            # forward
            forward_micro_step_id = micro_step + startup_steps
            output_tensor = self._forward_step_helper(forward_micro_step_id)

            # backward
            backward_micro_step_id = micro_step
            input_tensor_grad = self._backward_step_helper(
                backward_micro_step_id)

            # four directions comm
            # send output tensor to downstream
            # send input tensor grad to upstream
            # recv input tensor from upstream
            # recv output tensor grad from downstream

            # last stage doesn't send rst to downstream
            forward_virtual_pp_rank = self._get_virtual_pp_rank(
                forward_micro_step_id, forward=True)
            self.set_virtual_pipeline_rank(forward_virtual_pp_rank)
            if self.is_pipeline_last_stage():
                output_tensor = None

            # first stage doesn't send grad to upstream
            backward_virtual_pp_rank = self._get_virtual_pp_rank(
                backward_micro_step_id, forward=False)
            self.set_virtual_pipeline_rank(backward_virtual_pp_rank)
            if self.is_pipeline_first_stage():
                input_tensor_grad = None

            # determine whether to recv input tensor from upstream
            recv_prev = True
            if self.is_pipeline_first_stage(ignore_virtual=True):
                next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id - (self.num_stages - 1), forward=True)
                if next_forward_virtual_pp_rank == (self.num_model_chunks - 1):
                    # first pp stage and first virtual stage
                    recv_prev = False
                next_forward_virtual_pp_rank += 1
            else:
                next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id + 1, forward=True)

            # last iteration doesn't need recv from upstream
            if micro_step == (steady_steps - 1):
                recv_prev = False

            # determine whether to recv grad from downstream
            recv_next = True
            if self.is_pipeline_last_stage(ignore_virtual=True):
                next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    backward_micro_step_id - (self.num_stages - 1),
                    forward=False)
                if next_backward_virtual_pp_rank == 0:
                    # last pp stage and last virtual stage
                    recv_next = False
                next_backward_virtual_pp_rank -= 1
            else:
                next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    backward_micro_step_id + 1, forward=False)

            input_tensor, output_tensor_grad = p2p.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next)

            if recv_prev:
                self.input_tensors[next_forward_virtual_pp_rank].append(
                    input_tensor)
            if recv_next:
                self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                    output_tensor_grad)

        # remaining backward steps
        if not forward_only:
            if all_startup_steps:
                self.output_tensor_grads[self.num_model_chunks - 1].append(
                    p2p.recv_backward(self.is_pipeline_last_stage()))

            for micro_step in range(steady_steps, num_steps):
                # cooldown loop
                input_tensor_grad = self._backward_step_helper(micro_step)
                next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    micro_step + 1, forward=False)

                recv_next = True
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    if next_backward_virtual_pp_rank == (self.num_model_chunks -
                                                         1):
                        recv_next = False

                if micro_step == (num_steps - 1):
                    recv_next = False

                self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                    p2p.send_backward_recv_backward(input_tensor_grad,
                                                    recv_next=recv_next))

            self._layers.allreduce_shared_weight_gradients()

        if compute_loss:
            # return loss if compute loss
            with paddle.amp.auto_cast(enable=False):
                train_loss = self._broadcast_final_loss()
        else:
            # else just return all intermediate output tensor for all micro steps
            train_loss = self.output_tensors

        return train_loss

    def train_batch(self, data, optimizer, lr_scheduler=None, scaler=None):
        data = self._prepare_training(data, optimizer, lr_scheduler)
        # interleave scheduler for pipeline parallel
        train_loss = self.interleave_pipeline(data, scaler)

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss

    def eval_batch(self, data, compute_loss=False):
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        self._layers.eval()
        self._compute_loss = compute_loss

        return self.interleave_pipeline(data, None, forward_only=True)
