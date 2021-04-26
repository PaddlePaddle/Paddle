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

import time
import copy
import os

from types import MethodType

from numpy import prod

import paddle
import paddle.fluid as fluid
from .meta_parallel_base import MetaParallelBase
from .pp_utils.utils import get_tensor_bytes
from .pp_utils import utils
from .parallel_layers.pp_layers import PipelineLayer

FLOAT_TYPES = [
    paddle.float16,
    paddle.float32,
    paddle.float64,
]


class PipelineParallel(MetaParallelBase):
    def __init__(self, layers, hcg, strategy):
        super(PipelineParallel, self).__init__(layers, hcg, strategy)

        self.use_pipe_parallel = self._hcg.get_pipe_parallel_world_size() > 1
        self.use_data_parallel = self._hcg.get_data_parallel_world_size() > 1
        self.use_model_parallel = self._hcg.get_model_parallel_world_size() > 1

        self.num_caches = 0
        self.caches = {
            'inputs': [],
            'labels': [],
            'outputs': [],
            'backward_tensors': [],
        }
        self.recv_cache = None
        self.grad_tensors = None

        self.meta_buffer = None

        self.send_meta = True
        self.first_gradient_send = True

        self.current_loss = paddle.to_tensor(0.0)
        self.total_loss = None

    def _prepare_for_model(self):
        self.micro_batch_size = self._strategy.pipeline_configs[
            'micro_batch_size']
        self.accumulate_steps = self._strategy.pipeline_configs[
            'accumulate_steps']

        self.num_stages = self._hcg.get_pipe_parallel_world_size()
        self.stage_id = self._hcg.get_stage_id()
        self.prev_stage_id = self.stage_id - 1
        self.next_stage_id = self.stage_id + 1
        self._layers = PipelineLayer(
            layers=self._layers, num_stages=self.num_stages)
        #TODO: init process group

    def _allocate_caches(self, num_caches):
        if self.num_caches >= num_caches:
            return

        num = num_caches - self.num_caches
        self.num_caches = num_caches
        for key in self.caches:
            self.caches[key].extend([None] * num)

    def train_batch(self, data_iter, optimizer):
        self.optimizer = optimizer
        assert fluid.framework._dygraph_tracer()._has_grad, (
            'Please enable the generation of gradients.')

        if self.stage_id == 0 or self.stage_id == self.num_stages - 1:
            assert data_iter, (
                "For the first and the last stage, the data_iter must be set.")
        else:
            assert data_iter is None, (
                "For pipe stages other than the first and the last one, "
                "the data_iter must be None.")
        self.data_iter = data_iter
        self._layers.train()
        self.total_loss = None

        minibatch_cmds = utils.TrainGenerator(self.accumulate_steps,
                                              self.num_stages, self.stage_id)
        self._train(minibatch_cmds)
        return self.total_loss

    def _train(self, minibatch_cmds):
        self._allocate_caches(self.num_stages)
        for microbatch_cmds in minibatch_cmds:
            for cmd in microbatch_cmds:
                if type(cmd) not in self._COMMAND_MAP:
                    #FIXME:
                    continue

                self._apply_cmd = MethodType(self._COMMAND_MAP[type(cmd)], self)
                self._apply_cmd(**cmd.kwargs)

    def _allreduce_grads(self):
        self._modifying_grad = True
        assert self.use_data_parallel <= 1, ("Do not support data parallel "
                                             "with pipeline parallel now.")
        self._modifying_grad = False

    def _get_data(self):
        if self.use_model_parallel:
            mp_rank = self._hcg.get_model_parallel_rank()
        else:
            mp_rank = 0

        data = None

        # mp rank 0 loads the data and broadcat it to others.
        if mp_rank == 0:
            data = next(self.data_iter)
        if self.use_model_parallel:
            data = paddle.distributed.broadcast(
                data, group=self._hcg.get_model_parallel_group())
        return data

    def _forward(self, cache_id):
        if isinstance(self.caches['inputs'][cache_id], tuple):
            inputs = tuple(t.clone() for t in self.caches['inputs'][cache_id])
        else:
            inputs = self.caches['inputs'][cache_id].clone()

        self._clear_grads(inputs)
        outputs = self._layers.forward(inputs)

        self.caches['outputs'][cache_id] = outputs

        if self.stage_id == self.num_stages - 1:
            self.current_loss = outputs
            if isinstance(self.current_loss, paddle.Tensor):
                if self.total_loss is None:
                    self.total_loss = paddle.zeros_like(self.current_loss)
                self.total_loss += self.current_loss.detach()
            else:
                if self.total_loss is None:
                    self.total_loss = [
                        paddle.zeros_like(v) for v in self.current_loss
                    ]
                for idx, v in enumerate(self.current_loss):
                    self.total_loss[idx] += v.detach()

    def _backward(self, cache_id):
        assert self.optimizer is not None
        if self.stage_id == self.num_stages - 1:
            paddle.autograd.backward(self.current_loss)
            return

        outputs = self.caches['outputs'][cache_id]

        grad_tensors = self.grad_tensors
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.dtype in FLOAT_TYPES]
            assert len(out_tensors) == len(grad_tensors)
            paddle.autograd.backward(
                tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            paddle.autograd.backward(
                tensors=[outputs], grad_tensors=[grad_tensors])

        self.caches['outputs'][cache_id] = None
        grad_tensors = None

    def _load_micro_batch(self, cache_id):
        inputs = self._get_data()

        if self.stage_id == 0:
            data = None
            if isinstance(inputs[0], paddle.Tensor):
                data = inputs[0].clone().detach()
                data.stop_gradient = data.dtype == paddle.float32
            else:
                assert isinstance(inputs[0], tuple)
                # Assume list or tuple
                data = []
                for d in inputs[0]:
                    assert isinstance(d, paddle.Tensor)
                    d = d.clone().detach()
                    d.stop_gradient = d.dtype == paddle.float32
                    loaded.append(d)
                data = tuple(data)
            self.caches['inputs'][cache_id] = data

        if self.stage_id == self.num_stages - 1:
            label = None
            if isinstance(inputs[1], paddle.Tensor):
                label = inputs[1]
            elif isinstance(data[1], tuple):
                label = []
                for l in inputs[1]:
                    assert isinstance(l, paddle.Tensor)
                    l = l.detach()
                    label.append(l)
                label = tuple(label)
            self.caches['labels'][cache_id] = label

    def _send_meta(self, data, peer):
        """
        % type (0: tensor, 1: tuple)
        % num_tensors if type=tuple
        foreach tensor:
          % ndims
          % shape
        """
        if isinstance(data, paddle.Tensor):
            tensor_type = paddle.to_tensor([0])
            paddle.distributed.send(tensor_type, peer)
            dims = paddle.to_tensor(len(data.shape))
            paddle.distributed.send(dims, peer)
            shape = paddle.to_tensor(data.shape)
            paddle.distributed.send(shape, peer)
        elif isinstance(data, tuple):
            tensor_type = paddle.to_tensor([1])
            paddle.distributed.send(tensor_type, peer)
            nums = paddle.to_tensor(len(data))
            paddle.distributed.send(nums, peer)
            for idx, d in enumerate(data):
                assert isinstance(d, paddle.Tensor)
                dims = paddle.to_tensor(len(d.shape))
                paddle.distributed.send(dims, peer)
                shape = paddle.to_tensor(d.shape)
                paddle.distributed.send(shape, peer)

    def _recv_meta(self, peer):
        tensor_type = paddle.to_tensor([0])
        paddle.distributed.recv(tensor_type, peer)
        tensor_type = tensor_type.numpy()[0]

        if tensor_type == 0:
            dims = paddle.to_tensor([0])
            paddle.distributed.recv(dims, peer)
            dims = dims.numpy()[0]
            shape = paddle.to_tensor([0] * dims)
            paddle.distributed.recv(shape, peer)
            shape = shape.numpy().tolist()
            return self._allocate_buffer(
                shape, dtype="float32", num_caches=1)[0]
        elif tensor_type == 1:
            num = paddle.to_tensor([0])
            paddle.distributed.recv(num, peer)
            num = num.numpy()[0]
            shapes = []
            for i in range(num):
                dims = paddle.to_tensor([0])
                paddle.distributed.recv(dims, peer)
                dims = dims.numpy()[0]
                shape = paddle.to_tensor([0] * dims)
                paddle.distributed.recv(shape, peer)
                shapes.append(shape.numpy().tolist())

            dtypes = ["float32"] * len(shapes)
            caches = self._allocate_buffers(shapes, dtypes, num_buffers=1)[0]
            buffers = tuple(buffers)
            return buffers

    def _send_activations(self, cache_id):
        outputs = self.caches['outputs'][cache_id]

        if self.send_meta:
            self.send_meta = False
            self._send_meta(outputs, self.next_stage_id)

        if isinstance(outputs, paddle.Tensor):
            paddle.distributed.send(outputs, self.next_stage_id)
        elif isinstance(outputs, tuple):
            for output in outputs:
                paddle.distributed.send(output, self.next_stage_id)

    def _send_gradients(self, cache_id):
        inputs = self.caches['inputs'][cache_id]

        if isinstance(inputs, paddle.Tensor):
            assert inputs.grad is not None
            paddle.distributed.send(
                paddle.to_tensor(inputs.grad), self.prev_stage_id)
        else:
            for idx, d in enumerate(inputs):
                # Skip tensors that will not produce a grad
                if not d.dtype in FLOAT_TYPES:
                    assert d.grad is None
                    continue
                assert d.grad is not None
                paddle.distributed.send(d.grad, self.prev_stage_id)
        self.caches['inputs'][cache_id] = None

    def _recv_activations(self, cache_id):
        inputs = None

        # Allocate the buffer if necessary
        if self.recv_cache is None:
            self.recv_cache = self._recv_meta(self.prev_stage_id)

        if isinstance(self.recv_cache, paddle.Tensor):
            paddle.distributed.recv(self.recv_cache, self.prev_stage_id)
            inputs = self.recv_cache.clone().detach()
            inputs.stop_gradient = inputs.dtype not in FLOAT_TYPES
        else:
            assert isinstance(self.recv_cache, tuple)
            inputs = [None] * len(self.recv_cache)
            for idx, d in enumerate(self.recv_cache):
                assert isinstance(d, paddle.Tensor)

                paddle.distributed.recv(d, self.prev_stage_id)
                inputs[idx] = d.clone().detach()

            inputs = tuple(inputs)

            for d in inputs:
                d.stop_gradient = d.dtype not in FLOAT_TYPES

        self.caches['inputs'][cache_id] = inputs

    def _recv_gradients(self, cache_id):
        outputs = self.caches['outputs'][cache_id]
        if self.grad_tensors is None:
            if isinstance(outputs, paddle.Tensor):
                s = list(outputs.shape)
                dtype = 'float32'
                self.grad_tensors = self._allocate_buffer(
                    s, dtype, num_buffers=1)[0]
            else:
                sizes = [
                    list(d.shape) for d in outputs if d.dtype in FLOAT_TYPES
                ]
                dtypes = ['float32'] * len(sizes)
                self.grad_tensors = self._allocate_buffers(
                    sizes, dtypes, num_buffers=1)[0]

        if isinstance(self.grad_tensors, paddle.Tensor):
            paddle.distributed.recv(self.grad_tensors, self.next_stage_id)
        else:
            assert isinstance(outputs, tuple)
            for d in self.grad_tensors:
                paddle.distributed.recv(d, self.next_stage_id)

    def _step(self, lr_kwargs=None):
        self._modifying_grad = True
        self.optimizer.step()
        self.optimizer.clear_gradients()
        self._modifying_grad = False

    def _clear_grads(self, inputs):
        if isinstance(inputs, paddle.Tensor):
            if inputs.grad is not None:
                inputs.clear_gradient()
        else:
            for d in inputs:
                if d.grad is not None:
                    d.clear_gradient()

    def _allocate_zeros(self, shape, dtype):
        return paddle.zeros(shape, dtype)

    def _allocate_buffer(self, shape, dtype, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_caches
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, dtype))
        return buffers

    def _allocate_buffers(self, shapes, dtypes, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_caches
        for count in range(num_buffers):
            buffer = []
            for shape, dtype in zip(shapes, dtypes):
                buffer.append(
                    self._allocate_zeros(
                        shape, dtype, requires_grad=requires_grad))
            buffers.append(buffer)
        return buffers

    def save_state_dict(self, model_path):
        state_dict = self._layers.state_dict()
        paddle.save(state_dict, model_path)

    def load_state_dict(self, model_path):
        state_dict = paddle.load(self.model_path)
        self._layers.set_state_dict(state_dict)

    _COMMAND_MAP = {
        utils.Optimize: _step,
        #utils.ReduceGrads: _allreduce_grads,
        utils.Forward: _forward,
        utils.Backward: _backward,
    }

    def _pre_forward(self, *inputs, **kwargs):
        pass

    def forward(self, *inputs, **kwargs):
        raise RuntimeError("Call train_batch for pipeline instead of forward.")

    def _post_forward(self, output):
        pass

    def _pre_backward(self, loss):
        pass

    def backward_impl(self, loss, parameters):
        pass

    def _post_backward(self, loss):
        pass
