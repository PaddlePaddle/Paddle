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
from .pp_utils.utils import get_tensor_bytes, is_float_tensor
from .pp_utils import utils
from .parallel_layers.pp_layers import PipelineLayer

from ..utils.hybrid_parallel_util import broadcast_mp_parameters
from ..utils.hybrid_parallel_util import broadcast_dp_parameters
from ..utils.hybrid_parallel_util import fused_allreduce_gradients
from ..utils.log_util import logger

__all__ = []

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
        }

        self.recv_cache = None
        self.grad_tensors = None

        self.send_meta = True

        self.current_loss = paddle.to_tensor(0.0)
        self.total_loss = None

        self.use_amp = self._strategy.amp
        self.init_loss_scaling = self._strategy.amp_configs['init_loss_scaling']
        self.micro_batch_size = self._strategy.pipeline_configs[
            'micro_batch_size']
        self.accumulate_steps = self._strategy.pipeline_configs[
            'accumulate_steps']

        self.num_stages = self._hcg.get_pipe_parallel_world_size()
        self.stage_id = self._hcg.get_stage_id()
        self.prev_stage_id = self.stage_id - 1
        self.next_stage_id = self.stage_id + 1
        self.pp_group = self._hcg.get_pipe_parallel_group()
        logger.info("Pipeline Info -- num_stages: {}, stage_id: {}".format(
            self.num_stages, self.stage_id))

        if self.use_model_parallel:
            logger.info("start broadcast mp parameters")
            broadcast_mp_parameters(self._layers, self._hcg)

        if self.use_data_parallel:
            logger.info("start broadcast mp parameters")
            broadcast_dp_parameters(self._layers, self._hcg)

    def _allocate_caches(self, num_caches):
        if self.num_caches >= num_caches:
            return

        num = num_caches - self.num_caches
        self.num_caches = num_caches
        for key in self.caches:
            self.caches[key].extend([None] * num)

    def train_batch(self, data, optimizer):
        self.optimizer = optimizer
        assert fluid.framework._dygraph_tracer()._has_grad, (
            'Please enable the generation of gradients.')

        if self.stage_id == 0 or self.stage_id == self.num_stages - 1:
            assert data, (
                "For the first and the last stage, the data_iter must be set.")
        else:
            assert data is None, (
                "For pipe stages other than the first and the last one, "
                "the data_iter must be None.")
        self.data = data
        self._layers.train()
        self.total_loss = None

        minibatch_cmds = utils.TrainGenerator(self.accumulate_steps,
                                              self.num_stages, self.stage_id)
        self._train(minibatch_cmds)
        return self.total_loss

    def _train(self, minibatch_cmds):
        self._allocate_caches(self.accumulate_steps)
        for micro_cmds in minibatch_cmds:
            for cmd in micro_cmds:
                assert type(cmd) in self._COMMAND_MAP, "unknow cmd: {}".format(
                    type(cmd))
                self._apply_cmd = MethodType(self._COMMAND_MAP[type(cmd)], self)
                self._apply_cmd(**cmd.kwargs)

    def _allreduce_grads(self):
        if not self.use_data_parallel: return
        fused_allreduce_gradients(list(self._layers.parameters()), self._hcg)

    def _forward(self, cache_id):
        # load data
        self._load_micro_batch(cache_id)
        if self.stage_id != 0:
            self._recv_activations(cache_id)

        if isinstance(self.caches['inputs'][cache_id], tuple):
            inputs = tuple(t for t in self.caches['inputs'][cache_id])
        else:
            inputs = self.caches['inputs'][cache_id]

        self._clear_grads(inputs)
        outputs = self._layers.forward(inputs)
        self.caches['outputs'][cache_id] = outputs

        if self.stage_id == self.num_stages - 1:
            if self._layers._loss_fn is not None:
                labels = self.caches['labels'][cache_id]
                outputs = self._layers._loss_fn(outputs, labels)

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
            if self.use_data_parallel:
                self.current_loss = self.current_loss / self._hcg.get_data_parallel_world_size(
                )
            if self.accumulate_steps > 1:
                self.current_loss = self.current_loss / self.accumulate_steps
            self.caches['outputs'][cache_id] = self.current_loss.clone()
        else:
            self._send_activations(cache_id)

    def _backward(self, cache_id):
        assert self.optimizer is not None
        if self.stage_id == self.num_stages - 1:
            paddle.autograd.backward(self.caches['outputs'][cache_id])
            self._send_gradients(cache_id)
            return
        self._recv_gradients(cache_id)

        outputs = self.caches['outputs'][cache_id]

        grad_tensors = self.grad_tensors
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if is_float_tensor(t)]
            assert len(out_tensors) == len(grad_tensors)
            paddle.autograd.backward(
                tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            paddle.autograd.backward(
                tensors=[outputs], grad_tensors=[grad_tensors])

        grad_tensors = None
        if self.stage_id != 0: self._send_gradients(cache_id)
        self.caches['outputs'][cache_id] = None
        #self.caches['backward_tensors'][cache_id] = None

    def _get_data(self):
        if self.use_model_parallel:
            mp_rank = self._hcg.get_model_parallel_rank()
        else:
            mp_rank = 0

        # mp rank 0 loads the data and broadcat it to others.
        data = self.data
        if self.use_model_parallel and (self.stage_id == 0 or
                                        self.stage_id == self.num_stages - 1):
            assert isinstance(data, (tuple, paddle.Tensor))
            if isinstance(data, paddle.Tensor):
                paddle.distributed.broadcast(
                    data,
                    src=self._hcg.get_model_parallel_group_src_rank(),
                    group=self._hcg.get_model_parallel_group())
            else:
                data = []
                for d in self.data:
                    assert isinstance(d, paddle.Tensor)
                    paddle.distributed.broadcast(
                        d,
                        src=self._hcg.get_model_parallel_group_src_rank(),
                        group=self._hcg.get_model_parallel_group())
                    data.append(d)
            data = tuple(data)
        return data

    def _load_micro_batch(self, cache_id):
        inputs = self._get_data()

        if self.stage_id == 0:
            data = None
            #if isinstance(inputs[0], paddle.Tensor):
            if len(inputs) == 1:
                assert isinstance(inputs[0], paddle.Tensor)
                data = inputs[0].clone().detach()
                #data.stop_gradient = not is_float_tensor(data)
                data.stop_gradient = True
            else:
                assert isinstance(inputs, tuple)
                data = []
                for d in inputs:
                    assert isinstance(d, paddle.Tensor)
                    i = d.clone().detach()
                    #i.stop_gradient = not is_float_tensor(i)
                    i.stop_gradient = True
                    data.append(i)
                data = tuple(data)
            self.caches['inputs'][cache_id] = data

        if self.stage_id == self.num_stages - 1:
            labels = None
            #if isinstance(inputs[1], paddle.Tensor):
            if len(inputs) == 1:
                assert isinstance(inputs[0], paddle.Tensor)
                labels = inputs[0]
            elif isinstance(inputs, tuple):
                labels = []
                for label in inputs:
                    assert isinstance(label, paddle.Tensor)
                    label = label.detach()
                    labels.append(label)
                labels = tuple(labels)
            self.caches['labels'][cache_id] = labels

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
            paddle.distributed.send(
                tensor_type, peer, use_calc_stream=True, group=self.pp_group)
            dims = paddle.to_tensor(len(data.shape))
            paddle.distributed.send(
                dims, peer, use_calc_stream=True, group=self.pp_group)
            shape = paddle.to_tensor(data.shape)
            paddle.distributed.send(
                shape, peer, use_calc_stream=True, group=self.pp_group)
        elif isinstance(data, tuple):
            tensor_type = paddle.to_tensor([1])
            paddle.distributed.send(
                tensor_type, peer, use_calc_stream=True, group=self.pp_group)
            nums = paddle.to_tensor(len(data))
            paddle.distributed.send(
                nums, peer, use_calc_stream=True, group=self.pp_group)
            for idx, d in enumerate(data):
                assert isinstance(d, paddle.Tensor)
                dims = paddle.to_tensor(len(d.shape))
                paddle.distributed.send(
                    dims, peer, use_calc_stream=True, group=self.pp_group)
                shape = paddle.to_tensor(d.shape)
                paddle.distributed.send(
                    shape, peer, use_calc_stream=True, group=self.pp_group)

    def _recv_meta(self, peer):
        tensor_type = paddle.to_tensor([0])
        paddle.distributed.recv(
            tensor_type, peer, use_calc_stream=True, group=self.pp_group)
        tensor_type = tensor_type.numpy()[0]

        if tensor_type == 0:
            dims = paddle.to_tensor([0])
            paddle.distributed.recv(
                dims, peer, use_calc_stream=True, group=self.pp_group)
            dims = dims.numpy()[0]
            shape = paddle.to_tensor([0] * dims)
            paddle.distributed.recv(
                shape, peer, use_calc_stream=True, group=self.pp_group)
            shape = shape.numpy().tolist()
            return self._allocate_buffer(
                shape, dtype="float32", num_caches=1)[0]
        elif tensor_type == 1:
            num = paddle.to_tensor([0])
            paddle.distributed.recv(
                num, peer, use_calc_stream=True, group=self.pp_group)
            num = num.numpy()[0]
            shapes = []
            for i in range(num):
                dims = paddle.to_tensor([0])
                paddle.distributed.recv(
                    dims, peer, use_calc_stream=True, group=self.pp_group)
                dims = dims.numpy()[0]
                shape = paddle.to_tensor([0] * dims)
                paddle.distributed.recv(
                    shape, peer, use_calc_stream=True, group=self.pp_group)
                shapes.append(shape.numpy().tolist())

            dtypes = ["float32"] * len(shapes)
            caches = self._allocate_buffers(shapes, dtypes, num_caches=1)[0]
            caches = tuple(caches)
            return caches

    def _send_activations(self, cache_id):
        outputs = self.caches['outputs'][cache_id]

        if self.send_meta:
            self.send_meta = False
            self._send_meta(outputs, self.next_stage_id)

        if isinstance(outputs, paddle.Tensor):
            paddle.distributed.send(
                outputs,
                self.next_stage_id,
                use_calc_stream=True,
                group=self.pp_group)
        elif isinstance(outputs, tuple):
            for output in outputs:
                paddle.distributed.send(
                    output,
                    self.next_stage_id,
                    use_calc_stream=True,
                    group=self.pp_group)

    def _send_gradients(self, cache_id):
        inputs = self.caches['inputs'][cache_id]

        if isinstance(inputs, paddle.Tensor):
            assert inputs.grad is not None
            paddle.distributed.send(
                paddle.to_tensor(inputs.grad),
                self.prev_stage_id,
                use_calc_stream=True,
                group=self.pp_group)
        else:
            for idx, d in enumerate(inputs):
                # Skip tensors that will not produce a grad
                if not is_float_tensor(d):
                    assert d.grad is None
                    continue
                assert d.grad is not None
                paddle.distributed.send(
                    d.grad,
                    self.prev_stage_id,
                    use_calc_stream=True,
                    group=self.pp_group)
        self.caches['inputs'][cache_id] = None

    def _recv_activations(self, cache_id):
        inputs = None

        # Allocate the buffer if necessary
        if self.recv_cache is None:
            self.recv_cache = self._recv_meta(self.prev_stage_id)

        if isinstance(self.recv_cache, paddle.Tensor):
            paddle.distributed.recv(
                self.recv_cache,
                self.prev_stage_id,
                use_calc_stream=True,
                group=self.pp_group)
            inputs = self.recv_cache.clone().detach()
            inputs.stop_gradient = not is_float_tensor(inputs)
        else:
            assert isinstance(self.recv_cache, tuple)
            inputs = [None] * len(self.recv_cache)
            for idx, d in enumerate(self.recv_cache):
                assert isinstance(d, paddle.Tensor)

                paddle.distributed.recv(
                    d,
                    self.prev_stage_id,
                    use_calc_stream=True,
                    group=self.pp_group)
                inputs[idx] = d.clone().detach()

            inputs = tuple(inputs)

            for d in inputs:
                d.stop_gradient = not is_float_tensor(d)

        self.caches['inputs'][cache_id] = inputs

    def _recv_gradients(self, cache_id):
        outputs = self.caches['outputs'][cache_id]
        if self.grad_tensors is None:
            if isinstance(outputs, paddle.Tensor):
                s = list(outputs.shape)
                dtype = 'float16' if self.use_amp else "float32"
                self.grad_tensors = self._allocate_buffer(
                    s, dtype, num_buffers=1)[0]
            else:
                sizes = [list(d.shape) for d in outputs if is_float_tensor(d)]
                dtypes = ['float16'] * len(
                    sizes) if self.use_amp else ['float32'] * len(sizes)
                self.grad_tensors = self._allocate_buffers(
                    sizes, dtypes, num_caches=1)[0]

        if isinstance(self.grad_tensors, paddle.Tensor):
            paddle.distributed.recv(
                self.grad_tensors,
                self.next_stage_id,
                use_calc_stream=True,
                group=self.pp_group)
        else:
            assert isinstance(outputs, tuple)
            for d in self.grad_tensors:
                paddle.distributed.recv(
                    d,
                    self.next_stage_id,
                    use_calc_stream=True,
                    group=self.pp_group)

    def _step(self):
        self._allreduce_grads()
        self.optimizer.step()
        self.optimizer.clear_gradients()

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

    def _allocate_buffer(self, shape, dtype, num_caches=-1):
        caches = []
        if num_caches == -1:
            num_caches = self.num_caches
        for count in range(num_caches):
            caches.append(self._allocate_zeros(shape, dtype))
        return caches

    def _allocate_buffers(self, shapes, dtypes, num_caches=-1):
        caches = []
        if num_caches == -1:
            num_caches = self.num_caches
        for count in range(num_caches):
            cache = []
            for shape, dtype in zip(shapes, dtypes):
                cache.append(self._allocate_zeros(shape, dtype))
            caches.append(cache)
        return caches

    def save_state_dict(self, model_path):
        state_dict = self._layers.state_dict()
        paddle.save(state_dict, model_path)

    def load_state_dict(self, model_path):
        state_dict = paddle.load(self.model_path)
        self._layers.set_state_dict(state_dict)

    _COMMAND_MAP = {
        utils.Optimize: _step,
        utils.Forward: _forward,
        utils.Backward: _backward,
    }

    def forward(self, *inputs, **kwargs):
        raise RuntimeError("Call train_batch for pipeline instead of forward.")
