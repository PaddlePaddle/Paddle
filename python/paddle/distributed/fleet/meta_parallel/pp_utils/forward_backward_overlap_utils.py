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


import paddle

from .utils import dict_to_tuple_helper


class ScheduleChunk:
    # NOTE(zhangyuqin): ScheduleChunk is the atomic unit of pipeline scheduling.
    # A ScheduleChunk can contain several ScheduleNodes
    def __init__(self, nodes):
        self.nodes = nodes
        self._check_nodes_valid()

    def forward(self, inputs):
        for n in self.nodes:
            inputs = n.forward(inputs)
        return inputs

    def backward(self, output_grad):
        for n in reversed(self.nodes):
            output_grad = n.backward(output_grad)
        return output_grad

    def _check_nodes_valid(self):
        for n in self.nodes:
            assert isinstance(n, (ScheduleNode, ScheduleChunk))


def detach_and_requires_grad(inputs):
    if isinstance(inputs, (tuple, list)):
        is_tuple = isinstance(inputs, tuple)
        ret = []
        for input in inputs:
            if isinstance(input, (tuple, list)):
                ret.append(detach_and_requires_grad(input))
            elif isinstance(input, paddle.Tensor):
                tmp = input.detach() if input is not None else None
                if tmp is not None:
                    tmp.stop_gradient = input.stop_gradient
                ret.append(tmp)
            else:
                ret.append(input)
        if is_tuple:
            ret = tuple(ret)
        return ret
    elif isinstance(inputs, dict):
        ret = {}
        for key in inputs.keys():
            input = inputs[key]
            tmp = input.detach() if input is not None else None
            if tmp is not None:
                tmp.stop_gradient = input.stop_gradient
            ret[key] = tmp
        return ret
    else:
        tmp = inputs.detach()
        tmp.stop_gradient = inputs.stop_gradient
        return tmp


def clone_and_clear_dataptr(outputs, clear_dataptr=False):
    if isinstance(outputs, (tuple, list)):
        is_tuple = isinstance(outputs, tuple)
        ret = [
            FakeClone.apply(o)
            for o in outputs
            if o is not None and isinstance(o, paddle.Tensor)
        ]

        if clear_dataptr:
            for o in ret:
                o._clear_dataptr()
        if is_tuple:
            ret = tuple(ret)
        return ret
    elif isinstance(outputs, dict):
        ret = {}
        for key in outputs.keys():
            o = outputs[key]
            if o is not None and isinstance(o, paddle.Tensor):
                ret[key] = FakeClone.apply(o)
        if clear_dataptr:
            for key in ret:
                ret[key]._clear_dataptr()
        return ret
    else:
        ret = FakeClone.apply(outputs)
        if clear_dataptr:
            ret._clear_dataptr()
        return ret


class FakeClone(paddle.autograd.PyLayer):
    # NOTE(zhangyuqin): Some input tensors may not be used in the forward function, but their gradients
    # need to be retained. Therefore, we need a clone here. To avoid the DtoD copy, we need a FakeClone
    @staticmethod
    def forward(ctx, input):
        return paddle.empty_like(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ScheduleNode:
    # NOTE(zhangyuqin): ScheduleNode is a subgraph of the pipeline, capable of independently calling
    # forward and backward. Users should not use paddle.autograd.backward on the results of ScheduleNode.forward.
    # Instead, they should use ScheduleNode.backward. Otherwise, resource leakage may occur.
    def __init__(self, fwd_func, name=""):
        self.name = name
        self.fwd_func = fwd_func
        self.inputs = None
        self.outputs = None

        self.labels = None
        self.scale_loss_factor = None

    def forward(self, inputs=(), **kwargs):
        detached_inputs = detach_and_requires_grad(inputs)
        self.inputs = detached_inputs
        if self.labels is not None:
            outputs = self.fwd_func(self.inputs, self.labels, **kwargs)
        else:
            outputs = self.fwd_func(self.inputs, **kwargs)
        if self.scale_loss_factor is not None:
            outputs /= self.scale_loss_factor

        # Do not release the loss tensor.
        clear_dataptr = self.labels is None
        self.outputs = clone_and_clear_dataptr(outputs, clear_dataptr)
        return outputs

    def backward(self, output_grad=None, scaler=None):
        if output_grad is None:
            if isinstance(self.outputs, (tuple, list)):
                assert len(self.outputs) == 1
                outputs = self.outputs[0]
            else:
                outputs = self.outputs
            assert isinstance(outputs, paddle.Tensor)
            if scaler is not None:
                paddle.autograd.backward(scaler.scale(outputs))
            else:
                paddle.autograd.backward(outputs)
        else:
            # Record the original type (tuple or list) to preserve it after filtering
            is_output_grad_tuple = isinstance(output_grad, tuple)
            if not isinstance(output_grad, (tuple, list)):
                is_output_grad_tuple = True  # Single value becomes tuple
                output_grad = (output_grad,)

            outputs = dict_to_tuple_helper(self.outputs)
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            outputs = [t for t in outputs if not t.stop_gradient]

            # Filter None values from output_grad
            output_grad = [grad for grad in output_grad if grad is not None]
            # Preserve original type (tuple or list)
            output_grad = (
                tuple(output_grad)
                if is_output_grad_tuple
                else list(output_grad)
            )

            assert len(outputs) == len(output_grad), (
                f"{len(outputs)} of {type(outputs[0])} vs {len(output_grad)} of {type(output_grad[0])}"
            )

            paddle.autograd.backward(outputs, output_grad)

        inputs = dict_to_tuple_helper(self.inputs)
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        grad = tuple([e.grad if e is not None else None for e in inputs])
        # grad = tuple([e.grad if e is not None and not e.stop_gradient else None for e in inputs])
        self._reset_states()

        # if len(grad) == 1:
        #     grad = grad[0]
        return grad

    def _reset_states(self):
        self.inputs = None
        self.outputs = None
        self.labels = None
        self.scale_loss_factor = None
