# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import abc
import paddle
from ...utils import hybrid_parallel_util as hp_util

__all__ = []

FLOAT_TYPES = [
    paddle.float16,
    paddle.float32,
    paddle.float64,
]


def is_float_tensor(tensor):
    """Is a float tensor"""
    return tensor.dtype in FLOAT_TYPES


def get_tensor_bytes(tensor):
    """Get the bytes a tensor occupied."""
    elem_size = None
    if tensor.dtype == paddle.float32:
        elem_size = 4
    elif tensor.dtype == paddle.float64:
        elem_size = 8
    elif tensor.dtype == paddle.int64:
        elem_size = 8
    elif tensor.dtype == paddle.int32:
        elem_size = 4
    elif tensor.dtype == paddle.float16:
        elem_size = 2
    elif tensor.dtype == paddle.int8:
        elem_size = 1
    else:
        raise ValueError("unknown data type: {}".format(tensor.dtype))
    return tensor.numel() * elem_size


class Generator():
    def __init__(self, micro_batches, stages, stage_id):
        __metaclass__ = abc.ABCMeta

        self.micro_batches = micro_batches
        self.stages = stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

    @abc.abstractmethod
    def generate(self):
        pass

    def __iter__(self):
        self.iter = None
        return self

    def __next__(self):
        if self.iter is None:
            self.iter = self.generate()
        return next(self.iter)


class TrainGenerator(Generator):
    def generate(self):
        startup_steps = self.stages - self.stage_id - 1
        cmds = []
        forward_steps = 0
        backward_steps = 0
        #while (forward_steps < startup_steps):
        #    cmds.append(Forward(cache_id=forward_steps))
        #    forward_steps += 1
        #while (forward_steps < self.micro_batches):
        #    cmds.append(Forward(cache_id=forward_steps))
        #    forward_steps += 1
        #    cmds.append(Backward(cache_id=backward_steps))
        #    backward_steps += 1
        #while (backward_steps < self.micro_batches):
        #    cmds.append(Backward(cache_id=backward_steps))
        #    backward_steps += 1
        #cmds.append(Optimize())
        while (forward_steps < self.micro_batches):
            cmds.append(Forward(cache_id=forward_steps))
            forward_steps += 1
        while (backward_steps < self.micro_batches):
            cmds.append(Backward(cache_id=backward_steps))
            backward_steps += 1
        cmds.append(Optimize())
        yield cmds


class Command:
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return hp_util.call_to_str(self.name, **self.kwargs)


class Optimize(Command):
    pass


class Forward(Command):
    pass


class Backward(Command):
    pass
