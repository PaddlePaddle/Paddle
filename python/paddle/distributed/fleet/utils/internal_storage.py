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
# limitations under the License.

# The file has been adapted from fairscale file:
# https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/misc/param_bucket.py
# Git commit hash: 8acbec718f3c70a6b9785470bb9e05cd84fc3f8e
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import paddle
from paddle import framework

# (TODO: GhostScreaming) It will be removed later.
from paddle.fluid import core

from ..meta_parallel.sharding.sharding_utils import Type, device_guard


class InternalStorage:
    """
    This is a basic class, which is responsible for consolidating the basic storage tensor.

    """

    # Support integration parameter tensor
    def __init__(self, size, dtype, device, convert_cpu=False):
        self._params = []
        self._param_ids = []
        self._fill = 0
        self._device = device
        self._dtype = dtype

        # The actual flat tensor
        size = [size] if isinstance(size, int) else size
        if convert_cpu:
            value = (
                np.zeros(size, dtype=np.float16)
                if Type.fp16.value == dtype
                else np.zeros(size, dtype=np.float32)
            )
            self.buffer = core.VarBase(value=value, place=core.CPUPlace())
        else:
            self.buffer = paddle.zeros(size, dtype=dtype)

    def to(self, device, dtype=None, keep_alignment=True):
        """
        Move the underlying buffer
        """
        assert (
            self.buffer is not None
        ), "Cannot move a collapsed bucket, please rebuild it"
        assert (
            dtype == Type.fp32.value or Type.fp16.value
        ), "Conversion type is not supported now"

        dev_id = (
            0
            if paddle.get_device() == "cpu"
            else int(paddle.get_device().split(":")[1])
        )

        if self._device != device:
            tmp_buffer = (
                self.buffer.cuda(dev_id)
                if device == "gpu"
                else self.buffer.cpu()
            )
            for param in self._params:
                param.clear_gradient(False)
                param._gradient_set_empty(False)
            self.buffer.value().get_tensor()._clear()
            self.buffer = tmp_buffer
            self._device = device

        if dtype is not None:
            self.buffer = self.buffer.cast(dtype=dtype)
            self._dtype = dtype


class ParamStorage(InternalStorage):
    """
    This is a basic class to simplify the handling of parameter InternalStorages.
    """

    def __init__(self, size, dtype, device):
        super().__init__(size, dtype, device, convert_cpu=True)
        self.param2align = None

    def to(self, device, dtype=None, keep_alignment=True):
        """
        Move the underlying buffer
        """

        super().to(device, dtype)

        if keep_alignment:
            self._array_params()

    @framework.no_grad()
    def add_rank_params(self, trainable_params, param2align, convert_gpu=True):
        """
        Add new parameters to the InternalStorage. Params becomes a view of this InternalStorage buffer.
        """

        assert all(
            [id(param) not in self._param_ids for param in trainable_params]
        ), "The same param cannot be checked in twice"
        assert self.buffer is not None

        self.param2align = param2align

        cpu_param_shape = list()
        for param in trainable_params:
            p_shape = self._add_param_as_view(
                param, param2align[param.name], convert_gpu
            )
            cpu_param_shape.append(p_shape)

        if convert_gpu:
            # buffer convert from cpu to cuda
            dev_id = int(paddle.get_device().split(":")[1])
            self.buffer = self.buffer.cuda(dev_id)

        self._fill = 0

        for idx, param in enumerate(trainable_params):
            self._convert_buffer(
                param, cpu_param_shape[idx], param2align[param.name]
            )
            self._params.append(param)
            self._param_ids.append(id(param))

    @framework.no_grad()
    def _add_param_as_view(self, param, align, convert_gpu=True):

        assert (
            param.dtype == self.buffer.dtype
        ), "Different types for the InternalStorage and the param, cannot proceed: {} - {}".format(
            param.dtype, self.buffer.dtype
        )

        var_end = self._fill + np.prod(param.shape)
        offset = var_end + align
        assert offset <= np.prod(self.buffer.shape)

        p_shape = param.shape

        origin_state = param.stop_gradient
        param.stop_gradient = True
        param.flatten_()
        param.stop_gradient = origin_state

        # Copy the current param value
        dev_id = (
            0
            if paddle.get_device() == "cpu"
            else int(paddle.get_device().split(":")[1])
        )
        with device_guard(dev_id, "cpu"):
            tmp_var = core.VarBase(
                tensor=self.buffer._slice(self._fill, var_end)
            )
            if convert_gpu:
                param_cpu = param.cpu()
                param.value().get_tensor()._clear()
                tmp_var.set_value(param_cpu)
            else:
                tmp_var.set_value(param)

        self._fill = offset
        return p_shape

    @framework.no_grad()
    def _convert_buffer(self, param, p_shape, align):

        var_end = self._fill + np.prod(p_shape)
        offset = var_end + align
        assert offset <= np.prod(self.buffer.shape)

        # Convert the param value
        tmp_tensor = self.buffer._slice(self._fill, var_end)
        param.value().get_tensor()._share_data_with(tmp_tensor)
        param.value().get_tensor()._set_dims(p_shape)

        self._fill = offset

    @framework.no_grad()
    def _array_params(self):
        """
        Given the parameters which have been registered previously, rebuild the whole InternalStorage.
        """
        assert len(self._params) > 0
        assert self.param2align is not None

        self._fill = 0
        for p in self._params:
            self._convert_buffer(p, p.shape, self.param2align[p.name])  # modify


class GradStorage(InternalStorage):
    """
    This is a basic class to simplify the handling of gradient InternalStorages
    """

    def __init__(
        self, size, dtype, device, destination, parm2align, convert_cpu=False
    ):
        if isinstance(size, np.int64):
            size = size.tolist()
        super().__init__(size, dtype, device, convert_cpu)

        self._max_size = size
        self._release = False

        self.params_checked_in = 0
        self.destination = destination
        self._parm2align = parm2align
        self.sent = False

    def reset_checked_in(self):
        """Reset the counter of the parameter grads which have been checked in"""
        self.params_checked_in = 0
        self.sent = False

    @property
    def all_checked_in(self):
        """Judge all the expected gradient check-in happened"""
        return len(self._params) == self.params_checked_in

    def can_add_grad_view(self, param, align):
        """Is there enough InternalStorage to add this parameter gradient, and whether this param have already checked in."""
        return (
            self._fill + np.prod(param.shape) + align <= self._max_size
            and id(param) not in self._param_ids
        )

    def to(self, device, dtype=None, keep_alignment=True):
        """
        Move the underlying buffer
        """
        if self._release:
            self.rebuild()

        super().to(device, dtype)

        if keep_alignment:
            self._array_grads()

    @framework.no_grad()
    def add_grad(self, param, align):
        """
        Add a new parameter gradient to the InternalStorage. Param.grad becomes a view of this InternalStorage buffer.
        """

        assert (
            id(param) not in self._param_ids
        ), "The same gradients cannot be checked in twice"

        self._add_grad_as_view(param, align)
        self._params.append(param)
        self._param_ids.append(id(param))

    @framework.no_grad()
    def manumal_relase(self):
        """
        Release the buffer from InternalStorage. The InternalStorage will need to be rebuilt before use.
        """
        if not self._release:
            for p in self._params:
                if p.grad is not None:
                    p.clear_gradient(False)
                    p._gradient_set_empty(False)

            self.buffer = None
            self._fill = 0
            self.params_checked_in = 0
            self._release = True

    @framework.no_grad()
    def rebuild(self):
        """
        Given the parameter gradients which have been registered previously, rebuild the whole InternalStorage.
        """

        if self._release:
            self.buffer = paddle.zeros([self._max_size], dtype=self._dtype)

            for p in self._params:
                self._add_grad_as_view(p, self._parm2align[p.name])

            self._release = False

    @framework.no_grad()
    def _array_grads(self):
        """
        Given the parameters gradients which have been registered previously, rebuild the whole InternalStorage.
        """
        if len(self._params) > 0:
            self._fill = 0
            for p in self._params:
                self._add_grad_as_view(p, self._parm2align[p.name])

    @framework.no_grad()
    def _add_grad_as_view(self, param, align):
        assert (
            np.prod(self.buffer.shape) > 0
        ), "Cannot add a gradient to a released InternalStorage, please rebuild"
        assert param.dtype == self.buffer.dtype

        grad_end = self._fill + np.prod(param.shape)
        offset = grad_end + align
        assert offset <= np.prod(self.buffer.shape)

        # Copy the current grad value to InternalStorage
        dev_id = (
            0
            if paddle.get_device() == "cpu"
            else int(paddle.get_device().split(":")[1])
        )
        if self._device == "cpu":
            with device_guard(dev_id, self._device):
                tmp_var = core.VarBase(self.buffer._slice(self._fill, grad_end))
                param._copy_gradient_from(tmp_var)
                tmp_var.value().get_tensor()._clear()

        elif self._device == "gpu":
            tmp_var = core.VarBase(self.buffer._slice(self._fill, grad_end))
            param._copy_gradient_from(tmp_var)
            tmp_var.value().get_tensor()._clear()

        self._fill = offset
