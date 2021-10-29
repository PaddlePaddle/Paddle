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

import paddle
from paddle.fluid import core


class InternalStorage:
    """
    This is a basic class, which is responsible for consolidating the basic storage tensor.
    """

    # Support integration parameter tensor
    def __init__(self, size, dtype, device):
        self._params = []
        self._param_ids = []
        self._fill = 0
        self._device = device

        # The actual flat tensor
        size = [size] if isinstance(size, int) else size
        self.buffer = paddle.zeros(size, dtype=dtype)

    def to(self, device, local_rank, dtype=None, keep_alignment=True):
        """
        Move the underlying buffer
        """
        assert self.buffer is not None, "Cannot move a collapsed bucket, please rebuild it"

        if dtype and self.buffer.dtype != dtype:
            self.buffer = paddle.cast(self.buffer, dtype)

        self.buffer = self.buffer.cuda(
            local_rank) if device == "gpu" else self.buffer.cpu()


class ParamStorage(InternalStorage):
    """
    This is a basic class to simplify the handling of parameter InternalStorages.
    """

    def __init__(self, size, dtype, device):
        super().__init__(size, dtype, device)

    def to(self, device, local_rank, dtype=None, keep_alignment=True):
        """
        Move the underlying buffer
        """
        super().to(device, local_rank, dtype)

        if keep_alignment:
            self._array_params()

    @paddle.no_grad()
    def add_param(self, param):
        """
        Add a new parameter to the InternalStorage. Params becomes a view of this InternalStorage buffer.
        """

        assert id(
            param
        ) not in self._param_ids, "The same param cannot be checked in twice"

        self._add_param_as_view(param)
        self._params.append(param)
        self._param_ids.append(id(param))

    @paddle.no_grad()
    def _add_param_as_view(self, param):
        assert self.buffer is not None
        assert (
            param.dtype == self.buffer.dtype
        ), "Different types for the InternalStorage and the param, cannot proceed: {} - {}".format(
            param.dtype, self.buffer.dtype)

        fill_next = self._fill + param.numel()
        assert fill_next <= self.buffer.numel()

        # Copy the current param value
        tmp_var = core.VarBase(self.buffer.value().get_tensor()._slice(
            self._fill, fill_next))
        param = tmp_var.reshape_(param.shape)
        tmp_var.value().get_tensor()._clear()
        self._fill = fill_next

    @paddle.no_grad()
    def _array_params(self):
        """
        Given the parameters which have been registered previously, rebuild the whole InternalStorage.
        """
        assert len(self._params) > 0

        self._fill = 0
        for p in self._params:
            if p.dtype != self.buffer.dtype:
                p = paddle.cast(p, self.buffer.dtype)
            self._add_param_as_view(p)


class GradStorage(InternalStorage):
    """
    This is a basic class to simplify the handling of gradient InternalStorages
    """

    def __init__(self, size, dtype, device, destination):
        super().__init__(size, dtype, device)

        self._max_size = size
        self._release = False

        self.params_checked_in = 0
        self.destination = destination
        self.sent = False
        self.callback = None

    def reset_checked_in(self):
        """ Reset the counter of the parameter grads which have been checked in
        """
        self.params_checked_in = 0
        self.sent = False

    @property
    def all_checked_in(self):
        """ Judge all the expected gradient check-in happened """
        return len(self._params) == self.params_checked_in

    def can_add_grad_view(self, param):
        """ Is there enough InternalStorage to add this parameter gradient, and whether this param have already checked in.
        """
        return self._fill + param.numel() < self._max_size and id(
            param) not in self._param_ids

    def to(self, device, local_rank, dtype=None, keep_alignment=True):
        """
        Move the underlying buffer
        """
        if self._release:
            self.rebuild()

        super().to(device, local_rank, dtype)

        if keep_alignment:
            self._array_grads()

    @paddle.no_grad()
    def add_grad(self, param):
        """
        Add a new parameter gradient to the InternalStorage. Param.grad becomes a view of this InternalStorage buffer.
        """

        assert id(
            param
        ) not in self._param_ids, "The same gradients cannot be checked in twice"

        self._add_grad_as_view(param)
        self._params.append(param)
        self._param_ids.append(id(param))

    @paddle.no_grad()
    def manumal_relase(self):
        """
        Release the buffer from InternalStorage. The InternalStorage will need to be rebuilt before use.
        """
        if not self._release:
            for p in self._params:
                if p.grad is not None:
                    p.grad.detach()
                    p._clear_gradient()

            self.buffer = None
            self._fill = 0
            self.params_checked_in = 0
            self._release = True

    @paddle.no_grad()
    def rebuild(self):
        """
        Given the parameter gradients which have been registered previously, rebuild the whole InternalStorage.
        """
        assert len(self._params) > 0

        if self._release:
            self.buffer = paddle.zeros(
                self._max_size, dtype=self._params[0].dtype)

            for p in self._params:
                self._add_grad_as_view(p)

            self._release = False

    @paddle.no_grad()
    def rearrange(self):
        """
        Rearrange the InternalStorage to the size of the parameter gradients currently checked in, release the extra memory.
        """
        assert self.buffer.numel(
        ) > 0, "Cannot rearrange a released InternalStorage, please rebuild"

        tmp_var = core.VarBase(self.buffer.value().get_tensor()._slice(
            0, self._fill))
        self.buffer = tmp_var.clone()
        tmp_var.value().get_tensor()._clear()
        self._fill = 0
        for p in self._params:
            self._add_grad_as_view(p)

        self._max_size = self._fill

    @paddle.no_grad()
    def _array_grads(self):
        """
        Given the parameters gradients which have been registered previously, rebuild the whole InternalStorage.
        """
        assert len(self._params) > 0

        self._fill = 0
        for p in self._params:
            self._add_grad_as_view(p)

    @paddle.no_grad()
    def _add_grad_as_view(self, param):
        assert self.buffer.numel(
        ) > 0, "Cannot add a gradient to a released InternalStorage, please rebuild"
        assert param.dtype == self.buffer.dtype

        fill_next = self._fill + param.numel()
        assert fill_next <= self.buffer.numel()

        # Copy the current grad value to InternalStorage
        tmp_var = core.VarBase(self.buffer.value().get_tensor()._slice(
            self._fill, fill_next)).reshape_(param.shape)
        param._copy_from_gradient(tmp_var)
        tmp_var.value().get_tensor()._clear()
        self._fill = fill_next
