# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import math
import warnings
import numpy as np

import paddle
from .. import unique_name
from ..framework import Variable
from ..data_feeder import check_type

__all__ = []


class LearningRateDecay:
    """
    Base class of learning rate decay

    Define the common interface of an LearningRateDecay.
    User should not use this class directly,
    but need to use one of it's implementation.
    """

    def __init__(self, begin=0, step=1, dtype='float32'):
        self.step_num = begin
        self.step_size = step
        self.dtype = dtype

    def __call__(self):
        lr = self.step()
        if isinstance(lr, float):
            lr = self.create_lr_var(lr)
        self.step_num += self.step_size
        return lr

    def create_lr_var(self, lr):
        """
        convert lr from float to variable

        Args:
            lr: learning rate
        Returns:
            learning rate variable
        """
        from .. import layers

        lr = paddle.static.create_global_var(
            name=unique_name.generate("learning_rate"),
            shape=[1],
            value=float(lr),
            dtype=self.dtype,
            persistable=False,
        )
        return lr

    # Note: If you want to change what optimizer.state_dict stores, just overwrite this functions,
    # "self.step_num" will be stored by default.
    def state_dict(self):
        """
        Returns the state of the scheduler as a :class:`dict`.

        It is a subset of self.__dict__ .
        """
        self._state_keys()
        state_dict = {}
        for key in self.keys:
            if key not in self.__dict__:
                continue
            value = self.__dict__[key]
            if isinstance(value, Variable):
                assert (
                    value.size == 1
                ), "the size of Variable in state_dict must be 1, but its size is {} with shape {}".format(
                    value.size, value.shape
                )
                value = value.item()
            state_dict[key] = value

        return state_dict

    def _state_keys(self):
        """
        set the keys in self.__dict__ that are needed to be saved.
        """
        self.keys = ['step_num']

    def set_state_dict(self, state_dict):
        """
        Loads the schedulers state.
        """
        self._state_keys()
        for key in self.keys:
            if key in state_dict:
                self.__dict__[key] = state_dict[key]
            else:
                raise RuntimeError(
                    "Please check whether state_dict is correct for optimizer. Can't find [ {} ] in state_dict".format(
                        key
                    )
                )
        if len(state_dict) > len(self.keys):
            warnings.warn(
                "There are some unused values in state_dict. Maybe the optimizer have different 'LearningRateDecay' when invoking state_dict and set_dict"
            )

    # [aliases] Compatible with old method names
    set_dict = set_state_dict

    def step(self):
        raise NotImplementedError()


class _LearningRateEpochDecay(LearningRateDecay):
    """
    :api_attr: imperative

    Base class of learning rate decay, which is updated each epoch.

    Define the common interface of an _LearningRateEpochDecay.
    User should not use this class directly,
    but need to use one of it's implementation. And invoke method: `epoch()` each epoch.
    """

    def __init__(self, learning_rate, dtype=None):
        if not isinstance(learning_rate, (float, int)):
            raise TypeError(
                "The type of 'learning_rate' must be 'float, int', but received %s."
                % type(learning_rate)
            )
        if learning_rate < 0:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))

        self.base_lr = float(learning_rate)

        self.epoch_num = -1
        self.dtype = dtype
        if dtype is None:
            self.dtype = "float32"
        self.learning_rate = self.create_lr_var(self.base_lr)

        self.epoch()

    # For those subclass who overload _LearningRateEpochDecay, "self.epoch_num/learning_rate" will be stored by default.
    # you can change it for your subclass.
    def _state_keys(self):
        self.keys = ['epoch_num', 'learning_rate']

    def __call__(self):
        """
        Return last computed learning rate on current epoch.
        """
        if not isinstance(self.learning_rate, Variable):
            self.learning_rate = self.create_lr_var(self.learning_rate)
        return self.learning_rate

    def epoch(self, epoch=None):
        """
        compueted learning_rate and update it when invoked.
        """
        if epoch is None:
            self.epoch_num += 1
        else:
            self.epoch_num = epoch

        self.learning_rate = self.get_lr()

    def get_lr(self):
        raise NotImplementedError
