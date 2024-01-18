# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.nn import Layer

from .base_quanter import BaseQuanter


class ObserveWrapper(Layer):
    r"""
    Put an observer layer and an observed layer into a wrapping layer.
    It is used to insert layers into the model for QAT or PTQ.
    Args:
        observer(BaseQuanter): Observer layer
        observed(Layer): Observed layer
        observe_input(bool): If it is true the observer layer will be called before observed layer.
            If it is false the observed layer will be called before observer layer. Default: True.
    """

    def __init__(
        self,
        observer: BaseQuanter,
        observed: Layer,
        observe_input=True,
    ):
        super().__init__()
        self._observer = observer
        self._observed = observed
        self._observe_input = observe_input

    def forward(self, *inputs, **kwargs):
        if self._observe_input:
            out = self._observer(*inputs, **kwargs)
            return self._observed(out, **kwargs)
        else:
            out = self._observed(*inputs, **kwargs)
            return self._observer(out, **kwargs)
