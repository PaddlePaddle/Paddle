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

__all__ = ["ObserveWrapper"]


class ObserveWrapper(Layer):

    def __init__(self, observer, observed, observe_input=True):
        super(ObserveWrapper, self).__init__()
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
