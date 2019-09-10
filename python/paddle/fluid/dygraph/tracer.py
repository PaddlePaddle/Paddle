# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import six

from collections import defaultdict
from paddle.fluid import core
from paddle.fluid import framework

__all__ = ['Tracer']


class Tracer(core.Tracer):
    """
    Python wrapper of dygraph tracer
    """

    def __init__(self):
        super(Tracer, self).__init__()

        self._vars = defaultdict()
        self._train_mode = True

    def trace_var(self, name, var):
        self._vars[name] = var

    def all_parameters(self):
        return list((item for name, item in six.iteritems(self._vars)
                     if isinstance(item, framework.Parameter)))

    def trace_op(self, type, inputs, outputs, attrs, stop_gradient=False):
        self.trace(type, inputs, outputs, attrs,
                   framework._current_expected_place(), self._train_mode and
                   not stop_gradient)

    def train_mode(self):
        self._train_mode = True

    def eval_mode(self):
        self._train_mode = False
