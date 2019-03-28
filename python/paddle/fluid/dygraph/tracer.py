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


def release_op(op):
    del framework._dygraph_tracer()._ops[op._trace_id]


class Tracer(core.Tracer):
    """
    Python wrapper of dygraph tracer
    """

    def __init__(self, block):
        super(Tracer, self).__init__(block)

        self._ops = defaultdict()
        self._vars = defaultdict()
        self._trace_id = 0

    def trace_var(self, name, var):
        self._vars[name] = var

    def all_parameters(self):
        return list((item for name, item in six.iteritems(self._vars)
                     if isinstance(item, framework.Parameter)))

    def trace_op(self, op, stop_gradient=False):
        # record op's trace id
        op.iop._trace_id = self._trace_id

        backward_refs = self.trace(op.iop, op.inputs, op.outputs, op.attrs,
                                   framework._current_expected_place(),
                                   stop_gradient)

        if not stop_gradient:
            self._trace_id += 1
            self._ops[op.iop._trace_id] = op

            # register backward hooks and variables if needed
            if len(backward_refs) > 0:
                op.iop.register_backward_hooks(release_op)

                # TODO(minqiyang): remove all inputs and outputs after separate
                # var and grad
                op.backward_refs = defaultdict(list)
                for k, v in six.iteritems(op.inputs):
                    if k in backward_refs:
                        op.backward_refs[k] = op.inputs[k]

                for k, v in six.iteritems(op.outputs):
                    if k in backward_refs:
                        op.backward_refs[k] = op.outputs[k]
