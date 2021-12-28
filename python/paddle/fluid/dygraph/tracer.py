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
from paddle import _C_ops


class Tracer(core.Tracer):
    """
    :api_attr: imperative
    
    Tracer is used to execute and record the operators executed, to construct the 
    computation graph in dygraph model. Tracer has two mode, :code:`train_mode`
    and :code:`eval_mode`. In :code:`train_mode`, Tracer would add backward network 
    automatically and perform AutoGrad by method :code:`loss.backward()`. 
    In :code:`eval_mode`, Tracer would not add backward network.

    This is a low level API, users don't need to use it directly.
    """

    def __init__(self):
        super(Tracer, self).__init__()

        self._train_mode = True

    def trace_op(self,
                 type,
                 inputs,
                 outputs,
                 attrs,
                 stop_gradient=False,
                 inplace_map=None):
        if framework._in_eager_mode():
            function_ptr = _C_ops.__dict__[type]

            core_ops_args_info = _C_ops.get_core_ops_args_info()
            core_ops_returns_info = _C_ops.get_core_ops_returns_info()

            op_args = core_ops_args_info[type]
            op_returns = core_ops_returns_info[type]

            arg_list = []
            for arg in op_args:
                if arg in inputs.keys():
                    arg_list.append(inputs[arg])
                elif arg in outputs.keys():
                    arg_list.append(outputs[arg])
                else:
                    if "Num" in arg:
                        # Remove "Num" suffix to get out_name
                        out_name = arg[:-3]
                        assert out_name in outputs.keys()
                        num_outs = len(outputs[out_name])
                        arg_list.append(num_outs)
                    else:
                        arg_list.append(None)
            returns = function_ptr(*arg_list, **attrs)

            for i in range(len(op_returns)):
                retname = op_returns[i]
                if retname in outputs.keys():
                    # Replaced outputs by function returns
                    outputs[retname] = returns[i]
        else:
            self.trace(type, inputs, outputs, attrs,
                       framework._current_expected_place(), self._has_grad and
                       not stop_gradient, inplace_map if inplace_map else {})

    def train_mode(self):
        self._train_mode = True

    def eval_mode(self):
        self._train_mode = False
