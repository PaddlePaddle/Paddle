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

final_state_name_mapping = {
    "matmul_v2": {
        "final_op_name": "final_state_matmul",
        "transpose_x": "trans_x",
        "transpose_y": "trans_y",
        "x": "X",
        "y": "Y",
        "out": "Out",
    }
}


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

    def eager_trace_op(self,
                       type,
                       inputs,
                       outputs,
                       attrs,
                       stop_gradient=False,
                       inplace_map=None):
        function_ptr = _C_ops.__dict__[type]

        core_ops_args_info = _C_ops.get_core_ops_args_info()
        core_ops_args_type_info = _C_ops.get_core_ops_args_type_info()
        core_ops_returns_info = _C_ops.get_core_ops_returns_info()

        op_args = core_ops_args_info[type]
        op_args_type = core_ops_args_type_info[type]
        op_returns = core_ops_returns_info[type]

        arg_list = []
        for i in range(len(op_args)):
            arg_name = op_args[i]
            arg_type = op_args_type[i]
            if arg_name in inputs.keys():
                arg_to_append = inputs[arg_name]
            elif arg_name in outputs.keys():
                arg_to_append = outputs[arg_name]
            else:
                if "Num" in arg_name:
                    # Remove "Num" suffix to get out_name
                    out_name = arg_name[:-3]
                    assert out_name in outputs.keys()
                    num_outs = len(outputs[out_name])
                    arg_to_append = num_outs
                else:
                    arg_to_append = None

            if arg_to_append is None:
                arg_list.append(arg_to_append)
            elif arg_type == "tensor":
                if isinstance(arg_to_append, list):
                    arg_list.append(arg_to_append[0])
                else:
                    arg_list.append(arg_to_append)
            elif arg_type == "list":
                assert isinstance(arg_to_append, list)
                arg_list.append(arg_to_append)
            else:
                assert arg_type == "int"
                assert isinstance(arg_to_append, int)
                arg_list.append(arg_to_append)

        attrs_list = []
        for k, v in attrs.items():
            attrs_list.append(k)
            attrs_list.append(v)
        returns = function_ptr(*arg_list, *attrs_list)

        if isinstance(returns, tuple):
            for i in range(len(op_returns)):
                retname = op_returns[i]
                if retname in outputs.keys():
                    # Replaced outputs by function returns
                    if isinstance(returns[i], list):
                        for j in range(len(returns[i])):
                            outputs[retname][j].reconstruct_from_(returns[i][j],
                                                                  False)
                    else:
                        outputs[retname][0].reconstruct_from_(returns[i], False)
        elif isinstance(returns, list):
            assert len(outputs.keys()) == 1
            key = list(outputs.keys())[0]
            for j in range(len(returns)):
                outputs[key][j].reconstruct_from_(returns[j], False)
        else:
            assert len(outputs.keys()) == 1
            key = list(outputs.keys())[0]
            if isinstance(outputs[key], list):
                outputs[key][0].reconstruct_from_(returns, False)
            else:
                outputs[key].reconstruct_from_(returns, False)

    def eager_final_state_trace_op(self,
                                   type,
                                   inputs,
                                   outputs,
                                   attrs,
                                   stop_gradient=False,
                                   inplace_map=None):
        assert type in final_state_name_mapping.keys()

        final_state_type = final_state_name_mapping[type]["final_op_name"]
        function_ptr = _C_ops.__dict__[final_state_type]

        core_ops_args_info = _C_ops.get_final_state_core_ops_args_info()
        core_ops_args_type_info = _C_ops.get_final_state_core_ops_args_type_info(
        )
        core_ops_returns_info = _C_ops.get_final_state_core_ops_returns_info()

        op_args = core_ops_args_info[final_state_type]
        op_args_type = core_ops_args_type_info[final_state_type]
        op_returns = core_ops_returns_info[final_state_type]

        arg_list = []
        for i in range(len(op_args)):
            eager_arg_name = op_args[i]
            arg_type = op_args_type[i]

            assert eager_arg_name in final_state_name_mapping[type].keys()
            arg_name = final_state_name_mapping[type][eager_arg_name]

            if arg_name in inputs.keys():
                arg_to_append = inputs[arg_name]
            elif arg_name in outputs.keys():
                arg_to_append = outputs[arg_name]
            elif arg_name in attrs.keys() and arg_type == "":
                arg_to_append = attrs[arg_name]
            else:
                # dispensable
                arg_to_append = None

            if arg_type == "":
                # attribute
                arg_list.append(arg_to_append)
            elif arg_type == "tensor":
                if isinstance(arg_to_append, list):
                    arg_list.append(arg_to_append[0])
                else:
                    arg_list.append(arg_to_append)
            elif arg_type == "list":
                assert isinstance(arg_to_append, list)
                arg_list.append(arg_to_append)
            else:
                assert arg_to_append is None
                arg_list.append(arg_to_append)

        returns = function_ptr(*arg_list)

        if isinstance(returns, tuple):
            for i in range(len(op_returns)):
                eager_retname = op_returns[i]

                assert eager_retname in final_state_name_mapping[type].keys()
                retname = final_state_name_mapping[type][eager_retname]
                if retname in outputs.keys():
                    # Replaced outputs by function returns
                    if isinstance(returns[i], list):
                        for j in range(len(returns[i])):
                            outputs[retname][j].reconstruct_from_(returns[i][j],
                                                                  False)
                    else:
                        outputs[retname][0].reconstruct_from_(returns[i], False)
        elif isinstance(returns, list):
            assert len(outputs.keys()) == 1
            key = list(outputs.keys())[0]
            for j in range(len(returns)):
                outputs[key][j].reconstruct_from_(returns[j], False)
        else:
            assert len(outputs.keys()) == 1
            key = list(outputs.keys())[0]
            if isinstance(outputs[key], list):
                outputs[key][0].reconstruct_from_(returns, False)
            else:
                outputs[key].reconstruct_from_(returns, False)

    def trace_op(self,
                 type,
                 inputs,
                 outputs,
                 attrs,
                 stop_gradient=False,
                 inplace_map=None):
        if framework._in_eager_mode():
            # inputs : {"sum": [tensor], ...}
            # outputs : {"sum": [tensor], ...}

            if type in final_state_name_mapping.keys():
                final_state_type = final_state_name_mapping[type][
                    "final_op_name"]

                assert final_state_type in _C_ops.__dict__
                self.eager_final_state_trace_op(type, inputs, outputs, attrs,
                                                stop_gradient, inplace_map)
            else:
                self.eager_trace_op(type, inputs, outputs, attrs, stop_gradient,
                                    inplace_map)
        else:
            self.trace(type, inputs, outputs, attrs,
                       framework._current_expected_place(), self._has_grad and
                       not stop_gradient, inplace_map if inplace_map else {})

    def train_mode(self):
        self._train_mode = True

    def eval_mode(self):
        self._train_mode = False
