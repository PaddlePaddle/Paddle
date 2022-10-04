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

import six

from collections import defaultdict
from paddle.fluid import core
from paddle.fluid import framework
from paddle import _C_ops, _legacy_C_ops

name_mapping = {
    "graph_send_recv": {
        "final_op_name": "graph_send_recv",
        "x": "X",
        "src_index": "Src_index",
        "dst_index": "Dst_index",
        "out": "Out",
        "dst_count": "Dst_count"
    },
    "matmul_v2": {
        "final_op_name": "matmul",
        "transpose_x": "trans_x",
        "transpose_y": "trans_y",
        "x": "X",
        "y": "Y",
        "out": "Out",
    },
    # "elementwise_add": {
    #     "final_op_name": "add",
    #     "x": "X",
    #     "y": "Y",
    # },
    "trunc": {
        "final_op_name": "trunc",
        "x": "X",
        "out": "Out",
    },
    # "pool2d": {
    #     "final_op_name": "pool2d",
    #     "x": "X",
    #     "kernel_size": "ksize",
    #     "out": "Out",
    # },
    "abs": {
        "final_op_name": "abs",
        "x": "X",
        "out": "Out",
    },
    "digamma": {
        "final_op_name": "digamma",
        "x": "X",
        "out": "Out",
    },
    "diagonal": {
        "final_op_name": "diagonal",
        "x": "Input",
        "offset": "offset",
        "axis1": "axis1",
        "axis2": "axis2",
        "out": "Out",
    },
    "roi_align": {
        "final_op_name": "roi_align",
        "x": "X",
        "boxes": "ROIs",
        "boxes_num": "RoisNum",
        "pooled_height": "pooled_height",
        "pooled_width": "pooled_width",
        "spatial_scale": "spatial_scale",
        "sampling_ratio": "sampling_ratio",
        "aligned": "aligned",
    },
    # "one_hot": {
    #     "final_op_name": "one_hot",
    #     "x": "X",
    #     "num_class": "depth",
    #     "out": "Out",
    # }
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

    def eager_legacy_trace_op(self,
                              op_type,
                              inputs,
                              outputs,
                              attrs,
                              stop_gradient=False,
                              inplace_map=None):
        function_ptr = _legacy_C_ops.__dict__[op_type]

        core_ops_args_info = _legacy_C_ops.get_core_ops_args_info()
        core_ops_args_type_info = _legacy_C_ops.get_core_ops_args_type_info()
        core_ops_returns_info = _legacy_C_ops.get_core_ops_returns_info()

        op_args = core_ops_args_info[op_type]
        op_args_type = core_ops_args_type_info[op_type]
        op_returns = core_ops_returns_info[op_type]

        arg_list = []
        for i in range(len(op_args)):
            # initialized with None
            arg_to_append = None

            arg_name = op_args[i]
            arg_type = op_args_type[i]
            if arg_name in inputs.keys():
                arg_to_append = inputs[arg_name]
            elif arg_name in outputs.keys():
                arg_to_append = outputs[arg_name]
            else:
                if "Num" in arg_name[-3:]:
                    # Remove "Num" suffix to get out_name
                    out_name = arg_name[:-3]
                    assert out_name in outputs.keys()
                    num_outs = len(outputs[out_name])
                    arg_to_append = num_outs
                # NOTE(dev): For MasterParam/MasterParamOut in optimzer op
                elif "Var" in arg_name[-3:]:
                    out_name = arg_name[:-3]
                    print(out_name)
                    if out_name in outputs.keys():
                        arg_to_append = outputs[out_name]
                    elif out_name in inputs.keys():
                        arg_to_append = inputs[out_name]

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

        if op_type == 'load_combine':
            assert len(outputs.keys()) == 1
            key = list(outputs.keys())[0]
            for j in range(len(returns)):
                returns[j]._share_underline_tensor_to(outputs[key][j])
            return

        if isinstance(returns, tuple):
            for i in range(len(op_returns)):
                retname = op_returns[i]
                if retname in outputs.keys():
                    # Replaced outputs by function returns
                    if isinstance(returns[i], list):
                        for j in range(len(returns[i])):
                            outputs[retname][j].reconstruct_from_(
                                returns[i][j], False)
                    else:
                        if isinstance(outputs[retname], list):
                            outputs[retname][0].reconstruct_from_(
                                returns[i], False)
                        else:
                            outputs[retname].reconstruct_from_(
                                returns[i], False)
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

    def eager_trace_op(self,
                       op_type,
                       inputs,
                       outputs,
                       attrs,
                       stop_gradient=False,
                       inplace_map=None):
        assert op_type in name_mapping.keys()

        op_type = name_mapping[op_type]["final_op_name"]
        function_ptr = _C_ops.__dict__[op_type]

        core_ops_args_info = _C_ops.get_core_ops_args_info()
        core_ops_args_type_info = _C_ops.get_core_ops_args_type_info()
        core_ops_returns_info = _C_ops.get_core_ops_returns_info()

        op_args = core_ops_args_info[op_type]
        op_args_type = core_ops_args_type_info[op_type]
        op_returns = core_ops_returns_info[op_type]

        arg_list = []
        for i in range(len(op_args)):
            eager_arg_name = op_args[i]
            arg_type = op_args_type[i]

            assert eager_arg_name in name_mapping[op_type].keys()
            arg_name = name_mapping[op_type][eager_arg_name]

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

                assert eager_retname in name_mapping[op_type].keys()
                retname = name_mapping[op_type][eager_retname]
                if retname in outputs.keys():
                    # Replaced outputs by function returns
                    if isinstance(returns[i], list):
                        for j in range(len(returns[i])):
                            outputs[retname][j].reconstruct_from_(
                                returns[i][j], False)
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
        if not framework._in_legacy_dygraph():
            # inputs : {"sum": [tensor], ...}
            # outputs : {"sum": [tensor], ...}
            if type in name_mapping.keys():
                type = name_mapping[type]["final_op_name"]

                assert type in _legacy_C_ops.__dict__
                self.eager_trace_op(type, inputs, outputs, attrs, stop_gradient,
                                    inplace_map)
            else:
                self.eager_legacy_trace_op(type, inputs, outputs, attrs,
                                           stop_gradient, inplace_map)
        else:
            self.trace(type, inputs, outputs, attrs,
                       framework._current_expected_place(), self._has_grad
                       and not stop_gradient,
                       inplace_map if inplace_map else {})

    def train_mode(self):
        self._train_mode = True

    def eval_mode(self):
        self._train_mode = False
