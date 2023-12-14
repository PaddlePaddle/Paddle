# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os

from api_gen import NAMESPACE_TEMPLATE, CodeGen

CPP_FILE_TEMPLATE = """
#include <pybind11/pybind11.h>

#include "paddle/fluid/pybind/static_op_function.h"
#include "paddle/fluid/pybind/eager_op_function.h"
#include "paddle/fluid/pybind/manual_static_op_function.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"

{body}

"""

NAMESPACE_INNER_TEMPLATE = """
{function_impl}

static PyMethodDef OpsAPI[] = {{
{ops_api}
{{nullptr, nullptr, 0, nullptr}}
}};

void BindOpsAPI(pybind11::module *module) {{
  if (PyModule_AddFunctions(module->ptr(), OpsAPI) < 0) {{
    PADDLE_THROW(phi::errors::Fatal("Add C++ api to core.ops failed!"));
  }}
  if (PyModule_AddFunctions(module->ptr(), ManualOpsAPI) < 0) {{
    PADDLE_THROW(phi::errors::Fatal("Add C++ api to core.ops failed!"));
  }}
}}
"""

FUNCTION_IMPL_TEMPLATE = """
static PyObject *{name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {{
    VLOG(6) << "Call static_api_{name}";
    return static_api_{name}(self, args, kwargs);
  }} else {{
    VLOG(6) << "Call eager_api_{name}";
    return eager_api_{name}(self, args, kwargs);
  }}
}}"""

STATIC_ONLY_FUNCTION_IMPL_TEMPLATE = """
static PyObject *{name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
  VLOG(6) << "Call static_api_{name}";
  return static_api_{name}(self, args, kwargs);
}}"""

OPS_API_TEMPLATE = """
{{"{name}", (PyCFunction)(void (*)(void)){name}, METH_VARARGS | METH_KEYWORDS, "C++ interface function for {name}."}},"""

NEED_GEN_STATIC_ONLY_APIS = [
    'fetch',
    'fused_bias_dropout_residual_layer_norm',
    'fused_embedding_eltwise_layernorm',
    'fused_fc_elementwise_layernorm',
    'fused_multi_transformer_xpu',
    'fused_scale_bias_relu_conv_bn',
    'fused_scale_bias_add_relu',
    'fused_dconv_drelu_dbn',
    'fused_dot_product_attention',
    'fusion_transpose_flatten_concat',
    'skip_layernorm',
    'generate_sequence_xpu',
    'layer_norm_act_xpu',
    'memcpy',
    'batch_norm_',
    'multi_encoder_xpu',
    'multihead_matmul',
    'squeeze_excitation_block',
    'yolo_box_xpu',
    'fusion_gru',
    'fusion_seqconv_eltadd_relu',
    'fusion_seqexpand_concat_fc',
    'fused_conv2d_add_act',
    'fusion_repeated_fc_relu',
    'fusion_squared_mat_sub',
    'fused_attention',
    'fused_feedforward',
    'fc',
    'self_dp_attention',
    'get_tensor_from_selected_rows',
    'print',
    'sequence_mask',
]

NO_NEED_GEN_STATIC_ONLY_APIS = [
    'add_n_',
    'add_n_with_kernel',
    'assign_value',
    'c_allgather',
    'c_allreduce_max',
    'c_allreduce_sum',
    'c_embedding',
    'c_identity',
    'c_reduce_sum',
    'c_reducescatter',
    'decayed_adagrad',
    'dpsgd',
    'embedding_grad_sparse',
    'fused_batch_norm_act_',
    'fused_bn_add_activation_',
    'fused_elemwise_add_activation',
    'fused_scale_bias_relu_conv_bn',
    'fused_scale_bias_add_relu',
    'fused_dconv_drelu_dbn',
    'fused_dot_product_attention',
    'recv_v2',
    'rnn_',
    'row_conv',
    'seed',
    'send_v2',
    'shadow_feed',
    'sparse_momentum',
    'soft_relu',
    'uniform_random_batch_size_like',
]


class OpsAPIGen(CodeGen):
    def __init__(self) -> None:
        super().__init__()

    def _need_skip(self, op_info, op_name):
        return (
            super()._need_skip(op_info, op_name)
            or op_name.endswith(('_grad', '_grad_', 'xpu'))
            or op_name in NO_NEED_GEN_STATIC_ONLY_APIS
        )

    def _gen_one_function_impl(self, name):
        if name in NEED_GEN_STATIC_ONLY_APIS:
            return STATIC_ONLY_FUNCTION_IMPL_TEMPLATE.format(name=name)
        else:
            return FUNCTION_IMPL_TEMPLATE.format(name=name)

    def _gen_one_ops_api(self, name):
        return OPS_API_TEMPLATE.format(name=name)

    def gen_cpp_file(
        self, op_yaml_files, op_compat_yaml_file, namespaces, cpp_file_path
    ):
        if os.path.exists(cpp_file_path):
            os.remove(cpp_file_path)
        op_info_items = self._parse_yaml(op_yaml_files, op_compat_yaml_file)
        function_impl_str = ''
        ops_api_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                if self._need_skip(op_info, op_name):
                    continue
                function_impl_str += self._gen_one_function_impl(op_name)
                ops_api_str += self._gen_one_ops_api(op_name)

        inner_body = NAMESPACE_INNER_TEMPLATE.format(
            function_impl=function_impl_str, ops_api=ops_api_str
        )

        body = inner_body
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(cpp_file_path, 'w') as f:
            f.write(CPP_FILE_TEMPLATE.format(body=body))


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate Dialect Python C Files By Yaml'
    )
    parser.add_argument('--op_yaml_files', type=str)
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--namespaces', type=str)
    parser.add_argument('--ops_api_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArguments()
    op_yaml_files = args.op_yaml_files.split(",")
    op_compat_yaml_file = args.op_compat_yaml_file
    if args.namespaces is not None:
        namespaces = args.namespaces.split(",")
    ops_api_file = args.ops_api_file

    code_gen = OpsAPIGen()
    code_gen.gen_cpp_file(
        op_yaml_files, op_compat_yaml_file, namespaces, ops_api_file
    )
