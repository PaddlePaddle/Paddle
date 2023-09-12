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

import yaml
from backward_api_gen import BackwardAPI
from dist_api_gen import DistForwardAPI

######################
# Code Gen Templates #
######################

MAIN_DIST_BRANCH_TEMPLATE = """
  // Auto Parallel condition
  if ({}) {{
    // 1. Create API Output & Prepare Dist and Dense Output{}
    // 2. Infer DistTensor's Global Shape{}
    // 3. Select Kernel{}
    // 4. PrepareData (DataTransform & Prepare Dense Input){}
    // 5. Infer Local DenseTensor Meta{}
    // 6. DenseTensor Kernel Call{}
    // 7. Return
    {}
  }}
"""

# 1. Create API Outputs
SINGLE_OUT_CREATION_TEMPLATE = """
    auto dist_out = SetKernelDistOutput({});
    auto dense_out = dist_out->unsafe_mutable_value();
"""
VECTOR_OUT_CREATION_TEMPLATE = """
    auto dist_out = SetKernelDistOutput({name});
    std::vector<phi::DenseTensor*> dense_out(dist_out.size());
    for (size_t i=0; i<dist_out.size(); i++) {{
        dense_out[i] = const_cast<phi::DenseTensor*>(&dist_out[i]->value());
    }}
"""
INPLACE_OUT_CREATION_TEMPLATE = """
    *{} = {};
"""
MULTI_SINGLE_OUT_CREATION_TEMPLATE = """
    auto dist_out_{} = SetKernelDistOutput({});
    auto dense_out_{} = dist_out_{}->unsafe_mutable_value();
"""

# 2. Infer Global Shape
SINGLE_DIST_META_IN_TEMPLATE = """MakeDistMetaTensor(*{}.impl()), """
SINGLE_DIST_META_OUT_DECL_TEMPLATE = """
    phi::distributed::DistMetaTensor meta_{}({});"""
INFER_GLOBAL_SHAPE_TEMPLATE = """
    phi::{}({}{});
"""

# 4. PrepareData (DataTransform & Prepare Dist and Dense Input)
SINGLE_PREPARE_DATA_TEMPLATE = """
    auto dist_input_{arg} = PrepareDataForDistTensor({arg}, GetKernelInputArgDef(kernel.InputAt({idx}), kernel_backend), {flag}, kernel_result.is_stride_kernel);
    auto input_{arg} = &dist_input_{}->value();
"""


class DistBackwardAPI(DistForwardAPI, BackwardAPI):
    def __init__(self, backward_item_yaml):
        BackwardAPI.__init__(self, backward_item_yaml)
        self.init_dist_api_members()

    # override DistForwardAPI's method
    def generate_output_creation_code(self) -> str:
        # backward api only need to generate kernel outputs
        output_num = len(self.outputs['types'])
        output_creation_code = ""
        if output_num == 1:
            self.dist_output_args.append('dist_out')
            self.dense_output_args.append('dense_out')
            if self.outputs['types'][0] == 'Tensor':
                output_creation_code += SINGLE_OUT_CREATION_TEMPLATE.format(
                    self.outputs['names'][0]
                )
            elif self.outputs['types'][0] == 'std::vector<Tensor>':
                output_creation_code += VECTOR_OUT_CREATION_TEMPLATE.format(
                    name=self.outputs['names'][0]
                )
            else:
                self.vector_output_size_assertion_check()
        elif output_num > 1:
            for i, out_type in enumerate(self.outputs['types']):
                self.dist_output_args.append(f'dist_out_{i}')
                self.dense_output_args.append(f'dense_out_{i}')
                if out_type == 'Tensor':
                    output_creation_code += (
                        MULTI_SINGLE_OUT_CREATION_TEMPLATE.format(
                            i, self.outputs['names'][i], i, i
                        )
                    )
                else:
                    self.vector_output_size_assertion_check()
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api
                )
            )

        return output_creation_code

    # override DistForwardAPI's method
    def generate_return_code(self) -> str:
        return "return;"

    # override BaseAPI's method
    def get_api_func_name(self):
        return self.api

    # override BaseAPI's method
    # The method lookup order are: (DistBackwardAPI.__mro__)
    # <class '__main__.DistBackwardAPI'>,
    # <class 'dist_api_gen.DistForwardAPI'>,
    # <class 'api_gen.ForwardAPI'>,
    # <class 'backward_api_gen.BackwardAPI'>,
    # <class 'api_base.BaseAPI'>,
    # <class 'object'>
    # if don't override it, the ForwardAPI's gene_output wiil be called
    def gene_output(
        self,
        out_dtype_list,
        out_tensor_type_list=None,
        code_indent='',
        inplace_flag=False,
    ):
        return BackwardAPI.gene_output(
            self,
            out_dtype_list,
            out_tensor_type_list,
            code_indent,
            inplace_flag,
        )

    # override BaseAPI's method
    def get_return_type(self, inplace_flag=False):
        return BackwardAPI.get_return_type(self)

    # override BaseAPI's method
    def gene_return_code(self):
        return ""

    # override BaseAPI's method
    def gene_api_declaration(self) -> str:
        return BackwardAPI.gene_api_declaration(self)

    def generate_auto_paralel_branch(self) -> str:
        # if no tensor input, do not genetate auto parallel branch
        if len(self.inputs['names']) == 0:
            return ""
        return MAIN_DIST_BRANCH_TEMPLATE.format(
            self.generate_if_condition_code(),
            self.generate_output_creation_code(),
            self.generate_infer_global_shape_code(),
            self.generate_kernel_selection_code(),
            self.generate_prepare_data_code(),
            self.generate_infer_meta_code(),
            self.generate_kernel_call_code(),
            self.generate_return_code(),
        )


def header_include():
    return """
#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"
"""


def source_include(header_file_path, fw_header_file_path):
    return f"""
#include "{header_file_path}"
#include <memory>

#include "glog/logging.h"
#include "paddle/utils/flags.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "{fw_header_file_path}"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/api/profiler/supplement_tracing.h"

PD_DECLARE_bool(conv2d_disable_cudnn);
PD_DECLARE_int32(low_precision_op_list);
"""


def backward_api_namespace():
    return (
        """
namespace paddle {
namespace experimental {

""",
        """

}  // namespace experimental
}  // namespace paddle
""",
    )


def generate_backward_api(
    backward_yaml_path,
    is_fused_backward_yaml,
    header_file_path,
    source_file_path,
):
    bw_apis = []
    for each_api_yaml in backward_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                bw_apis.extend(api_list)

    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = backward_api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = (
        "paddle/phi/api/backward/fused_backward_api.h"
        if is_fused_backward_yaml
        else "paddle/phi/api/backward/backward_api.h"
    )
    include_fw_header_file = (
        "paddle/phi/api/include/fused_api.h"
        if is_fused_backward_yaml
        else "paddle/phi/api/include/api.h"
    )
    source_file.write(
        source_include(include_header_file, include_fw_header_file)
    )
    source_file.write(namespace[0])
    # not all fused ops supoort dygraph
    if is_fused_backward_yaml is True:
        new_bw_apis = [
            bw_api
            for bw_api in bw_apis
            if "support_dygraph_mode" in bw_api
            and bw_api["support_dygraph_mode"] is True
        ]
        bw_apis = new_bw_apis

    for bw_api in bw_apis:
        dist_bw_api = DistBackwardAPI(bw_api)
        header_file.write(dist_bw_api.gene_api_declaration())
        if is_fused_backward_yaml is True:
            source_file.write(dist_bw_api.gene_api_code())
        else:
            source_file.write(dist_bw_api.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ backward API files'
    )
    parser.add_argument(
        '--backward_yaml_path',
        help='path to backward yaml file',
        nargs='+',
        default=['paddle/phi/api/yaml/backward.yaml'],
    )

    parser.add_argument(
        '--is_fused_backward_yaml',
        help='flag of fused backward yaml',
        action='store_true',
    )

    parser.add_argument(
        '--backward_header_path',
        help='output of generated backward header code file',
        default='paddle/phi/api/backward/backward_api.h',
    )

    parser.add_argument(
        '--backward_source_path',
        help='output of generated backward source code file',
        default='paddle/phi/api/lib/backward_api.cc',
    )

    options = parser.parse_args()

    backward_yaml_path = options.backward_yaml_path
    is_fused_backward_yaml = options.is_fused_backward_yaml
    header_file_path = options.backward_header_path
    source_file_path = options.backward_source_path

    generate_backward_api(
        backward_yaml_path,
        is_fused_backward_yaml,
        header_file_path,
        source_file_path,
    )


if __name__ == '__main__':
    main()
