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
import re

import yaml
from api_base import PREFIX_TENSOR_NAME
from api_gen import (
    ForwardAPI,
    api_namespace,
    declare_extension_api,
    header_include,
    source_include,
)

######################
# Code Gen Templates #
######################

API_IMPL_TEMPLATE = """
PADDLE_API {} {}({}) {{
  // Kernel Key Construction{}
  // Kernel Dispatch Body{}
}}
"""
DIPATCH_END_GUARD_TEMPLATE = """
PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of ({}) for input tensors is unimplemented, please check the type of input tensors."));
"""

# TODO(chenweihang): add profile function code later
# TODO(chenweihang): add view support later
MAIN_DIST_BRANCH_TEMPLATE = """
  // Auto Parallel condition
  if (use_dist_branch) {{
    // 1. InferSpmd (Infer DistAttr of Inputs&Outputs){}
    // 2. Create API Output & Prepare Dist and Dense Output{}
    // 3. Infer DistTensor's Global Shape{}\n
    if (!computation_clip_for_pp){{
      // 4. Select Kernel{}
      // 5. Reshard Input{}\n
      // 6. PrepareData (DataTransform & Prepare Dense Input){}
      // 7. Infer Local DenseTensor Meta{}
      // 8. DenseTensor Kernel Call{}
      // 9. Reshard Partial Output to Replicated (Temporary){}\n
    }}
    // 10. Return
    {}
  }}
"""

# TODO(GhostScreaming): Support no-input operators.
# 1. Non computation rank clip
GET_MESH_TEMPLATE = """
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>({}impl())->dist_attr().process_mesh();
    computation_clip_for_pp = !phi::distributed::IsCurRankInMesh(mesh);"""

# Auto Parallel condition
AUTO_PARALLEL_COND_TEMPLATE = """
  bool use_dist_branch = AllInputsAreDistTensor({input_args});
  bool computation_clip_for_pp = false;
  if (use_dist_branch) {{{mesh}
  }}
  if (!computation_clip_for_pp) {{{kernel_code}
  }}
"""

# 1. InferSPMD
SINGLE_DIST_META_IN_TEMPLATE = """
    auto meta_dist_input_{} = MakeDistMetaTensor(*{}.impl());"""
INFER_SPMD_TEMPLATE = """
    auto spmd_info = phi::distributed::{}({});
"""
GENERAL_INFER_SPMD_TEMPLATE = """
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmd({});
"""
UNSUPPORTED_INFER_SPMD_COMMENT_TEMPLATE = """
    // API `{}` does not support InferSpmd now
"""

# 2. Create API Outputs
API_OUT_CREATION_TEMPLATE = """
    {} api_output{};
"""
INPLACE_API_OUT_CREATION_TEMPLATE = """
    {} api_output{{{}}};
"""
SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD = """
    auto dist_out = SetKernelDistOutput(&api_output);
    auto dense_out = dist_out->unsafe_mutable_value();
    if (computation_clip_for_pp) {{
      *dense_out = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }}
"""
MULTI_SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD = """
    auto dist_out_{idx} = SetKernelDistOutput({out});
    auto dense_out_{idx} = dist_out_{idx}->unsafe_mutable_value();
    if (computation_clip_for_pp) {{
      *dense_out_{idx} = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }}
"""
SINGLE_OUT_CREATION_TEMPLATE = """
    auto dist_out = SetKernelDistOutput(&api_output, spmd_info.second[0]);
    auto dense_out = dist_out->unsafe_mutable_value();
    if (computation_clip_for_pp) {{
      *dense_out = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }}
"""
MULTI_SINGLE_OUT_CREATION_TEMPLATE = """
    auto dist_out_{idx} = SetKernelDistOutput({out}, spmd_info.second[{idx}]);
    auto dense_out_{idx} = dist_out_{idx}->unsafe_mutable_value();
    if (computation_clip_for_pp) {{
      *dense_out_{idx} = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }}
"""
VECTOR_OUT_CREATION_TEMPLATE = """
    auto dist_out = SetKernelDistOutput({}, &api_output);
    std::vector<phi::DenseTensor*> dense_out(dist_out.size());
    for (size_t i = 0; i < dist_out.size(); ++i) {{
      dense_out[i] = const_cast<phi::DenseTensor*>(&dist_out[i]->value());
      if (computation_clip_for_pp) {{
        *dense_out[i] = phi::DenseTensor(
                std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
                phi::DenseTensorMeta());
      }}
    }}
"""
MULTI_VECTOR_OUT_CREATION_TEMPLATE = """
    auto dist_out_{out_name} = SetKernelDistOutput({size}, {in_name});
    std::vector<phi::DenseTensor*> dense_out_{out_name}(dist_out_{out_name}.size());
    for (size_t i = 0; i < dist_out_{out_name}.size(); ++i) {{
        dense_out_{out_name}[i] = const_cast<phi::DenseTensor*>(&dist_out_{out_name}[i]->value());
        if (computation_clip_for_pp) {{
          *dense_out_{out_name}[i] = phi::DenseTensor(
                  std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
                  phi::DenseTensorMeta());
        }}
    }}
"""
MULTI_VECTOR_INPLACE_AND_OPTIONAL_OUT_CREATION_TEMPLATE = """
    auto dist_out_{out_name} = {out_func}({size}, {in_name});
    std::vector<phi::DenseTensor*> dense_out_{out_name}(dist_out_{out_name}.size());
    for (size_t i = 0; i < dist_out_{out_name}.size(); ++i) {{
        dense_out_{out_name}[i] = dist_out_{out_name}[i] ? const_cast<phi::DenseTensor*>(&dist_out_{out_name}[i]->value()) : nullptr;
    }}
"""

# 3. Infer Global Shape
# TODO(chenweihang): the input MetaTensor created by Inferspmd can be reused
# for InferGlobalShape to avoid creating repeated inputs.
SINGLE_GLOBAL_META_IN_TEMPLATE = """MakeMetaTensor(*{}.impl()), """
VECTOR_GLOBAL_META_IN_TEMPLATE = """{}_meta_ptr_vec, """
VECTOR_GLOBAL_META_IN_DECL_TEMPLATE = """
    std::vector<phi::MetaTensor> {name}_meta_vec;
    for (auto tmp : {name}) {{
      {name}_meta_vec.emplace_back(MakeMetaTensor(*tmp.impl()));
    }}
    std::vector<const phi::MetaTensor*> {name}_meta_ptr_vec({name}_meta_vec.size());
    for (size_t i=0; i < {name}_meta_ptr_vec.size(); ++i) {{
      {name}_meta_ptr_vec[i] = &{name}_meta_vec[i];
    }}
"""
OPTIONAL_GLOBAL_SINGLE_META_IN_TEMPLATE = """meta_dist_{}, """
OPTIONAL_GLOBAL_SINGLE_META_IN_DECL_TEMPLATE = """
    phi::MetaTensor meta_dist_{name} = {name} ? MakeMetaTensor(*(*{name}).impl()) : phi::MetaTensor();
"""
OPTIONAL_GLOBAL_VECTOR_META_IN_TEMPLATE = """{}_meta_ptr_vec, """
OPTIONAL_GLOBAL_VECTOR_META_IN_DECL_TEMPLATE = """
    std::vector<phi::MetaTensor> {name}_meta_vec_tmp;
    if ({name}) {{
      for (auto tmp : *{name}) {{
        {name}_meta_vec_tmp.emplace_back(MakeMetaTensor(*tmp.impl()));
      }}
    }}
    std::vector<const phi::MetaTensor*> {name}_meta_ptr_vec_tmp({name}_meta_vec_tmp.size());
    for (size_t i = 0; i < {name}_meta_ptr_vec_tmp.size(); ++i) {{
      {name}_meta_ptr_vec_tmp[i] = &{name}_meta_vec_tmp[i];
    }}
    paddle::optional<std::vector<const phi::MetaTensor*>> {name}_meta_ptr_vec =
        {name} ? paddle::make_optional<std::vector<const phi::MetaTensor*>>({name}_meta_ptr_vec_tmp) : paddle::none;
"""
SINGLE_GLOBAL_META_OUT_DECL_TEMPLATE = """
    phi::MetaTensor meta_{}({});"""
VECTOR_GLOBAL_META_OUT_DECL_TEMPLATE = """
    std::vector<phi::MetaTensor> {name}_meta_vec;
    for (auto tmp : {name}) {{
      {name}_meta_vec.emplace_back(phi::MetaTensor(tmp));
    }}
    std::vector<phi::MetaTensor*> {name}_meta_ptr_vec({name}.size());
    for (size_t i = 0; i < {name}_meta_vec.size(); ++i) {{
      {name}_meta_ptr_vec[i] = &{name}_meta_vec[i];
    }}
"""
INFER_GLOBAL_SHAPE_TEMPLATE = """
    phi::{}({}{});
"""
# Dist Branch will not generated in the API that doesn't have input tensor.
SET_SINGLE_OUT_REPLICATED_DIST_ATTR = """
    SetReplicatedDistAttrForOutput({}, spmd_info.first[0].process_mesh());"""
SET_VECTOR_OUT_REPLICATED_DIST_ATTR = """
    auto current_process_mesh = spmd_info.first[0].process_mesh();
    for (size_t i = 0; i < dist_out.size(); ++i) {{
        SetReplicatedDistAttrForOutput(dist_out[i], current_process_mesh);
    }}
"""

# 4. Select Kernel
KERNEL_SELECTION_TEMPLATE = """
      VLOG(6) << "{} API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "{}", {{kernel_backend, kernel_layout, kernel_data_type}});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "{} kernel: " << kernel;
      auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
"""

# 5. Reshard Input
SINGLE_INPUT_RESHARD_TEMPLATE = """
      auto dist_input_{arg} = ReshardApiInputToKernelInput(dev_ctx, {arg}, spmd_info.first[{idx}]);"""
SINGLE_GENERAL_INPUT_RESHARD_TEMPLATE = """
      auto dist_input_{arg} = ReshardApiInputToReplicatedKernelInput(dev_ctx, {arg}, spmd_info.first[{idx}]);"""
UNSUPPORTED_RESHARD_INPUT_COMMENT_TEMPLATE = """
      // API `{}` does not need to support ReshardInput at this time
"""

# 6. PrepareData
SINGLE_PREPARE_DATA_TEMPLATE = """
      dist_input_{arg} = PrepareDataForDistTensor(dist_input_{arg}, GetKernelInputArgDef(kernel.InputAt({idx}), kernel_backend), {flag}, kernel_result.is_stride_kernel);
      auto input_{arg} = &dist_input_{arg}->value();
"""
SINGLE_PREPARE_DATA_TEMPLATE_NO_RESHARD = """
      auto dist_input_{arg} = PrepareDataForDistTensor({arg}, GetKernelInputArgDef(kernel.InputAt({idx}), kernel_backend), {flag}, kernel_result.is_stride_kernel);
      auto input_{arg} = &dist_input_{arg}->value();
"""
VECTOR_PREPARE_DATA_TEMPLATE = """
      auto dist_input_{name}_vec = PrepareDataForDistTensor({name}, GetKernelInputArgDef(kernel.InputAt({index}), kernel_backend), {trans_flag}, kernel_result.is_stride_kernel);
      std::vector<const phi::DenseTensor*> dense_input_{name}_vec;
      for (auto tmp : dist_input_{name}_vec) {{
        dense_input_{name}_vec.emplace_back(&tmp->value());
      }}
      std::vector<phi::MetaTensor> dense_input_{name}_meta_vec = MakeMetaTensor(dense_input_{name}_vec);
      std::vector<const phi::MetaTensor*> dense_input_{name}_meta_ptr_vec(dense_input_{name}_meta_vec.size());
      for (size_t i = 0; i < dense_input_{name}_meta_ptr_vec.size(); ++i) {{
        dense_input_{name}_meta_ptr_vec[i] = &dense_input_{name}_meta_vec[i];
      }}
"""
OPTIONAL_SINGLE_PREPARE_DATA_TEMPLATE = """
      auto dist_input_{name} = PrepareDataForDistTensor({name}, GetKernelInputArgDef(kernel.InputAt({index}), kernel_backend), {trans_flag}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_{name} = dist_input_{name} ? paddle::make_optional<phi::DenseTensor>(dist_input_{name}->value()) : paddle::none;
"""
OPTIONAL_VECTOR_PREPARE_DATA_TEMPLATE = """
      auto dist_input_{name}_vec = PrepareDataForDistTensor({name}, GetKernelInputArgDef(kernel.InputAt({index}), kernel_backend), {trans_flag}, kernel_result.is_stride_kernel);
      std::vector<const phi::DenseTensor*> dense_input_{name}_vec;
      if ({name}) {{
        for (auto tmp : *dist_input_{name}_vec) {{
          dense_input_{name}_vec.emplace_back(&tmp->value());
      }}
    }}
    paddle::optional<std::vector<const phi::DenseTensor*>> input_{name}(dense_input_{name}_vec);
    std::vector<phi::MetaTensor> dense_input_{name}_meta_vec = MakeMetaTensor(dense_input_{name}_vec);
    std::vector<const phi::MetaTensor*> dense_input_{name}_meta_ptr_vec_tmp(dense_input_{name}_meta_vec.size());
    for (size_t i = 0; i < dense_input_{name}_meta_ptr_vec_tmp.size(); ++i) {{
      dense_input_{name}_meta_ptr_vec_tmp[i] = &dense_input_{name}_meta_vec[i];
    }}
    paddle::optional<std::vector<const phi::MetaTensor*>> dense_input_{name}_meta_ptr_vec =
            {name} ? paddle::make_optional<std::vector<const phi::MetaTensor*>>(dense_input_{name}_meta_ptr_vec_tmp) : paddle::none;
"""
INFER_META_SINGLE_INPUT_TEMPLATE = """
    auto dist_input_{} = {}.impl();
    auto input_{} = &(static_cast<phi::distributed::DistTensor*>(dist_input_{}.get())->value());
"""
INFER_META_OPTIONAL_INPUT_TEMPLATE = """
    paddle::optional<phi::TensorBase> input_{} = {} ? paddle::optional<phi::TensorBase>(*{}->impl()) : paddle::none;
"""
INFER_META_VECTOR_INPUT_TEMPLATE = """
    auto input_{}_uq_ptr = TensorToDenseTensor({});
    const auto& input_{} = *input_{}_uq_ptr;
"""

# 7. Infer Local DenseTensor Meta
SINGLE_META_IN_TEMPLATE = """MakeMetaTensor(*input_{}), """
VECTOR_META_IN_TEMPLATE = """dense_input_{}_meta_ptr_vec, """
OPTIONAL_SINGLE_META_IN_TEMPLATE = """MakeMetaTensor(input_{}), """
OPTIONAL_VECTOR_META_IN_TEMPLATE = """dense_input_{}_meta_ptr_vec, """
SINGLE_META_OUT_DECL_TEMPLATE = """
      phi::MetaTensor meta_{}({});"""
VECTOR_META_OUT_DECL_TEMPLATE = """
      std::vector<phi::MetaTensor> {name}_meta_vec = MakeMetaTensor({name});
      std::vector<phi::MetaTensor*> {name}_meta_ptr_vec({name}_meta_vec.size());
      for (size_t i = 0; i < {name}_meta_vec.size(); ++i) {{
        {name}_meta_ptr_vec[i] = &{name}_meta_vec[i];
      }}
"""
INFER_META_TEMPLATE = """
      phi::{}({}{});
"""

# 8. DenseTensor Kernel Call
# TODO(chenweihang): support kernel fallback later
SINGLE_OUTPUT_NAME = """dense_out"""
# TODO(chenweihang): support vector and tuple output later
VECTOR_OUTPUT_NAME_TEMPLATE = """
"""
TUPLE_OUTPUT_NAME_TEMPLATE = """
"""
KERNEL_CALL_TEMPLATE = """
      using kernel_signature = {};
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)({}, {});
"""

# TODO(GhostScreaming): Some operators generate shape info in runtime,
# bincount. As a result, dist_output's global shape is set uncorrectly,
# because it's generated in InferMeta function. A temporally solution is
# use black op list to set DistTensor shape extra.
SINGLE_SET_DIST_OUT_DIMS = """
    dist_out->unsafe_set_dims(dense_out->dims());
"""
MULTI_SINGLE_SET_DIST_OUT_DIMS = """
    dist_out_{}->unsafe_set_dims(dense_out_{}->dims());
"""
VECTOR_SET_DIST_OUT_DIMS = """
    for (size_t i = 0; i < dist_out.size(); ++i) {{
        dist_out[i]->unsafe_set_dims(dense_out[i]->dims());
    }}
"""

PREFIX_VECTOR_TENSOR_NAME = "dense_input_"
SUFFIX_VECTOR_TENSOR_NAME = "_vec"

# 9. Reshard Partial Output to Replicated
RESHARD_P2R_SINGLE_OUTPUT_TEMPLATE = """
      ReshardOutputPartialAxisToReplicated(dev_ctx, dist_out);"""
RESHARD_P2R_MULTI_SINGLE_OUTPUT_TEMPLATE = """
      ReshardOutputPartialAxisToReplicated(dev_ctx, dist_out_{});"""
UNSUPPORTED_RESHARD_OUTPUT_COMMENT_TEMPLATE = """
      // API `{}` does not need to support ReshardOutput now
"""

# BaseAPI members:
# inputs:
#     names : [], list of input names
#     input_info : {input_name : type}
# attrs:
#     names : [], list of attribute names
#     attr_info : { attr_name : (type, default_values)}
# outputs:
#     names : [], list of output names
#     types : [], list of output types
#     out_size_expr : [], expression for getting size of vector<Tensor>


# TODO(GhostScreaming): Black list for operators which infer shape in runtime.
ops_infer_shape_in_runtime = [
    "bincount",
    "bicubic_interp",
    "bilinear_interp",
    "linear_interp",
    "nearest_interp",
    "trilinear_interp",
]


class DistForwardAPI(ForwardAPI):
    def __init__(self, api_item_yaml):
        super().__init__(api_item_yaml)
        self.init_dist_api_members()

    def init_dist_api_members(self):
        self.gene_dist_input_func = {
            "const Tensor&": {
                "dense": self.generate_single_dense_input,
            },
            "const std::vector<Tensor>&": {
                "dense": self.generate_vector_dense_input,
            },
            "const paddle::optional<Tensor>&": {
                "dense": self.generate_optional_single_dense_input,
            },
            "const paddle::optional<std::vector<Tensor>>&": {
                "dense": self.generate_optional_vector_dense_input,
            },
        }

        self.inplace_flag = False
        self.dist_output_args = []
        self.dense_output_args = []
        self.generate_infer_spmd = False
        self.generate_general_infer_spmd = False

    # override BaseAPI's method
    def parse_infer_meta(self, infer_meta_config):
        infer_meta = infer_meta_config
        if 'param' not in infer_meta_config:
            infer_meta['param'] = None
        if 'spmd_rule' not in infer_meta_config:
            infer_meta['spmd_rule'] = None

        return infer_meta

    def need_to_generate_code_for_inplace_impl(self, i):
        return (
            self.inplace_flag
            and self.inplace_map is not None
            and self.outputs['names'][i] in self.inplace_map
        )

    def need_to_generate_code_for_view_impl(self, i):
        return (
            not self.inplace_flag
            and self.view_map is not None
            and self.outputs['names'][i] in self.view_map
        )

    def is_inplace_output(self, i):
        return self.outputs['names'][i] in self.inplace_map

    def is_inplace_and_optional_output(self, i):
        return (
            self.outputs['names'][i] in self.inplace_map
            and self.inplace_map[self.outputs['names'][i]] in self.optional_vars
        )

    def vector_output_size_assertion_check(self):
        assert (
            self.outputs['out_size_expr'] is not None
        ), f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."

    def generate_non_computation_rank_clip_code(self) -> str:
        if len(self.inputs['names']) > 0:
            mesh = ""
            # All inputs have same mesh
            if (
                self.inputs['input_info'][self.inputs['names'][0]]
                == "const Tensor&"
            ):
                mesh = GET_MESH_TEMPLATE.format(
                    "{}.".format(self.inputs['names'][0])
                )
            elif (
                self.inputs['input_info'][self.inputs['names'][0]]
                == "const paddle::optional<Tensor>&"
            ):
                mesh = GET_MESH_TEMPLATE.format(
                    "{}->".format(self.inputs['names'][0])
                )
            elif (
                self.inputs['input_info'][self.inputs['names'][0]]
                == "const std::vector<Tensor>&"
            ):
                mesh = GET_MESH_TEMPLATE.format(
                    "{}[0].".format(self.inputs['names'][0])
                )
            elif (
                self.inputs['input_info'][self.inputs['names'][0]]
                == "const paddle::optional<std::vector<Tensor>>&"
            ):
                mesh = GET_MESH_TEMPLATE.format(
                    "{}->at(0).".format(self.inputs['names'][0])
                )
            return mesh
        else:
            return ""

    # Backward API Override this method
    def gene_kernel_backend_select(self):
        backend_select_code = ""
        if self.kernel['backend'] is not None:
            if '>' in self.kernel['backend']:
                vars_list = self.kernel['backend'].split('>')
                assert (
                    len(vars_list) == 2
                ), f"{self.api} api: The number of params to set backend with '>' only allows 2, but received {len(vars_list)}."
                assert (vars_list[0].strip() in self.attrs['names']) and (
                    self.attrs['attr_info'][vars_list[0].strip()][0]
                    == 'const Place&'
                ), f"{self.api} api: When use '>' to set kernel backend, the first param should be a attribute with Place type."
                backend_select_code = f"""
    kernel_backend = ParseBackendWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});
"""

            else:
                backend_args = [
                    ele.strip() for ele in self.kernel['backend'].split(',')
                ]
                backend_select_code = f"""
    kernel_backend = ParseBackend({", ".join(backend_args)});
"""

        return backend_select_code

    # Overload api_base.py gene_kernel_select function.
    def gene_kernel_select(self) -> str:
        api = self.api
        input_names = self.inputs['names']
        attrs = self.attrs
        kernel = self.kernel

        kernel_key_item_init = """
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;
"""

        # Check the tensor options
        attr_backend_count = 0
        attr_layout_count = 0
        attr_data_type_count = 0
        for attr_name in attrs['names']:
            if attrs['attr_info'][attr_name][0] == 'const Place&':
                assert (
                    kernel['backend'] is not None
                ), f"{api} api: When there is a parameter with 'Place' type in attributes, you must set backend of kernel manually."
                attr_backend_count = attr_backend_count + 1
            if attrs['attr_info'][attr_name][0] == 'DataLayout':
                assert (
                    kernel['layout'] is not None
                ), f"{api} api: When there is a parameter with 'DataLayout' type in attributes, you must set layout of kernel manually."
                attr_layout_count = attr_layout_count + 1
            if attrs['attr_info'][attr_name][0] == 'DataType':
                assert (
                    kernel['data_type'] is not None
                ), f"{api} api: When there is a parameter with 'DataType' type in attributes, you must set data_type of kernel manually."
                attr_data_type_count = attr_data_type_count + 1

        # preprocess kernel configures
        kernel_select_code = self.gene_kernel_backend_select()

        if kernel['layout'] is not None:
            if '>' in kernel['layout']:
                vars_list = kernel['layout'].split('>')
                assert (
                    len(vars_list) == 2
                ), f"{api} api: The number of params to set layout with '>' only allows 2, but received {len(vars_list)}."
                assert (
                    vars_list[0].strip() in attrs['names']
                    and attrs['attr_info'][vars_list[0].strip()][0]
                    == 'DataLayout'
                ), f"{api} api: When use '>' to set kernel layout, the first param should be a attribute with DataLayout type."
                kernel_select_code = (
                    kernel_select_code
                    + f"""
    kernel_layout = ParseLayoutWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});
"""
                )

            else:
                vars_list = kernel['layout'].split(',')
                assert (
                    len(vars_list) == 1
                ), f"{api} api: The number of params to set layout must be 1, but received {len(vars_list)}."
                kernel_select_code = (
                    kernel_select_code
                    + f"""
    kernel_layout = ParseLayout({vars_list[0].strip()});
"""
                )

        if kernel['data_type'] is not None:

            def process_data_type_args(args_item):
                args_item = args_item.strip()
                complex_match_result = re.match(
                    r"complex\((?P<param_name>\w+)\)", args_item
                )
                if complex_match_result:
                    return f"phi::dtype::ToComplex(ParseDataType({complex_match_result.group('param_name')}))"
                else:
                    return f"ParseDataType({args_item})"

            if '>' in kernel['data_type']:
                vars_list = kernel['data_type'].split('>')
                assert (
                    len(vars_list) == 2
                ), f"{api} api: The number of params to set data_type with '>' only allows 2, but received {len(vars_list)}."
                assert (
                    vars_list[0].strip() in attrs['names']
                    and attrs['attr_info'][vars_list[0].strip()][0]
                    == 'DataType'
                ), f"{api} api: When use '>' to set kernel data_type, the first param should be a attribute with DataType type."
                kernel_select_code = (
                    kernel_select_code
                    + f"""
    kernel_data_type = ParseDataTypeWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});
"""
                )

            else:
                vars_list = kernel['data_type'].split(',')
                assert (
                    len(vars_list) == 1
                ), f"{api} api: The number of params to set data_type only allows 1, but received {len(vars_list)}."
                kernel_select_code = (
                    kernel_select_code
                    + f"""
    kernel_data_type = {process_data_type_args(vars_list[0])};
"""
                )

        if len(input_names) == 0:
            assert (
                attr_backend_count > 0 and attr_data_type_count > 0
            ), f"{api} api: When there is no input tensor, the args must have 'Place' and 'DataType'."

        kernel_select_args = ""
        for input_name in input_names:
            kernel_select_args = kernel_select_args + input_name + ", "

        if len(kernel_select_args) > 2:
            kernel_select_args = kernel_select_args[:-2]

        # kernel_select_code = kernel_key_item_init + kernel_select_code

        if len(input_names) > 0:
            kernel_select_code = (
                kernel_select_code
                + f"""
    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {{
      auto kernel_key_set = ParseKernelKeyByInputArgs({kernel_select_args});
      auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
      if (kernel_backend == Backend::UNDEFINED) {{
        kernel_backend = kernel_key.backend();
      }}
      if (kernel_layout == DataLayout::UNDEFINED) {{
        kernel_layout = kernel_key.layout();
      }}
      if (kernel_data_type == DataType::UNDEFINED) {{
        kernel_data_type = kernel_key.dtype();
      }}
    }}"""
            )

        input_args = ""
        for input_name in self.inputs['names']:
            input_args = input_args + input_name + ", "
        if len(input_args) > 2:
            input_args = input_args[:-2]
        mesh = self.generate_non_computation_rank_clip_code()

        if_condition_code = AUTO_PARALLEL_COND_TEMPLATE.format(
            input_args=input_args, mesh=mesh, kernel_code=kernel_select_code
        )

        return kernel_key_item_init + if_condition_code

    def generate_specialized_infer_spmd_code(self) -> str:
        input_names = self.inputs['names']
        attr_names = self.attrs['names']

        # TODO(chenweihang): here we need to use infer_meta params,
        # if it is inconsistent, you need to change the infermeta func
        kernel_params = self.kernel['param']
        if kernel_params is None:
            kernel_params = input_names + attr_names

        input_decl_code = ""
        input_args_code = ""
        for param in kernel_params:
            if param in input_names:
                if self.inputs['input_info'][param] == "const Tensor&":
                    input_decl_code += SINGLE_DIST_META_IN_TEMPLATE.format(
                        param, param
                    )
                    input_args_code += "meta_dist_input_" + param + ", "
                else:
                    raise ValueError(
                        f"{self.api} : Param of infer_spmd error : {self.inputs['input_info'][param]} type is not supported."
                    )
            elif param in attr_names:
                input_args_code = input_args_code + param + ", "
            elif isinstance(param, str):
                input_args_code = input_args_code + "\"" + param + "\", "
            elif isinstance(param, bool):
                input_args_code = input_args_code + str(param).lower() + ", "
            else:
                input_args_code = input_args_code + str(param) + ", "

        infer_spmd_code = ""
        infer_spmd_func_code = self.infer_meta['spmd_rule']
        infer_spmd_code = INFER_SPMD_TEMPLATE.format(
            infer_spmd_func_code, input_args_code[:-2]
        )
        self.generate_infer_spmd = True

        return input_decl_code + infer_spmd_code

    def generate_general_infer_spmd_code(self) -> str:
        input_names = self.inputs['names']
        attr_names = self.attrs['names']

        # TODO(chenweihang): here we need use infer_meta params,
        # if it is inconsistent, you need to change the infermeta func
        kernel_params = self.kernel['param']
        if kernel_params is None:
            kernel_params = input_names + attr_names

        input_decl_code = ""
        input_args_code = ""
        for param in kernel_params:
            if param in input_names:
                if self.inputs['input_info'][param] == "const Tensor&":
                    input_decl_code += SINGLE_DIST_META_IN_TEMPLATE.format(
                        param, param
                    )
                    input_args_code += "meta_dist_input_" + param + ", "
                elif (
                    self.inputs['input_info'][param]
                    == "const std::vector<Tensor>&"
                    or self.inputs['input_info'][param]
                    == "const paddle::optional<Tensor>&"
                    or self.inputs['input_info'][param]
                    == "const paddle::optional<std::vector<Tensor>>&"
                ):
                    # TODO(chenweihang): support other input type later,
                    # now only support single tensor input api
                    input_decl_code = ""
                    input_args_code = ""
                    break
                else:
                    raise ValueError(
                        f"{self.api} : Param of infer_spmd error : {self.inputs['input_info'][param]} type is not supported."
                    )
            else:
                # do nothing
                pass

        if input_decl_code == "":
            return UNSUPPORTED_INFER_SPMD_COMMENT_TEMPLATE.format(self.api)

        infer_spmd_code = GENERAL_INFER_SPMD_TEMPLATE.format(
            input_args_code[:-2]
        )
        self.generate_infer_spmd = True
        self.generate_general_infer_spmd = True

        return input_decl_code + infer_spmd_code

    def generate_infer_spmd_code(self) -> str:
        if self.infer_meta['spmd_rule'] is not None:
            return self.generate_specialized_infer_spmd_code()
        else:
            return self.generate_general_infer_spmd_code()

    def generate_output_creation_code(self) -> str:
        # forward api need to generate api and kernel outputs
        output_num = len(self.outputs['types'])
        return_type = self.get_return_type_with_intermediate(self.inplace_flag)
        output_creation_code = ""
        if output_num == 1:
            # api output generate
            if self.need_to_generate_code_for_inplace_impl(0):
                inplace_assign_code = (
                    " = " + self.inplace_map[self.outputs['names'][0]]
                )
                output_creation_code += API_OUT_CREATION_TEMPLATE.format(
                    return_type, inplace_assign_code
                )
            else:
                output_creation_code += API_OUT_CREATION_TEMPLATE.format(
                    return_type, ""
                )
            # kernel output generate
            self.dist_output_args.append('dist_out')
            self.dense_output_args.append('dense_out')
            if self.outputs['types'][0] == 'Tensor':
                if self.infer_meta['spmd_rule'] is not None:
                    output_creation_code += SINGLE_OUT_CREATION_TEMPLATE
                else:
                    output_creation_code += SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD
            elif self.outputs['types'][0] == 'std::vector<Tensor>':
                output_creation_code += VECTOR_OUT_CREATION_TEMPLATE.format(
                    self.outputs['out_size_expr'][0]
                )
            else:
                self.vector_output_size_assertion_check()
        elif output_num > 1:
            # api output generate
            if self.inplace_flag:
                inplace_assign_code = ""
                for i, out_name in enumerate(self.outputs['names']):
                    if self.need_to_generate_code_for_inplace_impl(i):
                        inplace_assign_code += self.inplace_map[out_name] + ', '
                    else:
                        inplace_assign_code += 'Tensor(), '
                inplace_assign_code = inplace_assign_code[:-2]
                output_creation_code += (
                    INPLACE_API_OUT_CREATION_TEMPLATE.format(
                        return_type, inplace_assign_code
                    )
                )
            else:
                output_creation_code += API_OUT_CREATION_TEMPLATE.format(
                    return_type, ""
                )
            # kernel output generate
            for i, out_type in enumerate(self.outputs['types']):
                self.dist_output_args.append(f'dist_out_{i}')
                self.dense_output_args.append(f'dense_out_{i}')
                set_out_func = "SetKernelDistOutput"
                get_out_code = f"&std::get<{i}>(api_output)"
                if self.is_inplace_and_optional_output(i):
                    get_out_code = f"std::get<{i}>(api_output).get_ptr()"
                if out_type == 'std::vector<Tensor>':
                    self.vector_output_size_assertion_check()
                    # Special case for inplace vector and inplace optional<vector>
                    if self.is_inplace_output(i):
                        set_out_func = "SetKernelDistInplaceOutput"
                        if self.is_inplace_and_optional_output(i):
                            set_out_func = "SetKernelDistInplaceOptionalOutput"
                            get_out_code = f"std::get<{i}>(api_output)"
                        output_creation_code += MULTI_VECTOR_INPLACE_AND_OPTIONAL_OUT_CREATION_TEMPLATE.format(
                            out_func=set_out_func,
                            out_name=i,
                            size=self.outputs['out_size_expr'][i],
                            in_name=get_out_code,
                        )
                    else:
                        output_creation_code += (
                            MULTI_VECTOR_OUT_CREATION_TEMPLATE.format(
                                out_name=i,
                                size=self.outputs['out_size_expr'][i],
                                in_name=get_out_code,
                            )
                        )
                else:
                    if self.infer_meta['spmd_rule'] is not None:
                        output_creation_code += (
                            MULTI_SINGLE_OUT_CREATION_TEMPLATE.format(
                                idx=i, out=get_out_code
                            )
                        )
                    else:
                        output_creation_code += (
                            MULTI_SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD.format(
                                idx=i, out=get_out_code
                            )
                        )
        else:
            raise ValueError(
                f"{self.api} : Output error: the output should not be empty."
            )

        return output_creation_code

    def generate_infer_global_shape_code(self) -> str:
        input_names = self.inputs['names']
        attr_names = self.attrs['names']

        # 1. get infer meta func name
        infer_meta = self.infer_meta
        infer_meta_func_code = infer_meta['func']

        # 2. get meta tensor input args
        infer_meta_params = (
            infer_meta['param']
            if infer_meta['param'] is not None
            else input_names + attr_names
        )
        input_meta_code = ""
        input_args_code = ""
        for param in infer_meta_params:
            if param in input_names:
                if self.inputs['input_info'][param] == "const Tensor&":
                    input_args_code += SINGLE_GLOBAL_META_IN_TEMPLATE.format(
                        param
                    )
                elif (
                    self.inputs['input_info'][param]
                    == "const std::vector<Tensor>&"
                ):
                    input_args_code += VECTOR_GLOBAL_META_IN_TEMPLATE.format(
                        param
                    )
                    input_meta_code += (
                        VECTOR_GLOBAL_META_IN_DECL_TEMPLATE.format(name=param)
                    )
                elif (
                    self.inputs['input_info'][param]
                    == "const paddle::optional<Tensor>&"
                ):
                    input_args_code += (
                        OPTIONAL_GLOBAL_SINGLE_META_IN_TEMPLATE.format(param)
                    )
                    input_meta_code += (
                        OPTIONAL_GLOBAL_SINGLE_META_IN_DECL_TEMPLATE.format(
                            name=param
                        )
                    )
                elif (
                    self.inputs['input_info'][param]
                    == "const paddle::optional<std::vector<Tensor>>&"
                ):
                    input_args_code += (
                        OPTIONAL_GLOBAL_VECTOR_META_IN_TEMPLATE.format(param)
                    )
                    input_meta_code += (
                        OPTIONAL_GLOBAL_VECTOR_META_IN_DECL_TEMPLATE.format(
                            name=param
                        )
                    )

                else:
                    raise ValueError(
                        f"{self.api} : Param of infer_spmd error : {self.inputs['input_info'][param]} type is not supported."
                    )
            elif param in attr_names:
                input_args_code = input_args_code + param + ", "
            elif isinstance(param, str):
                input_args_code = input_args_code + "\"" + param + "\", "
            elif isinstance(param, bool):
                input_args_code = input_args_code + str(param).lower() + ", "
            else:
                input_args_code = input_args_code + str(param) + ", "

        # 3. get meta tensor output args
        output_decl_code = ""
        output_args_code = ""
        set_out_dist_attr_code = ""
        for i, out_name in enumerate(self.dist_output_args):
            if self.outputs['types'][i] == 'std::vector<Tensor>':
                output_decl_code += VECTOR_GLOBAL_META_OUT_DECL_TEMPLATE.format(
                    name=out_name
                )
                output_args_code += f"{out_name}_meta_ptr_vec, "
                if self.generate_general_infer_spmd is True:
                    set_out_dist_attr_code += (
                        SET_VECTOR_OUT_REPLICATED_DIST_ATTR
                    )
            else:
                output_decl_code += SINGLE_GLOBAL_META_OUT_DECL_TEMPLATE.format(
                    out_name, out_name
                )
                if len(self.dense_output_args) == 1:
                    output_args_code += f"&meta_{out_name}, "
                else:
                    output_args_code += (
                        f"{out_name} ? &meta_{out_name} : nullptr, "
                    )
                if self.generate_general_infer_spmd is True:
                    set_out_dist_attr_code += (
                        SET_SINGLE_OUT_REPLICATED_DIST_ATTR.format(out_name)
                    )
        output_args_code = output_args_code[:-2]

        return (
            output_decl_code
            + input_meta_code
            + INFER_GLOBAL_SHAPE_TEMPLATE.format(
                infer_meta_func_code, input_args_code, output_args_code
            )
            + set_out_dist_attr_code
        )

    def generate_kernel_selection_code(self) -> str:
        return KERNEL_SELECTION_TEMPLATE.format(
            self.api, self.kernel['func'][0], self.kernel['func'][0]
        )

    def generate_reshard_input_code(self) -> str:
        input_reshard_code = ""
        if self.generate_infer_spmd is True:
            input_names = self.inputs['names']

            kernel_params = (
                self.kernel['param']
                if self.kernel['param'] is not None
                else input_names
            )

            for i, param in enumerate(kernel_params):
                if param in input_names:
                    if self.inputs['input_info'][param] == "const Tensor&":
                        if self.generate_general_infer_spmd is True:
                            input_reshard_code += (
                                SINGLE_GENERAL_INPUT_RESHARD_TEMPLATE.format(
                                    arg=param, idx=i
                                )
                            )
                        else:
                            input_reshard_code += (
                                SINGLE_INPUT_RESHARD_TEMPLATE.format(
                                    arg=param, idx=i
                                )
                            )
                    else:
                        raise ValueError(
                            f"{self.api} : Param of reshard input error : {self.inputs['input_info'][param]} type is not supported."
                        )
                else:
                    # do nothing
                    pass
        else:
            input_reshard_code = (
                UNSUPPORTED_RESHARD_INPUT_COMMENT_TEMPLATE.format(self.api)
            )

        return input_reshard_code

    def generate_single_dense_input(
        self,
        input_name,
    ):
        input_tensor_code = ""
        trans_flag = self.gene_trans_flag(input_name)
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names

        if self.generate_infer_spmd is True:
            input_tensor_code += SINGLE_PREPARE_DATA_TEMPLATE.format(
                arg=input_name,
                idx=kernel_param.index(input_name),
                flag=trans_flag,
            )
        else:
            input_tensor_code += SINGLE_PREPARE_DATA_TEMPLATE_NO_RESHARD.format(
                arg=input_name,
                idx=kernel_param.index(input_name),
                flag=trans_flag,
            )

        return input_tensor_code

    def generate_vector_dense_input(
        self,
        input_name,
    ):
        input_tensor_code = ""
        trans_flag = self.gene_trans_flag(input_name)
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names

        input_tensor_code += VECTOR_PREPARE_DATA_TEMPLATE.format(
            name=input_name,
            index=kernel_param.index(input_name),
            trans_flag=trans_flag,
        )

        return input_tensor_code

    def generate_optional_single_dense_input(
        self,
        input_name,
    ):
        input_tensor_code = ""
        trans_flag = self.gene_trans_flag(input_name)
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names

        input_tensor_code += OPTIONAL_SINGLE_PREPARE_DATA_TEMPLATE.format(
            name=input_name,
            index=kernel_param.index(input_name),
            trans_flag=trans_flag,
        )

        return input_tensor_code

    def generate_optional_vector_dense_input(
        self,
        input_name,
    ):
        input_tensor_code = ""
        trans_flag = self.gene_trans_flag(input_name)
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names

        input_tensor_code += OPTIONAL_VECTOR_PREPARE_DATA_TEMPLATE.format(
            name=input_name,
            index=kernel_param.index(input_name),
            trans_flag=trans_flag,
        )

        return input_tensor_code

    def generate_prepare_data_code(self) -> str:
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names
        input_tensor_code = ""
        for i, input_name in enumerate(input_names):
            # set input code
            if input_name in kernel_param:
                # only support dense tensor
                api_tensor_type = self.inputs['input_info'][input_name]
                phi_tensor_type = 'dense'
                if api_tensor_type in self.gene_dist_input_func.keys():
                    input_tensor_code += self.gene_dist_input_func[
                        api_tensor_type
                    ][phi_tensor_type](input_name)
                else:
                    # do nothing
                    pass
            else:
                if input_name in self.infer_meta['param']:
                    if input_name in self.optional_vars:
                        input_tensor_code += (
                            INFER_META_OPTIONAL_INPUT_TEMPLATE.format(
                                input_name, input_name, input_name, input_name
                            )
                        )
                    else:
                        if (
                            self.inputs['input_info'][input_name]
                            == "const std::vector<Tensor>&"
                        ):
                            input_tensor_code += (
                                INFER_META_VECTOR_INPUT_TEMPLATE.format(
                                    input_name, input_name, input_name
                                )
                            )
                        else:
                            input_tensor_code += (
                                INFER_META_SINGLE_INPUT_TEMPLATE.format(
                                    input_name,
                                    input_name,
                                    input_name,
                                    input_name,
                                )
                            )

        return input_tensor_code

    def generate_infer_meta_code(self) -> str:
        input_names = self.inputs['names']
        attr_names = self.attrs['names']

        # 1. get infer meta func name
        infer_meta = self.infer_meta
        infer_meta_func_code = infer_meta['func']

        # 2. get meta tensor input args
        infer_meta_params = (
            infer_meta['param']
            if infer_meta['param'] is not None
            else input_names + attr_names
        )
        input_args_code = ""
        for param in infer_meta_params:
            if param in input_names:
                if self.inputs['input_info'][param] == "const Tensor&":
                    input_args_code += SINGLE_META_IN_TEMPLATE.format(param)
                elif (
                    self.inputs['input_info'][param]
                    == "const std::vector<Tensor>&"
                ):
                    input_args_code += VECTOR_META_IN_TEMPLATE.format(param)
                elif (
                    self.inputs['input_info'][param]
                    == "const paddle::optional<Tensor>&"
                ):
                    input_args_code += OPTIONAL_SINGLE_META_IN_TEMPLATE.format(
                        param
                    )
                elif (
                    self.inputs['input_info'][param]
                    == "const paddle::optional<std::vector<Tensor>>&"
                ):
                    input_args_code += OPTIONAL_VECTOR_META_IN_TEMPLATE.format(
                        param
                    )
                else:
                    raise ValueError(
                        f"{self.api} : Param of infer_meta error : {self.inputs['input_info'][param]} type is not supported."
                    )
            elif param in attr_names:
                input_args_code = input_args_code + param + ", "
            elif isinstance(param, str):
                input_args_code = input_args_code + "\"" + param + "\", "
            elif isinstance(param, bool):
                input_args_code = input_args_code + str(param).lower() + ", "
            else:
                input_args_code = input_args_code + str(param) + ", "

        # 3. get meta tensor output args
        output_decl_code = ""
        output_args_code = ""
        for i, out_name in enumerate(self.dense_output_args):
            if self.outputs['types'][i] == 'std::vector<Tensor>':
                output_decl_code += VECTOR_META_OUT_DECL_TEMPLATE.format(
                    name=out_name
                )
                output_args_code += f"{out_name}_meta_ptr_vec, "
            else:
                output_decl_code += SINGLE_META_OUT_DECL_TEMPLATE.format(
                    out_name, out_name
                )
                if len(self.dense_output_args) == 1:
                    output_args_code += f"&meta_{out_name}, "
                else:
                    output_args_code += (
                        f"{out_name} ? &meta_{out_name} : nullptr, "
                    )
        output_args_code = output_args_code[:-2]

        return output_decl_code + INFER_META_TEMPLATE.format(
            infer_meta_func_code, input_args_code, output_args_code
        )

    def generate_kernel_call_code(self) -> str:
        dense_input_trans_map = {
            'const Tensor&': 'const phi::DenseTensor&',
            'const std::vector<Tensor>&': 'const std::vector<const phi::DenseTensor*>&',
            'const paddle::optional<Tensor&>': 'paddle::optional<const phi::DenseTensor&>',
            'const paddle::optional<Tensor>&': 'const paddle::optional<phi::DenseTensor>&',
            'const paddle::optional<std::vector<Tensor>>&': 'const paddle::optional<std::vector<const phi::DenseTensor*>>&',
        }
        dense_output_trans_map = {
            'Tensor': 'phi::DenseTensor*',
            'std::vector<Tensor>': 'std::vector<phi::DenseTensor*>',
        }

        input_names = self.inputs['names']
        input_infos = self.inputs['input_info']
        kernel_args_type_list = ['const phi::DeviceContext&']

        attr_names = self.attrs['names']
        kernel_args = self.kernel['param']
        if kernel_args is None:
            kernel_args = input_names + attr_names

        # 1. generate input args list
        input_args = ["*dev_ctx"]
        for arg in kernel_args:
            if arg in input_names:
                if arg in self.optional_vars:
                    input_args.append(PREFIX_TENSOR_NAME + arg)
                else:
                    if input_infos[arg] == "const Tensor&":
                        input_args.append("*" + PREFIX_TENSOR_NAME + arg)
                    elif input_infos[arg] == "const std::vector<Tensor>&":
                        input_args.append(
                            PREFIX_VECTOR_TENSOR_NAME
                            + arg
                            + SUFFIX_VECTOR_TENSOR_NAME
                        )
                    else:
                        # do nothing
                        pass
                kernel_args_type_list.append(
                    dense_input_trans_map[input_infos[arg]]
                )
            elif arg in attr_names:
                if 'IntArray' in self.attrs['attr_info'][arg][0]:
                    kernel_args_type_list.append('const phi::IntArray&')
                    arg = 'phi::IntArray(' + arg + ')'
                elif 'vector<phi::Scalar>' in self.attrs['attr_info'][arg][0]:
                    kernel_args_type_list.append(
                        'const std::vector<phi::Scalar>&'
                    )
                elif 'Scalar' in self.attrs['attr_info'][arg][0]:
                    kernel_args_type_list.append('const phi::Scalar&')
                    arg = 'phi::Scalar(' + arg + ')'
                else:
                    kernel_args_type_list.append(
                        self.attrs['attr_info'][arg][0]
                    )
                input_args.append(arg)
            elif isinstance(arg, bool):
                input_args.append(str(arg).lower())
            else:
                input_args.append(str(arg))

        # 2. generate output args list
        # record into `self.dense_output_args` in `generate_output_creation_code` function

        # 3. generate kernel signature
        for i, out_type in enumerate(self.outputs['types']):
            kernel_args_type_list.append(dense_output_trans_map[out_type])
        kernel_signature = "void(*)(" + ", ".join(kernel_args_type_list) + ")"

        result = KERNEL_CALL_TEMPLATE.format(
            kernel_signature,
            ", ".join(input_args),
            ", ".join(self.dense_output_args),
        )
        global ops_infer_shape_in_runtime
        if self.kernel['func'][0] in ops_infer_shape_in_runtime:
            if len(self.outputs['types']) == 1:
                if self.outputs['types'][0] == 'Tensor':
                    result += SINGLE_SET_DIST_OUT_DIMS
                elif self.outputs['types'][0] == 'std::vector<Tensor>':
                    result += VECTOR_SET_DIST_OUT_DIMS
            else:
                for i in range(len(self.outputs['types'])):
                    result += MULTI_SINGLE_SET_DIST_OUT_DIMS.format(i, i)
        return result

    def generate_reshard_partial_out_to_replicated_code(self) -> str:
        reshard_p2r_code = ""
        if self.infer_meta['spmd_rule'] is not None:
            output_num = len(self.outputs['types'])
            if output_num == 1:
                if self.outputs['types'][0] == 'Tensor':
                    reshard_p2r_code += RESHARD_P2R_SINGLE_OUTPUT_TEMPLATE
                else:
                    self.vector_output_size_assertion_check()
            elif output_num > 1:
                for i, out_type in enumerate(self.outputs['types']):
                    if out_type == 'Tensor':
                        reshard_p2r_code += (
                            RESHARD_P2R_MULTI_SINGLE_OUTPUT_TEMPLATE.format(i)
                        )
                    else:
                        self.vector_output_size_assertion_check()
            else:
                raise ValueError(
                    f"{self.api} : Output error: the output should not be empty."
                )
        else:
            reshard_p2r_code = (
                UNSUPPORTED_RESHARD_OUTPUT_COMMENT_TEMPLATE.format(self.api)
            )

        return reshard_p2r_code

    def generate_return_code(self) -> str:
        return self.gene_return_code()

    def generate_auto_paralel_branch(self) -> str:
        # if no tensor input, do not genetate auto parallel branch
        if len(self.inputs['names']) == 0:
            return ""
        return MAIN_DIST_BRANCH_TEMPLATE.format(
            self.generate_infer_spmd_code(),
            self.generate_output_creation_code(),
            self.generate_infer_global_shape_code(),
            self.generate_kernel_selection_code(),
            self.generate_reshard_input_code(),
            self.generate_prepare_data_code(),
            self.generate_infer_meta_code(),
            self.generate_kernel_call_code(),
            self.generate_reshard_partial_out_to_replicated_code(),
            self.generate_return_code(),
        )

    def check_argument_whether_support_auto_parallel(self):
        for name in self.inputs['names']:
            if self.inputs['input_info'][name] not in [
                "const Tensor&",
                "const std::vector<Tensor>&",
                "const paddle::optional<Tensor>&",
                "const paddle::optional<std::vector<Tensor>>&",
            ]:
                return False
        for out_type in self.outputs['types']:
            if out_type not in ["Tensor", "std::vector<Tensor>"]:
                return False
        return True

    # override BaseAPI's method
    def gene_base_api_code(self, inplace_flag=False):
        # init status
        self.inplace_flag = inplace_flag
        self.dist_output_args = []
        self.dense_output_args = []
        # generate api body
        api_func_name = self.get_api_func_name()
        if inplace_flag and api_func_name[-1] != '_':
            api_func_name += '_'

        # All apis contains auto parallel branch default.
        # Auto parallel branch has following restrictions:
        # 1. doesn't support initialize ops now
        # 2. doesn't support stride/view api
        # 3. only for general forward and backward
        # 4. doesn't support double grad and triple grad
        # 5. for multi kernels functions, doesn't support sparse kernel
        if len(self.kernel['func']) > 1:
            kernel_dispatch_code = ''
            dist_branch_code = ""
            for kernel_name in self.kernel['func']:
                # Skip sparse kernels.
                if (
                    'sparse' not in kernel_name
                    and '_sr' not in kernel_name
                    and len(self.inputs['names']) > 0
                    and len(self.view_map) == 0
                    and self.check_argument_whether_support_auto_parallel()
                    and not self.api.endswith("_double_grad")
                    and not self.api.endswith("_triple_grad")
                ):
                    dist_branch_code += self.generate_auto_paralel_branch()
            kernel_dispatch_code += dist_branch_code
            for kernel_name in self.kernel['func']:
                kernel_dispatch_code += self.gene_dispatch_code(
                    kernel_name, inplace_flag
                )
            return API_IMPL_TEMPLATE.format(
                self.get_return_type(inplace_flag),
                api_func_name,
                self.get_define_args(inplace_flag),
                self.gene_kernel_select(),
                kernel_dispatch_code
                + DIPATCH_END_GUARD_TEMPLATE.format(self.api),
            )
        else:
            dist_branch_code = ""
            if (
                len(self.inputs['names']) > 0
                and len(self.view_map) == 0
                and self.check_argument_whether_support_auto_parallel()
                and not self.api.endswith("_double_grad")
                and not self.api.endswith("_triple_grad")
            ):
                dist_branch_code = self.generate_auto_paralel_branch()
            return API_IMPL_TEMPLATE.format(
                self.get_return_type(inplace_flag),
                api_func_name,
                self.get_define_args(inplace_flag),
                self.gene_kernel_select(),
                dist_branch_code
                + self.gen_kernel_code(
                    self.kernel['func'][0], '', inplace_flag
                ),
            )


def generate_api(
    api_yaml_path, is_fused_ops_yaml, header_file_path, source_file_path
):
    apis = []

    for each_api_yaml in api_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                apis.extend(api_list)

    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = (
        "paddle/phi/api/include/fused_api.h"
        if is_fused_ops_yaml is True
        else "paddle/phi/api/include/api.h"
    )
    # not all fused ops supoort dygraph
    if is_fused_ops_yaml is True:
        new_apis = [
            api
            for api in apis
            if "support_dygraph_mode" in api
            and api["support_dygraph_mode"] is True
        ]
        apis = new_apis

    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for api in apis:
        dist_foward_api = DistForwardAPI(api)
        if dist_foward_api.is_dygraph_api:
            dist_foward_api.is_dygraph_api = False

        header_file.write(dist_foward_api.gene_api_declaration())
        if is_fused_ops_yaml is True:
            source_file.write(dist_foward_api.gene_api_code())
        else:
            source_file.write(dist_foward_api.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    source_file.write(declare_extension_api())

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files'
    )
    parser.add_argument(
        '--api_yaml_path',
        help='path to api yaml file',
        nargs='+',
        default=['paddle/phi/api/yaml/ops.yaml'],
    )

    parser.add_argument(
        '--is_fused_ops_yaml',
        help='flag of fused ops yaml',
        action='store_true',
    )

    parser.add_argument(
        '--api_header_path',
        help='output of generated api header code file',
        default='paddle/phi/api/include/api.h',
    )

    parser.add_argument(
        '--api_source_path',
        help='output of generated api source code file',
        default='paddle/phi/api/lib/api.cc',
    )

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    is_fused_ops_yaml = options.is_fused_ops_yaml
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path

    generate_api(
        api_yaml_path, is_fused_ops_yaml, header_file_path, source_file_path
    )


if __name__ == '__main__':
    main()
