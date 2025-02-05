// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/tensorrt_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"

namespace paddle {
namespace dialect {

const char *TensorRTEngineOp::attributes_name[13] = {"engine_serialized_data",
                                                     "workspace_size",
                                                     "allow_build_at_runtime",
                                                     "input_names",
                                                     "output_names",
                                                     "outputs_rank",
                                                     "outputs_dtype",
                                                     "dynamic_shape_names",
                                                     "dynamic_shape_lens",
                                                     "min_input_shape_vector",
                                                     "max_input_shape_vector",
                                                     "opt_input_shape_vector",
                                                     "converter_debug_info"};

OpInfoTuple TensorRTEngineOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("x",
                  "pir::VectorType<paddle::dialect::DenseTensorType>",
                  false,
                  false,
                  false,
                  false)};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo(
          "engine_serialized_data", "pir::StrAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "workspace_size", "pir::Int64Attribute", ""),
      paddle::dialect::OpAttributeInfo(
          "allow_build_at_runtime", "pir::BoolAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "input_names", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "output_names", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "outputs_rank", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "outputs_dtype", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "dynamic_shape_names", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "dynamic_shape_lens", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "min_input_shape_vector", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "max_input_shape_vector", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "opt_input_shape_vector", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "converter_debug_info", "pir::StrAttribute", "")};

  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      OpOutputInfo("out",
                   "pir::VectorType<paddle::dialect::DenseTensorType>",
                   false,
                   false)};

  paddle::dialect::OpRunTimeInfo run_time_info =
      OpRunTimeInfo("", {""}, "", {""}, {}, {}, {}, {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "tensorrt_engine_op");
}

#define ADD_VEC_ATTRIBUTE(type, name)                                   \
  std::vector<pir::Attribute> name##_tmp;                               \
  name##_tmp.reserve(name.size());                                      \
  for (const auto &v : name) {                                          \
    name##_tmp.push_back(type::get(pir::IrContext::Instance(), v));     \
  }                                                                     \
  pir::Attribute attr_##name =                                          \
      pir::ArrayAttribute::get(pir::IrContext::Instance(), name##_tmp); \
  argument.AddAttribute(#name, attr_##name)

#define VERIFY_ATTRIBUTE(type, name)                              \
  PADDLE_ENFORCE_GT(                                              \
      attributes.count(#name),                                    \
      0,                                                          \
      common::errors::InvalidArgument(#name " does not exist.")); \
  PADDLE_ENFORCE_EQ(attributes.at(#name).isa<type>(),             \
                    true,                                         \
                    common::errors::InvalidArgument(              \
                        "Type of attribute: " #name " is not " #type))

void TensorRTEngineOp::Build(pir::Builder &builder,             // NOLINT
                             pir::OperationArgument &argument,  // NOLINT
                             pir::Value x,
                             paddle::platform::EngineParams trt_params,
                             std::vector<std::string> input_names,
                             std::vector<std::string> output_names,
                             std::vector<std::vector<int64_t>> outputs_shape,
                             std::vector<phi::DataType> outputs_dtype,
                             const std::string &converter_debug_info) {
  VLOG(4) << "Start build TensorRTEngineOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::Attribute attr_engine_serialized_data = pir::StrAttribute::get(
      pir::IrContext::Instance(), trt_params.engine_serialized_data);
  argument.AddAttribute("engine_serialized_data", attr_engine_serialized_data);
  pir::Attribute attr_workspace_size = pir::Int64Attribute::get(
      pir::IrContext::Instance(), trt_params.max_workspace_size);
  argument.AddAttribute("workspace_size", attr_workspace_size);
  pir::Attribute attr_allow_build_at_runtime = pir::BoolAttribute::get(
      pir::IrContext::Instance(), trt_params.allow_build_at_runtime);
  argument.AddAttribute("allow_build_at_runtime", attr_allow_build_at_runtime);

  std::vector<pir::Attribute> outputs_rank_tmp;
  outputs_rank_tmp.reserve(outputs_shape.size());
  for (const auto &v : outputs_shape) {
    outputs_rank_tmp.push_back(
        pir::Int32Attribute::get(pir::IrContext::Instance(), v.size()));
  }
  pir::Attribute attr_outputs_rank =
      pir::ArrayAttribute::get(pir::IrContext::Instance(), outputs_rank_tmp);
  argument.AddAttribute("outputs_rank", attr_outputs_rank);

  pir::Attribute attr_converter_debug_info =
      pir::StrAttribute::get(pir::IrContext::Instance(), converter_debug_info);
  argument.AddAttribute("converter_debug_info", attr_converter_debug_info);

  std::vector<std::string> dynamic_shape_names;
  std::vector<int> dynamic_shape_lens;
  std::vector<int> min_input_shape_vector;
  std::vector<int> max_input_shape_vector;
  std::vector<int> opt_input_shape_vector;
  for (const auto &it : trt_params.min_input_shape) {
    dynamic_shape_names.push_back(it.first);
    dynamic_shape_lens.push_back(it.second.size());
    for (const auto &value : it.second) {
      min_input_shape_vector.push_back(value);
    }
  }
  for (const auto &it : trt_params.max_input_shape) {
    for (const auto &value : it.second) {
      max_input_shape_vector.push_back(value);
    }
  }
  for (const auto &it : trt_params.optim_input_shape) {
    for (const auto &value : it.second) {
      opt_input_shape_vector.push_back(value);
    }
  }

  ADD_VEC_ATTRIBUTE(pir::StrAttribute, input_names);
  ADD_VEC_ATTRIBUTE(pir::StrAttribute, output_names);
  ADD_VEC_ATTRIBUTE(paddle::dialect::DataTypeAttribute, outputs_dtype);
  ADD_VEC_ATTRIBUTE(pir::StrAttribute, dynamic_shape_names);
  ADD_VEC_ATTRIBUTE(pir::Int32Attribute, dynamic_shape_lens);
  ADD_VEC_ATTRIBUTE(pir::Int32Attribute, min_input_shape_vector);
  ADD_VEC_ATTRIBUTE(pir::Int32Attribute, max_input_shape_vector);
  ADD_VEC_ATTRIBUTE(pir::Int32Attribute, opt_input_shape_vector);

  VLOG(4) << "Builder construction outputs";

  std::vector<pir::Type> argument_outputs;
  std::vector<pir::Type> out_types;
  for (size_t i = 0; i < static_cast<size_t>(outputs_shape.size()); i++) {
    if (outputs_dtype[i] == phi::DataType::UNDEFINED) {
      out_types.push_back(pir::Type());
    } else {
      out_types.push_back(pir::DenseTensorType::get(
          pir::IrContext::Instance(),
          TransToIrDataType(outputs_dtype[i]),
          phi::DDim(outputs_shape[i].data(), outputs_shape[i].size()),
          phi::DataLayout::kNCHW,
          phi::LegacyLoD(),
          0));
    }
  }
  pir::Type out_vector_type =
      pir::VectorType::get(pir::IrContext::Instance(), out_types);
  argument_outputs.push_back(out_vector_type);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void TensorRTEngineOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "TensorRTEngineOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(input_size,
                      1,
                      common::errors::InvalidArgument(
                          "The size of inputs must be equal to 1."));
    PADDLE_ENFORCE_EQ((*this)->operand_source(0).type().isa<pir::VectorType>(),
                      true,
                      common::errors::InvalidArgument(
                          "Type validation failed for the 0th input, got %s.",
                          (*this)->operand_source(0).type()));
    if (auto vec_type =
            (*this)->operand_source(0).type().dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            vec_type[i].isa<pir::DenseTensorType>(),
            true,
            common::errors::InvalidArgument(
                "Type validation failed for the 0th input, got %s.",
                (*this)->operand_source(0).type()));
      }
    }
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    VERIFY_ATTRIBUTE(pir::StrAttribute, engine_serialized_data);
    VERIFY_ATTRIBUTE(pir::Int64Attribute, workspace_size);
    VERIFY_ATTRIBUTE(pir::BoolAttribute, allow_build_at_runtime);
    VERIFY_ATTRIBUTE(pir::ArrayAttribute, input_names);
    VERIFY_ATTRIBUTE(pir::ArrayAttribute, output_names);
    VERIFY_ATTRIBUTE(pir::ArrayAttribute, outputs_rank);
    VERIFY_ATTRIBUTE(pir::ArrayAttribute, outputs_dtype);
    VERIFY_ATTRIBUTE(pir::ArrayAttribute, dynamic_shape_names);
    VERIFY_ATTRIBUTE(pir::ArrayAttribute, dynamic_shape_lens);
    VERIFY_ATTRIBUTE(pir::ArrayAttribute, min_input_shape_vector);
    VERIFY_ATTRIBUTE(pir::ArrayAttribute, max_input_shape_vector);
    VERIFY_ATTRIBUTE(pir::ArrayAttribute, opt_input_shape_vector);
    VERIFY_ATTRIBUTE(pir::StrAttribute, converter_debug_info);
  }

  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(output_size,
                      1,
                      common::errors::InvalidArgument(
                          "The size of outputs must be equal to 1."));
    auto output_type = (*this)->result(0).type();

    PADDLE_ENFORCE_EQ(output_type.isa<pir::VectorType>(),
                      true,
                      common::errors::InvalidArgument(
                          "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: TensorRTEngineOp.";
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::TensorRTEngineOp)
