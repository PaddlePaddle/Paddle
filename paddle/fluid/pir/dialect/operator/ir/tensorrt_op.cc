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
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/tensorrt_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"

namespace paddle {
namespace dialect {

const char* TensorRTEngineOp::attributes_name[8] = {"engine",
                                                    "max_batch_size",
                                                    "workspace_size",
                                                    "allow_build_at_runtime",
                                                    "input_names",
                                                    "output_names",
                                                    "origin_output_rank",
                                                    "origin_outputs_dtype"};

OpInfoTuple TensorRTEngineOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("x",
                  "pir::VectorType<paddle::dialect::DenseTensorType>",
                  false,
                  false,
                  false,
                  false)};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("engine", "pir::PointerAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "max_batch_size", "pir::Int32Attribute", ""),
      paddle::dialect::OpAttributeInfo(
          "workspace_size", "pir::Int64Attribute", ""),
      paddle::dialect::OpAttributeInfo(
          "allow_build_at_runtime", "pir::BoolAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "input_names", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "output_names", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "origin_output_rank", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "origin_outputs_dtype", "pir::ArrayAttribute", "")};

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

void TensorRTEngineOp::Build(
    pir::Builder& builder,
    pir::OperationArgument& argument,
    pir::Value x,
    void* engine,
    int max_batch_size,
    int64_t workspace_size,
    bool allow_build_at_runtime,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    std::vector<int> origin_output_rank,
    std::vector<phi::DataType> origin_outputs_dtype,
    const std::vector<paddle::dialect::IrTensor>& outs_meta) {
  VLOG(4) << "Start build TensorRTEngineOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::Attribute attr_engine =
      pir::PointerAttribute::get(pir::IrContext::Instance(), engine);
  argument.AddAttribute("engine", attr_engine);
  pir::Attribute attr_max_batch_size =
      pir::Int32Attribute::get(pir::IrContext::Instance(), max_batch_size);
  argument.AddAttribute("max_batch_size", attr_max_batch_size);
  pir::Attribute attr_workspace_size =
      pir::Int64Attribute::get(pir::IrContext::Instance(), workspace_size);
  argument.AddAttribute("workspace_size", attr_workspace_size);
  pir::Attribute attr_allow_build_at_runtime = pir::BoolAttribute::get(
      pir::IrContext::Instance(), allow_build_at_runtime);
  argument.AddAttribute("allow_build_at_runtime", attr_allow_build_at_runtime);

  std::vector<pir::Attribute> input_names_tmp;
  input_names_tmp.reserve(input_names.size());
  for (const auto& v : input_names) {
    input_names_tmp.push_back(
        pir::StrAttribute::get(pir::IrContext::Instance(), v));
  }
  pir::Attribute attr_input_names =
      pir::ArrayAttribute::get(pir::IrContext::Instance(), input_names_tmp);
  argument.AddAttribute("input_names", attr_input_names);

  std::vector<pir::Attribute> output_names_tmp;
  output_names_tmp.reserve(output_names.size());
  for (const auto& v : output_names) {
    output_names_tmp.push_back(
        pir::StrAttribute::get(pir::IrContext::Instance(), v));
  }
  pir::Attribute attr_output_names =
      pir::ArrayAttribute::get(pir::IrContext::Instance(), output_names_tmp);
  argument.AddAttribute("output_names", attr_output_names);

  std::vector<pir::Attribute> origin_output_rank_tmp;
  origin_output_rank_tmp.reserve(origin_output_rank.size());
  for (const auto& v : origin_output_rank) {
    origin_output_rank_tmp.push_back(
        pir::Int32Attribute::get(pir::IrContext::Instance(), v));
  }
  pir::Attribute attr_origin_output_rank = pir::ArrayAttribute::get(
      pir::IrContext::Instance(), origin_output_rank_tmp);
  argument.AddAttribute("origin_output_rank", attr_origin_output_rank);

  std::vector<pir::Attribute> origin_outputs_dtype_tmp;
  origin_outputs_dtype_tmp.reserve(origin_outputs_dtype.size());
  for (const auto& v : origin_outputs_dtype) {
    origin_outputs_dtype_tmp.push_back(
        paddle::dialect::DataTypeAttribute::get(pir::IrContext::Instance(), v));
  }
  pir::Attribute attr_origin_outputs_dtype = pir::ArrayAttribute::get(
      pir::IrContext::Instance(), origin_outputs_dtype_tmp);
  argument.AddAttribute("origin_outputs_dtype", attr_origin_outputs_dtype);

  VLOG(4) << "Builder construction outputs";

  std::vector<pir::Type> argument_outputs;

  std::vector<pir::Type> out_types;
  for (size_t i = 0; i < static_cast<size_t>(outs_meta.size()); i++) {
    out_types.push_back(
        pir::DenseTensorType::get(pir::IrContext::Instance(),
                                  TransToIrDataType(outs_meta[i].dtype()),
                                  outs_meta[i].dims(),
                                  outs_meta[i].layout(),
                                  outs_meta[i].lod(),
                                  outs_meta[i].offset()));
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
                      phi::errors::InvalidArgument(
                          "Type validation failed for the 0th input, got %s.",
                          (*this)->operand_source(0).type()));
    if (auto vec_type =
            (*this)->operand_source(0).type().dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            vec_type[i].isa<pir::DenseTensorType>(),
            true,
            phi::errors::InvalidArgument(
                "Type validation failed for the 0th input, got %s.",
                (*this)->operand_source(0).type()));
      }
    }
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto& attributes = this->attributes();
    PADDLE_ENFORCE_GT(attributes.count("engine"),
                      0,
                      phi::errors::InvalidArgument("engine does not exist."));
    PADDLE_ENFORCE_EQ(
        attributes.at("engine").isa<pir::PointerAttribute>(),
        true,
        phi::errors::InvalidArgument(
            "Type of attribute: engine is not pir::PointerAttribute."));
    PADDLE_ENFORCE_GT(
        attributes.count("max_batch_size"),
        0,
        phi::errors::InvalidArgument("max_batch_size does not exist."));
    PADDLE_ENFORCE_EQ(
        attributes.at("max_batch_size").isa<pir::Int32Attribute>(),
        true,
        phi::errors::InvalidArgument(
            "Type of attribute: max_batch_size is not pir::Int32Attribute."));
    PADDLE_ENFORCE_GT(
        attributes.count("workspace_size"),
        0,
        phi::errors::InvalidArgument("workspace_size does not exist."));
    PADDLE_ENFORCE_EQ(
        attributes.at("workspace_size").isa<pir::Int64Attribute>(),
        true,
        phi::errors::InvalidArgument(
            "Type of attribute: workspace_size is not pir::Int64Attribute."));
    PADDLE_ENFORCE_GT(
        attributes.count("allow_build_at_runtime"),
        0,
        phi::errors::InvalidArgument("allow_build_at_runtime does not exist."));
    PADDLE_ENFORCE_EQ(
        attributes.at("allow_build_at_runtime").isa<pir::BoolAttribute>(),
        true,
        phi::errors::InvalidArgument(
            "Type of attribute: allow_build_at_runtime is not "
            "pir::BoolAttribute."));
    PADDLE_ENFORCE_GT(
        attributes.count("input_names"),
        0,
        phi::errors::InvalidArgument("input_names does not exist."));
    PADDLE_ENFORCE_EQ(
        attributes.at("input_names").isa<pir::ArrayAttribute>(),
        true,
        phi::errors::InvalidArgument(
            "Type of attribute: input_names is not pir::ArrayAttribute."));
    PADDLE_ENFORCE_GT(
        attributes.count("output_names"),
        0,
        phi::errors::InvalidArgument("output_names does not exist."));
    PADDLE_ENFORCE_EQ(
        attributes.at("output_names").isa<pir::ArrayAttribute>(),
        true,
        phi::errors::InvalidArgument(
            "Type of attribute: output_names is not pir::ArrayAttribute."));
    PADDLE_ENFORCE_GT(
        attributes.count("origin_output_rank"),
        0,
        phi::errors::InvalidArgument("origin_output_rank does not exist."));
    PADDLE_ENFORCE_EQ(
        attributes.at("origin_output_rank").isa<pir::ArrayAttribute>(),
        true,
        phi::errors::InvalidArgument("Type of attribute: origin_output_rank is "
                                     "not pir::ArrayAttribute."));
    PADDLE_ENFORCE_GT(
        attributes.count("origin_outputs_dtype"),
        0,
        phi::errors::InvalidArgument("origin_outputs_dtype does not exist."));
    PADDLE_ENFORCE_EQ(
        attributes.at("origin_outputs_dtype").isa<pir::ArrayAttribute>(),
        true,
        phi::errors::InvalidArgument("Type of attribute: origin_outputs_dtype "
                                     "is not pir::ArrayAttribute."));
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
                      phi::errors::InvalidArgument(
                          "Type validation failed for the 0th output."));
    if (auto vec_type = output_type.dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); i++) {
        PADDLE_ENFORCE_EQ(vec_type[i].isa<pir::DenseTensorType>(),
                          true,
                          phi::errors::InvalidArgument(
                              "Type validation failed for the 0th output."));
      }
    }
  }
  VLOG(4) << "End Verifying for: TensorRTEngineOp.";
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::TensorRTEngineOp)
