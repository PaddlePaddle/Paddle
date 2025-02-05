// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/helper.h"
#include <cstdint>

#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/drr/src/ir_operation_factory.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

namespace paddle::inference {

template <>
std::string to_string<std::vector<float>>(
    const std::vector<std::vector<float>> &vec) {
  std::stringstream ss;
  for (const auto &piece : vec) {
    ss << to_string(piece) << "\n";
  }
  return ss.str();
}

template <>
std::string to_string<std::vector<std::vector<float>>>(
    const std::vector<std::vector<std::vector<float>>> &vec) {
  std::stringstream ss;
  for (const auto &line : vec) {
    for (const auto &rcd : line) {
      ss << to_string(rcd) << ";\t";
    }
    ss << '\n';
  }
  return ss.str();
}

void RegisterAllCustomOperator(bool use_pir) {
  const auto &meta_info_map = OpMetaInfoMap::Instance().GetMap();
  for (auto &pair : meta_info_map) {
    if (use_pir) {
      auto *custom_dialect =
          ::pir::IrContext::Instance()
              ->GetOrRegisterDialect<paddle::dialect::CustomOpDialect>();
      if (custom_dialect->HasRegistered(pair.first)) {
        VLOG(3) << "The operator `" << pair.first
                << "` has been registered. "
                   "Therefore, we will not repeat the registration here.";
        continue;
      }
      for (const auto &meta_info : pair.second) {
        VLOG(3) << "register pir custom op: " << pair.first;
        custom_dialect->RegisterCustomOp(meta_info);
      }

      std::string pir_op_name =
          paddle::framework::kCustomDialectPrefix + pair.first;
      paddle::drr::OperationFactory::Instance().RegisterOperationCreator(
          pir_op_name,
          [pair, pir_op_name](
              const std::vector<::pir::Value> &inputs,
              const ::pir::AttributeMap &attrs,
              ::pir::PatternRewriter &rewriter) mutable -> ::pir::Operation * {
            const auto &meta_inputs =
                paddle::OpMetaInfoHelper::GetInputs(pair.second[0]);
            const auto &meta_attrs =
                paddle::OpMetaInfoHelper::GetAttrs(pair.second[0]);
            const auto &meta_outputs =
                paddle::OpMetaInfoHelper::GetOutputs(pair.second[0]);
            const auto &inplace_map =
                paddle::OpMetaInfoHelper::GetInplaceMap(pair.second[0]);
            const auto &inplace_reverse_map =
                paddle::OpMetaInfoHelper::GetInplaceReverseMap(pair.second[0]);
            auto infershape_func =
                OpMetaInfoHelper::GetInferShapeFn(pair.second[0]);
            auto inferdtype_func =
                OpMetaInfoHelper::GetInferDtypeFn(pair.second[0]);

            PADDLE_ENFORCE_EQ(
                meta_inputs.size(),
                inputs.size(),
                common::errors::InvalidArgument(
                    "The number of inputs for the custom operator [%s] given "
                    "in the Pattern needs to be consistent with the number at "
                    "implementation time.",
                    pir_op_name));
            PADDLE_ENFORCE_EQ(
                meta_attrs.size(),
                attrs.size(),
                common::errors::InvalidArgument(
                    "The number of attrs for the custom operator [%s] given "
                    "in the Pattern needs to be consistent with the number at "
                    "implementation time.",
                    pir_op_name));

            if (!inplace_map.empty()) {
              pir_op_name += "_";
            }
            ::pir::OperationArgument argument(
                rewriter.ir_context()->GetRegisteredOpInfo(pir_op_name));
            argument.attributes = attrs;
            argument.inputs = inputs;

            std::vector<pir::Type> argument_outputs;
            std::vector<std::vector<int64_t>> input_shapes;
            std::vector<DataType> input_dtypes;
            std::unordered_map<std::string, int> input_name2id_map;
            std::vector<std::vector<std::vector<int64_t>>> vec_input_shapes;
            std::vector<std::vector<DataType>> vec_input_dtypes;
            std::unordered_map<std::string, int> vec_input_name2id_map;
            std::vector<paddle::any> custom_attrs;
            int input_index = 0;
            int vec_input_index = 0;

            for (size_t i = 0; i < meta_inputs.size(); ++i) {
              const auto &meta_input = meta_inputs.at(i);
              if (!inputs[i]) {
                VLOG(6) << "Add un-initialized tensor because the optional "
                           "input is None.";
                if (paddle::framework::detail::IsDuplicableVar(meta_input)) {
                  std::vector<std::vector<int64_t>> vec_input_shape;
                  std::vector<DataType> vec_input_dtype;
                  vec_input_shapes.emplace_back(vec_input_shape);
                  vec_input_dtypes.emplace_back(vec_input_dtype);
                  vec_input_name2id_map[meta_inputs[i]] = vec_input_index;
                  vec_input_index++;
                } else {
                  std::vector<int64_t> input_shape;
                  DataType input_dtype = DataType::UNDEFINED;
                  input_shapes.emplace_back(input_shape);
                  input_dtypes.emplace_back(input_dtype);
                  input_name2id_map[meta_inputs[i]] = input_index;
                  input_index++;
                }
                continue;
              }
              if (paddle::framework::detail::IsDuplicableVar(meta_input)) {
                PADDLE_ENFORCE_EQ(
                    inputs[i].type().isa<::pir::VectorType>(),
                    true,
                    common::errors::InvalidArgument(
                        "The [%d] input of the custom operator [%s] "
                        "should be a pir::VectorType.",
                        i,
                        pir_op_name));
                std::vector<std::vector<int64_t>> tmp_input_shapes;
                std::vector<phi::DataType> tmp_input_dtypes;
                vec_input_name2id_map[meta_inputs[i]] = vec_input_index;
                vec_input_index++;
                auto input_value_types =
                    inputs[i].type().dyn_cast<::pir::VectorType>().data();
                for (auto &input_value_type : input_value_types) {
                  auto input_tensor =
                      input_value_type
                          .dyn_cast<paddle::dialect::DenseTensorType>();
                  tmp_input_shapes.push_back(
                      phi::vectorize(input_tensor.dims()));
                  tmp_input_dtypes.push_back(
                      paddle::dialect::TransToPhiDataType(
                          input_tensor.dtype()));
                }
                vec_input_shapes.push_back(tmp_input_shapes);
                vec_input_dtypes.push_back(tmp_input_dtypes);
              } else {
                input_name2id_map[meta_inputs[i]] = input_index;
                input_index++;
                auto input_tensor =
                    inputs[i]
                        .type()
                        .dyn_cast<paddle::dialect::DenseTensorType>();
                input_shapes.push_back(phi::vectorize(input_tensor.dims()));
                input_dtypes.push_back(
                    paddle::dialect::TransToPhiDataType(input_tensor.dtype()));
              }
            }

            for (const auto &meta_attr : meta_attrs) {
              auto attr_name_and_type = paddle::ParseAttrStr(meta_attr);
              auto attr_name = attr_name_and_type[0];
              auto attr_type = attr_name_and_type[1];
              PADDLE_ENFORCE_EQ(attrs.count(attr_name),
                                true,
                                common::errors::InvalidArgument(
                                    "The attr [%s] in the custom operator [%s] "
                                    "specified in the Pattern needs to be "
                                    "consistent with the implementation",
                                    attr_name,
                                    pir_op_name));
              VLOG(6) << "Custom operator add attrs " << attr_name
                      << " to CustomOpKernelContext. Attribute type = "
                      << attr_type;
              if (attr_type == "bool") {
                auto bool_attr =
                    attrs.at(attr_name).dyn_cast<::pir::BoolAttribute>().data();
                custom_attrs.emplace_back(bool_attr);
              } else if (attr_type == "int") {
                int int_attr = attrs.at(attr_name)
                                   .dyn_cast<::pir::Int32Attribute>()
                                   .data();
                custom_attrs.emplace_back(int_attr);
              } else if (attr_type == "float") {
                float float_attr = attrs.at(attr_name)
                                       .dyn_cast<::pir::FloatAttribute>()
                                       .data();
                custom_attrs.emplace_back(float_attr);
              } else if (attr_type == "int64_t") {
                int64_t long_attr = attrs.at(attr_name)
                                        .dyn_cast<::pir::Int64Attribute>()
                                        .data();
                custom_attrs.emplace_back(long_attr);
              } else if (attr_type == "std::string") {
                std::string str_attr = attrs.at(attr_name)
                                           .dyn_cast<::pir::StrAttribute>()
                                           .AsString();
                custom_attrs.emplace_back(str_attr);
              } else if (attr_type == "std::vector<int>") {
                auto vec_attr = attrs.at(attr_name)
                                    .dyn_cast<::pir::ArrayAttribute>()
                                    .AsVector();
                std::vector<int> vec_int_attr;
                for (const auto &int_attr : vec_attr) {
                  vec_int_attr.push_back(
                      int_attr.dyn_cast<::pir::Int32Attribute>().data());
                }
                custom_attrs.emplace_back(vec_int_attr);
              } else if (attr_type == "std::vector<float>") {
                auto vec_attr = attrs.at(attr_name)
                                    .dyn_cast<::pir::ArrayAttribute>()
                                    .AsVector();
                std::vector<float> vec_float_attr;
                for (const auto &float_attr : vec_attr) {
                  vec_float_attr.push_back(
                      float_attr.dyn_cast<::pir::FloatAttribute>().data());
                }
                custom_attrs.emplace_back(vec_float_attr);
              } else if (attr_type == "std::vector<int64_t>") {
                auto vec_attr = attrs.at(attr_name)
                                    .dyn_cast<::pir::ArrayAttribute>()
                                    .AsVector();
                std::vector<int64_t> vec_long_attr;
                for (const auto &long_attr : vec_attr) {
                  vec_long_attr.push_back(
                      long_attr.dyn_cast<::pir::Int64Attribute>().data());
                }
                custom_attrs.emplace_back(vec_long_attr);
              } else if (attr_type == "std::vector<std::string>") {
                auto vec_attr = attrs.at(attr_name)
                                    .dyn_cast<::pir::ArrayAttribute>()
                                    .AsVector();
                std::vector<std::string> vec_string_attr;
                for (const auto &string_attr : vec_attr) {
                  vec_string_attr.push_back(
                      string_attr.dyn_cast<::pir::StrAttribute>().AsString());
                }
                custom_attrs.emplace_back(vec_string_attr);
              } else {
                PADDLE_THROW(common::errors::Unimplemented(
                    "Unsupported `%s` type value as custom attribute now. "
                    "Supported data types include `bool`, `int`, `float`, "
                    "`int64_t`, `std::string`, `std::vector<int>`, "
                    "`std::vector<float>`, `std::vector<int64_t>`, "
                    "`std::vector<std::string>`, Please check whether "
                    "the attribute data type and data type string are matched.",
                    attr_type));
              }
            }

            paddle::framework::CheckDefaultInferShapeDtype(
                infershape_func, inferdtype_func, pair.second[0]);
            std::vector<std::vector<int64_t>> output_shapes =
                paddle::framework::RunInferShape(infershape_func,
                                                 pair.second[0],
                                                 input_shapes,
                                                 input_name2id_map,
                                                 vec_input_shapes,
                                                 vec_input_name2id_map,
                                                 custom_attrs);
            std::vector<phi::DataType> output_dtypes =
                paddle::framework::RunInferDtype(inferdtype_func,
                                                 pair.second[0],
                                                 input_dtypes,
                                                 input_name2id_map,
                                                 vec_input_dtypes,
                                                 vec_input_name2id_map,
                                                 custom_attrs);

            size_t all_values_num = 0;
            // output name -> value num (that output should hold)
            std::unordered_map<std::string, size_t> output_name2value_num;
            for (const auto &output : meta_outputs) {
              if (paddle::framework::detail::IsDuplicableVar(output)) {
                PADDLE_ENFORCE_NE(inplace_reverse_map.find(output),
                                  inplace_reverse_map.end(),
                                  common::errors::InvalidArgument(
                                      "Only support vector output that is set "
                                      "for inplace, Please use "
                                      "`SetInplaceMap` in your output when "
                                      "registry custom operator."));
                const auto &input = inplace_reverse_map.at(output);
                auto index = vec_input_name2id_map[input];
                auto &vec_input_shape = vec_input_shapes[index];
                output_name2value_num[output] = vec_input_shape.size();
              } else {
                if (inplace_reverse_map.find(output) !=
                    inplace_reverse_map.end()) {
                  const auto &input = inplace_reverse_map.at(output);
                  auto index = input_name2id_map[input];
                  // input_shapes[index] is dim of tensor, if the dim doesn't
                  // have element, it must be a optional tensor that is None in
                  // custom operator
                  output_name2value_num[output] =
                      input_shapes[index].empty() ? 0 : 1;
                } else {
                  output_name2value_num[output]++;
                }
              }
              all_values_num += output_name2value_num[output];
            }

            PADDLE_ENFORCE_EQ(output_shapes.size(),
                              all_values_num,
                              common::errors::InvalidArgument(
                                  "The number of output shapes "
                                  "after running custom operator's "
                                  "InferShapeFunc is wrong, "
                                  "expected contains %d Tensors' "
                                  "shape, but actually contains %d "
                                  "Tensors' shape",
                                  all_values_num,
                                  output_shapes.size()));

            PADDLE_ENFORCE_EQ(output_dtypes.size(),
                              all_values_num,
                              common::errors::InvalidArgument(
                                  "The number of output dtypes "
                                  "after running custom operator's "
                                  "InferDtypeFunc is wrong, "
                                  "expected contains %d Tensors' "
                                  "dtype, but actually contains %d "
                                  "Tensors' dtype",
                                  all_values_num,
                                  output_dtypes.size()));

            size_t value_index = 0;
            for (const auto &output : meta_outputs) {
              auto value_num = output_name2value_num[output];
              if (value_num == 0) {
                // Optional value condition
                pir::Type out_type;
                argument_outputs.push_back(out_type);
                continue;
              }
              if (paddle::framework::detail::IsDuplicableVar(output)) {
                auto value_num = output_name2value_num[output];
                std::vector<pir::Type> out_types;
                for (size_t j = 0; j < value_num; ++j) {
                  auto ddims = phi::make_ddim(output_shapes[value_index]);
                  auto dtype = output_dtypes[value_index];
                  phi::DataLayout layout{DataLayout::NCHW};
                  phi::LegacyLoD lod;
                  out_types.push_back(paddle::dialect::DenseTensorType::get(
                      pir::IrContext::Instance(),
                      paddle::dialect::TransToIrDataType(dtype),
                      ddims,
                      layout,
                      lod,
                      0));
                  value_index++;
                }
                pir::Type out_vector_type =
                    pir::VectorType::get(pir::IrContext::Instance(), out_types);
                argument_outputs.push_back(out_vector_type);
              } else {
                auto ddims = phi::make_ddim(output_shapes[value_index]);
                auto dtype = output_dtypes[value_index];
                phi::DataLayout layout{DataLayout::NCHW};
                phi::LegacyLoD lod;
                auto out_type = paddle::dialect::DenseTensorType::get(
                    pir::IrContext::Instance(),
                    paddle::dialect::TransToIrDataType(dtype),
                    ddims,
                    layout,
                    lod,
                    0);
                argument_outputs.push_back(out_type);
                value_index++;
              }
            }

            argument.AddOutputs(argument_outputs.begin(),
                                argument_outputs.end());
            ::pir::PassStopGradientsDefaultly(argument);
            return rewriter.Build(std::move(argument));
          });
    }
    const auto &all_op_kernels{framework::OperatorWithKernel::AllOpKernels()};
    if (all_op_kernels.find(pair.first) == all_op_kernels.end()) {
      framework::RegisterOperatorWithMetaInfo(pair.second);
    } else {
      VLOG(3) << "The operator `" << pair.first
              << "` has been registered. Therefore, we will not repeat the "
                 "registration here.";
    }
  }
}

void InitGflagsFromEnv() {
  // support set gflags from environment.
  std::vector<std::string> gflags;
  const phi::ExportedFlagInfoMap &env_map = phi::GetExportedFlagInfoMap();
  std::ostringstream os;
  for (auto &pair : env_map) {
    os << pair.second.name << ",";
  }
  std::string tryfromenv_str = os.str();
  if (!tryfromenv_str.empty()) {
    tryfromenv_str.pop_back();
    tryfromenv_str = "--tryfromenv=" + tryfromenv_str;
    gflags.push_back(tryfromenv_str);
  }
  framework::InitGflags(gflags);
}

}  // namespace paddle::inference
