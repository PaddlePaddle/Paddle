// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <any>

#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/fluid/pir/drr/src/attr_type_uilts.h"
#include "paddle/fluid/pir/drr/src/ir_operation_factory.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace drr {

void OperationFactory::RegisterManualOpCreator() {
  RegisterOperationCreator(
      "pd_op.fused_gemm_epilogue",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::dialect::FusedGemmEpilogueOp>(
            inputs[0], inputs[1], inputs[2], attrs);
      });
  RegisterOperationCreator(
      "pd_op.fused_gemm_epilogue_grad",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::dialect::FusedGemmEpilogueGradOp>(
            inputs[0], inputs[1], inputs[2], inputs[3], attrs);
      });
  RegisterOperationCreator("builtin.combine",
                           [](const std::vector<pir::Value>& inputs,
                              const pir::AttributeMap& attrs,
                              pir::PatternRewriter& rewriter) {
                             return rewriter.Build<pir::CombineOp>(inputs);
                           });
  RegisterOperationCreator(
      "pd_op.scale",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::dialect::ScaleOp>(
            inputs[0],
            inputs[1],
            attrs.at("bias").dyn_cast<pir::FloatAttribute>().data(),
            attrs.at("bias_after_scale").dyn_cast<pir::BoolAttribute>().data());
      });

#ifdef PADDLE_WITH_DNNL
  RegisterOperationCreator(
      "onednn_op.conv2d_transpose_bias",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 4) {
          IR_ENFORCE(
              attributes.find("strides") != attributes.end(),
              "'strides' Attribute is expected for Conv2dTransposeBiasOp. ");
          std::vector<int> strides;
          for (size_t i = 0;
               i <
               attributes.at("strides").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            strides.push_back(attributes.at("strides")
                                  .dyn_cast<pir::ArrayAttribute>()
                                  .at(i)
                                  .dyn_cast<pir::Int32Attribute>()
                                  .data());
          }

          IR_ENFORCE(
              attributes.find("paddings") != attributes.end(),
              "'paddings' Attribute is expected for Conv2dTransposeBiasOp. ");
          std::vector<int> paddings;
          for (size_t i = 0;
               i <
               attributes.at("paddings").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            paddings.push_back(attributes.at("paddings")
                                   .dyn_cast<pir::ArrayAttribute>()
                                   .at(i)
                                   .dyn_cast<pir::Int32Attribute>()
                                   .data());
          }

          IR_ENFORCE(attributes.find("output_padding") != attributes.end(),
                     "'output_padding' Attribute is expected for "
                     "Conv2dTransposeBiasOp. ");
          std::vector<int> output_padding;
          for (size_t i = 0; i < attributes.at("output_padding")
                                     .dyn_cast<pir::ArrayAttribute>()
                                     .size();
               i++) {
            output_padding.push_back(attributes.at("output_padding")
                                         .dyn_cast<pir::ArrayAttribute>()
                                         .at(i)
                                         .dyn_cast<pir::Int32Attribute>()
                                         .data());
          }

          IR_ENFORCE(attributes.find("padding_algorithm") != attributes.end(),
                     "'padding_algorithm' Attribute is expected for "
                     "Conv2dTransposeBiasOp. ");
          std::string padding_algorithm = attributes.at("padding_algorithm")
                                              .dyn_cast<pir::StrAttribute>()
                                              .AsString();

          IR_ENFORCE(
              attributes.find("groups") != attributes.end(),
              "'groups' Attribute is expected for Conv2dTransposeBiasOp. ");
          int groups =
              attributes.at("groups").dyn_cast<pir::Int32Attribute>().data();

          IR_ENFORCE(
              attributes.find("dilations") != attributes.end(),
              "'dilations' Attribute is expected for Conv2dTransposeBiasOp. ");
          std::vector<int> dilations;
          for (size_t i = 0; i < attributes.at("dilations")
                                     .dyn_cast<pir::ArrayAttribute>()
                                     .size();
               i++) {
            dilations.push_back(attributes.at("dilations")
                                    .dyn_cast<pir::ArrayAttribute>()
                                    .at(i)
                                    .dyn_cast<pir::Int32Attribute>()
                                    .data());
          }

          IR_ENFORCE(attributes.find("data_format") != attributes.end(),
                     "'data_format' Attribute is expected for "
                     "Conv2dTransposeBiasOp. ");
          std::string data_format = attributes.at("data_format")
                                        .dyn_cast<pir::StrAttribute>()
                                        .AsString();

          IR_ENFORCE(
              attributes.find("is_test") != attributes.end(),
              "'is_test' Attribute is expected for Conv2dTransposeBiasOp. ");
          bool is_test =
              attributes.at("is_test").dyn_cast<pir::BoolAttribute>().data();

          return rewriter.Build<paddle::onednn::dialect::Conv2dTransposeBiasOp>(
              inputs[0],
              inputs[1],
              inputs[2],
              inputs[3],
              strides,
              paddings,
              output_padding,
              padding_algorithm,
              groups,
              dilations,
              data_format,
              is_test);
        }

        return rewriter.Build<paddle::onednn::dialect::Conv2dTransposeBiasOp>(
            inputs[0], inputs[1], inputs[2], attrs);
      });
#endif
}

pir::Attribute CreateIrAttribute(const std::any& obj) {
  if (obj.type() == typeid(bool)) {
    return IrAttributeCreator<bool>()(std::any_cast<bool>(obj));
  } else if (obj.type() == typeid(int32_t)) {
    return IrAttributeCreator<int32_t>()(std::any_cast<int32_t>(obj));
  } else if (obj.type() == typeid(int64_t)) {
    return IrAttributeCreator<int64_t>()(std::any_cast<int64_t>(obj));
  } else if (obj.type() == typeid(float)) {
    return IrAttributeCreator<float>()(std::any_cast<float>(obj));
  } else if (obj.type() == typeid(std::string)) {
    return IrAttributeCreator<std::string>()(std::any_cast<std::string>(obj));
  } else if (obj.type() == typeid(const char*)) {
    return IrAttributeCreator<std::string>()(std::any_cast<const char*>(obj));
  } else if (obj.type() == typeid(phi::DataType)) {
    return IrAttributeCreator<phi::DataType>()(
        std::any_cast<phi::DataType>(obj));
  } else if (obj.type() == typeid(phi::Place)) {
    return IrAttributeCreator<phi::Place>()(std::any_cast<phi::Place>(obj));
  } else if (obj.type() == typeid(std::vector<int32_t>)) {  // NOLINT
    return IrAttributeCreator<std::vector<int32_t>>()(
        std::any_cast<std::vector<int32_t>>(obj));
  } else if (obj.type() == typeid(std::vector<int64_t>)) {
    return IrAttributeCreator<std::vector<int64_t>>()(
        std::any_cast<std::vector<int64_t>>(obj));
  } else if (obj.type() == typeid(std::vector<float>)) {
    return IrAttributeCreator<std::vector<float>>()(
        std::any_cast<std::vector<float>>(obj));
  } else if (obj.type() == typeid(phi::IntArray)) {
    return IrAttributeCreator<phi::IntArray>()(
        std::any_cast<phi::IntArray>(obj));
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Type error. CreateIrAttribute for type(%s) "
                                   "is unimplemented CreateInCurrently.",
                                   obj.type().name()));
  }
}

pir::AttributeMap CreateAttributeMap(const OpCall& op_call,
                                     const MatchContextImpl& src_match_ctx) {
  pir::AttributeMap attr_map;
  for (const auto& kv : op_call.attributes()) {
    std::visit(
        [&](auto&& arg) {
          if constexpr (std::is_same_v<std::decay_t<decltype(arg)>,
                                       NormalAttribute>) {
            attr_map[kv.first] = src_match_ctx.GetIrAttr(arg.name());
          }
          if constexpr (std::is_same_v<std::decay_t<decltype(arg)>,
                                       ComputeAttribute>) {
            MatchContext ctx(std::make_shared<MatchContextImpl>(src_match_ctx));
            attr_map[kv.first] =
                CreateIrAttribute(arg.attr_compute_func()(ctx));
          }
        },
        kv.second);
  }
  return attr_map;
}

pir::Value GetIrValueByDrrTensor(const Tensor& tensor,
                                 const MatchContextImpl& res_match_ctx) {
  if (tensor.is_none()) {
    return pir::Value{};
  }
  return res_match_ctx.GetIrValue(tensor.name());
}

std::vector<pir::Value> GetIrValuesByDrrTensors(
    const std::vector<const Tensor*>& tensors,
    const MatchContextImpl& res_match_ctx) {
  std::vector<pir::Value> ir_values;
  ir_values.reserve(tensors.size());
  for (const auto* tensor : tensors) {
    ir_values.push_back(GetIrValueByDrrTensor(*tensor, res_match_ctx));
  }
  return ir_values;
}

void BindIrOutputs(const OpCall& op_call,
                   pir::Operation* op,
                   MatchContextImpl* match_ctx) {
  for (size_t i = 0; i < op_call.outputs().size(); ++i) {
    match_ctx->BindIrValue(op_call.outputs()[i]->name(), op->result(i));
  }
}

pir::Operation* CreateOperation(const OpCall& op_call,
                                const MatchContextImpl& src_match_ctx,
                                pir::PatternRewriter& rewriter,  // NOLINT
                                MatchContextImpl* res_match_ctx) {
  VLOG(6) << "Drr create [" << op_call.name() << "] op...";
  const auto& inputs = op_call.inputs();
  std::vector<pir::Value> ir_values =
      GetIrValuesByDrrTensors(inputs, *res_match_ctx);
  pir::Operation* op = OperationFactory::Instance().CreateOperation(
      op_call.name(),
      ir_values,
      CreateAttributeMap(op_call, src_match_ctx),
      rewriter);
  BindIrOutputs(op_call, op, res_match_ctx);
  VLOG(6) << "Drr create [" << op_call.name() << " @" << op << "] op done.";
  return op;
}

}  // namespace drr
}  // namespace paddle
