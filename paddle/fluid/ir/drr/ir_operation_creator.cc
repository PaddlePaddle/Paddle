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

#include "paddle/fluid/ir/drr/ir_operation_creator.h"

#include <any>

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_manual_op.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/fluid/ir/drr/attr_type_uilts.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/value.h"

namespace ir {
namespace drr {

Value GetIrValueByDrrTensor(const Tensor& tensor,
                            const MatchContextImpl& res_match_ctx) {
  if (tensor.is_none()) {
    return Value{};
  }
  return res_match_ctx.GetIrValue(tensor.name()).get();
}

std::vector<Value> GetIrValuesByDrrTensors(
    const std::vector<const Tensor*>& tensors,
    const MatchContextImpl& res_match_ctx) {
  std::vector<Value> ir_values;
  ir_values.reserve(tensors.size());
  for (const auto* tensor : tensors) {
    ir_values.push_back(GetIrValueByDrrTensor(*tensor, res_match_ctx));
  }
  return ir_values;
}

static ir::Attribute CreateIrAttribute(const std::any& obj) {
  if (obj.type() == typeid(bool)) {
    return IrAttrbuteCreator<bool>()(std::any_cast<bool>(obj));
  } else if (obj.type() == typeid(int32_t)) {
    return IrAttrbuteCreator<int32_t>()(std::any_cast<int32_t>(obj));
  } else if (obj.type() == typeid(int64_t)) {
    return IrAttrbuteCreator<int64_t>()(std::any_cast<int64_t>(obj));
  } else if (obj.type() == typeid(float)) {
    return IrAttrbuteCreator<float>()(std::any_cast<float>(obj));
  } else if (obj.type() == typeid(std::string)) {
    return IrAttrbuteCreator<const std::string&>()(
        std::any_cast<std::string>(obj));
  } else if (obj.type() == typeid(const char*)) {
    return IrAttrbuteCreator<const std::string&>()(
        std::any_cast<const char*>(obj));
  } else if (obj.type() == typeid(phi::DataType)) {
    return IrAttrbuteCreator<phi::DataType>()(
        std::any_cast<phi::DataType>(obj));
  } else if (obj.type() == typeid(phi::Place)) {
    return IrAttrbuteCreator<phi::Place>()(std::any_cast<phi::Place>(obj));
  } else if (obj.type() == typeid(std::vector<int>)) {
    return IrAttrbuteCreator<std::vector<int>>()(
        std::any_cast<std::vector<int>>(obj));
  } else if (obj.type() == typeid(std::vector<int64_t>)) {
    return IrAttrbuteCreator<std::vector<int64_t>>()(
        std::any_cast<std::vector<int64_t>>(obj));
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Type error. CreateIrAttribute for type(%s) "
                                   "is unimplemented CreateInCurrently.",
                                   obj.type().name()));
  }
}

ir::AttributeMap CreateAttributeMap(const OpCall& op_call,
                                    const MatchContextImpl& src_match_ctx) {
  ir::AttributeMap attr_map;
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

template <typename T>
T GetAttr(const std::string& attr_name,
          const OpCall& op_call,
          const MatchContextImpl& src_match_ctx) {
  IR_ENFORCE(op_call.attributes().count(attr_name),
             "Attr [%s] must exists in OpCall's attributes map.",
             attr_name);
  const auto& attr = op_call.attributes().at(attr_name);
  if (std::holds_alternative<NormalAttribute>(attr)) {
    return src_match_ctx.Attr<T>(std::get<NormalAttribute>(attr).name());
  } else if (std::holds_alternative<ComputeAttribute>(attr)) {
    MatchContext ctx(std::make_shared<MatchContextImpl>(src_match_ctx));
    try {
      return std::any_cast<T>(
          std::get<ComputeAttribute>(attr).attr_compute_func()(ctx));
    } catch (const std::bad_any_cast& e) {
      IR_THROW("Incorrect attribute [%s] with [%s] type.",
               attr_name,
               typeid(T).name());
    }
  } else {
    IR_THROW("Unknown attrbute type for : %s.", attr_name);
  }
}

void BindIrOutputs(const OpCall& op_call,
                   Operation* op,
                   MatchContextImpl* match_ctx) {
  for (size_t i = 0; i < op_call.outputs().size(); ++i) {
    std::shared_ptr<IrValue> ir_value = nullptr;
    if (op->result(i)) {
      ir_value = std::make_shared<IrValue>(op->result(i));
    }
    match_ctx->BindIrValue(op_call.outputs()[i]->name(), ir_value);
  }
}

void AutoSetInsertionPoint(const std::vector<Value>& ir_values,
                           ir::PatternRewriter& rewriter) {}  // NOLINT

Operation* CreateOperation(const OpCall& op_call,
                           const MatchContextImpl& src_match_ctx,
                           ir::PatternRewriter& rewriter,  // NOLINT
                           MatchContextImpl* res_match_ctx) {
  VLOG(6) << "Drr create [" << op_call.name() << "] op...";
  ir::Operation* op{nullptr};
  if (op_call.name() == "pd.reshape") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    // TODO(zyfncg): support attr in build op.
    if (ir_values.size() > 1) {
      op = rewriter.Build<paddle::dialect::ReshapeOp>(
          ir_values[0].dyn_cast<ir::OpResult>(),
          ir_values[1].dyn_cast<ir::OpResult>());
    } else {
      op = rewriter.Build<paddle::dialect::ReshapeOp>(
          ir_values[0].dyn_cast<ir::OpResult>(),
          GetAttr<std::vector<int64_t>>("shape", op_call, src_match_ctx));
    }
    res_match_ctx->BindIrValue(op_call.outputs()[0]->name(),
                               std::make_shared<IrValue>(op->result(0)));
    res_match_ctx->BindIrValue(op_call.outputs()[1]->name(),
                               std::make_shared<IrValue>(op->result(1)));
  } else if (op_call.name() == "pd.transpose") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    op = rewriter.Build<paddle::dialect::TransposeOp>(
        ir_values[0].dyn_cast<ir::OpResult>(),
        GetAttr<std::vector<int>>("perm", op_call, src_match_ctx));
    res_match_ctx->BindIrValue(op_call.outputs()[0]->name(),
                               std::make_shared<IrValue>(op->result(0)));
  } else if (op_call.name() == "pd.cast") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    op = rewriter.Build<paddle::dialect::CastOp>(
        ir_values[0].dyn_cast<ir::OpResult>(),
        GetAttr<phi::DataType>("dtype", op_call, src_match_ctx));
    res_match_ctx->BindIrValue(op_call.outputs()[0]->name(),
                               std::make_shared<IrValue>(op->result(0)));
  } else if (op_call.name() == "pd.full") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    op = rewriter.Build<paddle::dialect::FullOp>(
        CreateAttributeMap(op_call, src_match_ctx));
    res_match_ctx->BindIrValue(op_call.outputs()[0]->name(),
                               std::make_shared<IrValue>(op->result(0)));
    // } else if (op_call.name() == "pd.fused_gemm_epilogue") {
    //   const auto& inputs = op_call.inputs();
    //   std::vector<Value> ir_values =
    //       GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    //   Operation* op = rewriter.Build<paddle::dialect::FusedGemmEpilogueOp>(
    //       ir_values[0].dyn_cast<ir::OpResult>(),
    //       ir_values[1].dyn_cast<ir::OpResult>(),
    //       ir_values[2].dyn_cast<ir::OpResult>(),
    //       CreateAttributeMap(op_call, src_match_ctx));
    //   BindIrOutputs(op_call, op, res_match_ctx);
    // } else if (op_call.name() == "pd.fused_gemm_epilogue_grad") {
    //   const auto& inputs = op_call.inputs();
    //   std::vector<Value> ir_values =
    //       GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    //   op = rewriter.Build<paddle::dialect::FusedGemmEpilogueGradOp>(
    //       ir_values[0].dyn_cast<ir::OpResult>(),
    //       ir_values[1].dyn_cast<ir::OpResult>(),
    //       ir_values[2].dyn_cast<ir::OpResult>(),
    //       ir_values[3].dyn_cast<ir::OpResult>(),
    //       CreateAttributeMap(op_call, src_match_ctx));
    //   BindIrOutputs(op_call, op, res_match_ctx);
  } else if (op_call.name() == "builtin.combine") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    std::vector<ir::OpResult> ir_results;
    for (auto value : ir_values) {
      ir_results.push_back(value.dyn_cast<ir::OpResult>());
    }
    op = rewriter.Build<ir::CombineOp>(ir_results);
    res_match_ctx->BindIrValue(op_call.outputs()[0]->name(),
                               std::make_shared<IrValue>(op->result(0)));
  } else if (op_call.name() == "pd.concat") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    op = rewriter.Build<paddle::dialect::ConcatOp>(
        ir_values[0].dyn_cast<ir::OpResult>(),
        GetAttr<int>("axis", op_call, src_match_ctx));
    res_match_ctx->BindIrValue(op_call.outputs()[0]->name(),
                               std::make_shared<IrValue>(op->result(0)));
  } else if (op_call.name() == "pd.multihead_matmul") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    op = rewriter.Build<paddle::dialect::MultiheadMatmulOp>(
        ir_values[0].dyn_cast<ir::OpResult>(),
        ir_values[1].dyn_cast<ir::OpResult>(),
        ir_values[2].dyn_cast<ir::OpResult>(),
        ir_values[3].dyn_cast<ir::OpResult>(),
        false,
        true,
        false,
        GetAttr<float>("alpha", op_call, src_match_ctx),
        GetAttr<int>("head_number", op_call, src_match_ctx));
    res_match_ctx->BindIrValue(op_call.outputs()[0]->name(),
                               std::make_shared<IrValue>(op->result(0)));
  } else {
    PADDLE_THROW(phi::errors::Unavailable("Unknown op : " + op_call.name()));
  }
  VLOG(6) << "Drr create [" << op_call.name() << "] op done.";
  return op;
}

}  // namespace drr
}  // namespace ir
