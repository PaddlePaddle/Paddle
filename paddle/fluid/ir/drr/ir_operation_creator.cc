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

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"

namespace ir {
namespace drr {

Value GetIrValueByDrrTensor(const Tensor& tensor,
                            const MatchContextImpl& res_match_ctx) {
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
        },
        kv.second);
  }
  return attr_map;
}

template <typename T>
T GetAttr(const std::string& attr_name,
          const OpCall& op_call,
          const MatchContextImpl& src_match_ctx) {
  const auto& attr = op_call.attributes().at(attr_name);
  if (std::holds_alternative<NormalAttribute>(attr)) {
    return src_match_ctx.Attr<T>(std::get<NormalAttribute>(attr).name());
  } else if (std::holds_alternative<ComputeAttribute>(attr)) {
    MatchContext ctx(std::make_shared<MatchContextImpl>(src_match_ctx));
    return std::any_cast<T>(
        std::get<ComputeAttribute>(attr).attr_compute_func()(ctx));
  } else {
    IR_THROW("Unknown attrbute type for : %s.", attr_name);
  }
}

Operation* CreateOperation(const OpCall& op_call,
                           const MatchContextImpl& src_match_ctx,
                           ir::PatternRewriter& rewriter,  // NOLINT
                           MatchContextImpl* res_match_ctx) {
  if (op_call.name() == "pd.reshape") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    // TODO(zyfncg): support attr in build op.
    Operation* reshape_op = rewriter.Build<paddle::dialect::ReshapeOp>(
        ir_values[0].dyn_cast<ir::OpResult>(),
        ir_values[1].dyn_cast<ir::OpResult>());
    res_match_ctx->BindIrValue(
        op_call.outputs()[0]->name(),
        std::make_shared<IrValue>(reshape_op->result(0)));
    res_match_ctx->BindIrValue(
        op_call.outputs()[1]->name(),
        std::make_shared<IrValue>(reshape_op->result(1)));
    return reshape_op;

  } else if (op_call.name() == "pd.transpose") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    Operation* transpose_op = rewriter.Build<paddle::dialect::TransposeOp>(
        ir_values[0].dyn_cast<ir::OpResult>(),
        GetAttr<std::vector<int>>("perm", op_call, src_match_ctx));
    res_match_ctx->BindIrValue(
        op_call.outputs()[0]->name(),
        std::make_shared<IrValue>(transpose_op->result(0)));
    return transpose_op;

  } else if (op_call.name() == "pd.cast") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    Operation* cast_op = rewriter.Build<paddle::dialect::CastOp>(
        ir_values[0].dyn_cast<ir::OpResult>(),
        GetAttr<phi::DataType>("dtype", op_call, src_match_ctx));
    res_match_ctx->BindIrValue(op_call.outputs()[0]->name(),
                               std::make_shared<IrValue>(cast_op->result(0)));
    return cast_op;

  } else if (op_call.name() == "pd.full") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, *res_match_ctx);
    Operation* full_op = rewriter.Build<paddle::dialect::FullOp>(
        CreateAttributeMap(op_call, src_match_ctx));
    res_match_ctx->BindIrValue(op_call.outputs()[0]->name(),
                               std::make_shared<IrValue>(full_op->result(0)));
    return full_op;
  }

  PADDLE_THROW("Unknown op :" + op_call.name());
}

}  // namespace drr
}  // namespace ir
