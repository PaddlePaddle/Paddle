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

#include "paddle/fluid/pir/drr/ir_operation_creator.h"

#include <any>

#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/attr_type_uilts.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"

namespace pir {
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

static pir::Attribute CreateIrAttribute(const std::any& obj) {
  if (obj.type() == typeid(bool)) {
    return IrAttrbuteCreator<bool>()(std::any_cast<bool>(obj));
  } else if (obj.type() == typeid(int32_t)) {
    return IrAttrbuteCreator<int32_t>()(std::any_cast<int32_t>(obj));
  } else if (obj.type() == typeid(int64_t)) {
    return IrAttrbuteCreator<int64_t>()(std::any_cast<int64_t>(obj));
  } else if (obj.type() == typeid(float)) {
    return IrAttrbuteCreator<float>()(std::any_cast<float>(obj));
  } else if (obj.type() == typeid(std::string)) {
    return IrAttrbuteCreator<std::string>()(std::any_cast<std::string>(obj));
  } else if (obj.type() == typeid(const char*)) {
    return IrAttrbuteCreator<std::string>()(std::any_cast<const char*>(obj));
  } else if (obj.type() == typeid(phi::DataType)) {
    return IrAttrbuteCreator<phi::DataType>()(
        std::any_cast<phi::DataType>(obj));
  } else if (obj.type() == typeid(phi::Place)) {
    return IrAttrbuteCreator<phi::Place>()(std::any_cast<phi::Place>(obj));
  } else if (obj.type() == typeid(std::vector<int32_t>)) {
    return IrAttrbuteCreator<std::vector<int32_t>>()(
        std::any_cast<std::vector<int32_t>>(obj));
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
                   pir::Operation* op,
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
                           pir::PatternRewriter& rewriter) {}  // NOLINT

class OperationFactory {
 public:
  static OperationFactory& Instance() {
    static OperationFactory operation_factory;
    return operation_factory;
  }

  using operation_create_fn =
      std::function<pir::Operation*(const std::vector<Value>&,
                                    const pir::AttributeMap&,
                                    pir::PatternRewriter&)>;

  void RegisterOperationCreator(const std::string& op_name,
                                const operation_create_fn& create_fn) {
    op_creator_map.emplace(op_name, create_fn);
  }

  pir::Operation* CreateOperation(
      const std::string& op_name,
      const std::vector<Value>& inputs,
      const pir::AttributeMap& attrs,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    auto iter = op_creator_map.find(op_name);
    IR_ENFORCE(iter != op_creator_map.end(),
               "The create function for op: (%s) is not found.",
               op_name);
    return iter->second(inputs, attrs, rewriter);
  }

 private:
  OperationFactory() {
    RegisterGeneratedOpCreator();
    RegisterManualOpCreator();
  }

  void RegisterManualOpCreator();
  void RegisterGeneratedOpCreator() {}

  std::unordered_map<std::string, operation_create_fn> op_creator_map;
};

void OperationFactory::RegisterManualOpCreator() {
  RegisterOperationCreator(
      "pd_op.reshape",
      [](const std::vector<Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() > 1) {
          return rewriter.Build<paddle::dialect::ReshapeOp>(
              inputs[0].dyn_cast<pir::OpResult>(),
              inputs[1].dyn_cast<pir::OpResult>());
        } else {
          return rewriter.Build<paddle::dialect::ReshapeOp>(
              inputs[0].dyn_cast<pir::OpResult>(), attrs);
        }
      });

  RegisterOperationCreator(
      "pd_op.transpose",
      [](const std::vector<Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::dialect::TransposeOp>(
            inputs[0].dyn_cast<pir::OpResult>(), attrs);
      });
  RegisterOperationCreator("pd_op.cast",
                           [](const std::vector<Value>& inputs,
                              const pir::AttributeMap& attrs,
                              pir::PatternRewriter& rewriter) {
                             return rewriter.Build<paddle::dialect::CastOp>(
                                 inputs[0].dyn_cast<pir::OpResult>(), attrs);
                           });
  RegisterOperationCreator(
      "pd_op.full",
      [](const std::vector<Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::dialect::FullOp>(attrs);
      });
  RegisterOperationCreator(
      "pd_op.fused_gemm_epilogue",
      [](const std::vector<Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::dialect::FusedGemmEpilogueOp>(
            inputs[0].dyn_cast<pir::OpResult>(),
            inputs[1].dyn_cast<pir::OpResult>(),
            inputs[2].dyn_cast<pir::OpResult>(),
            attrs);
      });
  RegisterOperationCreator(
      "pd_op.fused_gemm_epilogue_grad",
      [](const std::vector<Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::dialect::FusedGemmEpilogueGradOp>(
            inputs[0].dyn_cast<pir::OpResult>(),
            inputs[1].dyn_cast<pir::OpResult>(),
            inputs[2].dyn_cast<pir::OpResult>(),
            inputs[3].dyn_cast<pir::OpResult>(),
            attrs);
      });
  RegisterOperationCreator(
      "builtin.combine",
      [](const std::vector<Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        std::vector<pir::OpResult> ir_results;
        for (auto value : inputs) {
          ir_results.push_back(value.dyn_cast<pir::OpResult>());
        }
        return rewriter.Build<pir::CombineOp>(ir_results);
      });
  RegisterOperationCreator("pd_op.concat",
                           [](const std::vector<Value>& inputs,
                              const pir::AttributeMap& attrs,
                              pir::PatternRewriter& rewriter) {
                             return rewriter.Build<paddle::dialect::ConcatOp>(
                                 inputs[0].dyn_cast<pir::OpResult>(), attrs);
                           });
  RegisterOperationCreator(
      "pd_op.multihead_matmul",
      [](const std::vector<Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::dialect::MultiheadMatmulOp>(
            inputs[0].dyn_cast<pir::OpResult>(),
            inputs[1].dyn_cast<pir::OpResult>(),
            inputs[2].dyn_cast<pir::OpResult>(),
            inputs[3].dyn_cast<pir::OpResult>(),
            attrs);
      });
}

pir::Operation* CreateOperation(const OpCall& op_call,
                                const MatchContextImpl& src_match_ctx,
                                pir::PatternRewriter& rewriter,  // NOLINT
                                MatchContextImpl* res_match_ctx) {
  VLOG(6) << "Drr create [" << op_call.name() << "] op...";
  const auto& inputs = op_call.inputs();
  std::vector<Value> ir_values =
      GetIrValuesByDrrTensors(inputs, *res_match_ctx);
  pir::Operation* op = OperationFactory::Instance().CreateOperation(
      op_call.name(),
      ir_values,
      CreateAttributeMap(op_call, src_match_ctx),
      rewriter);
  BindIrOutputs(op_call, op, res_match_ctx);
  VLOG(6) << "Drr create [" << op_call.name() << "] op done.";
  return op;
}

}  // namespace drr
}  // namespace pir
