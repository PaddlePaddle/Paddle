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

#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_to_pd_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
namespace cinn::dialect::details {

pir::Attribute ArrayAttributeToIntArrayAttribute(
    const ::pir::ArrayAttribute& array_attr) {
  std::vector<int64_t> data;
  for (size_t i = 0; i < array_attr.size(); ++i) {
    data.push_back(array_attr.at(i).dyn_cast<::pir::Int64Attribute>().data());
  }
  pir::Attribute attr_data = paddle::dialect::IntArrayAttribute::get(
      pir::IrContext::Instance(), phi::IntArray(data));
  return attr_data;
}

const auto& handler_reduce_max_op =
    [&](::pir::Operation* op,
        const ::pir::Builder& builder) -> ::pir::Operation* {
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto cinn_op = op->dyn_cast<cinn::dialect::ReduceMaxOp>();
  auto attr = cinn_op.attributes();

  // TODO(chenxi67): 1. CINN op Dialect Normalization；2.AST Op compute
  // Normalization
  pir::Attribute attr_axis = ArrayAttributeToIntArrayAttribute(
      attr.at("dim").dyn_cast<::pir::ArrayAttribute>());
  attr.insert({"axis", attr_axis});
  attr.insert({"keepdim", attr["keep_dim"]});
  attr.erase("dim");
  attr.erase("keep_dim");

  auto pd_op =
      const_cast<::pir::Builder*>(&builder)->Build<paddle::dialect::MaxOp>(
          cinn_op.operand_source(0), attr);
  return pd_op;
};

bool CanApplyOn(::pir::Operation* op) {
  return op->dialect()->name() == "cinn_op";
}

::pir::Operation* RewriteCinnOpToPdOp(::pir::Operation* op,
                                      const ::pir::Builder& builder) {
  VLOG(8) << "Rewrite CinnOp to PdOp for op: " << op->name();
  auto& op_transformers = TransformContext::Instance();
  return op_transformers[op->name()](op, builder);
}

void RewriteCinnOpToPdOp(const ::pir::Block& src_block,
                         ::pir::Block* target_block) {
  VLOG(8) << "Rewrite CinnOp to PdOp for block.";
  PADDLE_ENFORCE_NOT_NULL(
      target_block,
      ::common::errors::Fatal("target_block pointer is nullptr."));
  ::pir::IrMapping ir_mapping;
  ::pir::CloneOptions clone_options(/*clone_regions=*/true,
                                    /*clone_operands=*/true,
                                    /*clone_successors=*/true);
  auto* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, target_block);

  for (auto& op : src_block) {
    for (size_t i = 0; i < op.num_operands(); ++i) {
      if (!ir_mapping.GetMap<::pir::Value>().count(op.operand_source(i))) {
        ir_mapping.Add(op.operand_source(i), op.operand_source(i));
      }
    }
    ::pir::Operation* new_op;
    if (CanApplyOn(&op)) {
      new_op = RewriteCinnOpToPdOp(&op, builder);
      new_op->MoveTo(target_block, target_block->end());
    } else {
      new_op = op.Clone(ir_mapping, clone_options);
      new_op->MoveTo(target_block, target_block->end());
    }
    for (uint32_t i = 0; i < op.num_results(); ++i) {
      ir_mapping.Add(op.result(i), new_op->result(i));
    }
  }
}

}  // namespace cinn::dialect::details

REGISTER_TRANSFORM_RULES(reduce_max_op,
                         cinn::dialect::ReduceMaxOp::name(),
                         cinn::dialect::details::handler_reduce_max_op);
