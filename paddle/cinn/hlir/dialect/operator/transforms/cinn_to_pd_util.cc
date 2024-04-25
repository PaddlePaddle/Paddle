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

namespace cinn::dialect {

TransformContext::TransformContext() {
  const auto& handler_name = [&](::pir::Operation* op,
                                 ::pir::Builder builder) -> ::pir::Operation* {
    VLOG(10) << "transform " << op->name() << " from cinn_op to pd_op";
    auto cinn_op = op->dyn_cast<cinn::dialect::ReduceMaxOp>();
    auto attr = cinn_op.attributes();
    std::vector<int64_t> axis;
    for (size_t i = 0;
         i < attr.at("dim").dyn_cast<::pir::ArrayAttribute>().size();
         i++) {
      axis.push_back(attr.at("dim")
                         .dyn_cast<::pir::ArrayAttribute>()
                         .at(i)
                         .dyn_cast<::pir::Int64Attribute>()
                         .data());
    }
    pir::Attribute attr_axis = paddle::dialect::IntArrayAttribute::get(
        pir::IrContext::Instance(), phi::IntArray(axis));
    attr.insert({"axis", attr_axis});
    attr.insert({"keepdim", attr["keep_dim"]});
    attr.erase("dim");
    attr.erase("keep_dim");
    auto pd_op =
        builder.Build<paddle::dialect::MaxOp>(cinn_op.operand_source(0), attr);
    return pd_op;
  };
  op_transformers.insert({cinn::dialect::ReduceMaxOp::name(), handler_name});
}

bool CanApplyOn(::pir::Operation* op) {
  if (op->dialect()->name() == "cinn_op") {
    return true;
  }
  return false;
}

::pir::Operation* RewriteCinnOpToPdOp(::pir::Operation* op,
                                      ::pir::Builder builder) {
  VLOG(8) << "Rewrite CinnOp to PdOp for op: " << op->name();
  auto& op_transformers = TransformContext::Instance();
  return op_transformers[op->name()](op, builder);
}

void RewriteCinnOpToPdOp(::pir::Block* src_block, ::pir::Block* target_block) {
  VLOG(8) << "Rewrite CinnOp to PdOp for block.";
  if (src_block == nullptr || target_block == nullptr) {
    return;
  }
  ::pir::IrMapping ir_mapping;
  ::pir::CloneOptions clone_options(/*clone_regions=*/true,
                                    /*clone_operands=*/true,
                                    /*clone_successors=*/true);
  auto* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, target_block);

  for (auto& op : *src_block) {
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

}  // namespace cinn::dialect
