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
#include "paddle/pir/include/core/ir_mapping.h"
namespace cinn::dialect::details {

pir::Attribute ArrayAttributeToIntArrayAttribute(
    const ::pir::ArrayAttribute& array_attr) {
  std::vector<int64_t> data;
  for (size_t i = 0; i < array_attr.size(); ++i) {
    auto attr = array_attr.at(i);
    if (attr.isa<::pir::Int32Attribute>()) {
      data.push_back(attr.dyn_cast<::pir::Int32Attribute>().data());
    } else {
      data.push_back(attr.dyn_cast<::pir::Int64Attribute>().data());
    }
  }
  pir::Attribute attr_data = paddle::dialect::IntArrayAttribute::get(
      pir::IrContext::Instance(), phi::IntArray(data));
  return attr_data;
}

const auto& handler_reduce_sum_op =
    [](::pir::Operation* op,
       ::pir::IrMapping& ir_mapping,                    // NOLINT
       ::pir::Builder& builder) -> ::pir::Operation* {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();

  pir::Attribute attr_axis = ArrayAttributeToIntArrayAttribute(
      attrs.at("dim").dyn_cast<::pir::ArrayAttribute>());
  attrs.insert({"axis", attr_axis});
  attrs.insert({"dtype", attrs["dtype"]});
  attrs.insert({"keepdim", attrs["keep_dim"]});
  attrs.erase("dim");
  attrs.erase("keep_dim");

  auto pd_op = builder.Build<paddle::dialect::SumOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
};

const auto& handler_reduce_max_op =
    [](::pir::Operation* op,
       ::pir::IrMapping& ir_mapping,                    // NOLINT
       ::pir::Builder& builder) -> ::pir::Operation* {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();

  // TODO(chenxi67): 1. CINN op Dialect Normalization；2.AST Op compute
  // Normalization
  pir::Attribute attr_axis = ArrayAttributeToIntArrayAttribute(
      attrs.at("dim").dyn_cast<::pir::ArrayAttribute>());
  attrs.insert({"axis", attr_axis});
  attrs.insert({"keepdim", attrs["keep_dim"]});
  attrs.erase("dim");
  attrs.erase("keep_dim");

  auto pd_op = builder.Build<paddle::dialect::MaxOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
};

const auto& handler_reduce_min_op =
    [](::pir::Operation* op,
       ::pir::IrMapping& ir_mapping,                    // NOLINT
       ::pir::Builder& builder) -> ::pir::Operation* {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();

  pir::Attribute attr_axis = ArrayAttributeToIntArrayAttribute(
      attrs.at("dim").dyn_cast<::pir::ArrayAttribute>());
  attrs.insert({"axis", attr_axis});
  attrs.insert({"keepdim", attrs["keep_dim"]});
  attrs.erase("dim");
  attrs.erase("keep_dim");

  auto pd_op = builder.Build<paddle::dialect::MinOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
};

const auto& handler_reduce_prod_op =
    [](::pir::Operation* op,
       ::pir::IrMapping& ir_mapping,                    // NOLINT
       ::pir::Builder& builder) -> ::pir::Operation* {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();

  pir::Attribute attr_axis = ArrayAttributeToIntArrayAttribute(
      attrs.at("dim").dyn_cast<::pir::ArrayAttribute>());
  attrs.insert({"dims", attr_axis});
  attrs.erase("dim");

  auto pd_op = builder.Build<paddle::dialect::ProdOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
};

::pir::Operation* ConvertSliceOp(::pir::Operation* op,
                                 ::pir::IrMapping& ir_mapping,  // NOLINT
                                 ::pir::Builder& builder) {     // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();
  pir::Attribute starts = ArrayAttributeToIntArrayAttribute(
      attrs.at("starts").dyn_cast<::pir::ArrayAttribute>());
  pir::Attribute ends = ArrayAttributeToIntArrayAttribute(
      attrs.at("ends").dyn_cast<::pir::ArrayAttribute>());
  attrs["starts"] = starts;
  attrs["ends"] = ends;
  auto pd_op = builder.Build<paddle::dialect::SliceOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertReshapeOp(::pir::Operation* op,
                                   ::pir::IrMapping& ir_mapping,  // NOLINT
                                   ::pir::Builder& builder) {     // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();
  attrs.at("shape").dyn_cast<::pir::ArrayAttribute>();
  pir::Attribute shape = ArrayAttributeToIntArrayAttribute(
      attrs.at("shape").dyn_cast<::pir::ArrayAttribute>());
  attrs["shape"] = shape;
  auto pd_op = builder.Build<paddle::dialect::ReshapeOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertConcatOp(::pir::Operation* op,
                                  ::pir::IrMapping& ir_mapping,  // NOLINT
                                  ::pir::Builder& builder) {     // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();
  for (auto item : attrs) {
    VLOG(0) << item.first;
  }
  std::vector<pir::Value> vec_inputs;
  for (uint32_t i = 0; i < op->num_operands(); ++i) {
    vec_inputs.push_back(ir_mapping.Lookup(op->operand_source(i)));
  }
  auto op_input = builder.Build<pir::CombineOp>(vec_inputs).result(0);

  int axis = attrs.at("axis").dyn_cast<::pir::Int32Attribute>().data();

  auto pd_op = builder.Build<paddle::dialect::ConcatOp>(op_input, axis);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

bool CanApplyOn(::pir::Operation* op) {
  return op->dialect()->name() == "cinn_op";
}

::pir::Operation* RewriteCinnOpToPdOp(::pir::Operation* op,
                                      ::pir::IrMapping& ir_mapping,  // NOLINT
                                      ::pir::Builder& builder) {     // NOLINT
  VLOG(8) << "Rewrite CinnOp to PdOp for op: " << op->name();
  auto& op_transformers = TransformContext::Instance();
  return op_transformers[op->name()](op, ir_mapping, builder);
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
      new_op = RewriteCinnOpToPdOp(&op, ir_mapping, builder);
      new_op->MoveTo(target_block, target_block->end());
    } else {
      new_op = op.Clone(ir_mapping, clone_options);
      new_op->MoveTo(target_block, target_block->end());
    }
  }
}

}  // namespace cinn::dialect::details

REGISTER_TRANSFORM_RULES(reduce_sum_op,
                         cinn::dialect::ReduceSumOp::name(),
                         cinn::dialect::details::handler_reduce_sum_op);

REGISTER_TRANSFORM_RULES(reduce_max_op,
                         cinn::dialect::ReduceMaxOp::name(),
                         cinn::dialect::details::handler_reduce_max_op);

REGISTER_TRANSFORM_RULES(reduce_min_op,
                         cinn::dialect::ReduceMinOp::name(),
                         cinn::dialect::details::handler_reduce_min_op);

REGISTER_TRANSFORM_RULES(reduce_prod_op,
                         cinn::dialect::ReduceProdOp::name(),
                         cinn::dialect::details::handler_reduce_prod_op);

REGISTER_TRANSFORM_RULES(slice_op,
                         cinn::dialect::SliceOp::name(),
                         cinn::dialect::details::ConvertSliceOp);

REGISTER_TRANSFORM_RULES(reshape_op,
                         cinn::dialect::ReshapeOp::name(),
                         cinn::dialect::details::ConvertReshapeOp);

REGISTER_TRANSFORM_RULES(concat_op,
                         cinn::dialect::ConcatOp::name(),
                         cinn::dialect::details::ConvertConcatOp);
