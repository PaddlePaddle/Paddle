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
#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/ir_mapping.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

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
  ::pir::Attribute attr_data = paddle::dialect::IntArrayAttribute::get(
      ::pir::IrContext::Instance(), phi::IntArray(data));
  return attr_data;
}

const auto& handler_reduce_sum_op =
    [](::pir::Operation* op,
       ::pir::IrMapping& ir_mapping,                             // NOLINT
       ::pir::PatternRewriter& rewriter) -> ::pir::Operation* {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();

  ::pir::Attribute attr_axis = ArrayAttributeToIntArrayAttribute(
      attrs.at("axis").dyn_cast<::pir::ArrayAttribute>());
  attrs["axis"] = attr_axis;

  auto pd_op = rewriter.Build<paddle::dialect::SumOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
};

const auto& handler_reduce_max_op =
    [](::pir::Operation* op,
       ::pir::IrMapping& ir_mapping,                             // NOLINT
       ::pir::PatternRewriter& rewriter) -> ::pir::Operation* {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();

  // TODO(chenxi67): 1. CINN op Dialect Normalization；2.AST Op compute
  // Normalization
  ::pir::Attribute attr_axis = ArrayAttributeToIntArrayAttribute(
      attrs.at("axis").dyn_cast<::pir::ArrayAttribute>());
  attrs["axis"] = attr_axis;

  auto pd_op = rewriter.Build<paddle::dialect::MaxOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
};

const auto& handler_reduce_min_op =
    [](::pir::Operation* op,
       ::pir::IrMapping& ir_mapping,                             // NOLINT
       ::pir::PatternRewriter& rewriter) -> ::pir::Operation* {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();

  ::pir::Attribute attr_axis = ArrayAttributeToIntArrayAttribute(
      attrs.at("axis").dyn_cast<::pir::ArrayAttribute>());
  attrs["axis"] = attr_axis;

  auto pd_op = rewriter.Build<paddle::dialect::MinOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
};

const auto& handler_reduce_prod_op =
    [](::pir::Operation* op,
       ::pir::IrMapping& ir_mapping,                             // NOLINT
       ::pir::PatternRewriter& rewriter) -> ::pir::Operation* {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();

  ::pir::Attribute attr_axis = ArrayAttributeToIntArrayAttribute(
      attrs.at("axis").dyn_cast<::pir::ArrayAttribute>());
  attrs["axis"] = attr_axis;

  auto pd_op = rewriter.Build<paddle::dialect::ProdOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
};

::pir::Operation* ConvertSliceOp(::pir::Operation* op,
                                 ::pir::IrMapping& ir_mapping,        // NOLINT
                                 ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();
  ::pir::Attribute starts = ArrayAttributeToIntArrayAttribute(
      attrs.at("starts").dyn_cast<::pir::ArrayAttribute>());
  ::pir::Attribute ends = ArrayAttributeToIntArrayAttribute(
      attrs.at("ends").dyn_cast<::pir::ArrayAttribute>());
  attrs["starts"] = starts;
  attrs["ends"] = ends;
  auto pd_op = rewriter.Build<paddle::dialect::SliceOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertReshapeOp(
    ::pir::Operation* op,
    ::pir::IrMapping& ir_mapping,        // NOLINT
    ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();
  ::pir::Attribute shape = ArrayAttributeToIntArrayAttribute(
      attrs.at("shape").dyn_cast<::pir::ArrayAttribute>());
  attrs["shape"] = shape;
  auto pd_op = rewriter.Build<paddle::dialect::ReshapeOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertConcatOp(::pir::Operation* op,
                                  ::pir::IrMapping& ir_mapping,        // NOLINT
                                  ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  const auto& attrs = op->attributes();
  std::vector<pir::Value> vec_inputs;
  for (uint32_t i = 0; i < op->num_operands(); ++i) {
    vec_inputs.push_back(ir_mapping.Lookup(op->operand_source(i)));
  }
  auto op_input = rewriter.Build<pir::CombineOp>(vec_inputs).result(0);

  int axis = attrs.at("axis").dyn_cast<::pir::Int32Attribute>().data();

  auto pd_op = rewriter.Build<paddle::dialect::ConcatOp>(op_input, axis);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertGenerateShapeOp(
    ::pir::Operation* op,
    ::pir::IrMapping& ir_mapping,  // NOLINT
    ::pir::Builder& builder) {     // NOLINT
  auto* new_op = op->Clone(ir_mapping, {true, true, true});
  builder.Insert(new_op);
  return new_op;
}

::pir::Operation* ConvertScaleOp(::pir::Operation* op,
                                 ::pir::IrMapping& ir_mapping,        // NOLINT
                                 ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  const auto& attrs = op->attributes();

  float scale = attrs.at("scale").dyn_cast<::pir::FloatAttribute>().data();
  float bias = attrs.at("bias").dyn_cast<pir::FloatAttribute>().data();
  bool bias_after_scale =
      attrs.at("bias_after_scale").dyn_cast<pir::BoolAttribute>().data();
  auto pd_op = rewriter.Build<paddle::dialect::ScaleOp>(
      ir_mapping.Lookup(op->operand_source(0)), scale, bias, bias_after_scale);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertFlipOp(::pir::Operation* op,
                                ::pir::IrMapping& ir_mapping,        // NOLINT
                                ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  const auto& attrs = op->attributes();
  auto pd_op = rewriter.Build<paddle::dialect::FlipOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertPool2dOp(::pir::Operation* op,
                                  ::pir::IrMapping& ir_mapping,        // NOLINT
                                  ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = op->attributes();
  ::pir::Attribute kernel_size = ArrayAttributeToIntArrayAttribute(
      attrs.at("kernel_size").dyn_cast<::pir::ArrayAttribute>());
  attrs["kernel_size"] = kernel_size;
  attrs["strides"] = attrs.at("stride_size");
  attrs["paddings"] = attrs.at("padding_size");
  attrs.erase("stride_size");
  attrs.erase("padding_size");
  auto pd_op = rewriter.Build<paddle::dialect::Pool2dOp>(
      ir_mapping.Lookup(op->operand_source(0)), attrs);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertIscloseOp(
    ::pir::Operation* op,
    ::pir::IrMapping& ir_mapping,        // NOLINT
    ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  const auto& attrs = op->attributes();
  double rtol = attrs.at("atol").dyn_cast<pir::FloatAttribute>().data();
  double atol = attrs.at("atol").dyn_cast<pir::FloatAttribute>().data();
  bool equal_nan = attrs.at("equal_nan").dyn_cast<pir::BoolAttribute>().data();
  auto pd_op = rewriter.Build<paddle::dialect::IscloseOp>(
      ir_mapping.Lookup(op->operand_source(0)),
      ir_mapping.Lookup(op->operand_source(1)),
      rtol,
      atol,
      equal_nan);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertYieldStoreOp(
    ::pir::Operation* op,
    ::pir::IrMapping& ir_mapping,        // NOLINT
    ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  const auto& attrs = op->attributes();
  auto pd_op = rewriter.Build<paddle::dialect::ShareData_Op>(
      ir_mapping.Lookup(op->operand_source(0)));
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertExpandOp(::pir::Operation* op,
                                  ::pir::IrMapping& ir_mapping,        // NOLINT
                                  ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  const auto& attrs = op->attributes();

  std::vector<int64_t> shape_;
  auto attr_shape = attrs.at("out_shape").dyn_cast<::pir::ArrayAttribute>();
  for (size_t i = 0; i < attr_shape.size(); ++i) {
    shape_.push_back(attr_shape.at(i).dyn_cast<::pir::Int64Attribute>().data());
  }

  paddle::dialect::FullIntArrayOp full_shape_op =
      rewriter.Build<paddle::dialect::FullIntArrayOp>(
          shape_, phi::DataType::INT64, phi::CPUPlace());
  ::pir::Value out_shape = full_shape_op->result(0);
  auto pd_op = rewriter.Build<paddle::dialect::ExpandOp>(
      ir_mapping.Lookup(op->operand_source(0)), out_shape);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertUniformOp(
    ::pir::Operation* op,
    ::pir::IrMapping& ir_mapping,        // NOLINT
    ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  const auto& attrs = op->attributes();
  std::vector<int64_t> shape;
  auto attr_shape = attrs.at("out_shape").dyn_cast<::pir::ArrayAttribute>();
  for (size_t i = 0; i < attr_shape.size(); ++i) {
    shape.push_back(attr_shape.at(i).dyn_cast<::pir::Int64Attribute>().data());
  }
  ::phi::DataType dtype =
      attrs.at("dtype").dyn_cast<paddle::dialect::DataTypeAttribute>().data();

  float min = attrs.at("min").dyn_cast<pir::FloatAttribute>().data();
  float max = attrs.at("max").dyn_cast<pir::FloatAttribute>().data();
  float seed = attrs.at("diag_num").dyn_cast<pir::FloatAttribute>().data();
  ::phi::Place place =
      attrs.at("place").dyn_cast<paddle::dialect::PlaceAttribute>().data();

  auto pd_op = rewriter.Build<paddle::dialect::UniformOp>(
      shape, dtype, min, max, seed, place);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

::pir::Operation* ConvertGatherOp(::pir::Operation* op,
                                  ::pir::IrMapping& ir_mapping,        // NOLINT
                                  ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  const auto& attrs = op->attributes();
  int axis = attrs.at("axis").dyn_cast<pir::Int32Attribute>().data();
  auto pd_op = rewriter.Build<paddle::dialect::GatherOp>(
      ir_mapping.Lookup(op->operand_source(0)),
      ir_mapping.Lookup(op->operand_source(1)),
      axis);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    ir_mapping.Add(op->result(i), pd_op->result(i));
  }
  return pd_op;
}

bool CanApplyOn(::pir::Operation* op) {
  return op->dialect()->name() == "cinn_op";
}

::pir::Operation* RewriteCinnOpToPdOp(
    ::pir::Operation* op,
    ::pir::IrMapping& ir_mapping,        // NOLINT
    ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(8) << "Rewrite CinnOp to PdOp for op: " << op->name();
  auto& op_transformers = TransformContext::Instance();
  return op_transformers[op->name()](op, ir_mapping, rewriter);
}

void RewriteCinnOpToPdOp(const ::pir::Block& src_block,
                         ::pir::Block* target_block,
                         ::pir::PatternRewriter& rewriter) {  // NOLINT
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

  for (auto& op : src_block) {
    for (size_t i = 0; i < op.num_operands(); ++i) {
      if (!ir_mapping.GetMap<::pir::Value>().count(op.operand_source(i))) {
        ir_mapping.Add(op.operand_source(i), op.operand_source(i));
      }
    }
    ::pir::Operation* new_op;
    if (CanApplyOn(&op)) {
      new_op = RewriteCinnOpToPdOp(&op, ir_mapping, rewriter);
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

REGISTER_TRANSFORM_RULES(generate_shape_op,
                         cinn::dialect::GenerateShapeOp::name(),
                         cinn::dialect::details::ConvertGenerateShapeOp);
REGISTER_TRANSFORM_RULES(scale_op,
                         cinn::dialect::ScaleOp::name(),
                         cinn::dialect::details::ConvertScaleOp);

REGISTER_TRANSFORM_RULES(
    flip_op,
    cinn::dialect::ReverseOp::name(),  // cinn::dialect::ReverseOp <->
                                       // paddle::dialect::FlipOp
    cinn::dialect::details::ConvertFlipOp);

REGISTER_TRANSFORM_RULES(pool2d_op,
                         cinn::dialect::Pool2dOp::name(),
                         cinn::dialect::details::ConvertPool2dOp);

REGISTER_TRANSFORM_RULES(isclose_op,
                         cinn::dialect::IscloseOp::name(),
                         cinn::dialect::details::ConvertIscloseOp);

REGISTER_TRANSFORM_RULES(yield_store,
                         cinn::dialect::YieldStoreOp::name(),
                         cinn::dialect::details::ConvertYieldStoreOp);

REGISTER_TRANSFORM_RULES(
    expand_op,
    cinn::dialect::BroadcastOp::name(),
    cinn::dialect::details::ConvertExpandOp);  // cinn::dialect::BroadcastOp <->
                                               // paddle::dialect::ExpandOp

REGISTER_TRANSFORM_RULES(uniform_op,
                         cinn::dialect::UniformRandomOp::name(),
                         cinn::dialect::details::ConvertUniformOp);

REGISTER_TRANSFORM_RULES(gather_op,
                         cinn::dialect::GatherOp::name(),
                         cinn::dialect::details::ConvertGatherOp);
