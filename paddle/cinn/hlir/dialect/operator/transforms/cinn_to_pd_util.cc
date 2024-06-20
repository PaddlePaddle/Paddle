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

template <typename T, std::size_t... I>
auto VectorToTupleHelper(const std::vector<T>& vec, std::index_sequence<I...>) {
  return std::make_tuple(vec[I]...);
}

template <typename T, std::size_t N>
auto VectorToTuple(const std::vector<T>& vec) {
  return VectorToTupleHelper(vec, std::make_index_sequence<N>{});
}

template <typename Func, typename Tuple, std::size_t... I>
auto ApplyTupleArgsHelper(Func&& func,
                          Tuple&& tuple,
                          std::index_sequence<I...>) {
  return func(std::get<I>(std::forward<Tuple>(tuple))...);
}

template <typename Func, typename Tuple>
auto ApplyTupleArgs(Func&& func, Tuple&& tuple) {
  return ApplyTupleArgsHelper(
      std::forward<Func>(func),
      std::forward<Tuple>(tuple),
      std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}

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

enum class TransAttrType { IntArray, Int32, Int64, Float, Bool };
using OpAttrTypeMap = const std::unordered_map<
    std::string,
    const std::unordered_map<std::string, TransAttrType>>;

static OpAttrTypeMap OP_TRANS_ATTRTYPES_MAP = {
    {"reduce_max", {{"axis", TransAttrType::IntArray}}},
    {"reduce_min", {{"axis", TransAttrType::IntArray}}},
    {"reduce_sum", {{"axis", TransAttrType::IntArray}}},
    {"reduce_prod", {{"axis", TransAttrType::IntArray}}},
    {"gather", {{"axis", TransAttrType::Int32}}},
    {"isclose",
     {{"atol", TransAttrType::Float},
      {"rtol", TransAttrType::Float},
      {"equal_nan", TransAttrType::Bool}}},
    {"scale",
     {{"scale", TransAttrType::Float},
      {"bias", TransAttrType::Float},
      {"bias_after_scale", TransAttrType::Bool}}},
    {"slice",
     {{"starts", TransAttrType::IntArray}, {"ends", TransAttrType::IntArray}}},
};

pir::AttributeMap ConvertAttributes(::pir::Operation* op) {
  auto attrs = op->attributes();
  if (OP_TRANS_ATTRTYPES_MAP.count(op->name()) == 0) {
    return attrs;
  }
  const auto& trans_attrtypes = OP_TRANS_ATTRTYPES_MAP.at(op->name());
  for (const auto& trans_attrtype : trans_attrtypes) {
    auto attr_name = trans_attrtype.first;
    auto attr_type = trans_attrtype.second;
    if (attr_type == TransAttrType::IntArray) {
      auto attr = ArrayAttributeToIntArrayAttribute(
          attrs.at(attr_name).dyn_cast<::pir::ArrayAttribute>());
      attrs[attr_name] = attr;
    } else if (attr_type == TransAttrType::Int32) {
      attrs[attr_name] = attrs.at(attr_name).dyn_cast<::pir::Int32Attribute>();
    } else if (attr_type == TransAttrType::Int64) {
      attrs[attr_name] = attrs.at(attr_name).dyn_cast<::pir::Int64Attribute>();
    } else if (attr_type == TransAttrType::Float) {
      attrs[attr_name] = attrs.at(attr_name).dyn_cast<::pir::FloatAttribute>();
    } else if (attr_type == TransAttrType::Bool) {
      attrs[attr_name] = attrs.at(attr_name).dyn_cast<::pir::BoolAttribute>();
    } else {
      PADDLE_THROW(::common::errors::Unimplemented(
          "Unsupported attribute type in ConvertAttributes"));
    }
  }
  return attrs;
}

template <typename TARGET_OP, std::size_t INPUT_NUM>
::pir::Operation* ConvertCinnOp(::pir::Operation* op,
                                ::pir::IrMapping& ir_mapping,        // NOLINT
                                ::pir::PatternRewriter& rewriter) {  // NOLINT
  VLOG(6) << "transform " << op->name() << " from cinn_op to pd_op";
  auto attrs = ConvertAttributes(op);

  std::vector<pir::Value> inputs;
  for (uint32_t i = 0; i < INPUT_NUM; ++i) {
    inputs.push_back(ir_mapping.Lookup(op->operand_source(i)));
  }
  auto input_tuple = VectorToTuple<pir::Value, INPUT_NUM>(inputs);

  auto tuple_args = std::tuple_cat(input_tuple, std::make_tuple(attrs));
  const auto& build_op_func = [&rewriter](auto&&... args) {
    return rewriter.Build<TARGET_OP>(std::forward<decltype(args)>(args)...);
  };
  auto pd_op = ApplyTupleArgs(build_op_func, tuple_args);

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

REGISTER_TRANSFORM_RULES(
    reduce_sum_op,
    cinn::dialect::ReduceSumOp::name(),
    (cinn::dialect::details::ConvertCinnOp<paddle::dialect::SumOp, 1>));

REGISTER_TRANSFORM_RULES(
    reduce_max_op,
    cinn::dialect::ReduceMaxOp::name(),
    (cinn::dialect::details::ConvertCinnOp<paddle::dialect::MaxOp, 1>));

REGISTER_TRANSFORM_RULES(
    reduce_min_op,
    cinn::dialect::ReduceMinOp::name(),
    (cinn::dialect::details::ConvertCinnOp<paddle::dialect::MinOp, 1>));

REGISTER_TRANSFORM_RULES(
    reduce_prod_op,
    cinn::dialect::ReduceProdOp::name(),
    (cinn::dialect::details::ConvertCinnOp<paddle::dialect::ProdOp, 1>));

REGISTER_TRANSFORM_RULES(
    flip_op,
    cinn::dialect::ReverseOp::name(),
    (cinn::dialect::details::ConvertCinnOp<paddle::dialect::FlipOp, 1>));

REGISTER_TRANSFORM_RULES(
    gather_op,
    cinn::dialect::GatherOp::name(),
    (cinn::dialect::details::ConvertCinnOp<paddle::dialect::GatherOp, 2>));

REGISTER_TRANSFORM_RULES(
    isclose_op,
    cinn::dialect::IscloseOp::name(),
    (cinn::dialect::details::ConvertCinnOp<paddle::dialect::IscloseOp, 2>));

REGISTER_TRANSFORM_RULES(
    scale_op,
    cinn::dialect::ScaleOp::name(),
    (cinn::dialect::details::ConvertCinnOp<paddle::dialect::ScaleOp, 1>));

REGISTER_TRANSFORM_RULES(
    slice_op,
    cinn::dialect::SliceOp::name(),
    (cinn::dialect::details::ConvertCinnOp<paddle::dialect::SliceOp, 1>));

REGISTER_TRANSFORM_RULES(reshape_op,
                         cinn::dialect::ReshapeOp::name(),
                         cinn::dialect::details::ConvertReshapeOp);

REGISTER_TRANSFORM_RULES(concat_op,
                         cinn::dialect::ConcatOp::name(),
                         cinn::dialect::details::ConvertConcatOp);

REGISTER_TRANSFORM_RULES(generate_shape_op,
                         cinn::dialect::GenerateShapeOp::name(),
                         cinn::dialect::details::ConvertGenerateShapeOp);

REGISTER_TRANSFORM_RULES(pool2d_op,
                         cinn::dialect::Pool2dOp::name(),
                         cinn::dialect::details::ConvertPool2dOp);

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
