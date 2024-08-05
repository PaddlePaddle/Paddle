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

#include "paddle/cinn/hlir/framework/pir/op_mapper.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

namespace {

std::vector<int> GetVec32FromVec64Attr(::pir::Attribute attr) {
  auto attr_vec = attr.dyn_cast<::pir::ArrayAttribute>().AsVector();

  std::vector<int> dim;
  for (auto vec_element : attr_vec) {
    dim.push_back(vec_element.dyn_cast<::pir::Int64Attribute>().data());
  }

  return dim;
}

void AppendAttrForReduceOp(const ::pir::Operation& op,
                           utils::AttributeMap& attrs) {  // NOLINT
  auto attr = op.attributes().at("axis");
  attrs["axis"] = GetVec32FromVec64Attr(attr);
}

void AppendAttrForTransposeOp(const ::pir::Operation& op,
                              utils::AttributeMap& attrs) {  // NOLINT
  auto rank = op.operand_source(0)
                  .type()
                  .dyn_cast<paddle::dialect::DenseTensorType>()
                  .dims()
                  .size();
  auto attr = op.attributes().at("perm");

  auto attr_vec = attr.dyn_cast<::pir::ArrayAttribute>().AsVector();

  std::vector<int> dim;
  for (auto vec_element : attr_vec) {
    auto ele = vec_element.dyn_cast<::pir::Int32Attribute>().data();
    if (ele < 0) {
      ele += rank;
    }
    dim.push_back(ele);
  }

  attrs["axis"] = dim;
}

void AppendAttrForUniformOp(const ::pir::Operation& op,
                            utils::AttributeMap& attrs) {  // NOLINT
  auto attr = op.attributes().at("shape");

  attrs["shape"] = GetVec32FromVec64Attr(attr);
  attrs["dtype"] = "float32";
}

void AppendAttrForBroadcastToOp(const ::pir::Operation& op,
                                utils::AttributeMap& attrs) {  // NOLINT
  auto axes_attr = op.attributes().at("broadcast_axes");
  attrs["broadcast_axes"] = GetVec32FromVec64Attr(axes_attr);

  auto out_shape_attr = op.attributes().at("out_shape");
  attrs["out_shape"] = GetVec32FromVec64Attr(out_shape_attr);
}

void AppendAttrForSliceOp(const ::pir::Operation& op,
                          utils::AttributeMap& attrs) {  // NOLINT
  auto axes_attr = op.attributes().at("axes");
  attrs["axes"] = GetVec32FromVec64Attr(axes_attr);

  auto starts_attr = op.attributes().at("starts");
  attrs["starts"] = GetVec32FromVec64Attr(starts_attr);

  auto ends_attr = op.attributes().at("ends");
  attrs["ends"] = GetVec32FromVec64Attr(ends_attr);

  auto infer_flags_attr = op.attributes().at("infer_flags");
  attrs["infer_flags"] = GetVec32FromVec64Attr(infer_flags_attr);

  auto decrease_axis_attr = op.attributes().at("decrease_axis");
  attrs["decrease_axis"] = GetVec32FromVec64Attr(decrease_axis_attr);
}

}  // namespace

#define REGISTER_OPERAND_RULE(OP, args...)                                    \
  operand_funcs_[paddle::dialect::OP::name()] = []() -> std::vector<size_t> { \
    return {args};                                                            \
  };

#define REGISTER_ATTR_RULE(OP, func) \
  attr_funcs_[cinn::dialect::OP::name()] = func;

#define REGISTER_PD_ATTR_RULE(OP, func) \
  attr_funcs_[paddle::dialect::OP::name()] = func;

void OpMapper::RegisterMapRules() {
  // max(x, dim) -> reduce_max(x)
  REGISTER_OPERAND_RULE(MaxOp, 0);
  REGISTER_OPERAND_RULE(SumOp, 0);
  REGISTER_OPERAND_RULE(MinOp, 0);
  REGISTER_OPERAND_RULE(ProdOp, 0);
  REGISTER_ATTR_RULE(BroadcastOp, AppendAttrForBroadcastToOp);
  REGISTER_ATTR_RULE(UniformRandomOp, AppendAttrForUniformOp);
  REGISTER_PD_ATTR_RULE(TransposeOp, AppendAttrForTransposeOp);
  REGISTER_ATTR_RULE(SliceOp, AppendAttrForSliceOp);
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
