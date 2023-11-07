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

void AppendAttrForReduceOp(const ::pir::Operation& op,
                           utils::AttributeMap& attrs) {  // NOLINT
  auto attr = op.attributes().at("dim");
  auto attr_vec = attr.dyn_cast<::pir::ArrayAttribute>().AsVector();

  std::vector<int> dim;
  for (auto vec_element : attr_vec) {
    dim.push_back(vec_element.dyn_cast<::pir::Int64Attribute>().data());
  }

  attrs["dim"] = dim;
}

void AppendAttrForUniformOp(const ::pir::Operation& op,
                            utils::AttributeMap& attrs) {  // NOLINT
  auto attr = op.attributes().at("shape");
  auto attr_vec = attr.dyn_cast<::pir::ArrayAttribute>().AsVector();

  std::vector<int> shape;
  for (auto vec_element : attr_vec) {
    shape.push_back(vec_element.dyn_cast<::pir::Int64Attribute>().data());
  }

  std::cerr << "append dim" << std::endl;
  attrs["shape"] = shape;
  attrs["dtype"] = "float32";
}

void AppendAttrForBoadcastToOp(const ::pir::Operation& op,
                               utils::AttributeMap& attrs) {  // NOLINT
  auto axes_attr = op.attributes().at("broadcast_axes");
  auto attr_vec = axes_attr.dyn_cast<::pir::ArrayAttribute>().AsVector();

  std::vector<int> axis;
  for (auto vec_element : attr_vec) {
    axis.push_back(vec_element.dyn_cast<::pir::Int64Attribute>().data());
  }

  attrs["broadcast_axes"] = axis;

  auto out_shape_attr = op.attributes().at("out_shape");
  auto out_shape_attr_vec =
      out_shape_attr.dyn_cast<::pir::ArrayAttribute>().AsVector();

  std::vector<int> out_shape;
  for (auto vec_element : out_shape_attr_vec) {
    out_shape.push_back(vec_element.dyn_cast<::pir::Int64Attribute>().data());
  }

  attrs["out_shape"] = out_shape;
}

}  // namespace

#define REGISTER_OPERAND_RULE(OP, args...)                                    \
  operand_funcs_[paddle::dialect::OP::name()] = []() -> std::vector<size_t> { \
    return {args};                                                            \
  };

#define REGISTER_ATTR_RULE(OP, func) \
  attr_funcs_[cinn::dialect::OP::name()] = func;

void OpMapper::RegisterMapRules() {
  // max(x, dim) -> reduce_max(x)
  REGISTER_OPERAND_RULE(MaxOp, 0);
  REGISTER_OPERAND_RULE(SumOp, 0);
  REGISTER_OPERAND_RULE(MinOp, 0);
  REGISTER_OPERAND_RULE(ProdOp, 0);
  REGISTER_ATTR_RULE(ReduceMaxOp, AppendAttrForReduceOp);
  REGISTER_ATTR_RULE(ReduceSumOp, AppendAttrForReduceOp);
  REGISTER_ATTR_RULE(BroadcastOp, AppendAttrForBoadcastToOp);
  REGISTER_ATTR_RULE(UniformRandomOp, AppendAttrForUniformOp);
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
