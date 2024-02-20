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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

// -------------------
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_symbolic_shape.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
// -------------------

namespace paddle::dialect::details {

template <typename T = int64_t>
std::vector<T> GetVectorAttr(const ::pir::Operation *op,
                             const std::string &name) {
  using value_type = typename AttributeTrait<T>::value_type;

  const auto &attr_map = op->attributes();
  PADDLE_ENFORCE(
      attr_map.count(name),
      phi::errors::PreconditionNotMet(
          "attr [%s] MUST in attribute map for [%s] op", name, op->name()));
  const auto &val = attr_map.at(name);

  PADDLE_ENFORCE(val.isa<::pir::ArrayAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "axis Type MUST ArrayAttribute for [%s] op", op->name()));
  auto array_list = val.dyn_cast<::pir::ArrayAttribute>().AsVector();
  std::vector<T> vec_res;
  if (array_list.size() > 0) {
    PADDLE_ENFORCE_EQ(array_list[0].isa<value_type>(),
                      true,
                      phi::errors::Unimplemented(
                          "the 0th elementwise MUST be ir::Int64Attribute"));
    for (size_t i = 0; i < array_list.size(); ++i) {
      vec_res.push_back(array_list[i].dyn_cast<value_type>().data());
    }
  }
  return vec_res;
}

bool ReduceInferDim(pir::Operation *op,
                    pir::ShapeConstraintIRAnalysis *shape_analysis,
                    const std::vector<int64_t> &axis,
                    bool keep_dim,
                    bool reduce_all) {
  auto x = op->operand_source(0);
  int x_rank = x.type().dyn_cast<pir::DenseTensorType>().dims().size();

  const std::vector<int64_t> formated_axis = [&] {
    std::vector<int64_t> formated_axis = axis;
    for (size_t i = 0; i < axis.size(); ++i) {
      if (axis[i] < 0) {
        formated_axis[i] = axis[i] + x_rank;
      }
    }
    return formated_axis;
  }();

  bool full_dim = true;
  std::set<int64_t> dims_set(formated_axis.begin(), formated_axis.end());
  for (int64_t i = 0; i < x_rank; ++i) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  bool empty_dim = axis.size() == 0;
  reduce_all = reduce_all || full_dim || empty_dim;

  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(x);
  std::vector<symbol::DimExpr> input_shapes;
  if (x_shape_or_data.data() == std::nullopt ||
      x_shape_or_data.data()->size() == 0) {
    input_shapes = x_shape_or_data.shape();
  } else {
    input_shapes = *x_shape_or_data.data();
  }

  const std::vector<symbol::DimExpr> shapes = [&] {
    std::vector<symbol::DimExpr> shapes;
    for (int i = 0; i < x_rank; ++i) {
      if (reduce_all || dims_set.find(i) != dims_set.end()) {
        if (keep_dim) {
          shapes.push_back(1);
        } else {
          continue;
        }
      } else {
        shapes.push_back(input_shapes.at(i));
      }
    }
    return shapes;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(shapes)};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

}  // namespace paddle::dialect::details
