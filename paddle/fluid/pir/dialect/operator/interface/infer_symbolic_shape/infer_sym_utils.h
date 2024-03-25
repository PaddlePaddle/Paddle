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

#pragma once

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

COMMON_DECLARE_string(pir_dyshape_sym2value);

inline bool GetBoolAttr(const pir::Operation *op, const std::string &str) {
  const auto &attr_map = op->attributes();
  PADDLE_ENFORCE(
      attr_map.count(str),
      phi::errors::PreconditionNotMet(
          "attr [%s] MUST in attribute map for [%s] op", str, op->name()));
  return attr_map.at(str).dyn_cast<pir::BoolAttribute>().data();
}

// To make codes shorter
using ExprVec = std::vector<symbol::DimExpr>;
using ShapeOrData = symbol::ShapeOrDataDimExprs;
using TensorExprs = symbol::TensorShapeOrDataDimExprs;
using TensorListExprs = symbol::TensorListShapeOrDataDimExprs;

namespace paddle::dialect::details {
template <typename T>
struct AttributeTrait;

template <>
struct AttributeTrait<std::int64_t> {
  using value_type = ::pir::Int64Attribute;
};

template <>
struct AttributeTrait<int> {
  using value_type = ::pir::Int32Attribute;
};

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

inline ExprVec GetExprVecFromData(const ShapeOrData &shapeordata) {
  if (shapeordata.isa<TensorListExprs>()) {
    ExprVec result;
    TensorListExprs list =
        shapeordata.dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
    for (size_t i = 0; i < list.size(); i++) {
      for (auto expr : list[i].data().value()) {
        result.emplace_back(expr);
      }
    }
    return result;
  } else {
    return shapeordata.data().value();
  }
}

inline ExprVec GetExprVecFromShape(const ShapeOrData &shapeordata) {
  const auto GetShapeExprsFromList = [&]() {
    ExprVec result;
    TensorListExprs list =
        shapeordata.dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
    for (size_t i = 0; i < list.size(); i++) {
      for (auto expr : list[i].data().value()) {
        result.emplace_back(expr);
      }
    }
    return result;
  };
  if (shapeordata.isa<TensorListExprs>()) {
    return GetShapeExprsFromList();
  } else {
    return shapeordata.shape();
  }
}

std::optional<std::vector<int64_t>> VecExpr2Int64(const ExprVec &expr_vec);

ExprVec VecInt642Expr(const std::vector<int64_t> &int_vec);

template <typename T = int64_t>
inline std::string PrintVec(const std::vector<T> &vec) {
  std::ostringstream os;
  os << "[";

  for (size_t idx = 0; idx < vec.size(); idx++) {
    os << vec[idx];
    if (idx < vec.size() - 1) os << ", ";
  }
  os << "]";

  return os.str();
}

inline std::string PrintShapeOrData(const ShapeOrData &shapeordata) {
  std::ostringstream os;
  os << "{" << shapeordata << "]";
  return os.str();
}

inline std::vector<std::string> Split(const std::string &str,
                                      const std::string &splitter) {
  std::vector<std::string> results;
  std::string::size_type pos1, pos2;
  pos2 = str.find(splitter);
  pos1 = 0;
  while (std::string::npos != pos2) {
    results.push_back(str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + splitter.size();
    pos2 = str.find(splitter, pos1);
  }
  if (pos1 != str.length()) {
    results.push_back(str.substr(pos1));
  }
  return results;
}

inline std::unordered_map<symbol::DimExpr, symbol::DimExpr>
GetSymValueFromFlag() {
  std::unordered_map<symbol::DimExpr, symbol::DimExpr> map_sym2value;
  std::string &flag_str = FLAGS_pir_dyshape_sym2value;
  for (auto str : Split(flag_str, ",")) {
    std::vector<std::string> sym_value = Split(str, "=");
    PADDLE_ENFORCE_EQ(sym_value.size(),
                      2,
                      phi::errors::OutOfRange(
                          "FLAGS_pir_dyshape_sym2value's format should be like "
                          "'S0=1,S1=128', but receive too many '='s."));
    map_sym2value[sym_value[0]] = std::stoi(sym_value[1]);
  }

  return map_sym2value;
}

inline void CheckSymShapeByValue(
    const std::int64_t op_id,
    const std::string &op_name,
    const ::common::DDim &ddim,
    const ShapeOrData &shapeordata,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>
        &additional_cstrs = {}) {
  std::string op_info = "op_" + std::to_string(op_id) + "(" + op_name + ")";
  auto sym_value_map = GetSymValueFromFlag();
  if (shapeordata.isa<TensorListExprs>()) {
    VLOG(3) << "********** " << op_info
            << " 's shapeordata.isa<TensorListExprs>()";
  } else {
    auto sym_shape = shapeordata.shape();
    std::vector<std::int64_t> sym_value_shape;
    for (auto dim_expr : sym_shape) {
      symbol::DimExpr substitute_expr =
          symbol::SubstituteDimExpr(dim_expr, sym_value_map);
      symbol::DimExpr ret = symbol::SimplifyDimExpr(substitute_expr);
      PADDLE_ENFORCE_EQ(ret.Has<std::int64_t>(),
                        true,
                        platform::errors::PreconditionNotMet(
                            "after SubstituteDimExpr&SimplifyDimExpr, dim_expr "
                            "must have int value"));
      sym_value_shape.emplace_back(ret.Get<std::int64_t>());
    }

    auto real_shape = ::common::vectorize(ddim);
    std::ostringstream os;

    std::string real_shape_str = "real_shape" + PrintVec(real_shape);
    std::string sym_shape_str = "sym_val_shape" + PrintVec(sym_value_shape) +
                                " -> " + PrintShapeOrData(shapeordata);

    if (real_shape != sym_value_shape) {
      VLOG(3) << "!!!!! [ShapeCheckFailed] " << op_info << ": "
              << real_shape_str << " != " << sym_shape_str;
    } else {
      VLOG(3) << "===== [ShapeCheckPassed] op_" << op_info << ": "
              << real_shape_str << " == " << sym_shape_str;
    }
  }
}

bool ReduceInferDim(pir::Operation *op,
                    pir::ShapeConstraintIRAnalysis *shape_analysis,
                    const std::vector<int64_t> &axis,
                    bool keep_dim,
                    bool reduce_all);

void BuildCstrEqForTensorListAlongAxis(
    pir::ShapeConstraintIRAnalysis *shape_analysis,
    const symbol::TensorListShapeOrDataDimExprs &shape_data_list,
    int axis);

void BuildCstrEqForTensorListAlongAxis(
    pir::ShapeConstraintIRAnalysis *shape_analysis,
    const std::vector<pir::Value> &values,
    int axis);

}  // namespace paddle::dialect::details
