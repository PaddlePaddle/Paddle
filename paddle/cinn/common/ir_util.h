// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <absl/container/flat_hash_map.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/common/bfloat16.h"
#include "paddle/cinn/common/float16.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace common {

Expr IndiceToAbsOffset(const std::vector<Expr> &shape,
                       const std::vector<Expr> &indices);
Expr IndiceToAbsOffset(const std::vector<int> &shape,
                       const std::vector<Expr> &indices);

Expr PrecedingAxisToAbsOffset(const std::vector<Expr> &shape,
                              int preceding_n_axis);

Expr CastIfNeeded(Expr body, Type type);

//! Substitute vars to other expressions.
//! @param expr The expression to do modification.
//! @param var_map The map from variables to the target expressions.
void Substitute(Expr *expr, const std::map<const ir::_Var_ *, Expr> &var_map);

//! Get a stack of forloops(For and PolyFor nodes) to a Store node target to \p
//! tensor_name
std::vector<Expr *> GetForloopStackToStore(Expr *expr,
                                           const std::string &tensor_name);

// make const
// @{
inline Expr make_const(int32_t x) { return Expr(static_cast<int32_t>(x)); }
inline Expr make_const(int64_t x) { return Expr(static_cast<int64_t>(x)); }
inline Expr make_const(bfloat16 x) { return Expr(static_cast<bfloat16>(x)); }
inline Expr make_const(float16 x) { return Expr(static_cast<float16>(x)); }
inline Expr make_const(float x) { return Expr(static_cast<float>(x)); }
inline Expr make_const(double x) { return Expr(static_cast<double>(x)); }
inline Expr make_const(bool x) { return Expr(static_cast<bool>(x)); }
// @}

//! maker for some general consts.
// @{
template <typename T = int32_t>
inline Expr make_zero() {
  return make_const(static_cast<T>(0));
}
template <typename T = int32_t>
inline Expr make_one() {
  return make_const(static_cast<T>(1));
}
inline Expr make_bool(bool x) {
  return common::make_shared<ir::UIntImm>(Bool(), x);
}
inline Expr make_bool(bool x, int lanes) {
  return common::make_shared<ir::UIntImm>(Bool(lanes), x);
}
// @}

/**
 * \brief Check all the tensors are unique in an expression.
 */
void CheckTensorUniqueInExpr(Expr expr);

/**
 * \brief Check all the buffers are uniuqe in an expression.
 */
void CheckBufferUniqueInExpr(Expr expr);

std::vector<std::string> GatherItersToTensorProducer(
    const std::string &target_tensor_name, Expr *expr);

bool is_zero(Expr v);

bool MathEqual(const Expr &a, const Expr &b);

//! helper function to get a ir::Select node.
Expr select(Expr cond, Expr true_value, Expr false_value);

//! helper function to get the And of all the conditions.
Expr and_all(const std::vector<Expr> &conds);

//! helper function to get the Or of all the conditions.
Expr or_any(const std::vector<Expr> &conds);

//! Cast the expression \p e to type \type.
Expr cast(Expr e, Type type);

Expr max(Expr a, Expr b);

Expr min(Expr a, Expr b);

template <typename T>
Expr make_const(Type t, T v) {
  if (t.is_vector()) {
    if (t.is_int()) {
      return ir::Broadcast::Make(
          make_shared<ir::IntImm>(t.ElementOf(), static_cast<int64_t>(v)),
          t.lanes());
    } else if (t.is_uint()) {
      return ir::Broadcast::Make(
          make_shared<ir::UIntImm>(t.ElementOf(), static_cast<uint64_t>(v)),
          t.lanes());
    } else if (t.is_float()) {
      return ir::Broadcast::Make(
          make_shared<ir::FloatImm>(t.ElementOf(), static_cast<double>(v)),
          t.lanes());
    } else if (t.is_bool()) {
      return ir::Broadcast::Make(
          make_shared<ir::UIntImm>(t.ElementOf(), static_cast<bool>(v)),
          t.lanes());
    } else {
      CINN_NOT_IMPLEMENTED
    }
  } else {
    if (t.is_int()) {
      return make_shared<ir::IntImm>(t, static_cast<int64_t>(v));
    } else if (t.is_uint()) {
      return make_shared<ir::UIntImm>(t, static_cast<uint64_t>(v));
    } else if (t.is_float()) {
      return make_shared<ir::FloatImm>(t, static_cast<double>(v));
    } else if (t.is_bool()) {
      return make_shared<ir::UIntImm>(t, static_cast<bool>(v));
    } else {
      CINN_NOT_IMPLEMENTED
    }
  }
  return Expr();
}

template <typename FuncOp>
Expr FoldExpr(FuncOp func_op, const std::vector<Expr> &values) {
  Expr init_value;
  for (const Expr &val : values) {
    if (!init_value.defined()) {
      init_value = val;
    } else {
      init_value = func_op(val, init_value);
    }
  }
  return init_value;
}

}  // namespace common
}  // namespace cinn
