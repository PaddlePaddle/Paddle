// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/ir_base.h"
#include "paddle/pir/include/dialect/shape/ir/shape_op.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace cinn {
namespace ir {

struct _Dim_;

//! Wrapper for _Dim_
class Dim : public IrNodeRef {
 public:
  Dim() = default;

  explicit Dim(IrNode* n) : IrNodeRef(n) {}

  operator Expr() const { return Expr(ptr()); }

  Dim(const std::string& name, const symbol::DimExpr& sym_dim);

  const _Dim_* operator->() const;
  _Dim_* operator->();
};

/**
 * Definition of _Dim_.
 */
struct _Dim_ : ExprNode<_Dim_> {
  //! The name of this struct.
  std::string name;
  symbol::DimExpr sym_dim;
  Expr dim_expr;

  bool IsUniSymbolic() const;

  std::string ToString() const;

  Expr GetDimExpr() const;

  static Dim Make(const std::string& name, const symbol::DimExpr& sym_dim);

  static const IrNodeTy _node_type_ = IrNodeTy::_Dim_;
};

}  // namespace ir
}  // namespace cinn
