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

namespace cinn {
namespace ir {

struct _Dim_;

// This fake class is to pass the CI, and will be replaced by
// pir::shape::SymbolicDimOp when pir is completely integrated.
class SymbolicDimOp {
 public:
  const std::string GetSymName() const { return ""; }
  int64_t GetDimSize() const { return 0; }
  bool IsDynamic() const { return false; }
};

//! Wrapper for _Dim_
class Dim : public IrNodeRef {
 public:
  Dim() = default;

  explicit Dim(IrNode* n) : IrNodeRef(n) {}

  operator Expr() const { return Expr(ptr()); }

  const _Dim_* operator->() const;
  _Dim_* operator->();
};

/**
 * Definition of _Dim_.
 */
struct _Dim_ : ExprNode<_Dim_> {
  //! The name of this struct.
  std::string name;
  // (TODO: zhangzheng) Replace this fake class by pir::shape::SymbolicDimOp
  SymbolicDimOp sym_dim;
  Expr dim_expr;

  SymbolicDimOp GetSymbolicDim() const;

  bool IsDynamic() const;

  std::string GetSymbolName() const;

  int64_t GetRealDimSize() const;

  Expr GetDimExpr() const;

  static Dim Make(const std::string& name, const SymbolicDimOp& sym_dim);

  static const IrNodeTy _node_type_ = IrNodeTy::_Dim_;
};

}  // namespace ir
}  // namespace cinn
