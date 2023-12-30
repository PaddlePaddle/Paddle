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

#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace ir {

using pir::shape::SymbolicDimOp;

const _Dim_* Dim::operator->() const { return As<_Dim_>(); }
_Dim_* Dim::operator->() { return As<_Dim_>(); }

SymbolicDimOp _Dim_::GetSymbolicDim() const { return sym_dim; }

bool _Dim_::IsDynamic() const { return sym_dim.IsDynamic(); }

std::string _Dim_::GetSymbolName() const { return sym_dim.GetSymName(); }

int64_t _Dim_::GetRealDimSize() const { return sym_dim.GetDimSize(); }

Expr _Dim_::GetDimExpr() const { return dim_expr; }

Dim _Dim_::Make(const std::string& name, const SymbolicDimOp& sym_dim) {
  auto* n = make_shared<_Dim_>();
  n->name = name;
  n->sym_dim = sym_dim;
  if (sym_dim.IsDynamic()) {
    n->dim_expr = Expr(Var(sym_dim.GetSymName(), cinn::common::Int(32)));
  } else {
    n->dim_expr = Expr(static_cast<int32_t>(sym_dim.GetDimSize()));
  }

  return Dim(n);
}

Dim::Dim(const std::string& name, const SymbolicDimOp& sym_dim)
    : IrNodeRef(_Dim_::Make(name, sym_dim).self()) {}

}  // namespace ir
}  // namespace cinn
