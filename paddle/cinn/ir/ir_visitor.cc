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

#include <unordered_set>

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {

template <typename T>
static bool CompareExpressions(const ir::IndexExpr& a, const ir::IndexExpr& b) {
  auto aPart = common::GetFlattenExprs<T>(a);
  auto bPart = common::GetFlattenExprs<T>(b);

  std::sort(aPart.begin(), aPart.end(), common::ComparePriority);
  std::sort(bPart.begin(), bPart.end(), common::ComparePriority);

  if (aPart.size() != bPart.size()) return false;

  size_t i = 0;
  while (i < aPart.size()) {
    if (!common::ComparePriority(aPart[i], bPart[i])) return false;
    std::vector<std::pair<ir::IndexExpr, int>> aGroup, bGroup;

    do {
      aGroup.emplace_back(aPart[i], 0);
      bGroup.emplace_back(bPart[i], 0);
      ++i;
    } while (i < aPart.size() &&
             common::ComparePriority(aPart[i - 1], aPart[i]) == 1 &&
             common::ComparePriority(bPart[i - 1], bPart[i]) == 1);

    // compare expressions with same priority.
    for (size_t k = 0; k < aGroup.size(); ++k) {
      for (auto& b : bGroup) {
        if (b.second == 0 && aGroup[k].first == b.first) {
          b.second = 1;
          aGroup[k].second = 1;
          break;
        }
      }
      if (aGroup[k].second == 0) return false;
    }
  }

  return true;
}

bool operator==(Expr a, Expr b) {
  if (a.get() == b.get())
    return true;
  else if (a.is_index() && b.is_index())
    return a.as_index() == b.as_index();
  else
    return ir_utils::IRCompare(a, b);
}

bool operator!=(Expr a, Expr b) { return !(a == b); }

bool operator==(IndexExpr a, Expr b) {
  return b.is_index() ? a == b.as_index() : Expr(a) == b;
}

bool operator!=(IndexExpr a, Expr b) { return !(a == b); }

bool operator==(Expr a, IndexExpr b) {
  return a.is_index() ? a.as_index() == b : a == Expr(b);
}

bool operator!=(Expr a, IndexExpr b) { return !(a == b); }

bool operator==(IndexExpr a, IndexExpr b) {
  if (a.get() == b.get()) return true;
  if (a.node_type() != b.node_type()) return false;
  std::vector<ir::IndexExpr> aPart;
  std::vector<ir::IndexExpr> bPart;
  switch (a.node_type()) {
    case ir::IrNodeTy::IntImm: {
      return a.as_int64() == b.as_int64();
    }
    case ir::IrNodeTy::_Var_: {
      return a.as_var()->name == b.as_var()->name;
    }
    case ir::IrNodeTy::Cast: {
      auto lhs = a.As<ir::Cast>();
      auto rhs = b.As<ir::Cast>();
      return lhs->type() == rhs->type() && lhs->v() == rhs->v();
    }
    case ir::IrNodeTy::Load: {
      auto lhs = a.As<ir::Load>();
      auto rhs = b.As<ir::Load>();
      if (lhs->indices.size() != rhs->indices.size()) return false;
      if (lhs->tensor != rhs->tensor) return false;
      // compare indices
      for (int32_t i = 0; i < lhs->indices.size(); ++i) {
        if (lhs->indices[i] != rhs->indices[i]) return false;
      }
      return true;
    }
    case ir::IrNodeTy::Div:
    case ir::IrNodeTy::Mod: {
      return a.operand(0) == b.operand(0) && a.operand(1) == b.operand(1);
    }
    case ir::IrNodeTy::Add:
      return CompareExpressions<ir::Add>(a, b);
    case ir::IrNodeTy::Mul:
      return CompareExpressions<ir::Mul>(a, b);
    case ir::IrNodeTy::Min:
      return CompareExpressions<ir::Min>(a, b);
    case ir::IrNodeTy::Max:
      return CompareExpressions<ir::Max>(a, b);
  }
  return false;
}

bool operator!=(IndexExpr a, IndexExpr b) { return !(a == b); }

}  // namespace ir
}  // namespace cinn
