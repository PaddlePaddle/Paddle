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

#include "paddle/cinn/ir/utils/ir_replace.h"

#include <set>

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {
namespace ir_utils {
using utils::GetStreamCnt;

namespace {

struct IrReplaceMutator : ir::IRMutator<Expr*> {
  std::set<ir::IrNodeTy> valid_nodetys{
      {ir::IrNodeTy::Broadcast, ir::IrNodeTy::_Var_}};

  IrReplaceMutator(ir::Expr from, Expr to)
      : from_(from), to_(to), from_repr_(GetStreamCnt(from)) {
    CHECK(valid_nodetys.count(from->node_type()))
        << "Not valid node type got " << from->node_type();
  }
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* op, Expr* expr) override {
    if (op->node_type() == from_->node_type() &&
        from_repr_ == GetStreamCnt(*expr)) {
      *expr = ir::ir_utils::IRCopy(to_);
    }
  }

  void Visit(const ir::Broadcast* op, Expr* expr) override {
    if (op->node_type() == from_->node_type() &&
        from_repr_ == GetStreamCnt(*expr)) {
      *expr = ir::ir_utils::IRCopy(to_);
    }
  }

  std::string from_repr_;
  ir::Expr from_;
  Expr to_;
};

}  // namespace

void IrReplace(ir::Expr* expr, ir::Expr from, ir::Expr to) {
  CHECK(expr);
  IrReplaceMutator(from, to)(expr);
}

}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
