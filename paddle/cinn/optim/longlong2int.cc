// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/longlong2int.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_utils.h"
#include "paddle/cinn/ir/ir_visitor.h"

namespace cinn {
namespace optim {

class CheckOverflow : public ir::IRVisitor {
 public:
  bool is_overflow(Expr* expr) {
    ir::IRVisitor::Visit(expr);
    return is_overflow_;
  }

 private:
  void Visit(const ir::For* for_op) override {
    if (!for_op->extent.is_constant()) return;
    if (!for_op->extent.type().is_index_type()) return;
    if (curr_product_ > INT_MAX) {
      is_overflow_ = true;
      return;
    }
    curr_product_ *= for_op->extent.as_int64();
    ir::IRVisitor::Visit(&for_op->body);
    curr_product_ /= for_op->extent.as_int64();
  }
  int64_t curr_product_ = 1;
  bool is_overflow_ = false;
};

class NarrowLonglong2Int : public ir::IRMutator<> {
 public:
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* op, Expr* expr) override {
    if (expr->type().is_int(64)) {
      expr->get()->set_type(Int(32));
    }
  }
  void Visit(const ir::IntImm* op, Expr* expr) override {
    if (expr->type().is_int(64)) {
      expr->get()->set_type(Int(32));
    }
  }
};

void TryNarrowLonglong2Int(Expr* expr) {
  CheckOverflow check_overflow;
  if (!check_overflow.is_overflow(expr)) {
    NarrowLonglong2Int narrow;
    narrow(expr);
  }
}
}  // namespace optim
}  // namespace cinn
