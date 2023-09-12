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
#include <set>
#include <string>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/utils/ir_mutator.h"

namespace cinn {
namespace optim {

struct TensorWriteTeller : public ir::IRMutator<const Expr*> {
  //! Collect the write info in \p op.
  void Collect(const Expr* op) { Visit(op, op); }

  bool IsWrite(const std::string& tensor_name) const {
    return tensor_written.count(tensor_name);
  }

 private:
  std::set<std::string> tensor_written;

  void Visit(const Expr* expr, const Expr* op) override {
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::Store* expr, const Expr* op) override {
    auto* node = op->As<ir::Store>();
    CHECK(node);
    auto* tensor = node->tensor.As<ir::_Tensor_>();
    CHECK(tensor);
    tensor_written.insert(tensor->name);
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::_Tensor_* op, const Expr* expr) override {
    auto* node = expr->As<ir::_Tensor_>();
    if (node->is_call_node()) {
      tensor_written.insert(node->name);
    }
  }
};

}  // namespace optim
}  // namespace cinn
