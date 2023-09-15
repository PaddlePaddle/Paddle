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

#include "paddle/cinn/optim/remove_nested_block.h"

#include "paddle/cinn/ir/utils/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_printer.h"

namespace cinn {
namespace optim {

Expr GetExprInsideBlock(Expr op) {
  Expr node = op;
  while (node.As<ir::Block>()) {
    auto& stmts = node.As<ir::Block>()->stmts;
    if (stmts.size() == 1) {
      node = stmts.front();
    } else {
      break;
    }
  }
  return node;
}

// This will remove the nested blocks, but it will also remove the block outside
// the forloop's body.
struct NestedBlockSimplifer : public ir::IRMutator<Expr*> {
  void operator()(ir::Expr* expr) { Visit(expr); }

 private:
  void Visit(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Block* expr, Expr* op) override {
    auto* node = op->As<ir::Block>();
    if (node->stmts.size() == 1) {
      *op = GetExprInsideBlock(*op);
      IRMutator::Visit(op, op);
    } else {
      IRMutator::Visit(expr, op);
    }
  }
};

struct NestedBlockRemover : public ir::IRMutator<Expr*> {
  void operator()(ir::Expr* expr) { Visit(expr); }

 private:
  void Visit(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Block* expr, Expr* op) override {
    auto* node = op->As<ir::Block>();

    std::vector<ir::Expr> new_exprs;

    bool detect_nested = false;
    for (auto it = node->stmts.begin(); it != node->stmts.end(); it++) {
      auto* block = it->As<ir::Block>();
      if (block) {
        detect_nested = true;
        new_exprs.insert(
            std::end(new_exprs), block->stmts.begin(), block->stmts.end());
      } else {
        new_exprs.push_back(*it);
      }
    }

    node->stmts = new_exprs;

    IRMutator::Visit(expr, op);
  }
};

// add block outside forloop's body.
struct AddBlockToForloop : public ir::IRMutator<> {
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::For* expr, Expr* op) override {
    auto* node = op->As<ir::For>();
    if (!node->body.As<ir::Block>()) {
      node->body = ir::Block::Make({node->body});
    }

    ir::IRMutator<>::Visit(expr, op);
  }

  void Visit(const ir::PolyFor* expr, Expr* op) override {
    auto* node = op->As<ir::PolyFor>();
    if (!node->body.As<ir::Block>()) {
      node->body = ir::Block::Make({node->body});
    }

    ir::IRMutator<>::Visit(expr, op);
  }

  void Visit(const ir::_LoweredFunc_* expr, Expr* op) override {
    auto* node = op->As<ir::_LoweredFunc_>();
    if (!node->body.As<ir::Block>()) {
      node->body = ir::Block::Make({node->body});
    }

    ir::IRMutator<>::Visit(expr, op);
  }
};

void RemoveNestedBlock(Expr* e) {
  NestedBlockRemover()(e);
  NestedBlockSimplifer()(e);
  AddBlockToForloop()(e);
}

}  // namespace optim
}  // namespace cinn
