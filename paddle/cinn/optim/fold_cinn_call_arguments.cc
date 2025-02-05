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

#include "paddle/cinn/optim/fold_cinn_call_arguments.h"

#include <unordered_set>
#include <vector>

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

namespace {

/**
 * Fold the arguments of the Call nodes marked as CINN(calls an LoweredFunc).
 */
struct FoldCINNCallArgumentsMutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;
  void operator()(ir::LoweredFunc fn) { Visit(fn.As<ir::_LoweredFunc_>()); }

 private:
  void Visit(const ir::Block* op, Expr* expr) override {
    auto* node = expr->As<ir::Block>();
    for (auto it = node->stmts.begin(); it != node->stmts.end();) {
      if (it->As<ir::Store>()) {
        auto* call = it->As<ir::Store>()->value.As<ir::Call>();
        if (call && call->is_cinn_call()) {
          // remove the duplicate calls.
          std::string key = utils::GetStreamCnt(Expr(call));
          if (visited_call_.count(key)) {
            it = node->stmts.erase(it);
            continue;
          }

          ir::IRMutator<>::Visit(&(*it), &(*it));
          visited_call_.insert(key);
          continue;
        }
      }

      ir::IRMutator<>::Visit(&(*it), &(*it));
      ++it;
    }
  }
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();
    if (node->value.As<ir::Call>()) {
      auto* call = node->value.As<ir::Call>();
      switch (call->call_type) {
        case ir::CallType::CINN:
          MutateCall(call);
          *expr = node->value;
          break;
        case ir::CallType::Intrinsic:
          break;
        case ir::CallType::Extern:
          break;
        default:
          CINN_NOT_IMPLEMENTED
      }
    }
  }

  void MutateCall(ir::Call* call) {
    if (call->call_type == ir::CallType::Extern) return;

    std::vector<Expr> read_args;
    std::vector<Expr> write_args;
    for (auto& arg : call->read_args) {
      if (arg.as_tensor()) {
        PADDLE_ENFORCE_EQ(arg.as_tensor()->buffer.defined(),
                          true,
                          ::common::errors::InvalidArgument(
                              "Expected tensor [%s] to have a defined buffer, "
                              "but the buffer is not defined.",
                              arg.as_tensor()->name.c_str()));
        read_args.push_back(arg.as_tensor()->buffer);
      } else {
        read_args.push_back(arg);
      }
    }

    for (auto& arg : call->write_args) {
      if (arg.as_tensor()) {
        write_args.push_back(arg.as_tensor()->buffer);
      } else {
        write_args.push_back(arg);
      }
    }

    call->read_args = read_args;
    call->write_args = write_args;
  }

 private:
  // To avoid the same call triggered duplicately.
  std::unordered_set<std::string> visited_call_;
};

}  // namespace

void FoldCINNCallArguments(ir::LoweredFunc fn) {
  FoldCINNCallArgumentsMutator()(fn);
}

}  // namespace optim
}  // namespace cinn
