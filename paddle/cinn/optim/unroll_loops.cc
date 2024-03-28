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

#include "paddle/cinn/optim/unroll_loops.h"

#include <utility>
#include <vector>

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_replace.h"

namespace cinn {
namespace optim {

namespace {

struct UnrollMutator : public ir::IRMutator<Expr*> {
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  // update auto_max_step_ from the specific attribute of ScheduleBlock
  void Visit(const ir::ScheduleBlock* op, Expr* expr) override {
    auto attr_it = op->attrs.find(ir::attr::auto_unroll_max_step);
    if (attr_it != op->attrs.end()) {
      const int* attr_v = absl::get_if<int>(&attr_it->second);
      if (attr_v) {
        int value = *attr_v;
        std::swap(auto_max_step_, value);
        VLOG(5) << "auto_max_step is updated:" << auto_max_step_;
        ir::IRMutator<>::Visit(op, expr);
        std::swap(auto_max_step_, value);
        return;
      } else {
        LOG(WARNING) << "Get invalid value of attr:"
                     << ir::attr::auto_unroll_max_step;
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  // count a Store node as plain statement
  void Visit(const ir::Store* op, Expr* expr) override {
    IRMutator<>::Visit(op, expr);
    ++flat_step_;
  }

  // predicate whether a for-loop can be unrolled and do it
  void Visit(const ir::For* op, Expr* expr) override {
    IRMutator<>::Visit(op, expr);
    if (op->extent.As<ir::IntImm>() == nullptr) {
      VLOG(5) << "loop to be unrolled should have a constant extent";
      return;
    }
    int64_t extent = op->extent.as_int64();

    // predicate this for-loop can be unrolled by auto-unroll conditions
    bool unrollable =
        (op->is_serial() && extent >= 0 && not_unrolled_depth_ == 0 &&
         extent * flat_step_ <= auto_max_step_);

    // predicate this for-loop can be unrolled by the unrolled tag
    unrollable =
        (unrollable || op->is_unrolled()) && extent <= max_unroll_extent_;

    if (unrollable) {
      Unroll(op, expr);
      flat_step_ *= extent;
    } else {
      ++not_unrolled_depth_;
    }
  }

  //! Unroll a forloop.
  void Unroll(const ir::For* op, Expr* expr) {
    std::vector<Expr> body;

    auto* min = op->min.As<ir::IntImm>();
    auto* extent = op->extent.As<ir::IntImm>();
    if (!(min && extent)) return;

    for (int i = min->value; i < extent->value; i++) {
      Expr start = op->min + i;
      body.push_back(
          ir::ir_utils::IRCopy(op->body, /* copy_buffer_node = */ false));
      cinn::ir::ir_utils::IrReplaceVarBroadcast(
          &body.back(), op->loop_var, start);
    }

    *expr = ir::Block::Make(body);
  }

 private:
  // max permitted steps to be automatically unrolled in total
  int auto_max_step_ = 0;
  // max permitted extent of a loop to be unrolled
  int max_unroll_extent_ = 50;

  // the number of steps that have been unrolled or plain statement
  int64_t flat_step_ = 0;
  // the number of nested loops not to be unrolled
  int not_unrolled_depth_ = 0;
};

}  // namespace

void UnrollLoop(Expr* expr) { UnrollMutator()(expr); }

}  // namespace optim
}  // namespace cinn
