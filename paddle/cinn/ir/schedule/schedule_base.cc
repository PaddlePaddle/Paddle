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

#include "paddle/cinn/ir/schedule/schedule_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"

namespace cinn {
namespace ir {

/**
 * Replace a node to another node.
 * @param src_sref The node to be changed.
 * @param tgt_stmt The node we want.
 */
void ScheduleBase::Replace(const Expr& src_sref, const Expr& tgt_stmt) {
  CHECK(src_sref.As<ir::For>() || src_sref.As<ir::Block>() ||
        src_sref.As<ir::ScheduleBlockRealize>());
  CHECK(tgt_stmt.As<ir::For>() || tgt_stmt.As<ir::Block>() ||
        tgt_stmt.As<ir::ScheduleBlockRealize>());
  if (src_sref == tgt_stmt) {
    return;
  }
  struct ForLoopMutator : public ir::IRMutator<> {
    ForLoopMutator(const Expr& source, const Expr& target)
        : source_(source), target_(target) {}

    void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

    void Visit(const ir::For* op, Expr* expr) override {
      if (*expr == source_) {
        *expr = target_;
        return;
      }
      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) override {
      if (*expr == source_) {
        *expr = target_;
        return;
      }
      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::Block* op, Expr* expr) override {
      if (*expr == source_) {
        *expr = target_;
        return;
      }
      ir::IRMutator<>::Visit(op, expr);
    }

    const Expr& source_;
    const Expr& target_;
  };
  auto exprs = module_expr_.GetExprs();
  ForLoopMutator mutator(src_sref, tgt_stmt);
  for (auto& i : exprs) {
    mutator(&i);
  }
}

void ScheduleBase::BroadcastToElementwise(const std::string& block_name,
                                          const std::vector<int64_t>& axes) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  Expr broadcast_body = all_loops.back().As<ir::For>()->body;

  auto schedule_realize = broadcast_body.As<ir::Block>()
                              ->expr_fields()[0]
                              ->As<ir::ScheduleBlockRealize>();
  auto schedule_block =
      schedule_realize->schedule_block.As<ir::ScheduleBlock>();
  auto iter_vars = schedule_block->iter_vars;

  auto load_exprs = ir::ir_utils::CollectIRNodesInOrder(
      schedule_block->body, [&](const Expr* x) { return x->As<ir::Load>(); });

  for (auto load_expr : load_exprs) {
    auto load = load_expr.As<ir::Load>();
    load->indices.resize(all_loops.size(), Expr(0));

    for (size_t i = 0; i < axes.size(); ++i) {
      load->indices[axes[i]] = schedule_block->iter_vars[axes[i]];
    }
  }
}

void ScheduleBase::Broadcast(const std::string& block_name,
                             const BroadcastInfo& info) {
  auto axes = info.broadcast_axes;
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  if (axes[0] >= all_loops.size()) {
    throw std::runtime_error("axes execeed loop size");
  }

  // Get Last loop
  Expr broadcast_body = all_loops.back().As<ir::For>()->body;

  auto schedule_realize = broadcast_body.As<ir::Block>()
                              ->expr_fields()[0]
                              ->As<ir::ScheduleBlockRealize>();
  auto schedule_block =
      schedule_realize->schedule_block.As<ir::ScheduleBlock>();

  auto iter_vars = schedule_block->iter_vars;
  auto iter_values = schedule_realize->iter_values;

  auto factors = info.output_shape;
  auto full_broadcast = info.full_broadcast;
  auto first_broadcast = info.first_broadcast;
  if (info.split_first) {
    // iter value is one
    for (size_t i = 0; i < axes.size(); ++i) {
      // new_extent
      auto axis = axes[i];
      auto loop_temp = all_loops[axis].As<ir::For>();
      int extent = factors[i];
      loop_temp->extent = Expr(extent);

      if (info.with_constrain) {
        auto check = ir::EQ::Make(loop_temp->loop_var, Expr(0));
        schedule_block->body =
            ir::IfThenElse::Make(check, schedule_block->body);
      }
    }

    // change load and store
    // get new offset
    all_loops = this->GetLoops(block_name);
    auto offset = Expr(0);
    auto stride = Expr(1);
    auto in_offset = Expr(0);

    std::set<int> brodacast_set(info.broadcast_axes.begin(),
                                info.broadcast_axes.end());
    for (int i = all_loops.size() - 1; i >= 0; --i) {
      auto loop_temp = all_loops[i].As<ir::For>();
      offset = offset + loop_temp->loop_var * stride;

      stride = stride * loop_temp->extent;
      if (!brodacast_set.count(i)) {
        in_offset = in_offset + loop_temp->loop_var * stride;
      }
    }

    auto exprs = ir::ir_utils::CollectIRNodesInOrder(
        schedule_block->body,
        [&](const Expr* x) { return x->As<ir::Store>(); });
    for (auto expr : exprs) {
      auto store = expr.As<ir::Store>();
      store->indices[0] = offset;
    }

    exprs = ir::ir_utils::CollectIRNodesInOrder(
        schedule_block->body, [&](const Expr* x) { return x->As<ir::Load>(); });

    for (auto expr : exprs) {
      auto load = expr.As<ir::Load>();
      if (!info.first_broadcast) {
        load->indices[0] = offset;
      } else {
        load->indices[0] = in_offset;
      }
    }

    return;
  }

  for (size_t i = 0; i < axes.size(); ++i) {
    // new_extent
    auto axis = axes[i];
    auto loop_temp = all_loops[axis].As<ir::For>();
    int extent = factors[i];
    loop_temp->extent = Expr(extent);

    if (!full_broadcast && (!(info.with_constrain))) {
      schedule_realize->iter_values[axis] = loop_temp->loop_var;
    }

    if (info.with_constrain) {
      auto check = ir::EQ::Make(loop_temp->loop_var, Expr(0));
      schedule_block->body = ir::IfThenElse::Make(check, schedule_block->body);
    }
  }

  if (first_broadcast && !full_broadcast) {
    auto exprs = ir::ir_utils::CollectIRNodesInOrder(
        schedule_block->body, [&](const Expr* x) { return x->As<ir::Load>(); });

    if (info.op_name == "cinn_op.reshape") {
      for (auto expr : exprs) {
        auto load = expr.As<ir::Load>();
        for (size_t k = 0; k < load->indices.size(); ++k) {
          for (size_t i = 0; i < axes.size(); ++i) {
            ReplaceExpr(&load->indices[k],
                        {schedule_block->iter_vars[axes[i]]},
                        {Expr(0)});
          }
        }
      }

      return;
    }
    for (auto expr : exprs) {
      auto load = expr.As<ir::Load>();
      if (load->indices.size() == schedule_realize->iter_values.size()) {
        for (size_t i = 0; i < axes.size(); ++i) {
          load->indices[axes[i]] = Expr(0);
        }
      } else if (load->indices.size() < schedule_realize->iter_values.size()) {
        // only one element
        // replace t zeros
        for (size_t k = 0; k < load->indices.size(); ++k) {
          for (size_t i = 0; i < axes.size(); ++i) {
            ReplaceExpr(&load->indices[k],
                        {schedule_block->iter_vars[axes[i]]},
                        {Expr(0)});
          }
        }
      } else {
        throw std::runtime_error("not support broadcast type yet");
      }
    }
  }
}

}  // namespace ir
}  // namespace cinn
