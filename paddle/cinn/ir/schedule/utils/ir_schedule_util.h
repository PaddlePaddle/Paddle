// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/schedule/ir_schedule_error.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/utils/random_engine.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {

// The struct used to reconstruct the new For node to replace the old For node.
struct LoopReconstructor : public ir::IRMutator<> {
 public:
  explicit LoopReconstructor(const Expr& root,
                             const Expr& block,
                             const Expr& loop)
      : root_(root), block_(block), loop_(loop) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

  /* \param inserted_pos The position index of the new_loop_ body `stmts` to be
   * inserted:
   *        - `index = -1` means inserted into the tail
   *        - otherwise, it should be a index between [0, stmts size)
   */
  std::string MakeNewLoop(const std::vector<IterRange>& iter_ranges,
                          bool keep_unit_loops,
                          int inserted_pos = -1) {
    int n_iters = iter_ranges.size();
    std::vector<Var> loop_vars;
    std::vector<Expr> loop_extents;
    std::vector<Expr> iter_values;
    loop_vars.reserve(n_iters);
    loop_extents.reserve(n_iters);
    iter_values.reserve(n_iters);
    std::vector<std::string> new_var_names;
    for (int i = 0; i < n_iters; ++i) {
      const auto& range = iter_ranges[i];
      if (keep_unit_loops || range.extent != Expr(1)) {
        std::string var_name =
            common::UniqName("ax" + std::to_string(loop_vars.size()));
        new_var_names.push_back(var_name);
        Var var(var_name, Int(32));
        loop_vars.push_back(var);
        loop_extents.push_back(range.extent);
        iter_values.push_back(common::AutoSimplify(range.min) + var);
      } else {
        iter_values.push_back(common::AutoSimplify(range.min));
      }
    }
    auto schedule_block_node =
        block_.As<ir::ScheduleBlockRealize>()->schedule_block;
    new_block_ = ScheduleBlockRealize::Make(std::move(iter_values),
                                            std::move(schedule_block_node));
    Expr loop_body = new_block_;
    for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
      auto loop_var = loop_vars[i];
      auto loop_extent = loop_extents[i];
      if (!loop_body.As<ir::Block>()) loop_body = Block::Make({loop_body});
      loop_body = For::Make(loop_var,
                            Expr(0),
                            loop_extent,
                            ForType::Serial,
                            loop_.As<ir::For>()->device_api,
                            std::move(loop_body));
    }
    new_loop_ = ir::ir_utils::IRCopy(loop_);

    // Replace the copied Tensor object with the original Tensor object,
    // to ensure that the same Tensor in a AST is the same object.
    std::unordered_map<std::string, ir::Expr> tensors_map;
    ir::ir_utils::CollectIRNodesWithoutTensor(
        loop_, [&tensors_map](const Expr* x) {
          if (x->as_tensor()) {
            tensors_map.insert({x->as_tensor()->name, *x});
            return true;
          }
          return false;
        });
    auto find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_loop_, [](const Expr* x) { return x->As<ir::Store>(); });
    for (auto store : find_store) {
      store.As<ir::Store>()->tensor =
          tensors_map.at(store.As<ir::Store>()->tensor.as_tensor()->name);
    }
    auto find_load = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_loop_, [](const Expr* x) { return x->As<ir::Load>(); });
    for (auto load : find_load) {
      load.As<ir::Load>()->tensor =
          tensors_map.at(load.As<ir::Load>()->tensor.as_tensor()->name);
    }

    InsertBlock(new_loop_, loop_body, inserted_pos);
    return utils::Join(new_var_names, ",");
  }

 public:
  /*! \brief The root block */
  Expr root_;
  /*! \brief The given block to be moved */
  Expr block_;
  /*! \brief The given loop the block and its loop nest to be put under */
  Expr loop_;
  /*! \brief The new loop to replace the original loop */
  Expr new_loop_{nullptr};
  /*! \brief The new block realize to the moved block */
  Expr new_block_{nullptr};
  /*! \brief The plan to remove the given block by replacing this loop/block in
   * the AST */
  Expr source_expr{nullptr};
  /*! \brief The plan to remove the given block by replacing to this loop/block
   * in the AST */
  Expr target_expr{nullptr};
};

// The struct used to create all stmts after rfactor transformation.
struct RfCreater : public ir::IRMutator<> {
 public:
  RfCreater(const Expr& root, const Expr& rf_loop, const int& rf_axis)
      : root_(root), rf_loop_(rf_loop), rf_axis_(rf_axis) {}
  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

  Expr CreateRfAllStmts() {
    auto root_realize = root_.As<ScheduleBlockRealize>();
    CHECK(root_realize);
    auto root_block = root_realize->schedule_block.As<ScheduleBlock>();
    CHECK(root_block);
    Expr root_loop = ir::ir_utils::IRCopy(root_block->body);
    if (auto block = root_loop.As<Block>()) {
      CHECK_EQ(block->stmts.size(), 1U)
          << "rfactor root should only have one block stmt";
      root_loop = block->stmts[0];
    }
    auto* root_for = root_loop.As<For>();
    CHECK(root_for);
    auto rf_for = rf_loop_.As<For>();
    CHECK(rf_for);
    // create new rfactor forloops
    Expr new_rf_forloop = ir::ir_utils::IRCopy(root_loop);
    RfMutator rf_mutator(rf_loop_, rf_axis_);
    rf_mutator(&new_rf_forloop);
    VLOG(3) << "After RfMutator, new rf_forloop is\n" << new_rf_forloop;
    auto new_rf_tensor = rf_mutator.GetNewRfTensor();
    // create final write-back forloops
    Expr final_forloop = ir::ir_utils::IRCopy(root_loop);
    FinalMutator final_mutator(rf_loop_, rf_axis_, new_rf_tensor);
    final_mutator(&final_forloop);
    VLOG(3) << "After FinalMuator, final write-back forloop is\n"
            << final_forloop;
    // combine the new created rfactor forloops with the final write-back
    // forloops and replace
    root_block->body = Block::Make({new_rf_forloop, final_forloop});
    return new_rf_tensor;
  }

  Expr root_;
  Expr rf_loop_;
  int rf_axis_;
};

//! Visit all ScheduleBlock and change its body to ir::Block if it is not.
struct ChangeBodyToBlock : public ir::IRMutator<> {
 public:
  static void Change(Expr* expr) {
    ChangeBodyToBlock mutator;
    mutator(expr);
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::ScheduleBlock* expr, Expr* op) override {
    if (!op->As<ScheduleBlock>()->body.As<Block>()) {
      op->As<ScheduleBlock>()->body =
          Block::Make({op->As<ScheduleBlock>()->body});
    }
    IRMutator::Visit(expr, op);
  }
};

struct CacheReadRewriter : public ir::IRMutator<> {
 public:
  static Expr Rewrite(const Expr& root, CacheBlockInfo* info) {
    CacheReadRewriter rewriter(root, info);
    Expr new_root = ir::ir_utils::IRCopy(root);
    rewriter(&new_root);
    return new_root;
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  explicit CacheReadRewriter(const Expr& root, CacheBlockInfo* info)
      : root_(root), info_(info) {}

  void Visit(const ir::Block* expr, Expr* op) override {
    if (*op == info_->loc_block) {
      IRMutator::Visit(expr, op);
      op->As<Block>()->stmts.insert(
          op->As<Block>()->stmts.begin() + info_->loc_pos, info_->cache_block);
    } else {
      IRMutator::Visit(expr, op);
    }
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    if (expr->tensor == Expr(info_->read_tensor)) {
      IRMutator::Visit(expr, op);
      op->As<Load>()->tensor = Expr(info_->write_tensor);
    } else {
      IRMutator::Visit(expr, op);
    }
  }

 private:
  /*! \brief The parent scope of the insertion */
  const Expr& root_;
  /*! \brief The info for inserting cache stage */
  CacheBlockInfo* info_;
};

struct CacheWriteRewriter : public ir::IRMutator<> {
 public:
  static Expr Rewrite(const Expr& root, CacheBlockInfo* info) {
    CacheWriteRewriter rewriter(root, info);
    Expr new_root = ir::ir_utils::IRCopy(root);
    rewriter.mutate_cache_block = true;
    rewriter(&info->cache_block);
    rewriter.mutate_cache_block = false;
    rewriter(&new_root);
    auto find_tensor = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_root,
        [&](const Expr* x) {
          return x->As<Store>() &&
                 (x->As<Store>()->tensor == Expr(info->read_tensor));
        },
        true);
    if (!find_tensor.empty()) {
      auto find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
          (*find_tensor.begin()), [&](const Expr* x) {
            return x->As<Load>() &&
                   (x->As<Load>()->tensor == Expr(info->write_tensor));
          });
      for (auto load_ir : find_store) {
        load_ir.As<Load>()->tensor = Expr(info->read_tensor);
      }
    }
    return new_root;
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  explicit CacheWriteRewriter(const Expr& root, CacheBlockInfo* info)
      : root_(root), info_(info) {}

  void Visit(const ir::Block* expr, Expr* op) override {
    if (*op == info_->loc_block) {
      IRMutator::Visit(expr, op);
      op->As<Block>()->stmts.insert(
          op->As<Block>()->stmts.begin() + info_->loc_pos, info_->cache_block);
    } else {
      IRMutator::Visit(expr, op);
    }
  }

  void Visit(const ir::ScheduleBlock* expr, Expr* op) override {
    if (op->As<ScheduleBlock>()->name == info_->write_tensor->name) {
      op->As<ScheduleBlock>()->name = info_->read_tensor->name;
    } else if (op->As<ScheduleBlock>()->name == info_->read_tensor->name) {
      op->As<ScheduleBlock>()->name = info_->write_tensor->name;
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
    if (op->As<Load>()->tensor == Expr(info_->write_tensor) &&
        mutate_cache_block) {
      op->As<Load>()->tensor = Expr(info_->read_tensor);
    } else if (op->As<Load>()->tensor == Expr(info_->read_tensor) &&
               mutate_cache_block) {
      op->As<Load>()->tensor = Expr(info_->write_tensor);
    }
  }

  void Visit(const ir::Store* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
    if (op->As<Store>()->tensor == Expr(info_->write_tensor)) {
      op->As<Store>()->tensor = Expr(info_->read_tensor);
    } else if (op->As<Store>()->tensor == Expr(info_->read_tensor) &&
               mutate_cache_block) {
      op->As<Store>()->tensor = Expr(info_->write_tensor);
    }
  }

 private:
  /*! \brief The parent scope of the insertion */
  const Expr& root_;
  /*! \brief The info for inserting cache stage */
  CacheBlockInfo* info_;
  /*! \brief Are we mutating the cache tensor's block */
  bool mutate_cache_block{true};
};

struct FixLocalBufferSize : public ir::IRMutator<> {
 public:
  explicit FixLocalBufferSize(const std::string& tensor_name)
      : tensor_name_(tensor_name) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::Store* expr, Expr* op) override {
    if (op->As<Store>()->tensor.As<_Tensor_>()->name == tensor_name_) {
      op->As<Store>()->tensor.As<_Tensor_>()->shape = {Expr(1)};
      op->As<Store>()->tensor.As<_Tensor_>()->domain = {Expr(1)};
      op->As<Store>()->tensor.As<_Tensor_>()->buffer->shape = {Expr(1)};
      op->As<Store>()->indices = {Expr(0)};
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    if (op->As<Load>()->tensor.As<_Tensor_>()->name == tensor_name_) {
      op->As<Load>()->tensor.As<_Tensor_>()->shape = {Expr(1)};
      op->As<Load>()->tensor.As<_Tensor_>()->domain = {Expr(1)};
      op->As<Load>()->tensor.As<_Tensor_>()->buffer->shape = {Expr(1)};
      op->As<Load>()->indices = {Expr(0)};
    }
    IRMutator::Visit(expr, op);
  }
  std::string tensor_name_;
};

struct InsertExpr : public ir::IRMutator<> {
 public:
  static void Insert(const Expr& ir_node,
                     const Expr& insert_node,
                     bool after_node,
                     Expr* expr) {
    InsertExpr mutator(ir_node, insert_node, after_node);
    mutator(expr);
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  explicit InsertExpr(const Expr& ir_node,
                      const Expr& insert_node,
                      bool after_node)
      : ir_node_(ir_node), insert_node_(insert_node), after_node_(after_node) {}

  void Visit(const ir::Block* expr, Expr* op) override {
    for (int i = 0; i < expr->stmts.size(); i++) {
      if (expr->stmts[i] == ir_node_) {
        if (after_node_) {
          op->As<ir::Block>()->stmts.insert(
              op->As<ir::Block>()->stmts.begin() + i + 1, insert_node_);
        } else {
          op->As<ir::Block>()->stmts.insert(
              op->As<ir::Block>()->stmts.begin() + i, insert_node_);
        }
        return;
      }
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::For* expr, Expr* op) override {
    if (expr->body == ir_node_) {
      if (after_node_)
        op->As<ir::For>()->body =
            ir::Block::Make({op->As<ir::For>()->body, insert_node_});
      else
        op->As<ir::For>()->body =
            ir::Block::Make({insert_node_, op->As<ir::For>()->body});
      return;
    }
    IRMutator::Visit(expr, op);
  }

 private:
  const Expr& ir_node_;
  const Expr& insert_node_;
  bool after_node_;
};

}  // namespace ir
}  // namespace cinn
