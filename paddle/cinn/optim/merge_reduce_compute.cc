// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/merge_reduce_compute.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace optim {

namespace {

struct ForVarExtent {
  ir::Var loop_var;
  ir::Expr extent;
};

struct IndicesAndExtent {
  std::vector<ir::Expr> indices;
  std::vector<ForVarExtent> for_var_extents;
};

struct RootAndBlock {
  std::string root;
  std::unordered_set<std::string> schedule_blocks;
};

struct AliveBuffers {
  std::unordered_set<std::string> read_buffer_names;
  std::unordered_set<std::string> write_buffer_names;
};

std::unordered_map<ir::Var, ir::Var> ConstructForVarReplaceMap(
    const std::vector<ForVarExtent>& lhs_extents,
    const std::vector<ForVarExtent>& rhs_extents) {
  std::unordered_map<ir::Var, ir::Var> ret;
  std::unordered_set<std::size_t> visited_rhs_index;
  for (const auto& [lhs_var, lhs_extent] : lhs_extents) {
    for (std::size_t i = 0; i < rhs_extents.size(); ++i) {
      const auto& [rhs_var, rhs_extent] = rhs_extents[i];
      if (cinn::common::AutoSimplify(ir::Sub::Make(lhs_extent, rhs_extent)) ==
              ir::Expr(0) &&
          visited_rhs_index.count(i) == 0) {
        ret[lhs_var] = rhs_var;
        visited_rhs_index.insert(i);
        break;
      }
    }
  }
  return ret;
}

bool ForVarExtentEqual(const std::vector<ForVarExtent>& for_var_extent1,
                       const std::vector<ForVarExtent>& for_var_extent2) {
  if (for_var_extent1.size() != for_var_extent2.size()) return false;

  for (size_t i = 0; i < for_var_extent1.size(); ++i) {
    const ir::Expr lhs = for_var_extent1[i].extent;
    const ir::Expr rhs = for_var_extent2[i].extent;
    if (cinn::common::AutoSimplify(ir::Sub::Make(lhs, rhs)) != ir::Expr(0)) {
      return false;
    }
  }
  return true;
}

struct ReduceInitCollector : public ir::IRMutator<Expr*> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  RootAndBlock GetMergeRootAndBlock() {
    auto AllForVarExtentEqual =
        [&](const std::unordered_map<std::string, std::vector<ForVarExtent>>&
                block_name_to_for_var_extents) -> bool {
      std::vector<ForVarExtent> root_for_var_extent =
          block_name_to_for_var_extents.at(merge_sb_root_);
      for (const auto& [block_name, for_var_extent] :
           block_name_to_for_var_extents) {
        if (ForVarExtentEqual(root_for_var_extent, for_var_extent)) continue;
        return false;
      }
      return true;
    };

    auto IsReduceInitNeedMerge =
        [&](const std::unordered_map<std::string, std::vector<ForVarExtent>>&
                block_name_to_for_var_extents) -> bool {
      if (block_name_to_for_var_extents.size() <= 1) return false;
      return AllForVarExtentEqual(block_name_to_for_var_extents);
    };

    RootAndBlock root_and_block;
    root_and_block.root = merge_sb_root_;
    if (IsReduceInitNeedMerge(block_name_to_for_var_extents_)) {
      root_and_block.schedule_blocks =
          [&]() -> std::unordered_set<std::string> {
        std::unordered_set<std::string> init_block_name;
        for (const auto& [block_name, for_var_extent] :
             block_name_to_for_var_extents_) {
          if (block_name != merge_sb_root_) {
            init_block_name.insert(block_name);
          }
        }
        return init_block_name;
      }();
    }

    return root_and_block;
  }

 private:
  void Visit(const ir::For* op, ir::Expr* expr) {
    auto* node = expr->As<ir::For>();
    for_var_extents_.push_back({node->loop_var, node->extent});
    ir::IRMutator<>::Visit(op, expr);
    for_var_extents_.pop_back();
  }

  void Visit(const ir::ScheduleBlock* op, ir::Expr* expr) {
    auto* node = expr->As<ir::ScheduleBlock>();
    if (utils::EndsWith(node->name, "__reduce_init")) {
      if (block_name_to_for_var_extents_.empty()) {
        merge_sb_root_ = node->name;
      }
      block_name_to_for_var_extents_[node->name] = for_var_extents_;
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  std::vector<ForVarExtent> for_var_extents_;
  std::unordered_map<std::string, std::vector<ForVarExtent>>
      block_name_to_for_var_extents_;

  std::string merge_sb_root_;
};

struct BlockMerger : public ir::IRMutator<Expr*> {
 public:
  explicit BlockMerger(const RootAndBlock& root_and_block)
      : root_and_block_(root_and_block) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Block* op, ir::Expr* expr) {
    auto InsertRootAndRemoveCurrentStmts = [&](ir::Block* current_block) {
      if (block_to_new_stmts_.find(current_block) !=
          block_to_new_stmts_.end()) {
        current_block->stmts = block_to_new_stmts_[current_block];
      }
      while (!insert_root_schedule_blocks_.empty()) {
        VLOG(6) << "Insert to root block: "
                << insert_root_schedule_blocks_.back();
        root_block_->stmts.insert(root_block_->stmts.begin(),
                                  insert_root_schedule_blocks_.back());
        insert_root_schedule_blocks_.pop_back();
      }
    };

    auto* node = expr->As<ir::Block>();
    current_block_ = node;
    ir::IRMutator<>::Visit(op, expr);
    InsertRootAndRemoveCurrentStmts(node);
  }

  void Visit(const ir::ScheduleBlockRealize* op, ir::Expr* expr) override {
    auto* sbr_node = expr->As<ir::ScheduleBlockRealize>();
    current_sbr_ = sbr_node;
    const auto* sb_node = sbr_node->schedule_block.As<ir::ScheduleBlock>();
    if (sb_node->name == root_and_block_.root) {
      root_block_ = current_block_;
      root_for_var_extents_ = for_var_extents_;
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::For* op, ir::Expr* expr) {
    auto* node = expr->As<ir::For>();
    for_var_extents_.push_back({node->loop_var, node->extent});
    ir::IRMutator<>::Visit(op, expr);
    for_var_extents_.pop_back();
  }

  void Visit(const ir::Store* op, ir::Expr* expr) {
    auto* node = expr->As<ir::Store>();
    ir::Expr sb = ir::ir_utils::IRCopy(current_sbr_->schedule_block);
    ir::ScheduleBlock* sb_node = sb.As<ir::ScheduleBlock>();
    std::string sb_name = sb_node->name;
    if (root_and_block_.schedule_blocks.find(sb_name) !=
        root_and_block_.schedule_blocks.end()) {
      MergeInitAndRemoveOrigin();
    }
  }

  void MergeInitAndRemoveOrigin() {
    // Merge current sbr to root block.
    insert_root_schedule_blocks_.push_back(ReplaceSbrIterValues());

    // Record and will remove current sbr later.
    block_to_new_stmts_[current_block_] = [&]() -> std::vector<ir::Expr> {
      std::vector<ir::Expr> new_stmts;
      for (const ir::Expr& expr : current_block_->stmts) {
        if (expr.As<ir::ScheduleBlockRealize>()) {
          const ir::Expr sb = ir::ir_utils::IRCopy(
              expr.As<ir::ScheduleBlockRealize>()->schedule_block);
          const ir::ScheduleBlock* sb_node = sb.As<ir::ScheduleBlock>();
          if (root_and_block_.schedule_blocks.find(sb_node->name) !=
              root_and_block_.schedule_blocks.end()) {
            continue;
          }
        }
        new_stmts.push_back(expr);
      }
      return new_stmts;
    }();
  }

  ir::Expr ReplaceSbrIterValues() {
    std::unordered_map<cinn::ir::Var, cinn::ir::Var> for_var_map =
        ConstructForVarReplaceMap(for_var_extents_, root_for_var_extents_);
    std::vector<ir::Expr> new_iter_values;
    for (const ir::Expr& iter_value : current_sbr_->iter_values) {
      ir::Expr new_iter_value = ir::ir_utils::IRCopy(iter_value);
      for (const auto& [lhs_var, rhs_var] : for_var_map) {
        ReplaceVarWithExpr(
            &new_iter_value, lhs_var, ir::ir_utils::IRCopy(rhs_var));
      }
      new_iter_values.push_back(new_iter_value);
    }
    return ir::ScheduleBlockRealize::Make(
        new_iter_values, ir::ir_utils::IRCopy(current_sbr_->schedule_block));
  }

  std::vector<ForVarExtent> for_var_extents_;
  std::vector<ForVarExtent> root_for_var_extents_;
  std::vector<ir::Expr> insert_root_schedule_blocks_;
  std::unordered_map<ir::Block*, std::vector<ir::Expr>> block_to_new_stmts_;

  ir::Block* root_block_;
  ir::Block* current_block_;
  ir::ScheduleBlockRealize* current_sbr_;
  RootAndBlock root_and_block_;
};

struct GlobalTensorInfoCollector : public ir::IRMutator<Expr*> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  std::unordered_set<std::string> GetCommonGlobalBufferNames() const {
    auto IndiceToExprWithForVar =
        [&](const ir::Expr indice,
            const std::unordered_map<ir::Var, ir::Var>& for_var_map)
        -> ir::Expr {
      ir::Expr ret = ir::ir_utils::IRCopy(indice);
      for (const auto& [lhs_var, rhs_var] : for_var_map) {
        ReplaceVarWithExpr(&ret, lhs_var, ir::ir_utils::IRCopy(rhs_var));
      }
      return ret;
    };

    auto IndiceAndExtentEqual =
        [&](const IndicesAndExtent& indice_and_extent1,
            const IndicesAndExtent& indice_and_extent2) {
          if (!ForVarExtentEqual(indice_and_extent1.for_var_extents,
                                 indice_and_extent2.for_var_extents))
            return false;
          const auto& indice1 = indice_and_extent1.indices;
          const auto& indice2 = indice_and_extent2.indices;
          if (indice1.size() != indice2.size()) return false;

          std::unordered_map<ir::Var, ir::Var> for_var_map =
              ConstructForVarReplaceMap(indice_and_extent1.for_var_extents,
                                        indice_and_extent2.for_var_extents);

          for (size_t i = 0; i < indice1.size(); ++i) {
            ir::Expr lhs = IndiceToExprWithForVar(indice1.at(i), for_var_map);
            ir::Expr rhs = IndiceToExprWithForVar(indice2.at(i), for_var_map);
            if (cinn::common::AutoSimplify(ir::Sub::Make(lhs, rhs)) !=
                ir::Expr(0)) {
              return false;
            }
          }
          return true;
        };

    auto AllIndiceAndExtentEqual =
        [&](const std::vector<IndicesAndExtent>& indice_and_extent) -> bool {
      for (size_t i = 1; i < indice_and_extent.size(); ++i) {
        if (!IndiceAndExtentEqual(indice_and_extent[0], indice_and_extent[i]))
          return false;
      }
      return true;
    };

    auto IndiceContainsLoad =
        [&](const IndicesAndExtent& indice_and_extent) -> bool {
      for (const auto& index : indice_and_extent.indices) {
        std::set<Expr> load_tensors = ir::ir_utils::CollectLoadTensors(
            index, /*teller=*/[&](const Expr*) -> bool { return true; });
        if (load_tensors.size() > 0) {
          return true;
        }
      }
      return false;
    };

    auto IsCommonGlobalTensor =
        [&](const std::vector<IndicesAndExtent>& indice_and_extent) {
          if (indice_and_extent.size() <= 1) return false;
          if (IndiceContainsLoad(indice_and_extent[0])) return false;
          if (contains_select_) return false;
          return AllIndiceAndExtentEqual(indice_and_extent);
        };

    std::unordered_set<std::string> global_buffer_name;
    for (const auto& [buffer_name, indice_and_extent] :
         buffer_to_indice_and_extent_) {
      if (global_store_buffer_names_.find(buffer_name) !=
          global_store_buffer_names_.end()) {
        continue;
      }
      if (IsCommonGlobalTensor(indice_and_extent)) {
        global_buffer_name.insert(buffer_name);
      }
    }
    return global_buffer_name;
  }

 private:
  void Visit(const ir::ScheduleBlockRealize* op, ir::Expr* expr) override {
    const auto* sbr_node = expr->As<ir::ScheduleBlockRealize>();
    const auto& iter_values = sbr_node->iter_values;
    const auto* sb_node = sbr_node->schedule_block.As<ir::ScheduleBlock>();
    const auto& iter_vars = sb_node->iter_vars;

    for (std::size_t i = 0; i < iter_values.size(); ++i) {
      var_to_sb_expr_[iter_vars[i]] = iter_values[i];
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::For* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::For>();
    for_var_extents_.push_back(
        {node->loop_var, ir::ir_utils::IRCopy(node->extent)});
    ir::IRMutator<>::Visit(op, expr);
    for_var_extents_.pop_back();
  }

  void Visit(const ir::Load* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Load>();
    const auto& load_buffer = node->tensor.as_tensor_ref()->buffer;
    if (load_buffer->memory_type == ir::MemoryType::Heap) {
      std::vector<ir::Expr> tensor_indices;
      for (const auto& indice : node->indices) {
        ir::Expr new_indice = ir::ir_utils::IRCopy(indice);
        for (const auto& [var, sb_expr] : var_to_sb_expr_) {
          ReplaceVarWithExpr(&new_indice, var, ir::ir_utils::IRCopy(sb_expr));
        }
        tensor_indices.push_back(new_indice);
      }
      buffer_to_indice_and_extent_[load_buffer->name].push_back(
          {tensor_indices, for_var_extents_});
    }
  }

  void Visit(const ir::Store* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Store>();
    const auto& store_buffer = node->tensor.as_tensor_ref()->buffer;
    if (store_buffer->memory_type == ir::MemoryType::Heap) {
      global_store_buffer_names_.insert(store_buffer->name);
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Select* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Select>();
    contains_select_ = true;
    ir::IRMutator<>::Visit(op, expr);
  }

  std::vector<ForVarExtent> for_var_extents_;
  std::unordered_map<ir::Var, ir::Expr> var_to_sb_expr_;
  std::unordered_map<std::string, std::vector<IndicesAndExtent>>
      buffer_to_indice_and_extent_;
  std::unordered_set<std::string> global_store_buffer_names_;

  bool contains_select_ = false;
};

struct GlobalLoadMergeCollector : public ir::IRMutator<Expr*> {
 public:
  GlobalLoadMergeCollector(
      const std::unordered_set<std::string>& common_global_buffer_names)
      : common_global_buffer_names_(common_global_buffer_names) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  std::unordered_map<std::string, RootAndBlock> GetBufferToRootAndBlock() {
    return buffer_to_root_and_block_;
  }

 private:
  void Visit(const ir::ScheduleBlock* op, ir::Expr* expr) {
    auto* node = expr->As<ir::ScheduleBlock>();
    current_sb_ = node;
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load* op, ir::Expr* expr) {
    auto* node = expr->As<ir::Load>();
    const auto& buffer_name = node->tensor.as_tensor_ref()->buffer->name;
    if (common_global_buffer_names_.count(buffer_name) == 0) {
      return;
    }
    if (buffer_to_root_and_block_.find(buffer_name) ==
        buffer_to_root_and_block_.end()) {
      buffer_to_root_and_block_[buffer_name].root = current_sb_->name;
    } else if (buffer_to_root_and_block_[buffer_name].root !=
               current_sb_->name) {
      buffer_to_root_and_block_[buffer_name].schedule_blocks.insert(
          current_sb_->name);
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  ir::ScheduleBlock* current_sb_;
  std::unordered_set<std::string> common_global_buffer_names_;
  std::unordered_map<std::string, RootAndBlock> buffer_to_root_and_block_;
};

struct AliveBufferAnalyer : public ir::IRMutator<Expr*> {
 public:
  explicit AliveBufferAnalyer(const RootAndBlock& root_and_block)
      : root_and_block_(root_and_block) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  bool IsBlockNeedFuse() { return block_need_fuse_; }

 private:
  void Visit(const ir::ScheduleBlock* op, ir::Expr* expr) {
    auto* node = expr->As<ir::ScheduleBlock>();
    auto RecordBuffers = [&](const std::vector<ir::Expr>& node_buffers)
        -> std::unordered_set<std::string> {
      std::unordered_set<std::string> alive_buffers;
      for (const auto& buffer_region : node_buffers) {
        auto* buffer =
            buffer_region.As<ir::_BufferRange_>()->buffer.As<ir::_Buffer_>();
        alive_buffers.insert(buffer->name);
      }
      return alive_buffers;
    };

    auto RecordReadAndWriteBuffers = [&](const ir::ScheduleBlock* node) {
      const std::unordered_set<std::string> node_read_buffer_names =
          RecordBuffers(node->read_buffers);
      alive_buffers_.read_buffer_names.insert(node_read_buffer_names.begin(),
                                              node_read_buffer_names.end());
      const std::unordered_set<std::string> node_write_buffer_names =
          RecordBuffers(node->write_buffers);
      alive_buffers_.write_buffer_names.insert(node_write_buffer_names.begin(),
                                               node_write_buffer_names.end());
    };

    auto IsContainsBuffer =
        [&](const ir::Expr& buffer_region,
            const std::unordered_set<std::string>& alive_buffers) -> bool {
      const auto* buffer =
          buffer_region.As<ir::_BufferRange_>()->buffer.As<ir::_Buffer_>();
      return alive_buffers.count(buffer->name) != 0;
    };

    auto AliveBufferEmpty = [&]() -> bool {
      return (alive_buffers_.read_buffer_names.empty() &&
              alive_buffers_.write_buffer_names.empty());
    };

    auto CheckReadAndWriteBuffers = [&](const ir::ScheduleBlock* node) -> bool {
      if (AliveBufferEmpty()) return false;
      // Read buffers should not be written in alive_buffers.
      for (const auto& buffer_region : node->read_buffers) {
        if (IsContainsBuffer(buffer_region, alive_buffers_.write_buffer_names))
          return false;
      }
      // Write buffers should not be read or written in alive_buffers.
      for (const auto& buffer_region : node->write_buffers) {
        if (IsContainsBuffer(buffer_region, alive_buffers_.read_buffer_names))
          return false;
        if (IsContainsBuffer(buffer_region, alive_buffers_.write_buffer_names))
          return false;
      }
      return true;
    };

    if (node->name == root_and_block_.root) {
      RecordReadAndWriteBuffers(node);
    } else if (root_and_block_.schedule_blocks.count(node->name) != 0) {
      block_need_fuse_ = block_need_fuse_ && CheckReadAndWriteBuffers(node);
    }

    if (!AliveBufferEmpty()) {
      RecordReadAndWriteBuffers(node);
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  AliveBuffers alive_buffers_;
  RootAndBlock root_and_block_;
  bool block_need_fuse_ = true;
};

/**
 * 1. Get reduce init root and block.
 * Reduce init block like `buffer[0] = 0.0f`.
 *
 * 2. Merge init block with root.
 *
 * @param expr The expression to mutate.
 */
void MergeReduceInitBlock(ir::Expr* e) {
  ReduceInitCollector reduce_init_collector;
  reduce_init_collector(e);

  const auto& root_and_block = reduce_init_collector.GetMergeRootAndBlock();

  BlockMerger block_merger(root_and_block);
  block_merger(e);
}

/**
 * 1. Get common global buffer.
 * The buffer should have exactly the same index in different ScheduleBlocks.
 *
 * 2. Get root of common global buffer.
 * Root and block indicate different ScheduleBlocks.
 *
 * 3. Analyze alive buffer within root and block.
 * Only load buffer stmts can change order, like `load-load`
 * ignore `load-store`, `store-load`, `store-store` which will change
 * calculation results,
 *
 * 4. Merge block with root.
 *
 * @param expr The expression to mutate.
 */
void MergeCommonLoadBlock(ir::Expr* e) {
  GlobalTensorInfoCollector global_tensor_info_collector;
  global_tensor_info_collector(e);
  const auto& global_tensor_names =
      global_tensor_info_collector.GetCommonGlobalBufferNames();

  GlobalLoadMergeCollector global_load_merge_collector(global_tensor_names);
  global_load_merge_collector(e);
  const auto& buffers = global_load_merge_collector.GetBufferToRootAndBlock();

  for (const auto& [buffer_name, root_and_block] : buffers) {
    AliveBufferAnalyer alive_buffer_analyzer(root_and_block);
    alive_buffer_analyzer(e);
    if (alive_buffer_analyzer.IsBlockNeedFuse()) {
      BlockMerger block_merger(root_and_block);
      block_merger(e);
    }
  }
}

}  // namespace

void MergeReduceCompute(Expr* e) {
  VLOG(4) << "Before MergeReduceCompute: \n" << *e;

  MergeReduceInitBlock(e);
  MergeCommonLoadBlock(e);

  VLOG(4) << "After MergeReduceCompute: \n" << *e;
}

}  // namespace optim
}  // namespace cinn
