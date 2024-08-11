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

#include "paddle/cinn/optim/reduce_init_merge.h"

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

class ForLevel {};
class KernelLevel {};

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

struct ReduceInitCollector : public ir::IRMutator<Expr*> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  RootAndBlock GetMergeRootAndBlock() {
    auto ForVarExtentEqual =
        [&](const std::vector<ForVarExtent>& for_var_extent1,
            const std::vector<ForVarExtent>& for_var_extent2) -> bool {
      if (for_var_extent1.size() != for_var_extent2.size()) {
        return false;
      }
      for (size_t i = 0; i < for_var_extent1.size(); ++i) {
        const ir::Expr lhs = for_var_extent1[i].extent;
        const ir::Expr rhs = for_var_extent2[i].extent;
        if (cinn::common::AutoSimplify(ir::Sub::Make(lhs, rhs)) !=
            ir::Expr(0)) {
          return false;
        }
      }
      return true;
    };

    auto AllForVarExtentEqual =
        [&](const std::unordered_map<std::string, std::vector<ForVarExtent>>&
                block_name_to_for_var_extents) -> bool {
      std::vector<ForVarExtent> root_for_var_extent =
          block_name_to_for_var_extents.at(merge_sb_root_);
      for (const auto& [block_name, for_var_extent] :
           block_name_to_for_var_extents) {
        if (ForVarExtentEqual(root_for_var_extent, for_var_extent)) {
          continue;
        }
        return false;
      }
      return true;
    };

    auto IsReduceInitCanMerge =
        [&](const std::unordered_map<std::string, std::vector<ForVarExtent>>&
                block_name_to_for_var_extents) -> bool {
      if (block_name_to_for_var_extents.size() <= 1) return false;
      return AllForVarExtentEqual(block_name_to_for_var_extents);
    };

    auto CollectInitBlockName =
        [&](const std::unordered_map<std::string, std::vector<ForVarExtent>>&
                block_name_to_for_var_extents)
        -> std::unordered_set<std::string> {
      std::unordered_set<std::string> init_block_name;
      for (const auto& [block_name, for_var_extent] :
           block_name_to_for_var_extents) {
        if (block_name != merge_sb_root_) {
          init_block_name.insert(block_name);
        }
      }
      return init_block_name;
    };

    RootAndBlock root_and_block;
    root_and_block.root = merge_sb_root_;
    if (IsReduceInitCanMerge(block_name_to_for_var_extents_)) {
      root_and_block.schedule_blocks =
          CollectInitBlockName(block_name_to_for_var_extents_);
    }

    VLOG(6) << "Merge reduce init size: "
            << root_and_block.schedule_blocks.size();
    for (const auto& name : root_and_block.schedule_blocks) {
      VLOG(6) << "Schedule block names: " << name;
    }
    return root_and_block;
  }

 private:
  void Visit(const ir::For* op, ir::Expr* expr) {
    auto* node = expr->As<ir::For>();
    CHECK(node);
    for_var_extents_.push_back({node->loop_var, node->extent});
    ir::IRMutator<>::Visit(op, expr);
    for_var_extents_.pop_back();
  }

  void Visit(const ir::ScheduleBlock* op, ir::Expr* expr) {
    auto* node = expr->As<ir::ScheduleBlock>();
    CHECK(node);
    const std::string init_block = "__reduce_init";
    if (utils::EndsWith(node->name, init_block)) {
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
    auto insert_root_and_remove_current_stmts = [&](ir::Block* current_block) {
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
    CHECK(node);
    current_block_ = node;
    ir::IRMutator<>::Visit(op, expr);
    insert_root_and_remove_current_stmts(node);
  }

  void Visit(const ir::ScheduleBlockRealize* op, ir::Expr* expr) override {
    auto* sbr_node = expr->As<ir::ScheduleBlockRealize>();
    CHECK(sbr_node);
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
    CHECK(node);
    for_var_extents_.push_back({node->loop_var, node->extent});
    ir::IRMutator<>::Visit(op, expr);
    for_var_extents_.pop_back();
  }

  void Visit(const ir::Store* op, ir::Expr* expr) {
    auto* node = expr->As<ir::Store>();
    CHECK(node);
    ir::Expr sb = ir::ir_utils::IRCopy(current_sbr_->schedule_block);
    ir::ScheduleBlock* sb_node = sb.As<ir::ScheduleBlock>();
    CHECK(sb_node);
    std::string sb_name = sb_node->name;
    if (root_and_block_.schedule_blocks.find(sb_name) !=
        root_and_block_.schedule_blocks.end()) {
      VLOG(6) << "Reduce init schedule block: " << sb_name
              << " with root: " << root_and_block_.root;
      MergeInitAndRemoveOrigin();
    }
  }

  void MergeInitAndRemoveOrigin() {
    // Merge current sbr to root block
    insert_root_schedule_blocks_.push_back(ReplaceSbrIterValues());

    // Remove current sbr
    std::vector<ir::Expr> new_stmts;
    for (const ir::Expr& expr : current_block_->stmts) {
      if (expr.As<ir::ScheduleBlockRealize>()) {
        const ir::Expr sb = ir::ir_utils::IRCopy(
            expr.As<ir::ScheduleBlockRealize>()->schedule_block);
        const ir::ScheduleBlock* sb_node = sb.As<ir::ScheduleBlock>();
        if (root_and_block_.schedule_blocks.find(sb_node->name) !=
            root_and_block_.schedule_blocks.end()) {
          VLOG(6) << "Remove sbr name: " << sb_node->name;
          continue;
        }
      }
      new_stmts.push_back(expr);
    }
    block_to_new_stmts_[current_block_] = new_stmts;
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
      VLOG(6) << "Old sbr iter value: " << iter_value
              << " New sbr iter value: " << new_iter_value;
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
    CHECK(node);
    current_sb_ = node;
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load* op, ir::Expr* expr) {
    auto* node = expr->As<ir::Load>();
    CHECK(node);
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

  bool GetBlockNeedFuse() { return block_need_fuse_; }

 private:
  void Visit(const ir::ScheduleBlock* op, ir::Expr* expr) {
    auto* node = expr->As<ir::ScheduleBlock>();
    CHECK(node);

    auto RecordBuffers = [&](const std::vector<ir::Expr>& node_buffers)
        -> std::unordered_set<std::string> {
      std::unordered_set<std::string> alive_buffers;
      for (const auto& buffer_region : node_buffers) {
        auto* buffer =
            buffer_region.As<ir::_BufferRange_>()->buffer.As<ir::_Buffer_>();
        PADDLE_ENFORCE_NOT_NULL(
            buffer,
            phi::errors::InvalidArgument("The buffer is not valid. "
                                         "Please ensure that `x->buffer` is "
                                         "properly assigned to a buffer."));
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

    auto CheckReadAndWriteBuffers = [&](const ir::ScheduleBlock* node) -> bool {
      for (const auto& buffer_region : node->read_buffers) {
        if (IsContainsBuffer(buffer_region, alive_buffers_.write_buffer_names))
          return false;
      }
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

    if (!alive_buffers_.read_buffer_names.empty()) {
      RecordReadAndWriteBuffers(node);
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  AliveBuffers alive_buffers_;
  RootAndBlock root_and_block_;
  bool block_need_fuse_ = true;
};

template <typename Op>
struct GlobalTensorTrait;

template <typename Op>
struct GlobalTensorChecker {
  static bool ExtentEqual(const std::vector<ForVarExtent>& for_var_extents1,
                          const std::vector<ForVarExtent>& for_var_extents2) {
    if (for_var_extents1.size() != for_var_extents2.size()) return false;

    for (size_t i = 0; i < for_var_extents1.size(); ++i) {
      ForVarExtent lhs = for_var_extents1[i];
      ForVarExtent rhs = for_var_extents2[i];
      if (cinn::common::AutoSimplify(ir::Sub::Make(lhs.extent, rhs.extent)) !=
          ir::Expr(0)) {
        return false;
      }
    }
    return true;
  }

  static bool IndiceEqual(const IndicesAndExtent& indice_and_extent1,
                          const IndicesAndExtent& indice_and_extent2) {
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

    if (!ExtentEqual(indice_and_extent1.for_var_extents,
                     indice_and_extent2.for_var_extents)) {
      return false;
    }
    const auto& indice1 = indice_and_extent1.indices;
    const auto& indice2 = indice_and_extent2.indices;
    if (indice1.size() != indice2.size()) return false;

    std::unordered_map<ir::Var, ir::Var> for_var_map =
        ConstructForVarReplaceMap(indice_and_extent1.for_var_extents,
                                  indice_and_extent2.for_var_extents);

    for (size_t i = 0; i < indice1.size(); ++i) {
      ir::Expr lhs = IndiceToExprWithForVar(indice1.at(i), for_var_map);
      ir::Expr rhs = IndiceToExprWithForVar(indice2.at(i), for_var_map);
      if (cinn::common::AutoSimplify(ir::Sub::Make(lhs, rhs)) != ir::Expr(0)) {
        return false;
      }
    }
    return true;
  }

  static bool LoopVarAndExtentEqual(
      const IndicesAndExtent& indice_and_extent1,
      const IndicesAndExtent& indice_and_extent2) {
    const auto& for_var_extents1 = indice_and_extent1.for_var_extents;
    const auto& for_var_extents2 = indice_and_extent2.for_var_extents;
    if (!ExtentEqual(for_var_extents1, for_var_extents2)) return false;

    for (size_t i = 0; i < for_var_extents1.size(); ++i) {
      ForVarExtent lhs = for_var_extents1[i];
      ForVarExtent rhs = for_var_extents2[i];
      if (lhs.loop_var != rhs.loop_var) return false;
    }
    return true;
  }

  static bool IsCommonGlobalTensor(
      const std::vector<IndicesAndExtent>& indice_and_extent) {
    auto AllIndiceAndExtentEqual =
        [&](const std::vector<IndicesAndExtent>& indice_and_extent) -> bool {
      PADDLE_ENFORCE_GE(
          indice_and_extent.size(),
          2,
          ::common::errors::InvalidArgument(
              "The size of indice_and_extent should greater_equal to 2"));
      for (size_t i = 1; i < indice_and_extent.size(); ++i) {
        if (!GlobalTensorTrait<Op>::IndiceAndExtentEqual(indice_and_extent[0],
                                                         indice_and_extent[i]))
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

    if (indice_and_extent.size() <= 1) return false;
    if (IndiceContainsLoad(indice_and_extent[0])) return false;
    return AllIndiceAndExtentEqual(indice_and_extent);
  }
};

template <>
struct GlobalTensorTrait<ForLevel> {
  static bool IndiceAndExtentEqual(const IndicesAndExtent& indice_and_extent1,
                                   const IndicesAndExtent& indice_and_extent2) {
    return GlobalTensorChecker<ForLevel>::IndiceEqual(indice_and_extent1,
                                                      indice_and_extent2) &&
           GlobalTensorChecker<ForLevel>::LoopVarAndExtentEqual(
               indice_and_extent1, indice_and_extent2);
  }
};

template <>
struct GlobalTensorTrait<KernelLevel> {
  static bool IndiceAndExtentEqual(const IndicesAndExtent& indice_and_extent1,
                                   const IndicesAndExtent& indice_and_extent2) {
    return GlobalTensorChecker<KernelLevel>::IndiceEqual(indice_and_extent1,
                                                         indice_and_extent2);
  }
};

struct GlobalTensorInfoCollector : public ir::IRMutator<Expr*> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  template <typename Op>
  std::unordered_set<std::string> GetCommonGlobalBufferNames() const {
    std::unordered_set<std::string> global_buffer_name;
    for (const auto& [buffer_name, indice_and_extent] :
         buffer_to_indice_and_extent_) {
      if (global_store_buffer_names_.find(buffer_name) !=
          global_store_buffer_names_.end()) {
        continue;
      }
      if (contains_select_) continue;
      if (GlobalTensorChecker<Op>::IsCommonGlobalTensor(indice_and_extent)) {
        global_buffer_name.insert(buffer_name);
      }
    }
    return global_buffer_name;
  }

 private:
  void Visit(const ir::ScheduleBlockRealize* op, ir::Expr* expr) override {
    const auto* sbr_node = expr->As<ir::ScheduleBlockRealize>();
    CHECK(sbr_node);
    const auto& iter_values = sbr_node->iter_values;
    const auto* sb_node = sbr_node->schedule_block.As<ir::ScheduleBlock>();
    const auto& iter_vars = sb_node->iter_vars;
    PADDLE_ENFORCE_EQ(
        iter_values.size(),
        iter_vars.size(),
        ::common::errors::InvalidArgument(
            "The size of iter_values should equal to the size of iter_vars, as "
            "they comes from the same ScheduleBlockRealize"));

    for (std::size_t i = 0; i < iter_values.size(); ++i) {
      var_to_sb_expr_[iter_vars[i]] = iter_values[i];
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::For* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::For>();
    CHECK(node);
    for_var_extents_.push_back(
        {node->loop_var, ir::ir_utils::IRCopy(node->extent)});
    ir::IRMutator<>::Visit(op, expr);
    for_var_extents_.pop_back();
  }

  void Visit(const ir::Load* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Load>();
    CHECK(node);
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
    CHECK(node);
    const auto& store_buffer = node->tensor.as_tensor_ref()->buffer;
    if (store_buffer->memory_type == ir::MemoryType::Heap) {
      global_store_buffer_names_.insert(store_buffer->name);
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Select* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Select>();
    CHECK(node);
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

struct SubstituteTensorWithVar : public ir::IRMutator<Expr*> {
  SubstituteTensorWithVar(
      const std::unordered_set<std::string>& merge_buffer_names)
      : merge_buffer_names_(merge_buffer_names) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Block* op, Expr* expr) override {
    auto* node = expr->As<ir::Block>();
    CHECK(node);
    current_block_ = node;
    IRMutator<>::Visit(op, expr);

    if (block_to_var_declare_sbrs_.find(node) ==
        block_to_var_declare_sbrs_.end()) {
      return;
    }
    while (!block_to_var_declare_sbrs_[node].empty()) {
      const ir::Expr declare_sbr = block_to_var_declare_sbrs_[node].back();
      block_to_var_declare_sbrs_[node].pop_back();
      PADDLE_ENFORCE_NE(
          declare_sbr,
          ir::Expr(nullptr),
          ::common::errors::InvalidArgument(
              "common global var declare %s should not be none", declare_sbr));
      node->stmts.insert(node->stmts.begin(), declare_sbr);
    }
  }

  void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) override {
    auto* node = expr->As<ir::ScheduleBlockRealize>();
    CHECK(node);
    current_sbr_ = node;
    IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load* op, Expr* expr) override {
    auto* node = expr->As<ir::Load>();
    CHECK(node);
    const auto& buffer_name = node->tensor.as_tensor_ref()->buffer->name;
    if (merge_buffer_names_.count(buffer_name) == 0) {
      return;
    }

    if (buffer_to_var_.count(buffer_name) == 0) {
      RecordLocalVarBlock(node, buffer_name);
    }
    *expr = buffer_to_var_[buffer_name];
  }

  void RecordLocalVarBlock(ir::Load* load_node,
                           const std::string& buffer_name) {
    const ir::Expr sb = ir::ir_utils::IRCopy(current_sbr_->schedule_block);
    const ir::ScheduleBlock* sb_node = sb.As<ir::ScheduleBlock>();
    CHECK(sb_node);

    auto local_var =
        ir::_Var_::Make(common::UniqName("cse_global_var"), load_node->type());
    auto let_op = ir::Let::Make(local_var, const_cast<ir::Load*>(load_node));
    ir::Expr new_sb = ir::ScheduleBlock::Make(
        sb_node->iter_vars, {}, {}, sb_node->name + "_merge_local", let_op);

    ir::Expr new_sbr = ir::ScheduleBlockRealize::Make(
        ir::ir_utils::IRCopy(current_sbr_->iter_values), new_sb);
    PADDLE_ENFORCE_EQ(
        buffer_to_var_.count(buffer_name),
        0,
        ::common::errors::InvalidArgument(
            "buffer_name %s should not be in buffer_to_var_", buffer_name));
    buffer_to_var_[buffer_name] = local_var;
    block_to_var_declare_sbrs_[current_block_].push_back(new_sbr);
  }

  std::unordered_set<std::string> merge_buffer_names_;
  std::unordered_map<std::string, ir::Expr> buffer_to_var_;
  std::unordered_map<ir::Block*, std::vector<ir::Expr>>
      block_to_var_declare_sbrs_;

  ir::ScheduleBlockRealize* current_sbr_;
  ir::Block* current_block_;
};

void MergeReduceInitScheduleBlock(ir::Expr* e) {
  VLOG(6) << "Before ReduceInitMerge_MergeReduceInitScheduleBlock: \n" << *e;
  ReduceInitCollector reduce_init_collector;
  reduce_init_collector(e);

  const auto& root_and_block = reduce_init_collector.GetMergeRootAndBlock();

  BlockMerger block_merger(root_and_block);
  block_merger(e);
  VLOG(6) << "After ReduceInitMerge_MergeReduceInitScheduleBlock: \n" << *e;
}

void MergeGlobalTensorLoad(ir::Expr* e) {
  VLOG(6) << "Before ReduceInitMerge_MergeGlobalTensorLoad: \n" << *e;
  GlobalTensorInfoCollector global_tensor_info_collector;
  global_tensor_info_collector(e);
  const auto& global_tensor_names =
      global_tensor_info_collector.GetCommonGlobalBufferNames<KernelLevel>();

  GlobalLoadMergeCollector global_load_merge_collector(global_tensor_names);
  global_load_merge_collector(e);
  const auto& buffers = global_load_merge_collector.GetBufferToRootAndBlock();

  for (const auto& [buffer_name, root_and_block] : buffers) {
    AliveBufferAnalyer alive_buffer_analyzer(root_and_block);
    alive_buffer_analyzer(e);
    if (alive_buffer_analyzer.GetBlockNeedFuse()) {
      BlockMerger block_merger(root_and_block);
      block_merger(e);
    }
  }
  VLOG(6) << "After ReduceInitMerge_MergeGlobalTensorLoad: \n" << *e;
}

void SubstituteGlobalTensor(ir::Expr* e) {
  VLOG(6) << "Before ReduceInitMerge_SubstituteGlobalTensor: \n" << *e;
  GlobalTensorInfoCollector collector;
  collector(e);

  const auto& substitute_buffer_names =
      collector.GetCommonGlobalBufferNames<ForLevel>();

  SubstituteTensorWithVar substitute_functor(substitute_buffer_names);
  substitute_functor(e);
  VLOG(6) << "After ReduceInitMerge_SubstituteGlobalTensor: \n" << *e;
}

}  // namespace

void ReduceInitMerge(Expr* e) {
  VLOG(4) << "Before ReduceInitMerge: \n" << *e;

  MergeReduceInitScheduleBlock(e);
  MergeGlobalTensorLoad(e);
  SubstituteGlobalTensor(e);

  VLOG(4) << "After ReduceInitMerge: \n" << *e;
}

}  // namespace optim
}  // namespace cinn
