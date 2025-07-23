// Copyright (c) 2025 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/utils/string.h"

PD_DECLARE_bool(cinn_enable_tile_transpose);

namespace cinn {
namespace ir {
namespace {

/**
 * Tiling template for Elementwise+Transpose fusion graph.
 *
 * This tactic accelerates transposes by doing inner-block transpose in shared
 * memory, making both reads and writes coaleased, therefore achieving nearly
 * copy-like throughput.
 *
 * This tactic literally supports any permutation, as long as the last dimension
 * is permuted. If the last dimension is consistent, the general tactic has been
 * good. However, this tactic has limitations in the fusion:
 * 1) Reduce is not supported, because reduce usually requires a larger inner
 *    loop (~32) for better performance, while transpose prefers a smaller inner
 *    loop (~4) to restrict the shared memory size.
 * 2) All transposes in the graph must have the same permutation, because the
 *    size of the shared memory we need is 32^(n+1), where n is the number of
 *    different permutations. More than one permutation makes it impossible to
 *    allocate such a huge space.
 *
 *
 * How does this tactic work:
 *
 * First, we generalize a transpose `src => dst` as:
 *    src [ ..., dst_low_axis, ..., src_low_axis ]
 *    dst [ ..., src_low_axis, ..., dst_low_axis ]
 * The rest `...` are called high_axis, which may contain any permutation, and
 * can be transposed by simple index mapping without impacting performance.
 *
 * Second, we split both src_low_axis and dst_low_axis into {-1, 32}:
 *    src [ ..., d_h, d32, ... s_h, s32 ]
 *    dst [ ..., s_h, s32, ... d_h, d32 ]
 *
 * Third, we create a shared memory of shape [32, 32], and bind (s32, d32) to
 * (thread.y, thread.x) to transpose them in the shared memory. The (s_h, d_h)
 * also become high_axis. We transpose high_axis using (block.y, block.x).
 *    src [ block.x, thread.y, block.y, thread.x ]
 *                        \              /
 *                          \          /
 *                 shm [ thread.y, thread.x ]  (write cache)
 *                            |      |
 *                 shm [ thread.x, thread.y ]  (read cache)
 *                               \ /
 *                          _____/ \_____
 *                         /             \
 *    dst [ block.y, thread.y, block.x, thread.x ]
 *
 * Finally, the IR is like:
 *    shm[thread.y][thread.x] = src[block.x, thread.y, block.y, thread.x]
 *    __syncthreads()
 *    dst[block.y, thread.y, block.x, thread.x] = shm[thread.x][thread.y]
 *
 * Notes:
 * 1) For simplicity, the high_axis are actually all bound to block.x.
 * 2) For performance, thread.y is actually composed of 4 loops * 8 threads.
 * 3) To support multiple transpose inputs, we actually store the transposed
 *    value to a local buffer before later computation, so that all inputs can
 *    reuse the same shared memory.
 */
class TileTransposeTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context, ir::IRSchedule* sch) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "TileTransposeTactic"; }

 private:
  struct Candidate {
    // The target load to do CacheRead.
    ir::Expr load;

    // The block where this load first appears. We will do CacheRead on this
    // block, and later blocks will simply reuse the first load's value.
    std::string first_appear_block_id;

    // The buffer index of this load in the first block it appears.
    int buffer_index;
  };

  void InitCandidates(ir::IRSchedule* sch);
  void InitUnconditionalLoads(ir::IRSchedule* sch);
  void InitAxisInfo();

  std::vector<int> GetSrcLowAxis(const std::vector<int>& perm);
  std::vector<int> GetDstLowAxis(const std::vector<int>& perm);

  std::string CreateCacheBlock(ir::IRSchedule* sch,
                               const std::string& block_id,
                               int buffer_index,
                               const std::string& memory_type);
  void TileCacheBlock(ir::IRSchedule* sch,
                      const std::string& block_id,
                      int buffer_index);
  void TileBlock(ir::IRSchedule* sch, const std::string& block_id);
  void CanonicalizeLayout(ir::IRSchedule* sch, const std::string& block_id);
  void FuseAndBind(ir::IRSchedule* sch,
                   const std::string& block_id,
                   bool need_sync = false);

 private:
  ScheduleContext* context_;
  bool can_apply_;

  // The sub iter space apart from the main iter space.
  std::vector<int> sub_iter_space_;

  // Groups of axis as illustrated in the above graph.
  std::vector<int> high_axis_;
  std::vector<int> src_low_axis_;
  std::vector<int> dst_low_axis_;

  struct LoadHash {
    size_t operator()(const ir::Expr& load) const {
      auto& tensor_name = load.As<ir::Load>()->tensor.as_tensor()->name;
      return std::hash<std::string>()(tensor_name);
    }
  };

  // Map from each candidate load to the corresponding Candidate struct.
  // Note: the same tensor name doesn't necessarily refers to the same load,
  // because the load indices may differ. Therefore, we hash loads by their
  // tensor names but check equality by their indices.
  std::unordered_map<ir::Expr, Candidate, LoadHash> load2candidates_;

  // Map from each block's id to the candidates in the block.
  // Note: this map records all possible candidates for a block, including
  // candidates whose first appearing block are not the block.
  std::unordered_map<std::string, std::vector<Candidate>> block2candidates_;

  // Candidate loads that have been cache-read and tiled.
  std::unordered_set<ir::Expr, LoadHash> processed_loads_;

  // Loads that are executed unconditionally (not inside Select).
  std::unordered_set<ir::Expr, LoadHash> unconditional_loads_;
};

std::vector<int> OffsetVec(const std::vector<int>& vec, int offset) {
  std::vector<int> new_vec = vec;
  for (auto& e : new_vec) e += offset;
  return new_vec;
}

std::vector<int> ArangeVec(int count, int begin = 0) {
  std::vector<int> vec(count);
  std::iota(vec.begin(), vec.end(), begin);
  return vec;
}

int64_t GetLoopRangeProduct(const std::vector<ir::Expr>& loops,
                            const std::vector<int>& loops_index) {
  int64_t prod = 1;
  for (int i : loops_index) {
    auto* node = loops[i].As<ir::For>();
    if (!node->extent.is_constant()) return -1;
    prod *= node->extent.as_int64();
  }
  return prod;
}

/**
 * Get the relative iter space of the load according to the loops.
 *
 * This class currently supports the following cases:
 *   1) var[i, k, m, j]  (index mapping)
 *         iter space: [i, k, m, j]
 *   2) var[i, k % 32, k % 32, j]  (simple splitting)
 *        iter space: [i, k, j]
 *   3) var[i, k * 32 + m, j]  (simple fusion)
 *        iter space: [i, k, m, j]
 *   4) var[i, k + 128, j]  (simple offsetting)
 *        iter space: [i, k, j]
 *
 * The result is translated to the corresponding loop_index instead of returning
 * loop_vars directly.
 */
struct IterSpaceGetter {
  IterSpaceGetter(const ir::Load* load, const std::vector<ir::Expr>& loops)
      : load_(load), loops_(loops), indices_vars_(load->indices.size()) {
    for (int i = 0; i < load_->indices.size(); ++i) {
      ir::ir_utils::CollectIRNodes(load_->indices[i], [&](const ir::Expr* x) {
        if (x->is_var() && !x->as_var()->is_symbolic_constant) {
          indices_vars_[i].insert(x->as_var_ref());
        }
        return false;
      });
    }
  }

  std::vector<int> operator()() {
    // Try to arrange the iter vars in the order of the iter space
    std::vector<ir::Var> iter_space_vars;
    for (int i = 0; i < load_->indices.size(); ++i) {
      // Case 1. constant
      if (indices_vars_[i].size() == 0) {
        continue;
      }

      // Case 2. single variable
      if (indices_vars_[i].size() == 1) {
        int cover_range = CheckSingleVar(i);
        if (cover_range < 0) return {};
        iter_space_vars.push_back(*indices_vars_[i].begin());
        i += cover_range - 1;
        continue;
      }

      // Case 3. no more than 3 variables
      if (indices_vars_[i].size() <= 3) {
        std::vector<ir::Var> arranged_vars = CheckMultipleVars(i);
        if (arranged_vars.empty()) return {};
        iter_space_vars.insert(
            iter_space_vars.end(), arranged_vars.begin(), arranged_vars.end());
        continue;
      }

      return {};
    }

    // Construct the iter space
    std::vector<int> iter_space;
    for (auto& var : iter_space_vars) {
      int loop_index =
          std::stoi(var->name.substr(std::strlen(analyzer::kLoopVar)));
      iter_space.push_back(loop_index);
    }
    return iter_space;
  }

 private:
  int CheckSingleVar(int begin) {
    ir::Var var = *indices_vars_[begin].begin();

    // Check that var exclusively covers a continuous range, such as:
    //   [ ..., i / 32, i % 32, ... ]
    // The following cases are not supported:
    //   [ ..., i / 32, (i % 32) * 4 + j, ... ]  # not exclusive
    //   [ ..., i / 32, ..., i % 32, ... ]       # not continuous
    int end;
    for (end = begin + 1; end < indices_vars_.size(); ++end) {
      if (indices_vars_[end].count(var) == 0) break;
      if (indices_vars_[end].size() > 1) return -1;
    }
    for (int i = end + 1; i < indices_vars_.size(); ++i) {
      if (indices_vars_[i].count(var) > 0) return -1;
    }

    // Try to fuse the indices that contain `var` into one expression
    ir::Expr fused_index;
    if (end - begin == 1) {
      fused_index = optim::ArithSimplify(load_->indices[begin]);
    } else {
      auto shape_it = load_->tensor.as_tensor()->shape.begin();
      auto indices_it = load_->indices.begin();
      std::vector<ir::Expr> sub_shape(shape_it + begin, shape_it + end);
      std::vector<ir::Expr> sub_indices(indices_it + begin, indices_it + end);
      fused_index = common::IndiceToAbsOffset(sub_shape, sub_indices);
    }

    // Check that fused_index is either a single `var` or `var + offset`
    if (fused_index != ir::Expr(var)) {
      auto* add_node = fused_index.As<ir::Add>();
      if (!add_node || add_node->a() != ir::Expr(var)) return -1;
    }

    return end - begin;
  }

  std::vector<ir::Var> CheckMultipleVars(int pos) {
    // Check that vars at this pos only appear at this pos, such as:
    //   [ ..., i * 32 + j, ... ]
    // The following case is not supported:
    //   [ ..., (i * 32 + j) / 8, j % 8, ... ]
    // because j appears at multiple positions.
    for (int i = 0; i < indices_vars_.size(); ++i) {
      if (i == pos) continue;
      for (auto& var : indices_vars_[i]) {
        if (indices_vars_[pos].count(var) > 0) return {};
      }
    }

    // Collect vars in this index in ast order
    std::vector<ir::Var> vars_in_ast_order;
    ir::ir_utils::CollectIRNodes(load_->indices[pos], [&](const ir::Expr* x) {
      if (x->is_var() && !x->as_var()->is_symbolic_constant) {
        vars_in_ast_order.push_back(x->as_var_ref());
      }
      return false;
    });

    // Re-construct the index using the vars in ast order
    std::vector<ir::Expr> sub_shape;
    std::vector<ir::Expr> sub_indices;
    for (auto& var : vars_in_ast_order) {
      int loop_index =
          std::stoi(var->name.substr(std::strlen(analyzer::kLoopVar)));
      sub_shape.push_back(loops_[loop_index].As<ir::For>()->extent);
      sub_indices.push_back(var);
    }
    ir::Expr sub_index = common::IndiceToAbsOffset(sub_shape, sub_indices);

    // Compare the re-constructed index with the actual index
    if (sub_index == load_->indices[pos]) {
      return vars_in_ast_order;
    }
    return {};
  }

 private:
  const ir::Load* load_;
  const std::vector<ir::Expr>& loops_;

  // iter vars in each of the load's indices
  std::vector<std::set<ir::Var>> indices_vars_;
};

void TileTransposeTactic::Init(ScheduleContext* context, ir::IRSchedule* sch) {
  context_ = context;
  can_apply_ = false;
  if (!FLAGS_cinn_enable_tile_transpose) return;

  ir::Expr module_root = sch->GetModule().GetExprs().front();
  ir::Expr root_block = ir::analyzer::GetRootSBlock(module_root);
  auto* root_node = root_block.As<ir::ScheduleBlockRealize>()
                        ->schedule_block.As<ir::ScheduleBlock>();

  if (root_node->attrs.count(kTileMethod) > 0) return;
  if (!context->config.base_info->reduce_axis.empty()) return;

  // There must be at least 8 warps (256 threads) to perform this tactic.
  if (context->config.tile_config.warp_num < 8) return;

  InitUnconditionalLoads(sch);
  InitCandidates(sch);

  VLOG(4) << "sub_iter_space: " << utils::Join(sub_iter_space_, ", ");
  if (sub_iter_space_.empty()) return;

  can_apply_ = true;
  root_node->attrs[kTileMethod] = TacticName();

  InitAxisInfo();
}

void TileTransposeTactic::InitUnconditionalLoads(ir::IRSchedule* sch) {
  struct Collector : public ir::IRMutator<> {
    void operator()(ir::Expr* expr) { IRMutator<>::Visit(expr, expr); }

    void Visit(const ir::Select* op, ir::Expr* expr) override {
      auto* select = expr->As<ir::Select>();
      IRMutator<>::Visit(&select->condition, &select->condition);
    }

    void Visit(const ir::Load* op, ir::Expr* expr) override {
      results_.insert(*expr);
      IRMutator<>::Visit(op, expr);
    }

    std::unordered_set<ir::Expr, LoadHash> results_;
  };

  Collector collector;
  for (auto& block : sch->GetAllBlocks()) {
    std::vector<ir::Expr> loops = sch->GetLoops(block);
    ir::Expr store = ir::analyzer::GetStoreOfSBlock(block);
    store = ir::analyzer::ExpandIterVar(store, block);
    store = ir::analyzer::CanonicalizeLoopVar(store, loops);
    collector(&store.As<ir::Store>()->value);
  }
  unconditional_loads_ = std::move(collector.results_);
}

void TileTransposeTactic::InitCandidates(ir::IRSchedule* sch) {
  sub_iter_space_.clear();
  load2candidates_.clear();
  block2candidates_.clear();
  processed_loads_.clear();

  for (auto& block : sch->GetAllBlocks()) {
    std::vector<ir::Expr> loops = sch->GetLoops(block);
    std::string block_id = ir::analyzer::GetBlockName(block);

    ir::Expr store = ir::analyzer::GetStoreOfSBlock(block);
    store = ir::analyzer::ExpandIterVar(store, block);
    store = ir::analyzer::CanonicalizeLoopVar(store, loops);

    std::vector<ir::Expr> loads = ir::ir_utils::CollectIRNodesInOrder(
        store.As<ir::Store>()->value,
        [](const ir::Expr* x) { return x->As<ir::Load>(); });

    // Find candidate loads in this schedule block using the following rules.
    for (int i = 0; i < loads.size(); ++i) {
      ir::Expr load = loads[i];

      // 1. Skip loads that have been added.
      auto candidate_it = load2candidates_.find(load);
      if (candidate_it != load2candidates_.end()) {
        block2candidates_[block_id].push_back(candidate_it->second);
        continue;
      }

      // 2. Skip conditional loads (loads inside Select). As we are currently
      //    unable to analyze the Select's condition, these loads may lead to
      //    out-of-bound accesses.
      if (unconditional_loads_.count(load) == 0) continue;

      // 3. The load tensor should not be defined by a previous schedule block,
      //    otherwise we should do CacheRead on that block rather than here.
      auto* tensor = load.As<ir::Load>()->tensor.as_tensor();
      if (sch->HasBlock(tensor->name)) continue;

      IterSpaceGetter iter_space_getter(load.As<ir::Load>(), loops);
      std::vector<int> iter_space = iter_space_getter();

      // 4. This is a critical transpose, including:
      // 1) its dim size equals to the loop size (not a broadcast).
      // 2) its last dim is changed in permutation (incurs discrete access).
      // 3) both the src/dst_low_axis are non-unit (not a squeeze/unsqueeze).
      if (iter_space.size() != loops.size()) continue;
      if (iter_space.back() == iter_space.size() - 1) continue;
      if (GetLoopRangeProduct(loops, GetSrcLowAxis(iter_space)) == 1) continue;
      if (GetLoopRangeProduct(loops, GetDstLowAxis(iter_space)) == 1) continue;

      // 5. All transposes in this graph should be in the same sub iter space,
      //    because we only support the alignment of two iter spaces.
      if (sub_iter_space_.empty()) {
        sub_iter_space_ = iter_space;
      } else if (sub_iter_space_ != iter_space) {
        sub_iter_space_.clear();
        return;
      }

      Candidate candidate{load, block_id, i};
      load2candidates_.emplace(load, candidate);
      block2candidates_[block_id].push_back(candidate);
    }
  }
}

void TileTransposeTactic::InitAxisInfo() {
  src_low_axis_ = GetSrcLowAxis(sub_iter_space_);
  dst_low_axis_ = GetDstLowAxis(sub_iter_space_);

  std::set<int> high_axis;
  for (int i = 0; i < sub_iter_space_.size(); ++i) high_axis.insert(i);
  for (auto i : src_low_axis_) high_axis.erase(i);
  for (auto i : dst_low_axis_) high_axis.erase(i);
  high_axis_.assign(high_axis.begin(), high_axis.end());
}

std::vector<int> TileTransposeTactic::GetSrcLowAxis(
    const std::vector<int>& iter_space) {
  std::set<int> src_low_axis{iter_space.back()};
  for (int i = iter_space.size() - 2; i >= 0; --i) {
    if (iter_space[i] + 1 != iter_space[i + 1]) break;
    src_low_axis.insert(iter_space[i]);
  }
  return {src_low_axis.begin(), src_low_axis.end()};
}

std::vector<int> TileTransposeTactic::GetDstLowAxis(
    const std::vector<int>& iter_space) {
  std::set<int> dst_low_axis;
  auto it =
      std::find(iter_space.begin(), iter_space.end(), iter_space.size() - 1);
  if (it != iter_space.end()) {
    dst_low_axis.insert(*it);
    while (it != iter_space.begin()) {
      if (*(it - 1) != *it - 1) break;
      --it;
      dst_low_axis.insert(*it);
    }
  }
  return {dst_low_axis.begin(), dst_low_axis.end()};
}

void TileTransposeTactic::Apply(ir::IRSchedule* sch,
                                const std::string& block_id) {
  if (!can_apply_) return;

  // Handle all candidate loads in the block before tiling the block itself,
  // otherwise we will lose track of some occurrences of the loads due to
  // indices change.
  for (auto& candidate : block2candidates_[block_id]) {
    if (processed_loads_.count(candidate.load) == 0) {
      TileCacheBlock(
          sch, candidate.first_appear_block_id, candidate.buffer_index);
      processed_loads_.insert(candidate.load);
    }
  }

  // Tile the block itself.
  TileBlock(sch, block_id);

  VLOG(4) << "After TileTransposeTactic on [" << block_id
          << "]: " << sch->GetModule().GetExprs().front();
}

std::string TileTransposeTactic::CreateCacheBlock(
    ir::IRSchedule* sch,
    const std::string& block_id,
    int buffer_index,
    const std::string& memory_type) {
  ir::Expr block = sch->GetBlock(block_id);
  ir::Expr cache_block = sch->CacheRead(block, buffer_index, memory_type);

  std::string transpose_stage = (memory_type == "shared") ? "write" : "read";
  sch->Annotate(cache_block, "transpose_stage", transpose_stage);

  // Mark the cache block as a virtual output to prevent inlining. This doesn't
  // affect the actual outputs of the graph.
  std::string cache_block_id = ir::analyzer::GetBlockName(cache_block);
  context_->output_names.insert(cache_block_id);

  return cache_block_id;
}

void TileTransposeTactic::TileCacheBlock(ir::IRSchedule* sch,
                                         const std::string& block_id,
                                         int buffer_index) {
  // Step 1. Create the shared and local buffers.
  std::string shared_cache_block_id =
      CreateCacheBlock(sch, block_id, buffer_index, "shared");
  std::string local_cache_block_id =
      CreateCacheBlock(sch, block_id, buffer_index, "local");

  // Step 2. Convert the layout to [ high_axis, src_low_axis, dst_low_axis ].
  CanonicalizeLayout(sch, shared_cache_block_id);
  CanonicalizeLayout(sch, local_cache_block_id);

  // Step 3. Do inner-block transpose.
  int offset = high_axis_.size();
  sch->Split(shared_cache_block_id, offset + 1, {-1, 4, 8});
  sch->Split(shared_cache_block_id, offset, {-1, 32});

  sch->Split(local_cache_block_id, offset + 1, {-1, 32});
  sch->Split(local_cache_block_id, offset, {-1, 4, 8});

  sch->Reorder(shared_cache_block_id, OffsetVec({0, 2, 3, 4, 1}, offset));
  sch->Reorder(local_cache_block_id, OffsetVec({0, 3, 1, 2, 4}, offset));

  // Step 4. Fuse and bind as [ block_x, inner_loop, thread_y, thread_x ].
  FuseAndBind(sch, shared_cache_block_id, /* need_sync = */ true);
  FuseAndBind(sch, local_cache_block_id, /* need_sync = */ true);
}

void TileTransposeTactic::TileBlock(ir::IRSchedule* sch,
                                    const std::string& block_id) {
  CanonicalizeLayout(sch, block_id);

  int offset = high_axis_.size();
  sch->Split(block_id, offset + 1, {-1, 32});
  sch->Split(block_id, offset, {-1, 4, 8});

  sch->Reorder(block_id, OffsetVec({0, 3, 1, 2, 4}, offset));

  FuseAndBind(sch, block_id);

  if (context_->output_names.count(block_id) == 0) {
    ir::Expr block = sch->GetBlock(block_id);
    sch->SetBuffer(block, "local");
  }
}

void TileTransposeTactic::CanonicalizeLayout(ir::IRSchedule* sch,
                                             const std::string& block_id) {
  std::vector<int> order = high_axis_;
  order.insert(order.end(), src_low_axis_.begin(), src_low_axis_.end());
  order.insert(order.end(), dst_low_axis_.begin(), dst_low_axis_.end());

  sch->Reorder(block_id, order);

  std::vector<int> src_low_axis =
      ArangeVec(src_low_axis_.size(), high_axis_.size());
  std::vector<int> dst_low_axis =
      ArangeVec(dst_low_axis_.size(), high_axis_.size() + src_low_axis_.size());

  sch->Fuse(block_id, dst_low_axis);
  sch->Fuse(block_id, src_low_axis);
}

void TileTransposeTactic::FuseAndBind(ir::IRSchedule* sch,
                                      const std::string& block_id,
                                      bool need_sync) {
  int offset = high_axis_.size();
  sch->Fuse(block_id, ArangeVec(offset + 2));

  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  sch->Bind(loops[0], "blockIdx.x");
  sch->Bind(loops[2], "threadIdx.y");
  sch->Bind(loops[3], "threadIdx.x");

  if (need_sync) {
    sch->SyncThreads(sch->GetLoops(block_id)[0], /* after_node = */ false);
  }
}

}  // namespace

std::unique_ptr<ScheduleTactic> CreateTileTransposeTactic() {
  return std::make_unique<TileTransposeTactic>();
}

}  // namespace ir
}  // namespace cinn
