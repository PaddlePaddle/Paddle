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

#include "paddle/cinn/ir/group_schedule/tactic/tile_broadcast_tactic.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"

PD_DECLARE_bool(cinn_enable_tile_broadcast);

namespace cinn {
namespace ir {
namespace {

/**
 * Tiling template for NCHW broadcasts.
 *
 * This tactic performs tiling for NCHW broadcasts that have the form:
 *    [1, C, 1, 1] => [N, C, H, W].
 * More generally, if we classify the axis into two types:
 *    B - broadcast axis, i.e. N, H, and W
 *    P - preserved axis, i.e. C
 * then this tactic can handle (in the sense of axis fusion):
 *    [1, P, 1] => [B, P, B],
 * as long as the last axis is B.
 *
 *
 * Performance Impact:
 * This tactic primarily addresses the issue of redundant computation in
 * broadcasts. Without this tactic, the general tiling tactic doesn't treat the
 * P axis as a special axis, and fuses all axis into one before tiling. As a
 * result, P's index is often interleaved with the inner loop, such as:
 *    ... = exp(var[k * 256 + threadIdx.x]),
 * thus for every loop iteration `k`, we needs to load and compute a new value.
 *
 * On the contrary, this tactic assigns a dedicated axis (blockIdx.x or
 * blockIdx.y) for P, such as:
 *    ... = exp(var[blockIdx.x]),
 * so that each thread only need to load and compute it once. When dealing with
 * complex ops (e.g. div, exp, rsqrt), this tactic can save much computation
 * bandwidth and bring up to 30% speedup.
 *
 *
 * Limitations:
 * The implementation of this tactic has been tailored for various layouts.
 * However, we have not yet observed consistent performance improvements with
 * layouts other than NCHW. Therefore, it is exclusive for NCHW now.
 *
 * To avoid unexpected performance degradation, this tactic also imposes
 * constraints on dim sizes. See Init for details.
 *
 *
 * Example:
 *   for (i, 0, 128):       # N
 *     for (j, 0, 256):     # C
 *       for (k, 0, 32):    # H
 *         for (a, 0, 32):  # W
 *           ScheduleBlock(var_1):
 *             var_1[i, j, k, a] = exp(var[j])
 * =>
 *   for (blockIdx.y, 0, 128):         # N
 *     for (blockIdx.x, 0, 256):       # C
 *       for (k, 0, 4):                # HW / 256
 *         for (threadIdx.x, 0, 256):  # HW % 256
 *           ScheduleBlock(var_1):
 *              var_1[blockIdx.y, blockIdx.x, ...] = exp(var[blockIdx.x])
 *
 * Note: there are 3 ways of axis binding in this tactic for different dim
 * sizes. See Apply for details.
 */
class TileBroadcastTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context, ir::IRSchedule* sch) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "TileBroadcastTactic"; }

 private:
  void InitBroadcastAxisInfo(ir::IRSchedule* sch);
  void InitBroadcastSizeInfo();
  void FuseAxisGroups(ir::IRSchedule* sch, const std::string& block_id);
  void ApplyVectorize(ir::IRSchedule* sch,
                      const std::string& block_id,
                      const int block_size);
  bool CanEnableVectorize(const int block_size);
  std::vector<std::string> TileNCHW(ir::IRSchedule* sch,
                                    const std::string& block_id,
                                    int block_size);
  std::vector<std::string> TileNHWC(ir::IRSchedule* sch,
                                    const std::string& block_id,
                                    int block_size);
  std::vector<std::string> TileVectorizeNCHW(ir::IRSchedule* sch,
                                             const std::string& block_id,
                                             const int block_size,
                                             const int vetorize_factor);

 private:
  ScheduleContext* context_;

  enum class BroadcastLayout { Invalid = 0x0, NCHWLayout, NHWCLayout };

  BroadcastLayout applied_layout_;  // applied tactic

  // list of broadcast axis in ascending order
  std::vector<int> broadcast_axis_;
  // one-hot representation of broadcast_axis
  std::vector<bool> is_broadcast_axis_;

  // list of 3 groups of axis, as the following graph shows:
  //   high_broadcast_axis
  //        v     v
  //       [B, P, B, P, ..., B, B]
  //           ^     ^       ^  ^
  //           |     |       low_broadcast_axis
  //        preserved_axis

  std::vector<int> high_broadcast_axis_;
  std::vector<int> preserved_axis_;
  std::vector<int> low_broadcast_axis_;

  // product of all the broadcast axis's dim sizes
  int64_t broadcast_size_;
  // product of the low broadcast axis's dim sizes
  int64_t low_broadcast_size_;
  // product of the preserved axis's dim sizes
  int64_t preserved_size_;
};

std::unordered_set<ir::Var> CollectIterVars(
    const std::vector<ir::Expr>& exprs) {
  std::unordered_set<ir::Var> result;
  for (auto& expr : exprs) {
    ir::ir_utils::CollectIRNodes(expr, [&](const ir::Expr* x) {
      if (x->is_var() && !x->as_var()->is_symbolic_constant) {
        result.insert(x->as_var_ref());
      }
      return false;
    });
  }
  return result;
}

std::unordered_map<ir::Var, int> GetVar2LoopIdxMap(
    const ir::Expr& block, const std::vector<ir::Expr>& loops) {
  std::unordered_map<ir::Var, int> var2loopidx;
  auto* block_realize = block.As<ir::ScheduleBlockRealize>();
  auto* block_node = block_realize->schedule_block.As<ir::ScheduleBlock>();

  // map loop vars
  for (int i = 0; i < loops.size(); i++) {
    auto& loop_var = loops[i].As<ir::For>()->loop_var;
    var2loopidx[loop_var] = i;
  }

  // map iter vars
  for (int i = 0; i < block_node->iter_vars.size(); i++) {
    auto& iter_var = block_node->iter_vars[i];
    auto& iter_value = block_realize->iter_values[i];
    if (iter_value.is_var()) {
      var2loopidx[iter_var] = var2loopidx[iter_value.as_var_ref()];
    }
  }
  return var2loopidx;
}

std::vector<ir::Expr> CollectLoads(const ir::Expr& expr) {
  return ir::ir_utils::CollectIRNodesInOrder(
      expr, [](const ir::Expr* x) { return x->As<ir::Load>(); });
}

/**
 * This pair of store and load is elementwise or broadcast if the indices are
 * injective (no reduce) and the iter vars in both's indices are in the same
 * order (no transpose).
 *
 * For example, the following cases are elementwise or broadcast:
 *    var_1[i, j, k] = var_0[i, j, k]  # elementwise
 *    var_3[i, j, k] = var_2[0, j, 0]  # broadcast in {i, k}
 *
 * The following cases are not:
 *    var_1[i, j, k] = var_0[i, k, j]  # transpose in {j, k}
 *    var_3[0, j, k] = var_2[i, j, k]  # reduce in {i}
 */
bool IsElementwiseOrBroadcast(const ir::Store& dst, const ir::Load& src) {
  auto dst_index_it = dst.indices.cbegin();

  for (auto& src_index : src.indices) {
    if (!src_index.is_index()) return false;
    ir::IndexExpr src_index_expr = src_index.as_index().Normalize();
    if (src_index_expr.is_constant()) continue;
    if (!src_index_expr.is_var()) return false;

    dst_index_it = std::find(dst_index_it, dst.indices.end(), src_index_expr);
    if (dst_index_it == dst.indices.cend()) return false;
    ++dst_index_it;
  }
  return true;
}

bool CheckAllElementwiseOrBroadcast(ir::IRSchedule* sch) {
  for (auto& block : sch->GetAllBlocks()) {
    if (ir::analyzer::IsReductionSBlock(block)) return false;
    ir::Expr store = ir::analyzer::GetStoreOfSBlock(block);
    auto* store_node = store.As<ir::Store>();
    for (auto& load : CollectLoads(store_node->value)) {
      auto* load_node = load.As<ir::Load>();
      if (!IsElementwiseOrBroadcast(*store_node, *load_node)) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Get the common set of broadcast axis of all schedule blocks in the graph.
 *
 * For example, given loop axis {i, j, k, a} in order and 3 schedule blocks:
 *    var_1[i, j, k, a] = var_0[k, 0]  # broadcast axis: {i, j, a}
 *    var_3[i, j, k, a] = var_2[i, k]  # broadcast axis: {j, a}
 *    var_5[i, j, k, a] = var_4[0]     # broadcast axis: {i, j, k, a}
 * The common broadcast axis are: {j, a}. After mapping axis to loop index,
 * this function returns {1, 3}.
 */
std::vector<int> GetCommonBroadcastAxis(ir::IRSchedule* sch) {
  std::vector<int> common_broadcast_axis;
  bool is_first_op = true;

  for (auto& block : sch->GetAllBlocks()) {
    std::vector<ir::Expr> loops = sch->GetLoops(block);
    std::unordered_map<ir::Var, int> var2loopidx =
        GetVar2LoopIdxMap(block, loops);

    ir::Expr store = ir::analyzer::GetStoreOfSBlock(block);
    auto* store_node = store.As<ir::Store>();
    std::unordered_set<ir::Var> vars_in_store =
        CollectIterVars(store_node->indices);

    // Visit each load in the store's value to find broadcasts.
    for (auto& load : CollectLoads(store_node->value)) {
      auto* load_node = load.As<ir::Load>();
      std::unordered_set<ir::Var> vars_in_load =
          CollectIterVars(load_node->indices);

      // Get broadcast axis of the load by comparing the iter_vars in the
      // store's indices and the load's indices.
      // Note: if an iter_var only appears in the store's indices but not the
      // load's indices, its corresponding axis is a broadcast axis.
      std::set<int> broadcast_axis;
      for (auto& var : vars_in_store) {
        if (vars_in_load.count(var) == 0) {
          broadcast_axis.insert(var2loopidx[var]);
        }
      }
      if (broadcast_axis.empty()) {  // not a broadcast, just continue
        continue;
      }

      // Do set intersection to update the common broadcast axis.
      if (is_first_op) {
        is_first_op = false;
        common_broadcast_axis.assign(broadcast_axis.begin(),
                                     broadcast_axis.end());
      } else {
        std::vector<int> result(loops.size());
        auto end = std::set_intersection(common_broadcast_axis.begin(),
                                         common_broadcast_axis.end(),
                                         broadcast_axis.begin(),
                                         broadcast_axis.end(),
                                         result.begin());
        common_broadcast_axis.assign(result.begin(), end);
      }
    }
  }
  return common_broadcast_axis;
}

static int CalcNumWarps(int64_t num_warps) {
  // NHWC layout: calculate number of warps per block
  constexpr int MAX_WARP_BLOCK = 32;
  // the largest preserved size is 1024, for size bigger than 1024
  // TODO(heqianyue): the code should be revised to be a DP version
  if (num_warps > 1024) {
    return -1;
  }
  // several rules to decide the thread block size
  // (1) the preserved size (channel) is too small
  // we need more threads in a block
  if (num_warps <= 3) {
    return std::max(4l, num_warps * 2);
  }
  // (2) if the num_warps is power of 2, use thread block size 256
  if ((num_warps & (num_warps - 1)) == 0) {
    return 8;
  }
  // (3) if num_warps <= 32, use the num_warps * 32 as thread block size
  if (num_warps <= MAX_WARP_BLOCK) {
    return num_warps;
  }
  // (4) otherwise, find the largest divisor `best` of num_warps that is smaller
  // than 32 and use `best` * 32 as the block size
  int best = -1;
  for (int x = MAX_WARP_BLOCK; x >= 4; x--) {
    if (num_warps % x == 0) {
      best = x;
      break;
    }
  }
  return best;
}

void TileBroadcastTactic::Init(ScheduleContext* context, ir::IRSchedule* sch) {
  context_ = context;
  applied_layout_ = BroadcastLayout::Invalid;
  if (!FLAGS_cinn_enable_tile_broadcast) {
    return;
  }

  // Check whether we can apply this tactic.
  // We can apply if all the following conditions are met:
  // 1. This graph contains only elementwise or broadcast ops (no reduce or
  //    transpose).
  if (!CheckAllElementwiseOrBroadcast(sch)) {
    return;
  }
  InitBroadcastAxisInfo(sch);
  // 2. There exists common broadcast axis in this graph.
  if (broadcast_axis_.empty()) {
    return;
  }

  if (is_broadcast_axis_.back()) {
    // 3. It is an NCHW broadcast. We check this by checking that he last axis
    // is a broadcast axis, and all the 3 groups of axis exist.
    if (high_broadcast_axis_.empty() || low_broadcast_axis_.empty() ||
        preserved_axis_.empty()) {
      return;
    }
    InitBroadcastSizeInfo();
    // 4. The low_broadcast_size should be a multiple of 32 (the CUDA warp
    // size).
    // Otherwise, memory access will not be fully coalesced, leading to
    // performance degradation.
    // TODO(liangshuhao): we may allow aligning to 16 if further optimizations
    //    can compensate for the cost of non-coalesced access.
    if (low_broadcast_size_ % 32 != 0) {
      return;
    }
    applied_layout_ = BroadcastLayout::NCHWLayout;
    VLOG(4) << "TileBroadcastTactic::Init: matched NCHWLayout\n";
  } else {
    // 3. For NHWC layout, we need valid preserved axis and broadcast axis
    // also, to be more stable, we only handles NHW(C) broadcast
    if (preserved_axis_.empty() || broadcast_axis_.empty() ||
        preserved_axis_.size() > 1) {
      return;
    }
    InitBroadcastSizeInfo();
    // 4. compatible channel size is the multiple of 32
    if (preserved_size_ % 32 != 0) {
      return;
    }
    int num_warps = CalcNumWarps(preserved_size_ >> 5);
    // 5. check whether NHWC layout can be successfully applied
    if (num_warps < 0) {
      return;
    }
    applied_layout_ = BroadcastLayout::NHWCLayout;
    VLOG(4) << "TileBroadcastTactic::Init: matched NHWCLayout\n";
  }
  // Now we can apply this tactic
  ir::Expr module_root = sch->GetModule().GetExprs().front();
  ir::Expr root_block = ir::analyzer::GetRootSBlock(module_root);
  auto* root_node = root_block.As<ir::ScheduleBlockRealize>()
                        ->schedule_block.As<ir::ScheduleBlock>();
  root_node->attrs[kTileMethod] = TacticName();
}

void TileBroadcastTactic::InitBroadcastAxisInfo(ir::IRSchedule* sch) {
  broadcast_axis_ = GetCommonBroadcastAxis(sch);

  int data_rank = context_->config.base_info->loop_ranges.size();
  is_broadcast_axis_.assign(data_rank, false);
  for (int axis : broadcast_axis_) {
    is_broadcast_axis_[axis] = true;
  }

  int low_bc_axis_offset = 0;
  for (int i = broadcast_axis_.size() - 1; i > 0; --i) {
    if (broadcast_axis_[i - 1] != broadcast_axis_[i] - 1) {
      low_bc_axis_offset = i;
      break;
    }
  }
  high_broadcast_axis_.assign(broadcast_axis_.begin(),
                              broadcast_axis_.begin() + low_bc_axis_offset);
  low_broadcast_axis_.assign(broadcast_axis_.begin() + low_bc_axis_offset,
                             broadcast_axis_.end());

  preserved_axis_.clear();
  for (int axis = 0; axis < data_rank; ++axis) {
    if (!is_broadcast_axis_[axis]) {
      preserved_axis_.push_back(axis);
    }
  }
}

void TileBroadcastTactic::InitBroadcastSizeInfo() {
  auto& loop_ranges = context_->config.base_info->loop_ranges;

  const auto MulDimSize = [](int64_t a, int64_t b) {
    return (a == -1 || b == -1) ? -1 : a * b;
  };

  broadcast_size_ = 1;
  for (int axis : broadcast_axis_) {
    broadcast_size_ = MulDimSize(broadcast_size_, loop_ranges[axis]);
  }

  low_broadcast_size_ = 1;
  for (int axis : low_broadcast_axis_) {
    low_broadcast_size_ = MulDimSize(low_broadcast_size_, loop_ranges[axis]);
  }

  preserved_size_ = 1;
  for (int axis : preserved_axis_) {
    preserved_size_ = MulDimSize(preserved_size_, loop_ranges[axis]);
  }
}

bool TileBroadcastTactic::CanEnableVectorize(const int block_size) {
  if (!context_->config.base_info->can_apply_vectorize) return false;
  const int vectorize_factor =
      static_cast<int>(context_->config.tile_config.vectorize_factor);
  const int block_element_nums = block_size * vectorize_factor;
  if (applied_layout_ == BroadcastLayout::NCHWLayout) {
    if (low_broadcast_size_ <= block_element_nums &&
        low_broadcast_size_ % vectorize_factor == 0) {
      return true;
    } else if (low_broadcast_size_ > block_element_nums &&
               low_broadcast_size_ % block_element_nums == 0) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> TileBroadcastTactic::TileNCHW(
    ir::IRSchedule* sch, const std::string& block_id, int block_size) {
  // To achieve best performance, we apply different tiling templates based on
  // low_broadcast_size. The key is which axis to allocate inner loop:
  // 1. For small size:
  //        [B, P, B<=256]
  //     => [(blockY, loop), blockX, threadX].
  // 2. For medium size:
  //        [B, P, 256<B<=2048],
  //     => [blockY, blockX, (loop, threadX)].
  // 3. For large size:
  //        [B, P, B>2048]
  //     => [blockX', blockY, (blockX, loop, threadX)].
  VLOG(4) << "TileBroadcastTactic using original NCHW layout\n";
  if (low_broadcast_size_ <= 256) {
    sch->Split(block_id, 0, {-1, 4});
    return {"blockIdx.y", "", "blockIdx.x", "threadIdx.x"};
  } else if (low_broadcast_size_ <= 2048) {
    sch->Split(block_id, 2, {-1, block_size});
    return {"blockIdx.y", "blockIdx.x", "", "threadIdx.x"};
  } else {
    sch->Reorder(block_id, {1, 0});
    sch->Fuse(block_id, {1, 2});
    sch->Split(block_id, 1, {-1, 4, block_size});
    return {"blockIdx.y", "blockIdx.x", "", "threadIdx.x"};
  }
}

std::vector<std::string> TileBroadcastTactic::TileNHWC(
    ir::IRSchedule* sch, const std::string& block_id, int block_size) {
  // NHWC layout will have 2 fused loops, so we start with (blockIdx.x,
  // threadIdx.x)
  VLOG(4) << "TileBroadcastTactic using NHWC layout, block size: " << block_size
          << "\n";
  if (broadcast_size_ <= 64) {
    /**
     * if the broadcast size is smaller than 64
     * this means we need more blocks to increase the occupancy
     * so no thread coarsening anyway
     */
    sch->Split(block_id, 1, {-1, block_size});
    sch->Fuse(block_id, {0, 1});
    return {"blockIdx.x", "threadIdx.x"};
  } else {
    if (preserved_size_ == block_size) {
      // block size is enough to cover
      sch->Split(block_id, 1, {-1, block_size});
      sch->Fuse(block_id, {0, 1});
      sch->Split(block_id, 0, {-1, 4});
      return {"blockIdx.x", "", "threadIdx.x"};
    } else if (preserved_size_ < block_size) {
      // block size is larger (deliberately, to have enough threads)
      // than preserved size
      sch->Fuse(block_id, {0, 1});  // single fused loop
      sch->Split(
          block_id, 0, {-1, 4, block_size / preserved_size_, preserved_size_});
      return {"blockIdx.x", "", "threadIdx.y", "threadIdx.x"};
    } else {
      /**
       * block size is not enough to cover the preserved size
       * make the load index invariant to inner loop
       * (-1, 4, p_size / block_size, block_size)
       */
      sch->Split(block_id, 1, {-1, block_size});
      sch->Fuse(block_id, {0, 1});
      sch->Split(block_id, 0, {-1, 4, preserved_size_ / block_size});
      return {"blockIdx.x", "", "blockIdx.y", "threadIdx.x"};
    }
  }
}

std::vector<std::string> TileBroadcastTactic::TileVectorizeNCHW(
    ir::IRSchedule* sch,
    const std::string& block_id,
    const int block_size,
    const int vectorize_factor) {
  /**
   * block_size = 256, vectorize_loop factor = 4
   * block_element_nums = block_size * vectorize_factor = 1024
   * 1. For small size:
   *        [B, P, B<=block_element_nums]
   *     => [blockY, blockX, (threadX, vectorize_loop)].
   * 2. For medium size:
   *        [B, P, block_element_nums<B<=2048],
   *     => [blockY, blockX, (threadX, vectorize_loop)].
   * 2. For large size:
   *        [B, P, 2048<B],
   *     => [blockX', blockY, (blockX, threadX, vectorize_loop)].
   */
  const int block_element_nums = block_size * vectorize_factor;
  VLOG(4) << "TileBroadcastTactic using original NCHW layout, "
             "low_broadcast_size_ = "
          << low_broadcast_size_;
  if (low_broadcast_size_ <= block_element_nums) {
    sch->Split(block_id, 2, {-1, vectorize_factor});
    return {"blockIdx.y", "blockIdx.x", "threadIdx.x", ""};
  } else if (low_broadcast_size_ <= 2048) {
    sch->Split(block_id, 2, {-1, block_size, vectorize_factor});
    sch->Fuse(block_id, {1, 2});
    return {"blockIdx.y", "blockIdx.x", "threadIdx.x", ""};
  } else {
    sch->Reorder(block_id, {1, 0});
    sch->Fuse(block_id, {1, 2});
    sch->Split(block_id, 1, {-1, block_size, vectorize_factor});
    return {"blockIdx.y", "blockIdx.x", "threadIdx.x", ""};
  }
}

void TileBroadcastTactic::Apply(ir::IRSchedule* sch,
                                const std::string& block_id) {
  if (applied_layout_ == BroadcastLayout::Invalid) return;
  int block_size = 256;

  if (CanEnableVectorize(block_size)) {
    ApplyVectorize(sch, block_id, block_size);
    return;
  }
  // check the number of warps here, if not a applicable
  // preserved_size, func will return later
  if (applied_layout_ == BroadcastLayout::NHWCLayout) {
    block_size = CalcNumWarps(preserved_size_ >> 5);
    if (block_size == -1) {
      applied_layout_ = BroadcastLayout::Invalid;
      return;
    }
    block_size = std::clamp(block_size << 5, 128, 1024);
  }

  // Cluster and fuse axis of the same type to get exactly 3 loops.
  // [B, P, B, P, ..., B, B] => [B, P, B]
  FuseAxisGroups(sch, block_id);

  // Do tiling.
  std::vector<std::string> axis_bind;
  if (applied_layout_ == BroadcastLayout::NCHWLayout) {
    axis_bind = TileNCHW(sch, block_id, block_size);
  } else {
    axis_bind = TileNHWC(sch, block_id, block_size);
  }

  // Do binding.
  auto loops = sch->GetLoops(block_id);
  for (int i = 0; i < axis_bind.size(); ++i) {
    if (!axis_bind[i].empty()) {
      sch->Bind(loops[i], axis_bind[i]);
    }
  }

  // Set to use local buffer if this schedule block is not output.
  if (context_->output_names.count(block_id) == 0) {
    auto block = sch->GetBlock(block_id);
    sch->SetBuffer(block, "local");
  }

  VLOG(4) << "After TileBroadcast on block: [" << block_id << "]:\n"
          << sch->GetLoops(block_id)[0];
}

void TileBroadcastTactic::FuseAxisGroups(ir::IRSchedule* sch,
                                         const std::string& block_id) {
  // Reorder high-dim axis to cluster axis of the same type.
  // [B, P, B, P, ..., B, B] => [B, B, ..., P, P, ..., B, B]
  // Fuse continuous axis of the same type.
  // [B, B, ..., P, P, ..., B, B] => [B, P, B] (for NCHW layout)
  // [B, B, P, B, ..., P] => [B, P] (for NHWC layout)
  if (applied_layout_ == BroadcastLayout::Invalid) return;
  const auto FuseRange = [&](int start, int count) {
    if (count > 1) {
      std::vector<int> loops_index(count);
      std::iota(loops_index.begin(), loops_index.end(), start);
      sch->Fuse(block_id, loops_index);
    }
  };
  std::vector<int> axis_perm;
  int high_axis_num = 0;
  int low_axis_num = 0;
  int mid_axis_num = preserved_axis_.size();
  if (applied_layout_ == BroadcastLayout::NCHWLayout) {
    axis_perm = high_broadcast_axis_;
    high_axis_num = high_broadcast_axis_.size();
    low_axis_num = low_broadcast_axis_.size();
  } else {
    axis_perm = broadcast_axis_;
    high_axis_num = broadcast_axis_.size();
  }
  axis_perm.insert(
      axis_perm.end(), preserved_axis_.begin(), preserved_axis_.end());
  sch->Reorder(block_id, axis_perm);

  FuseRange(high_axis_num + mid_axis_num, low_axis_num);
  FuseRange(high_axis_num, mid_axis_num);
  FuseRange(0, high_axis_num);
}

/*
Broadcast layout support vectorize situation:
To apply vectorize, we should consider threads and
vectorize factor  dimension is continue, we need to meet
the following conditions:
1. can_apply_vectorize must be true
2. In NCHW layout situation, list of 3 groups of axis,
as the following graph shows:
  high_broadcast_axis
      v     v
      [B, P, B, P, ..., B, B]
          ^     ^       ^  ^
          |     |       low_broadcast_axis
      preserved_axis

case 1: low_broadcast_axis less than 1024,
according to low_broadcast_axis be divided by 32 in Init function,
so low_broadcast_axis must be divisible by vectorize_factor.
block_size = low_broadcast_axis / vectorize_factor.

case 2: low_broadcast_axis greater than 1024
block_size is set to 256, so low_broadcast_axis must
be divisible by vectorize_factor * block_size.

so, after dealing with TileBroadcastTactic threads,
threads and vectorize dimension is continuous.

3. In NHWC layout situation，current not support vectorize.
*/

void TileBroadcastTactic::ApplyVectorize(ir::IRSchedule* sch,
                                         const std::string& block_id,
                                         const int block_size) {
  const auto vectorize_factor =
      static_cast<int>(context_->config.tile_config.vectorize_factor);

  FuseAxisGroups(sch, block_id);

  const auto ApplyVectorization = [&](const std::string& block_id, int factor) {
    auto loops = sch->GetLoops(block_id);
    auto vectorize_axis = loops.size() - 1;
    sch->Vectorize(loops[vectorize_axis], factor);
  };

  std::vector<std::string> axis_bind;
  if (applied_layout_ == BroadcastLayout::NCHWLayout) {
    // [B, P, B] (for NCHW layout)
    axis_bind = TileVectorizeNCHW(sch, block_id, block_size, vectorize_factor);
  } else {
    // [B, P] (for NHWC layout)
    // TODO(baoqiwen): Need support TileVectorizeNHWC
  }

  // set vectorize schedule primitives
  ApplyVectorization(block_id, vectorize_factor);

  // Do binding.
  auto loops = sch->GetLoops(block_id);
  for (int i = 0; i < axis_bind.size(); ++i) {
    if (!axis_bind[i].empty()) {
      sch->Bind(loops[i], axis_bind[i]);
    }
  }

  // Set to use local buffer if this schedule block is not output.
  if (context_->output_names.count(block_id) == 0) {
    auto block = sch->GetBlock(block_id);
    sch->SetBuffer(block, "local");
  }

  VLOG(4) << "After TileBroadcast with Vectorize on block: [" << block_id
          << "]:\n"
          << sch->GetLoops(block_id)[0];
}

}  // namespace

std::unique_ptr<ScheduleTactic> CreateTileBroadcastTactic() {
  return std::make_unique<TileBroadcastTactic>();
}

}  // namespace ir
}  // namespace cinn
