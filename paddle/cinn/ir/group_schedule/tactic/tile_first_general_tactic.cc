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

#include "paddle/cinn/ir/group_schedule/tactic/tile_first_general_tactic.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"

PD_DECLARE_bool(support_reduce_stride_read);
PD_DECLARE_bool(support_trivial_stride_read);

namespace cinn {
namespace ir {

bool IsInnerThreadSpatialLoopGT(const ScheduleConfig& config, int num) {
  return config.tile_config.spatial_inner_num > num;
}

bool IsReduceBlock(const ScheduleConfig& config, const std::string& block_id) {
  return config.base_info->reduce_tensor_names.count(block_id) > 0;
}

bool HasReduceAxis(const ScheduleConfig& config) {
  return config.base_info->reduce_axis.size() > 0;
}

bool IsWarpReduce(const ScheduleConfig& config) {
  const auto& MatchWarpReduce = cinn::adt::match{
      [&](const ir::NoneReduceMethod&) { return false; },
      [&](const ir::WarpReduceMethod&) { return true; },
      [&](const ir::BlockReduceMethod&) { return false; },
  };
  return std::visit(MatchWarpReduce, config.tile_config.reduce_method);
}

class TileFirstGeneralTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "TileFirstGeneralTactic"; }

 private:
  void MergeFlattenAxis(ir::IRSchedule* sch, const std::string& block_id);
  void MergeReduceAxis(ir::IRSchedule* sch, const std::string& block_id);
  void SplitSptialInner(ir::IRSchedule* sch, const std::string& block_id);
  void SplitReduceInner(ir::IRSchedule* sch, const std::string& block_id);
  void ReorderFlattenInnerWithReduceAxis(ir::IRSchedule* sch,
                                         const std::string& block_id);
  void SplitWarpNumber(ir::IRSchedule* sch, const std::string& block_id);
  void Unroll(ir::IRSchedule* sch, const std::string& block_id);
  void VariableTypeAssignment(ir::IRSchedule* sch, const std::string& block_id);
  void SetReduceType(ir::IRSchedule* sch, const std::string& block_id);
  void BindCudaInfo(ir::IRSchedule* sch, const std::string& block_id);

 private:
  ScheduleContext* context_;
  std::vector<int32_t> vec_flatten_axis_;
  std::vector<int32_t> vec_reduce_axis_;
  int reduce_current_axis_{0};
};

void TileFirstGeneralTactic::Init(ScheduleContext* context) {
  context_ = context;
  reduce_current_axis_ =
      IsInnerThreadSpatialLoopGT(context_->config, 1) ? 2 : 1;
  if (context_->config.base_info->is_reduce_all) {
    reduce_current_axis_ = 0;
  }
  // reduce axis have be re-order to last
  vec_flatten_axis_.clear();
  vec_reduce_axis_.clear();
  int32_t reduce_start_idx = context_->config.base_info->data_rank -
                             context_->config.base_info->reduce_axis.size();
  for (int32_t i = 0; i < context_->config.base_info->data_rank; ++i) {
    if (i >= reduce_start_idx) {
      vec_reduce_axis_.push_back(i);
    } else {
      vec_flatten_axis_.push_back(i);
    }
  }
}

void TileFirstGeneralTactic::Apply(ir::IRSchedule* sch,
                                   const std::string& block_id) {
  if (ir::IsReduceInitTensorName(block_id)) return;
  MergeReduceAxis(sch, block_id);
  VLOG(6) << "After MergeReduceAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  MergeFlattenAxis(sch, block_id);
  VLOG(6) << "After MergeFlattenAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SplitSptialInner(sch, block_id);
  VLOG(6) << "After SplitSptialInner on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SplitReduceInner(sch, block_id);
  VLOG(6) << "After SplitReduceInner on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  ReorderFlattenInnerWithReduceAxis(sch, block_id);
  VLOG(6) << "After ReorderFlattenInnerWithReduceAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SplitWarpNumber(sch, block_id);
  VLOG(6) << "After SplitWarpNumber on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  BindCudaInfo(sch, block_id);
  VLOG(6) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  VariableTypeAssignment(sch, block_id);
  VLOG(6) << "After VariableTypeAssignment on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  Unroll(sch, block_id);
  VLOG(6) << "After Unroll on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SetReduceType(sch, block_id);
}

void TileFirstGeneralTactic::MergeFlattenAxis(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  if (vec_flatten_axis_.size() >= 2) {
    sch->Fuse(block_id, vec_flatten_axis_);
  }
}

void TileFirstGeneralTactic::MergeReduceAxis(ir::IRSchedule* sch,
                                             const std::string& block_id) {
  if (vec_reduce_axis_.size() >= 2 && !ir::IsReduceInitTensorName(block_id)) {
    sch->Fuse(block_id, vec_reduce_axis_);
  }
}

void TileFirstGeneralTactic::SplitSptialInner(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  if (IsInnerThreadSpatialLoopGT(context_->config, 1)) {
    if (FLAGS_support_trivial_stride_read) {
      auto loops = sch->GetLoops(block_id);
      std::vector<int> split_factors{
          static_cast<int>(context_->config.tile_config.spatial_inner_num), -1};
      sch->Split(loops[0], split_factors);
      loops = sch->GetLoops(block_id);
      sch->Reorder({loops[1], loops[0]});
    } else {
      auto loops = sch->GetLoops(block_id);
      auto split_loops = sch->Split(
          loops[0],
          std::vector<int>(
              {-1,
               static_cast<int>(
                   context_->config.tile_config.spatial_inner_num)}));
    }
  }
}

void TileFirstGeneralTactic::SplitReduceInner(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  if (!HasReduceAxis(context_->config)) return;

  auto loops = sch->GetLoops(block_id);
  auto reduce_loop = loops[reduce_current_axis_].As<ir::For>();

  if (FLAGS_support_reduce_stride_read) {
    if (context_->config.base_info->reduce_numel <= 256) {
      std::vector<int> split_factors{
          -1, static_cast<int>(context_->config.tile_config.tree_reduce_num)};
      sch->Split(loops[reduce_current_axis_], split_factors);
      loops = sch->GetLoops(block_id);
      sch->Reorder(
          {loops[reduce_current_axis_ + 1], loops[reduce_current_axis_]});
    } else {
      // split warp num first
      std::vector<int> split_factors{
          static_cast<int>(context_->config.tile_config.warp_num), -1, 32};
      sch->Split(loops[reduce_current_axis_], split_factors);
      loops = sch->GetLoops(block_id);
      sch->Reorder(
          {loops[reduce_current_axis_ + 2], loops[reduce_current_axis_ + 1]});
      loops = sch->GetLoops(block_id);
      sch->Fuse({loops[reduce_current_axis_], loops[reduce_current_axis_ + 1]});
    }
  } else {
    std::vector<int> split_factors{
        static_cast<int>(context_->config.tile_config.tree_reduce_num), -1};
    sch->Split(loops[reduce_current_axis_], split_factors);
  }
  loops = sch->GetLoops(block_id);
  if (IsReduceBlock(context_->config, block_id)) {
    sch->FactorizeReduction(loops[reduce_current_axis_],
                            0,
                            /* with_write_back_block_init = */ false);
  }
}

void TileFirstGeneralTactic::ReorderFlattenInnerWithReduceAxis(
    ir::IRSchedule* sch, const std::string& block_id) {
  // re-order flatten inner num with last dim
  auto loops = sch->GetLoops(block_id);
  if (IsInnerThreadSpatialLoopGT(context_->config, 1) &&
      HasReduceAxis(context_->config)) {
    sch->Reorder({loops[2], loops[1]});
    if (IsReduceBlock(context_->config, block_id) &&
        sch->HasBlock(block_id + "_rf")) {
      loops = sch->GetLoops(block_id + "_rf");
      sch->Reorder({loops[2], loops[1]});
    }
  }
}

void TileFirstGeneralTactic::SplitWarpNumber(ir::IRSchedule* sch,
                                             const std::string& block_id) {
  const auto IsWarpNumGT = [&](int64_t num) {
    return context_->config.tile_config.warp_num > num;
  };
  if (!IsWarpNumGT(1)) return;

  const auto GetMinimalWarpNum = [&](const ir::Expr& loop,
                                     const ScheduleConfig& config) -> int {
    ir::Expr extent = loop.As<ir::For>()->extent;
    common::cas_intervals_t var_intervals =
        common::CollectVarIntervalsOfExprs({extent});
    common::SymbolicExprAnalyzer analyzer(var_intervals);
    const auto& proved_gt =
        analyzer.ProveGT(ir::Expr(config.tile_config.warp_num * 32), extent);
    if (proved_gt.value_or(false)) {
      ir::Expr upper_bound = analyzer.UpperBound(extent);
      if (upper_bound.is_constant()) {
        return (static_cast<int>(upper_bound.get_constant()) + 31) / 32;
      }
    }
    return config.tile_config.warp_num;
  };

  auto loops = sch->GetLoops(block_id);
  if (!HasReduceAxis(context_->config)) {
    if (context_->config.tile_config.warp_num ==
        -1) {  // only in bucket spatial_numel <= 1024
      sch->Split(loops[0], std::vector<int>({1, -1}));
    } else {
      sch->Split(
          loops[0],
          std::vector<int>(
              {-1,
               static_cast<int>(context_->config.tile_config.warp_num * 32)}));
    }
  } else if (IsWarpReduce(context_->config)) {
    // get num warp from flatten num
    int minimal_warp_number = GetMinimalWarpNum(loops[0], context_->config);
    int thread_y =
        minimal_warp_number * 32 / context_->config.tile_config.tree_reduce_num;
    sch->Split(loops[0], std::vector<int>({-1, thread_y}));

    if (IsReduceBlock(context_->config, block_id) &&
        sch->HasBlock(block_id + "_rf")) {
      auto loops = sch->GetLoops(block_id + "_rf");
      sch->Split(loops[0], std::vector<int>({-1, thread_y}));
    }
  } else {
    return;
  }
}

void TileFirstGeneralTactic::Unroll(ir::IRSchedule* sch,
                                    const std::string& block_id) {
  std::vector<size_t> unroll_loops_idx = [&] {
    if (IsWarpReduce(context_->config)) {
      return std::vector<size_t>{3, 4};
    } else {
      return std::vector<size_t>{2, 3};
    }
  }();

  const auto DoUnroll = [&](const std::vector<ir::Expr>& loops) {
    for (size_t loop_idx : unroll_loops_idx) {
      if (loops.size() > loop_idx &&
          loops[loop_idx].As<ir::For>()->extent.is_constant()) {
        sch->Unroll(loops[loop_idx]);
      }
    }
  };

  DoUnroll(sch->GetLoops(block_id));
  if (IsReduceBlock(context_->config, block_id) &&
      sch->HasBlock(block_id + "_rf")) {
    DoUnroll(sch->GetLoops(block_id + "_rf"));
  }
}

void TileFirstGeneralTactic::VariableTypeAssignment(
    ir::IRSchedule* sch, const std::string& block_id) {
  const auto IsOutputTensor = [&](const std::string& tensor_name) -> bool {
    return context_->config.base_info->direct_output_var_names.count(
               tensor_name) > 0;
  };
  const auto HasConsumers = [&](const ir::Expr& block) -> bool {
    return !ir::analyzer::GetConsumerSBlocks(block, sch->GetRootBlock(block))
                .empty();
  };

  auto block = sch->GetBlock(block_id);
  if (!IsOutputTensor(block_id) && HasConsumers(block)) {
    sch->SetBuffer(block, "local", false);
  }

  if (IsReduceBlock(context_->config, block_id) &&
      sch->HasBlock(block_id + "_rf")) {
    auto block = sch->GetBlock(block_id + "_rf");
    sch->SetBuffer(block, "local", false);
  }
}

void TileFirstGeneralTactic::SetReduceType(ir::IRSchedule* sch,
                                           const std::string& block_id) {
  if (IsReduceBlock(context_->config, block_id)) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = context_->config.tile_config.reduce_method;
  }
}

void TileFirstGeneralTactic::BindCudaInfo(ir::IRSchedule* sch,
                                          const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);
  if (loops.size() == 1 || context_->config.base_info->is_reduce_all) {
    sch->Split(loops[0], std::vector<int>({1, -1}));
  }

  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    sch->Bind(loops[0], "blockIdx.x");
    if (IsWarpReduce(context_->config)) {
      sch->Bind(loops[1], "threadIdx.y");
      sch->Bind(loops[2], "threadIdx.x");
    } else {
      sch->Bind(loops[1], "threadIdx.x");
    }
  };

  DoBind(sch->GetLoops(block_id));

  if (IsReduceBlock(context_->config, block_id) &&
      sch->HasBlock(block_id + "_rf")) {
    auto loops = sch->GetLoops(block_id + "_rf");
    if (context_->config.base_info->is_reduce_all) {
      sch->Split(loops[0], std::vector<int>({1, -1}));
    }
    DoBind(sch->GetLoops(block_id + "_rf"));
  }
}

std::unique_ptr<ScheduleTactic> CreateTileFirstGeneralTactic() {
  return std::make_unique<TileFirstGeneralTactic>();
}

}  // namespace ir
}  // namespace cinn
