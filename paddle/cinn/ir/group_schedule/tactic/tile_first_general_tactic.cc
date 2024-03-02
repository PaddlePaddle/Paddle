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
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace ir {

void TileFirstGeneralTactic::Init(ScheduleContext* context) {
  context_ = context;
  reduce_current_axis_ =
      context_->group_tile_info->flatten_inner_num > 1 ? 2 : 1;
  auto vec_axis = context_->group_tile_info->reduce_axis_;
  // reduce axis have be re-order to last
  vec_flatten_axis_.clear();
  vec_reduce_axis_.clear();
  int32_t reduce_start_idx =
      context_->group_tile_info->data_rank - vec_axis.size();
  for (int32_t i = 0; i < context_->group_tile_info->data_rank; ++i) {
    if (i >= reduce_start_idx) {
      vec_reduce_axis_.push_back(i);
    } else {
      vec_flatten_axis_.push_back(i);
    }
  }
}

void TileFirstGeneralTactic::Apply(ir::IRSchedule* sch,
                                   const std::string& block_id) {
  MergeFlattenAxis(sch, block_id);
  MergeReduceAxis(sch, block_id);
  SplitFlattenInner(sch, block_id);
  SplitReduceInner(sch, block_id);
  ReorderFlattenInnerWithReduceAxis(sch, block_id);
  SplitWarpNumber(sch, block_id);
  BindCudaInfo(sch, block_id);
  VariableTypeAssignment(sch, block_id);
  Unroll(sch, block_id);
  SetReduceType(sch, block_id);

  if (sch->HasBlock(block_id)) {
    std::cerr << "process:  " << block_id << "\n"
              << sch->GetBlock(block_id) << std::endl;
  }
}

void TileFirstGeneralTactic::MergeFlattenAxis(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  if (vec_flatten_axis_.size() >= 2 && !ir::IsReduceInitTensorName(block_id)) {
    sch->Fuse(block_id, vec_flatten_axis_);
  }
}

void TileFirstGeneralTactic::MergeReduceAxis(ir::IRSchedule* sch,
                                             const std::string& block_id) {
  // should down reduce axis
  std::vector<int32_t> fuse_axis = vec_reduce_axis_;
  if (vec_reduce_axis_.size() >= 2) {
    for (size_t i = 0; i < fuse_axis.size(); ++i) {
      fuse_axis[i] -= (vec_flatten_axis_.size() - 1);
    }
  }
  if (vec_reduce_axis_.size() >= 2 && !ir::IsReduceInitTensorName(block_id)) {
    sch->Fuse(block_id, fuse_axis);
  }
}

void TileFirstGeneralTactic::SplitFlattenInner(ir::IRSchedule* sch,
                                               const std::string& block_id) {
  if (context_->group_tile_info->flatten_inner_num > 1 &&
      !ir::IsReduceInitTensorName(block_id)) {
    // split flatten inner
    auto loops = sch->GetLoops(block_id);
    auto split_loops = sch->Split(
        loops[0],
        std::vector<int>({-1, context_->group_tile_info->flatten_inner_num}));
  }
}

void TileFirstGeneralTactic::SplitReduceInner(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  if (context_->group_tile_info->reduce_inner_num > 1 &&
      !ir::IsReduceInitTensorName(block_id)) {
    auto loops = sch->GetLoops(block_id);
    auto split_expr = loops[reduce_current_axis_].As<ir::For>();

    if (split_expr->extent.as_int64() == 1) {
      return;
    }

    std::vector<int> split_factors;
    if (context_->group_tile_info->reduce_block >= 2048) {
      split_factors.emplace_back(
          std::ceil(context_->group_tile_info->reduce_numel * 1.0 /
                    context_->group_tile_info->reduce_inner_num));
      split_factors.emplace_back(context_->group_tile_info->reduce_inner_num);
    } else {
      split_factors.emplace_back(
          std::ceil(context_->group_tile_info->reduce_block * 1.0 /
                    context_->group_tile_info->reduce_inner_num));
      split_factors.emplace_back(context_->group_tile_info->reduce_inner_num);
    }

    auto split_loops = sch->Split(loops[reduce_current_axis_], split_factors);

    if (context_->group_tile_info->reduce_var_names.count(block_id)) {
      sch->FactorizeReduction(split_loops[0], 0);
    }
  }
}

void TileFirstGeneralTactic::ReorderFlattenInnerWithReduceAxis(
    ir::IRSchedule* sch, const std::string& block_id) {
  // re-order flatten inner num with last dim
  if (context_->group_tile_info->flatten_inner_num > 1 &&
      context_->group_tile_info->reduce_axis_.size() > 0 &&
      !ir::IsReduceInitTensorName(block_id)) {
    auto loops = sch->GetLoops(block_id);
    sch->Reorder({loops[2], loops[1]});
    if (context_->group_tile_info->reduce_var_names.count(block_id)) {
      auto loops = sch->GetLoops(block_id + "_rf");
      sch->Reorder({loops[2], loops[1]});
    }
  }
}

void TileFirstGeneralTactic::SplitWarpNumber(ir::IRSchedule* sch,
                                             const std::string& block_id) {
  if (context_->group_tile_info->warp_num > 1 &&
      !ir::IsReduceInitTensorName(block_id)) {
    if (context_->group_tile_info->reduce_axis_.size() == 0) {
      // get num warp from flatten num
      auto loops = sch->GetLoops(block_id);
      sch->Split(loops[0],
                 std::vector<int>({context_->group_tile_info->block_num,
                                   context_->group_tile_info->warp_num * 32}));
    } else if (context_->group_tile_info->flatten_inner_num > 1) {
      // get num warp from flatten num
      auto loops = sch->GetLoops(block_id);
      sch->Split(loops[0],
                 std::vector<int>({-1, context_->group_tile_info->warp_num}));

      loops = sch->GetLoops(block_id);
      sch->Fuse({loops[1], loops[2]});

      if (context_->group_tile_info->reduce_var_names.count(block_id)) {
        auto loops = sch->GetLoops(block_id + "_rf");
        sch->Split(loops[0],
                   std::vector<int>({-1, context_->group_tile_info->warp_num}));

        loops = sch->GetLoops(block_id + "_rf");
        sch->Fuse({loops[1], loops[2]});
      }
    }
  }
}

void TileFirstGeneralTactic::Unroll(ir::IRSchedule* sch,
                                    const std::string& block_id) {
  // set unroll
  if (ir::IsReduceInitTensorName(block_id)) {
    return;
  }
  auto loops = sch->GetLoops(block_id);
  if (loops.size() > 2) {
    sch->Unroll(loops[2]);
  }
  if (loops.size() > 3) {
    sch->Unroll(loops[3]);
  }

  if (context_->group_tile_info->reduce_var_names.count(block_id)) {
    auto loops = sch->GetLoops(block_id + "_rf");
    if (loops.size() > 2) {
      sch->Unroll(loops[2]);
    }
    if (loops.size() > 3) {
      sch->Unroll(loops[3]);
    }
  }
}

void TileFirstGeneralTactic::VariableTypeAssignment(
    ir::IRSchedule* sch, const std::string& block_id) {
  if (ir::IsReduceInitTensorName(block_id)) {
    return;
  }

  auto block = sch->GetBlock(block_id);
  {
    if (!context_->group_tile_info->direct_output_var_names.count(block_id)) {
      sch->SetBuffer(block, "local", false);
    }
  }

  if (context_->group_tile_info->reduce_var_names.count(block_id)) {
    auto block = sch->GetBlock(block_id + "_rf");
    sch->SetBuffer(block, "local", false);
  }
}

void TileFirstGeneralTactic::SetReduceType(ir::IRSchedule* sch,
                                           const std::string& block_id) {
  if (ir::IsReduceInitTensorName(block_id)) {
    return;
  }

  if (context_->group_tile_info->reduce_var_names.count(block_id)) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    if (context_->group_tile_info->reduce_type == 0) {
      block->reduce_type = 0;
    }
  }
}

void TileFirstGeneralTactic::BindCudaInfo(ir::IRSchedule* sch,
                                          const std::string& block_id) {
  // bind cuda block and thread info
  if (ir::IsReduceInitTensorName(block_id)) {
    return;
  }
  auto loops = sch->GetLoops(block_id);
  if (loops.size() == 1) {
    sch->Split(loops[0], std::vector<int>({1, -1}));
  }

  loops = sch->GetLoops(block_id);
  sch->Bind(loops[0], "blockIdx.x");
  sch->Bind(loops[1], "threadIdx.x");

  if (context_->group_tile_info->reduce_var_names.count(block_id)) {
    auto loops = sch->GetLoops(block_id + "_rf");
    sch->Bind(loops[0], "blockIdx.x");
    sch->Bind(loops[1], "threadIdx.x");
  }
}

}  // namespace ir
}  // namespace cinn
