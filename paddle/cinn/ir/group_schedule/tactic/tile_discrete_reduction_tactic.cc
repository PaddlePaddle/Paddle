// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/tactic/tile_discrete_reduction_tactic.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"

namespace cinn {
namespace ir {

using cinn::ir::analyzer::IsReductionSBlock;

bool UseDiscreteDataTile(const ScheduleConfig& config) {
  // use discrete data tile for [...RS]
  auto& iter_spaces = config.base_info->iter_space_type;
  size_t size = iter_spaces.size();
  return size >= 2 && iter_spaces[size - 2].first == "R" &&
         iter_spaces[size - 1].first == "S";
}

class TileDiscreteReductionTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context, ir::IRSchedule* sch) override;
  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;
  std::string TacticName() const override {
    return "TileDiscreteReductionTactic";
  }

 private:
  void MergeDiscreteFlattenAxis(ir::IRSchedule* sch,
                                const std::string& block_id);
  void MergeReduceAxis(ir::IRSchedule* sch, const std::string& block_id);
  void SplitSptialInner(ir::IRSchedule* sch, const std::string& block_id);
  void SplitReduceInner(ir::IRSchedule* sch, const std::string& block_id);
  void SetBufferType(ir::IRSchedule* sch, const std::string& block_id);
  void SetReduceType(ir::IRSchedule* sch, const std::string& block_id);
  void BindCudaInfo(ir::IRSchedule* sch, const std::string& block_id);

 private:
  ScheduleContext* context_;
  bool can_apply_;
  std::vector<int32_t> vec_spatial_axis_first_;
  std::vector<int32_t> vec_spatial_axis_last_;
  std::vector<int32_t> vec_reduce_axis_;
  std::unordered_map<std::string, std::string> map_rf_block_;
  std::unordered_map<std::string, std::string> map_global_rf_block_;
};

void TileDiscreteReductionTactic::Init(ScheduleContext* context,
                                       ir::IRSchedule* sch) {
  context_ = context;
  can_apply_ = false;

  // Check whether this group has been tiled by previous tactic.
  ir::Expr module_root = sch->GetModule().GetExprs().front();
  ir::Expr root_block = ir::analyzer::GetRootSBlock(module_root);
  auto* root_node = root_block.As<ir::ScheduleBlockRealize>()
                        ->schedule_block.As<ir::ScheduleBlock>();
  if (root_node->attrs.count(kTileMethod) > 0) {
    return;
  }
  if (!UseDiscreteDataTile(context_->config)) {
    return;
  }
  can_apply_ = true;
  root_node->attrs[kTileMethod] = TacticName();

  // reduce axes have been re-ordered to the last
  vec_reduce_axis_.clear();
  int data_rank = context_->config.base_info->loop_ranges.size();
  int32_t reduce_start_idx =
      data_rank - context_->config.base_info->reduce_axis.size();
  for (int32_t i = 0; i < data_rank; ++i) {
    if (i >= reduce_start_idx) {
      vec_reduce_axis_.push_back(i);
    }
  }
  vec_spatial_axis_first_.clear();
  vec_spatial_axis_last_.clear();

  if (!context_->config.base_info->reduce_axis.empty()) {
    int64_t first_reduce_axis = context_->config.base_info->reduce_axis.front();
    for (auto axis : context_->config.base_info->reduce_axis) {
      if (context->config.base_info->loop_strides[axis] >
          context->config.base_info->loop_strides[first_reduce_axis]) {
        first_reduce_axis = axis;
      }
    }
    for (int32_t i = 0; i < reduce_start_idx; ++i) {
      if (i < first_reduce_axis) {
        vec_spatial_axis_first_.push_back(i);
      } else {
        vec_spatial_axis_last_.push_back(i);
      }
    }
  }

  map_rf_block_.clear();
  map_global_rf_block_.clear();
}

void TileDiscreteReductionTactic::Apply(ir::IRSchedule* sch,
                                        const std::string& block_id) {
  if (!can_apply_) return;
  if (ir::IsReduceInitTensorName(block_id)) return;

  MergeReduceAxis(sch, block_id);
  VLOG(6) << "After MergeReduceAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  MergeDiscreteFlattenAxis(sch, block_id);
  VLOG(6) << "After MergeDiscreteFlattenAxis on block: [" << block_id
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
  BindCudaInfo(sch, block_id);
  VLOG(6) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SetBufferType(sch, block_id);
  SetReduceType(sch, block_id);
}

void TileDiscreteReductionTactic::MergeDiscreteFlattenAxis(
    ir::IRSchedule* sch, const std::string& block_id) {
  // Note: We need to fuse loops from bottom to top,
  // because the loop index will be changed when the upper loops fused.
  if (vec_spatial_axis_last_.size() >= 2) {
    sch->Fuse(block_id, vec_spatial_axis_last_);
  }
  if (vec_spatial_axis_first_.size() >= 2) {
    sch->Fuse(block_id, vec_spatial_axis_first_);
  }
}

void TileDiscreteReductionTactic::MergeReduceAxis(ir::IRSchedule* sch,
                                                  const std::string& block_id) {
  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  int32_t max_loop_idx = 0;
  for (int32_t idx : vec_reduce_axis_) {
    max_loop_idx = std::max(max_loop_idx, idx);
    PADDLE_ENFORCE_EQ(idx < loops.size() || loops.size() == 1,
                      true,
                      ::common::errors::InvalidArgument(
                          "The reduce axis should meet: axis's idx < "
                          "loops.size() or loops.size() == 1, but received "
                          "idx= %d ,loops.size() = %d",
                          idx,
                          loops.size()));
  }
  if (max_loop_idx < loops.size() && vec_reduce_axis_.size() >= 2) {
    sch->Fuse(block_id, vec_reduce_axis_);
  }
}

void TileDiscreteReductionTactic::SplitSptialInner(
    ir::IRSchedule* sch, const std::string& block_id) {
  const int64_t sp_thread = context_->config.tile_config.warp_num * 32 /
                            context_->config.tile_config.tree_reduce_num;
  auto loops = sch->GetLoops(block_id);
  if (loops.size() == 3) {
    // [S, S', R] => [S, S'(-1), S'(sp_thread), R]
    sch->Split(loops[1], std::vector<int>({-1, sp_thread}));
    // [S, S'(-1), S'(sp_thread), R] => [S, S'(sp_thread), R]
    sch->Fuse(block_id, std::vector<int>{0, 1});
  } else if (loops.size() == 2) {
    // [S, R] => [S(-1), S(sp_thread), R]
    sch->Split(loops[0], std::vector<int>({-1, sp_thread}));
  }
}

void TileDiscreteReductionTactic::SplitReduceInner(
    ir::IRSchedule* sch, const std::string& block_id) {
  const int64_t rd_block = context_->config.tile_config.grid_reduce_num;
  const int64_t rd_thread = context_->config.tile_config.tree_reduce_num;
  const int cur_reduce_axis = 2;

  // [ R ] => [ rd_block*rd_thread, rd_inner ]
  auto loops = sch->GetLoops(block_id);
  sch->Split(loops[cur_reduce_axis],
             std::vector<int>{-1, rd_block * rd_thread});
  loops = sch->GetLoops(block_id);
  sch->Reorder({loops[cur_reduce_axis + 1], loops[cur_reduce_axis]});

  loops = sch->GetLoops(block_id);
  if (IsReductionSBlock(sch->GetBlock(block_id))) {
    ir::Expr rf_tensor =
        sch->FactorizeReduction(loops[cur_reduce_axis],
                                /* rf_axis = */ 0,
                                /* with_write_back_block_init = */ false);
    map_rf_block_[block_id] = rf_tensor.as_tensor_ref()->name;
  }

  // [ rd_block*rd_thread ] => [ rd_block, rd_thread ]
  if (rd_block > 1) {
    loops = sch->GetLoops(block_id);
    sch->Split(loops[cur_reduce_axis], {rd_block, rd_thread});

    if (IsReductionSBlock(sch->GetBlock(block_id))) {
      loops = sch->GetLoops(map_rf_block_[block_id]);
      sch->Split(loops[cur_reduce_axis], {rd_block, rd_thread});

      loops = sch->GetLoops(block_id);
      ir::Expr rf_tensor =
          sch->FactorizeReduction(loops[cur_reduce_axis],
                                  /* rf_axis = */ 0,
                                  /* with_write_back_block_init = */ false);
      std::string tensor_name = rf_tensor.as_tensor_ref()->name;
      map_global_rf_block_[block_id] = tensor_name;
      rf_tensor.as_tensor_ref()->WithBuffer("global", "_" + tensor_name);
    }
  }
}

void TileDiscreteReductionTactic::SetBufferType(ir::IRSchedule* sch,
                                                const std::string& block_id) {
  auto block = sch->GetBlock(block_id);
  if (context_->output_names.count(block_id) > 0) {
    sch->SetBuffer(block, "global");
  } else {
    sch->SetBuffer(block, "local");
  }

  if (map_rf_block_.count(block_id) > 0) {
    auto block = sch->GetBlock(map_rf_block_[block_id]);
    sch->SetBuffer(block, "local");
  }
}

void TileDiscreteReductionTactic::SetReduceType(ir::IRSchedule* sch,
                                                const std::string& block_id) {
  if (IsReductionSBlock(sch->GetBlock(block_id))) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = context_->config.tile_config.reduce_method;
  }
  if (map_global_rf_block_.count(block_id) > 0) {
    auto block = sch->GetBlock(map_global_rf_block_[block_id])
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = context_->config.tile_config.reduce_method;
  }
}

void TileDiscreteReductionTactic::BindCudaInfo(ir::IRSchedule* sch,
                                               const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);

  // [S(-1), S(32), R(16), R(-1)] =>
  // [S(blockIdx.x), S(threadIdx.x), R(threadIdx.y), R(inner_loop)]
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    sch->Bind(loops[0], "blockIdx.x");
    sch->Bind(loops[1], "threadIdx.x");
    if (context_->config.tile_config.grid_reduce_num > 1) {
      sch->Bind(loops[2], "blockIdx.y");
      if (loops.size() > 3) {
        sch->Bind(loops[3], "threadIdx.y");
      }
    } else {
      sch->Bind(loops[2], "threadIdx.y");
    }
  };

  DoBind(sch->GetLoops(block_id));

  if (map_rf_block_.count(block_id) > 0) {
    DoBind(sch->GetLoops(map_rf_block_[block_id]));
  }
  if (map_global_rf_block_.count(block_id) > 0) {
    DoBind(sch->GetLoops(map_global_rf_block_[block_id]));
  }
}

std::unique_ptr<ScheduleTactic> CreateTileDiscreteReductionTactic() {
  return std::make_unique<TileDiscreteReductionTactic>();
}

}  // namespace ir
}  // namespace cinn
