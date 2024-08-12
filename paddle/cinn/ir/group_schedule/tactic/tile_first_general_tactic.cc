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
      [&](const ir::DiscreteReduceMethod&) { return false; },
      [&](const ir::IntervalReduceMethod&) { return false; },
  };
  return std::visit(MatchWarpReduce, config.tile_config.reduce_method);
}

bool UseContinuousDataTile(const ScheduleConfig& config) {
  if (config.base_info->reduce_axis.empty()) {
    return true;
  }
  int64_t min_stride = INT_MAX;
  int64_t min_reduce_stride = INT_MAX;
  int64_t last_axis = 0;
  int64_t last_reduce_axis = 0;
  for (size_t i = 0; i < config.base_info->loop_strides.size(); i++) {
    if (config.base_info->loop_strides[i] < min_stride &&
        config.base_info->loop_strides[i] != 0) {
      min_stride = config.base_info->loop_strides[i];
      last_axis = i;
    }
  }
  for (int64_t axis : config.base_info->reduce_axis) {
    if (config.base_info->loop_strides[axis] < min_reduce_stride) {
      min_reduce_stride = config.base_info->loop_strides[axis];
      last_reduce_axis = axis;
    }
  }
  return last_axis == last_reduce_axis;
}

class TileFirstGeneralTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "TileFirstGeneralTactic"; }

 private:
  void ApplyContinuousReduce(ir::IRSchedule* sch, const std::string& block_id);
  void ApplyDiscreteReduce(ir::IRSchedule* sch, const std::string& block_id);
  void ApplyIntervalReduce(ir::IRSchedule* sch, const std::string& block_id);

  void AlignToReduceInput(ir::IRSchedule* sch, const std::string& block_id);
  void MergeFlattenAxis(ir::IRSchedule* sch, const std::string& block_id);
  void MergeDiscreteFlattenAxis(ir::IRSchedule* sch,
                                const std::string& block_id);
  void MergeReduceAxis(ir::IRSchedule* sch, const std::string& block_id);
  void SplitSptialInner(ir::IRSchedule* sch, const std::string& block_id);
  void SplitReduceInner(ir::IRSchedule* sch, const std::string& block_id);
  void VariableTypeAssignment(ir::IRSchedule* sch, const std::string& block_id);
  void SetReduceType(ir::IRSchedule* sch, const std::string& block_id);
  void SetDiscreteReduceType(ir::IRSchedule* sch, const std::string& block_id);
  void SetIntervalReduceType(ir::IRSchedule* sch, const std::string& block_id);
  void BindCudaInfo(ir::IRSchedule* sch, const std::string& block_id);

 private:
  ScheduleContext* context_;
  std::vector<int32_t> vec_spatial_axis_first_;
  std::vector<int32_t> vec_spatial_axis_last_;
  std::vector<int32_t> vec_flatten_axis_;
  std::vector<int32_t> vec_reduce_axis_;

  std::vector<bool> vec_is_reduce_;
  std::vector<int64_t> loop_perm_;
  int64_t num_lower_reduce_axes_;
  int64_t lower_reduce_extend_;  // -1 means dynamic shape
};

void TileFirstGeneralTactic::Init(ScheduleContext* context) {
  context_ = context;

  const size_t num_loops = context_->config.base_info->data_space.size();
  vec_is_reduce_.assign(num_loops, false);
  for (int64_t axis : context_->config.base_info->reduce_axis) {
    vec_is_reduce_[axis] = true;
  }

  const auto& loop_strides = context_->config.base_info->loop_strides;
  loop_perm_.clear();
  if (!loop_strides.empty()) {
    loop_perm_.resize(num_loops);
    std::iota(loop_perm_.begin(), loop_perm_.end(), 0);
    std::sort(loop_perm_.begin(), loop_perm_.end(), [&](int64_t a, int64_t b) {
      return loop_strides[a] > loop_strides[b];
    });
  }

  num_lower_reduce_axes_ = 0;
  lower_reduce_extend_ = 1;
  for (int i = loop_perm_.size() - 1; i >= 0; i--) {
    int axis = loop_perm_[i];
    if (loop_strides[axis] == 0) {
      continue;
    }
    if (!vec_is_reduce_[axis]) {
      break;
    }
    num_lower_reduce_axes_++;

    const int64_t data_space = context_->config.base_info->data_space[axis];
    if (data_space == -1) {
      lower_reduce_extend_ = -1;
    } else if (lower_reduce_extend_ != -1) {
      lower_reduce_extend_ =
          std::max(data_space * loop_strides[axis], lower_reduce_extend_);
    }
  }

  VLOG(4) << "loop_perm: " << utils::Join(loop_perm_, ", ");
  VLOG(4) << "num_lower_reduce_axes: " << num_lower_reduce_axes_;
  VLOG(4) << "lower_reduce_extend: " << lower_reduce_extend_;

  // reduce axis have be re-order to last
  vec_flatten_axis_.clear();
  vec_reduce_axis_.clear();
  int32_t reduce_start_idx =
      num_loops - context_->config.base_info->reduce_axis.size();
  for (int32_t i = 0; i < num_loops; ++i) {
    if (i >= reduce_start_idx) {
      vec_reduce_axis_.push_back(i);
    } else {
      vec_flatten_axis_.push_back(i);
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
}

void TileFirstGeneralTactic::Apply(ir::IRSchedule* sch,
                                   const std::string& block_id) {
  if (ir::IsReduceInitTensorName(block_id)) return;

  AlignToReduceInput(sch, block_id);
  VLOG(4) << "After AlignToReduceInput on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  if (num_lower_reduce_axes_) {
    if (lower_reduce_extend_ <= 16 && lower_reduce_extend_ != -1) {
      ApplyIntervalReduce(sch, block_id);
    } else {
      ApplyContinuousReduce(sch, block_id);
    }
  } else {
    if (vec_reduce_axis_.empty()) {
      ApplyContinuousReduce(sch, block_id);
    } else {
      ApplyDiscreteReduce(sch, block_id);
    }
  }
}

void TileFirstGeneralTactic::ApplyContinuousReduce(
    ir::IRSchedule* sch, const std::string& block_id) {
  const auto sp_thread = context_->config.tile_config.warp_num * 32 /
                         context_->config.tile_config.tree_reduce_num;
  const auto sp_loop = context_->config.tile_config.spatial_inner_num;
  const auto rd_thread = context_->config.tile_config.tree_reduce_num;
  VLOG(4) << "ApplyContinuousDataTile sp_thread=" << sp_thread;
  VLOG(4) << "ApplyContinuousDataTile sp_loop=" << sp_loop;
  VLOG(4) << "ApplyContinuousDataTile rd_thread=" << rd_thread;
  VLOG(4) << "ApplyContinuousDataTile vec_flatten_axis: "
          << utils::Join(vec_flatten_axis_, ", ");
  VLOG(4) << "ApplyContinuousDataTile vec_reduce_axis: "
          << utils::Join(vec_reduce_axis_, ", ");

  MergeReduceAxis(sch, block_id);
  VLOG(4) << "After MergeReduceAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  MergeFlattenAxis(sch, block_id);
  VLOG(4) << "After MergeFlattenAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Split spatial axes -> [sp_block, sp_loop, sp_thread]
  int current_reduce_axis = 0;
  if (vec_flatten_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    if (sp_loop > 1 && sp_thread > 1) {
      // [S, R] => [S(-1), S(inner_loop), S(thread), R]
      sch->Split(loops[0], {-1, sp_loop, sp_thread});
      current_reduce_axis = 3;
    } else if (sp_loop > 1 || sp_thread > 1) {
      // [S, R] => [S(-1), S(thread), R]
      sch->Split(loops[0], {-1, sp_loop > 1 ? sp_loop : sp_thread});
      current_reduce_axis = 2;
    } else {
      // [S, R] => [S, R]
      current_reduce_axis = 1;
    }
  }
  VLOG(4) << "After SplitSptial on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Split reduce axes -> [rd_loop, rd_thread]
  if (vec_reduce_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    auto reduce_loop = loops[current_reduce_axis].As<ir::For>();
    // [S..S, R] => [S..S, R(-1), R(thread)]
    sch->Split(loops[current_reduce_axis], {-1, rd_thread});

    // TODO(lshpku): the Reorder is unneeded if the later FactorizeReduction
    // supports rf_axis=1.
    loops = sch->GetLoops(block_id);
    // [S..S, R(-1), R(thread)] => [S..S, R(thread), R(-1)]
    sch->Reorder({loops[current_reduce_axis + 1], loops[current_reduce_axis]});

    if (IsReduceBlock(context_->config, block_id)) {
      loops = sch->GetLoops(block_id);
      sch->FactorizeReduction(loops[current_reduce_axis],
                              /* rf_axis = */ 0,
                              /* with_write_back_block_init = */ false);
    }
  }
  VLOG(4) << "After SplitReduce on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Bind CUDA info
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    std::string sp_axis_type = "threadIdx.y";
    std::string rd_axis_type = "threadIdx.x";
    sch->Bind(loops[0], "blockIdx.x");
    if (!vec_flatten_axis_.empty() && sp_thread > 1) {
      if (vec_reduce_axis_.empty()) {
        // [S..S] => [S(blockIdx.x), optional(inner_loop), S(threadIdx.x)]
        sch->Bind(loops[current_reduce_axis - 1], rd_axis_type);
      } else {
        // [S..S, R..R] =>
        // [S(blockIdx.x), optional(inner_loop), S(threadIdx.y), R..R]
        sch->Bind(loops[current_reduce_axis - 1], sp_axis_type);
      }
    }
    if (!vec_reduce_axis_.empty() && current_reduce_axis > 0) {
      // [S(blockIdx.x), optional(inner_loop), S(threadIdx.y), R..R] =>
      // [S(blockIdx.x), optional(inner_loop), S(threadIdx.y), R(threadIdx.x),
      // R(inner_loop)]
      sch->Bind(loops[current_reduce_axis], rd_axis_type);
    }
  };
  DoBind(sch->GetLoops(block_id));
  if (IsReduceBlock(context_->config, block_id) &&
      sch->HasBlock(block_id + "_rf")) {
    DoBind(sch->GetLoops(block_id + "_rf"));
  }
  VLOG(4) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  VariableTypeAssignment(sch, block_id);
  SetReduceType(sch, block_id);
}

void TileFirstGeneralTactic::ApplyDiscreteReduce(ir::IRSchedule* sch,
                                                 const std::string& block_id) {
  MergeReduceAxis(sch, block_id);
  VLOG(4) << "After MergeReduceAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  MergeDiscreteFlattenAxis(sch, block_id);
  VLOG(4) << "After MergeDiscreteFlattenAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SplitSptialInner(sch, block_id);
  VLOG(4) << "After SplitSptialInner on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SplitReduceInner(sch, block_id);
  VLOG(4) << "After SplitReduceInner on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  BindCudaInfo(sch, block_id);
  VLOG(4) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  VariableTypeAssignment(sch, block_id);
  VLOG(4) << "After VariableTypeAssignment on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SetDiscreteReduceType(sch, block_id);
}

void TileFirstGeneralTactic::ApplyIntervalReduce(ir::IRSchedule* sch,
                                                 const std::string& block_id) {
  const int64_t sp_thread = context_->config.tile_config.warp_num * 32 /
                            context_->config.tile_config.tree_reduce_num;
  const int64_t sp_loop = context_->config.tile_config.spatial_inner_num;
  const int64_t rd_thread = context_->config.tile_config.tree_reduce_num;
  const int64_t tx = lower_reduce_extend_;
  const int64_t ty = std::max(32 / tx, sp_thread);
  const int64_t tz = sp_thread * rd_thread / (tx * ty);
  VLOG(4) << "ApplyIntervalReduce tx=" << tx << " ty=" << ty << " tz=" << tz;

  MergeReduceAxis(sch, block_id);
  VLOG(4) << "After MergeReduceAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  MergeFlattenAxis(sch, block_id);
  VLOG(4) << "After MergeFlattenAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Split spatial axes -> [sp_block, sp_loop, ty]
  int current_reduce_axis = 0;
  if (vec_flatten_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    if (sp_loop > 1 && ty > 1) {
      sch->Split(loops[0], {-1, sp_loop, ty});
      current_reduce_axis = 3;
    } else if (sp_loop > 1 || ty > 1) {
      sch->Split(loops[0], {-1, sp_loop > 1 ? sp_loop : ty});
      current_reduce_axis = 2;
    } else {
      current_reduce_axis = 1;
    }
  }
  VLOG(4) << "After SplitSptial on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Split reduce axes -> [rd_loop, tz, tx]
  {
    auto loops = sch->GetLoops(block_id);
    auto reduce_loop = loops[current_reduce_axis].As<ir::For>();
    sch->Split(loops[current_reduce_axis], {-1, tz * tx});

    loops = sch->GetLoops(block_id);
    sch->Reorder({loops[current_reduce_axis + 1], loops[current_reduce_axis]});

    if (IsReduceBlock(context_->config, block_id)) {
      loops = sch->GetLoops(block_id);
      sch->FactorizeReduction(loops[current_reduce_axis],
                              /* rf_axis = */ 0,
                              /* with_write_back_block_init = */ false);
      loops = sch->GetLoops(block_id + "_rf");
      sch->Split(loops[current_reduce_axis], {tz, tx});
    }

    loops = sch->GetLoops(block_id);
    sch->Split(loops[current_reduce_axis], {tz, tx});
  }
  VLOG(4) << "After SplitReduce on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Bind CUDA info
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    sch->Bind(loops[0], "blockIdx.x");
    if (!vec_flatten_axis_.empty() && ty > 1) {
      sch->Bind(loops[current_reduce_axis - 1], "threadIdx.y");
    }
    sch->Bind(loops[current_reduce_axis], "threadIdx.z");
    sch->Bind(loops[current_reduce_axis + 1], "threadIdx.x");
  };
  DoBind(sch->GetLoops(block_id));
  if (IsReduceBlock(context_->config, block_id) &&
      sch->HasBlock(block_id + "_rf")) {
    DoBind(sch->GetLoops(block_id + "_rf"));
  }
  VLOG(4) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  VariableTypeAssignment(sch, block_id);
  SetIntervalReduceType(sch, block_id);
}

void TileFirstGeneralTactic::AlignToReduceInput(ir::IRSchedule* sch,
                                                const std::string& block_id) {
  const auto& loop_strides = context_->config.base_info->loop_strides;
  if (loop_strides.empty()) {
    return;
  }

  // Reorder S/R loops seperately, otherwise reduce_init will be de-inlined.
  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  std::vector<Expr> sp_loops, rd_loops;
  for (auto i : loop_perm_) {
    if (vec_is_reduce_[i]) {
      rd_loops.push_back(loops[i]);
    } else if (loop_strides[i] != 0) {
      sp_loops.push_back(loops[i]);
    }
  }
  sch->Reorder(sp_loops);
  sch->Reorder(rd_loops);
}

void TileFirstGeneralTactic::MergeFlattenAxis(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  if (vec_flatten_axis_.size() >= 2) {
    sch->Fuse(block_id, vec_flatten_axis_);
  }
}

void TileFirstGeneralTactic::MergeDiscreteFlattenAxis(
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

void TileFirstGeneralTactic::MergeReduceAxis(ir::IRSchedule* sch,
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

void TileFirstGeneralTactic::SplitSptialInner(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);
  if (loops.size() == 3) {
    // [S, S', R] => [S, S'(-1), S'(32), R]
    auto split_loops = sch->Split(loops[1], std::vector<int>({-1, 32}));
    // [S, S'(-1), S'(32), R] => [S, S'(32), R]
    sch->Fuse(block_id, std::vector<int>{0, 1});
  } else if (loops.size() == 2) {
    // [S, R] => [S(-1), S(32), R]
    auto split_loops = sch->Split(loops[0], std::vector<int>({-1, 32}));
  }
}

void TileFirstGeneralTactic::SplitReduceInner(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);
  // [S(-1), S(32), R] => [S(-1), S(32), R(16), R(-1)]
  sch->Split(loops[2], std::vector<int>{16, -1});

  loops = sch->GetLoops(block_id);
  if (IsReduceBlock(context_->config, block_id)) {
    sch->FactorizeReduction(loops[2],
                            0,
                            /* with_write_back_block_init = */ false);
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

void TileFirstGeneralTactic::SetDiscreteReduceType(
    ir::IRSchedule* sch, const std::string& block_id) {
  if (IsReduceBlock(context_->config, block_id)) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = cinn::ir::DiscreteReduceMethod();
  }
}

void TileFirstGeneralTactic::SetIntervalReduceType(
    ir::IRSchedule* sch, const std::string& block_id) {
  if (IsReduceBlock(context_->config, block_id)) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = cinn::ir::IntervalReduceMethod();
  }
}

void TileFirstGeneralTactic::BindCudaInfo(ir::IRSchedule* sch,
                                          const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);

  // [S(-1), S(32), R(16), R(-1)] =>
  // [S(blockIdx.x), S(threadIdx.x), R(threadIdx.y), R(inner_loop)]
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    sch->Bind(loops[0], "blockIdx.x");
    sch->Bind(loops[1], "threadIdx.x");
    sch->Bind(loops[2], "threadIdx.y");
  };

  DoBind(sch->GetLoops(block_id));

  if (IsReduceBlock(context_->config, block_id) &&
      sch->HasBlock(block_id + "_rf")) {
    auto loops = sch->GetLoops(block_id + "_rf");
    DoBind(sch->GetLoops(block_id + "_rf"));
  }
}

std::unique_ptr<ScheduleTactic> CreateTileFirstGeneralTactic() {
  return std::make_unique<TileFirstGeneralTactic>();
}

}  // namespace ir
}  // namespace cinn
