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

#include "paddle/cinn/ir/group_schedule/tactic/loop_reorder_alignment_tactic.h"
#include <set>
#include <unordered_map>
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace ir {

class LoopReorderAlignmentTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override {
    return "LoopReorderAlignmentTactic";
  }

 private:
  bool NeedReorderLoops();

  std::vector<int32_t> GetNewOrder();

  void UpdateBaseRank(ir::IRSchedule* sch, const std::string& block_id);

  void DoBroadcastLoop(ir::IRSchedule* sch, const std::string& block_id);

  void DoReorder(ir::IRSchedule* sch, const std::string& block_id);

 private:
  ScheduleContext* context_;
  size_t base_rank_;
  bool need_reorder_loops_;
  std::vector<int32_t> new_order_;
};

void LoopReorderAlignmentTactic::Init(ScheduleContext* context) {
  context_ = context;
  base_rank_ = 0;
  need_reorder_loops_ = NeedReorderLoops();
  new_order_ = GetNewOrder();
}

void LoopReorderAlignmentTactic::Apply(ir::IRSchedule* sch,
                                       const std::string& block_id) {
  DoBroadcastLoop(sch, block_id);

  if (!ir::IsReduceInitTensorName(block_id)) {
    UpdateBaseRank(sch, block_id);
  }

  if (need_reorder_loops_ && !ir::IsReduceInitTensorName(block_id)) {
    DoReorder(sch, block_id);
  }
}

void LoopReorderAlignmentTactic::UpdateBaseRank(ir::IRSchedule* sch,
                                                const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);
  if (base_rank_ == 0) {
    base_rank_ = loops.size();
  } else {
    if (base_rank_ != loops.size()) {
      throw std::runtime_error("loops  rank not same ");
    }
  }
}

bool LoopReorderAlignmentTactic::NeedReorderLoops() {
  const auto HasReduceAxis = [&]() {
    return context_->config.base_info->reduce_axis.size() > 0;
  };
  if (!HasReduceAxis()) {
    return false;
  }

  const auto HasNonLastDimReduce = [&]() {
    std::vector<int64_t> vec_reduce_axis =
        context_->config.base_info->reduce_axis;
    std::sort(vec_reduce_axis.begin(), vec_reduce_axis.end());
    return vec_reduce_axis.front() !=
           context_->config.base_info->data_rank - vec_reduce_axis.size();
  };

  return HasNonLastDimReduce();
}

std::vector<int32_t> LoopReorderAlignmentTactic::GetNewOrder() {
  std::set<int64_t> reduce_set(context_->config.base_info->reduce_axis.begin(),
                               context_->config.base_info->reduce_axis.end());

  std::vector<int32_t> new_order;
  for (int32_t i = 0; i < context_->config.base_info->data_rank; ++i) {
    if (!reduce_set.count(i)) {
      new_order.push_back(i);
    }
  }
  for (auto axis : context_->config.base_info->reduce_axis) {
    new_order.push_back(axis);
  }

  return new_order;
}

void LoopReorderAlignmentTactic::DoBroadcastLoop(ir::IRSchedule* sch,
                                                 const std::string& block_id) {
  const auto HasBroadcastInfo = [&](const std::string& block_id) {
    return context_->config.base_info->broadcast_info.count(block_id) > 0;
  };
  const auto HasBroadcastToElementwiseInfo = [&](const std::string& block_id) {
    return context_->config.base_info->broadcast_to_elementwise.count(
               block_id) > 0;
  };
  const auto IsFullBroadcast = [&](const std::string& block_id) {
    return context_->config.base_info->broadcast_info[block_id].full_broadcast;
  };
  const auto IsSplitFirst = [&](const std::string& block_id) {
    return context_->config.base_info->broadcast_info[block_id].split_first;
  };

  if (HasBroadcastInfo(block_id)) {
    if (IsFullBroadcast(block_id)) {
      std::vector<int32_t> vec_out_split(
          context_->config.base_info->broadcast_info[block_id]
              .output_shape.size(),
          1);

      auto loops = sch->GetLoops(block_id);
      sch->Split(loops[0], vec_out_split);
      loops = sch->GetLoops(block_id);
    } else if (IsSplitFirst(block_id)) {
      for (auto& info :
           context_->config.base_info->broadcast_info[block_id].split_info) {
        auto axis = info.first;
        auto split_res = info.second;

        auto loops = sch->GetLoops(block_id);
        sch->Split(loops[axis], split_res);
        loops = sch->GetLoops(block_id);
      }
    } else {
      // Do nothing
    }

    sch->Broadcast(block_id,
                   context_->config.base_info->broadcast_info[block_id]);
  }

  if (HasBroadcastToElementwiseInfo(block_id)) {
    sch->BroadcastToElementwise(
        block_id,
        context_->config.base_info->broadcast_to_elementwise[block_id]
            .broadcast_axes);
  }
}

void LoopReorderAlignmentTactic::DoReorder(ir::IRSchedule* sch,
                                           const std::string& block_id) {
  const auto IsReduceBlock = [&](const std::string& block_id) {
    return context_->config.base_info->reduce_tensor_names.count(block_id) > 0;
  };
  if (IsReduceBlock(block_id)) {
    return;
  }

  sch->Reorder(block_id, new_order_);
}

std::unique_ptr<ScheduleTactic> CreateLoopReorderAlignmentTactic() {
  return std::make_unique<LoopReorderAlignmentTactic>();
}

}  // namespace ir
}  // namespace cinn
