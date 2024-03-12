// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/tactic/bind_cuda_tactic.h"
#include <unordered_map>
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace ir {

class BindCudaTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "BindCudaTactic"; }

 private:
  ScheduleContext* context_;
};

void BindCudaTactic::Init(ScheduleContext* context) { context_ = context; }

const std::unordered_map<IterativeSpaceInfo::AxisType, std::string>
    axis_type2bind_info = {
        {IterativeSpaceInfo::AxisType::kCudaBlockX, "blockIdx.x"},
        {IterativeSpaceInfo::AxisType::kCudaBlockY, "blockIdx.y"},
        {IterativeSpaceInfo::AxisType::kCudaBlockZ, "blockIdx.z"},
        {IterativeSpaceInfo::AxisType::kCudaThreadX, "threadIdx.x"},
        {IterativeSpaceInfo::AxisType::kCudaThreadY, "threadIdx.y"},
        {IterativeSpaceInfo::AxisType::kCudaThreadZ, "threadIdx.z"},
};

void BindCudaTactic::Apply(ir::IRSchedule* sch, const std::string& block_id) {
  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  int loop_idx = 0;
  for (int i = 0;
       i < context_->iter_space_info.sp_space.size() && loop_idx < loops.size();
       ++i, ++loop_idx) {
    const auto& axis = context_->iter_space_info.sp_space[i];
    const IterativeSpaceInfo::AxisType& axis_type = std::get<1>(axis);
    if (axis_type2bind_info.count(axis_type) != 0 &&
        loops[loop_idx].As<ir::For>()->is_serial()) {
      sch->Bind(loops[loop_idx], axis_type2bind_info.at(axis_type));
    }
  }
  for (int i = 0;
       i < context_->iter_space_info.rb_space.size() && loop_idx < loops.size();
       ++i, ++loop_idx) {
    const auto& axis = context_->iter_space_info.rb_space[i];
    const IterativeSpaceInfo::AxisType& axis_type = std::get<1>(axis);
    if (axis_type2bind_info.count(axis_type) != 0 &&
        loops[loop_idx].As<ir::For>()->is_serial()) {
      sch->Bind(loops[loop_idx], axis_type2bind_info.at(axis_type));
    }
  }
}

std::unique_ptr<ScheduleTactic> CreateBindCudaTactic() {
  return std::make_unique<BindCudaTactic>();
}

}  // namespace ir
}  // namespace cinn
