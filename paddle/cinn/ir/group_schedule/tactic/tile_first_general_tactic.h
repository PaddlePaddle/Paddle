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

#pragma once

#include <string>
#include "paddle/cinn/ir/group_schedule/tactic/schedule_tactic.h"

namespace cinn {
namespace ir {

class TileFirstGeneralTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "TileFirstGeneralTactic"; }

 private:
  void MergeFlattenAxis(ir::IRSchedule* sch, const std::string& block_id);
  void MergeReduceAxis(ir::IRSchedule* sch, const std::string& block_id);
  void SplitFlattenInner(ir::IRSchedule* sch, const std::string& block_id);
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

}  // namespace ir
}  // namespace cinn
