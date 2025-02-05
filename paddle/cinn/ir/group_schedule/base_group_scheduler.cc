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

#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/dy_shape_group_scheduler.h"

namespace cinn {
namespace ir {

std::unique_ptr<GroupScheduler> GroupScheduler::Make(
    ir::IRSchedule* ir_sch,
    const std::unordered_set<std::string>& output_tensor_names,
    const cinn::common::Target& target,
    bool is_dy_shape,
    const std::shared_ptr<FusionGroupInfo>& group_info) {
  return std::make_unique<DynamicShapeGroupScheduler>(
      ir_sch, output_tensor_names, target, group_info);
}

std::unordered_set<std::string> GroupScheduler::OutputTensorNames() const {
  std::unordered_set<std::string> output_tensor_names{output_tensor_names_};
  for (ir::ScheduleBlockNode* node : schedule_block_graph_->EndPoints()) {
    output_tensor_names.insert(node->id());
  }
  return output_tensor_names;
}

}  // namespace ir
}  // namespace cinn
