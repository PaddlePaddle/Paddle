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

#pragma once
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule_block_graph.h"

namespace cinn {
namespace ir {

using SymbolicPredicate = Expr;

/**
 * The base class used for scheduling fusion groups.
 */
class GroupScheduler {
 public:
  GroupScheduler(ir::IRSchedule* ir_sch,
                 const std::unordered_set<std::string>& output_tensor_names,
                 const cinn::common::Target& target)
      : ir_sch_(ir_sch),
        output_tensor_names_(output_tensor_names),
        target_(target) {
    schedule_block_graph_ = std::make_unique<ir::ScheduleBlockGraph>(*ir_sch_);
  }

  static std::unique_ptr<GroupScheduler> Make(
      ir::IRSchedule* ir_sch,
      const std::unordered_set<std::string>& output_tensor_names,
      const cinn::common::Target& target,
      bool is_dy_shape = false);

  virtual ~GroupScheduler() = default;

  virtual void Schedule() = 0;

  virtual std::vector<std::pair<SymbolicPredicate, ir::Expr>> GetIRs() = 0;

 protected:
  ir::IRSchedule* ir_sch_;
  const std::unordered_set<std::string>& output_tensor_names_;
  const cinn::common::Target& target_;
  // Graph in units of ScheduleBlockNode, each node corresponds to a
  // ScheduleBlock in IR.
  std::unique_ptr<ir::ScheduleBlockGraph> schedule_block_graph_;
};

}  // namespace ir
}  // namespace cinn
