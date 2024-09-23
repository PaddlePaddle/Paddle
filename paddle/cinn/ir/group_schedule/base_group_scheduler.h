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
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"
#include "paddle/cinn/ir/group_schedule/tactic/schedule_tactic.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule_block_graph.h"

namespace cinn {

namespace hlir::framework::pir {
struct FusionGroupInfo;
}  // namespace hlir::framework::pir

using hlir::framework::pir::FusionGroupInfo;

namespace ir {

using SymbolicPredicate = Expr;

/**
 * The base class used for scheduling fusion groups.
 */
class GroupScheduler {
 public:
  GroupScheduler(ir::IRSchedule* ir_sch,
                 const std::unordered_set<std::string>& output_tensor_names,
                 const cinn::common::Target& target,
                 const std::shared_ptr<FusionGroupInfo>& group_info)
      : ir_sch_(ir_sch),
        output_tensor_names_(output_tensor_names),
        target_(target),
        group_info_(group_info) {
    schedule_block_graph_ = std::make_unique<ir::ScheduleBlockGraph>(*ir_sch_);
  }

  static std::unique_ptr<GroupScheduler> Make(
      ir::IRSchedule* ir_sch,
      const std::unordered_set<std::string>& output_tensor_names,
      const cinn::common::Target& target,
      bool is_dy_shape = false,
      const std::shared_ptr<FusionGroupInfo>& group_info = nullptr);

  virtual ~GroupScheduler() = default;

  virtual void Schedule() = 0;

  virtual std::vector<std::pair<SymbolicPredicate, ir::Expr>> GetIRs() = 0;
  virtual std::vector<int> GetPriorities() = 0;
  virtual std::vector<std::pair<SymbolicPredicate, ir::Expr>> GetCX86IRs() {
    CINN_NOT_IMPLEMENTED;
  }

  std::unordered_set<std::string> OutputTensorNames() const;

 protected:
  ir::IRSchedule* ir_sch_;
  const std::unordered_set<std::string>& output_tensor_names_;
  const cinn::common::Target& target_;
  // Graph in units of ScheduleBlockNode, each node corresponds to a
  // ScheduleBlock in IR.
  std::unique_ptr<ir::ScheduleBlockGraph> schedule_block_graph_;

  std::shared_ptr<FusionGroupInfo> group_info_;
};

}  // namespace ir
}  // namespace cinn
