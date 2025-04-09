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
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/tactic/schedule_tactic.h"

namespace cinn {
namespace ir {

/**
 * The class used for scheduling fusion groups with dynamic shape.
 * Note: Currently only CUDA backend is supported.
 */
class DynamicShapeGroupScheduler : public GroupScheduler {
 public:
  DynamicShapeGroupScheduler(
      ir::IRSchedule* ir_sch,
      const std::unordered_set<std::string>& output_tensor_names,
      const cinn::common::Target& target,
      const std::shared_ptr<FusionGroupInfo>& group_info)
      : GroupScheduler(ir_sch, output_tensor_names, target, group_info) {
    Init();
  }

  void Schedule() override;

  std::vector<std::pair<SymbolicPredicate, ir::Expr>> GetIRs() override;
  std::vector<std::pair<SymbolicPredicate, ir::Expr>> GetCX86IRs() override;
  std::vector<int> GetPriorities() override;

  struct BucketContext {
    SymbolicPredicate predicate;
    int priority;
    std::unique_ptr<ir::IRSchedule> ir_sch;
    std::unique_ptr<ir::ScheduleBlockGraph> schedule_block_graph;
    ScheduleContext schedule_context;
  };

 private:
  void Init();

  void InitBuckets();

  void ApplyTactics(BucketContext* bucket_context);

  ir::ScheduleBlockNode* FindGlobalMasterNode(
      const std::unique_ptr<ir::ScheduleBlockGraph>& schedule_block_graph);

  SymbolicPredicate MakeBucketPredicate(const BucketInfo& bucket_info,
                                        ScheduleBlockNode* node);

 private:
  std::vector<BucketContext> bucket_contexts_;
  std::vector<std::unique_ptr<ScheduleTactic>> tactics_;
};

}  // namespace ir
}  // namespace cinn
