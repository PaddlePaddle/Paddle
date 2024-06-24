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

namespace cinn {
namespace ir {

// The priority of the ScheduleBlockNode,
// prioritizing whether it has been bound to the cuda axis,
// and secondly considering the amount of calculated data.
struct NodePriority {
  bool has_loop_binded;
  int64_t score;

  bool operator<(const NodePriority& other) const {
    if (has_loop_binded ^ other.has_loop_binded) {
      return !has_loop_binded;
    } else {
      return score < other.score;
    }
  }
};

/**
 * The class used for scheduling fusion groups with static shape.
 * Its responsibility is to perform loop alignment,
 * automatic inline, automatic loop fusion,
 * and optimize the storage location of intermediate variables.
 * Note: Currently only CUDA backend is supported.
 */
class StaticShapeGroupScheduler : public GroupScheduler {
 public:
  StaticShapeGroupScheduler(
      ir::IRSchedule* ir_sch,
      const std::unordered_set<std::string>& output_tensor_names,
      const cinn::common::Target& target,
      const std::shared_ptr<hlir::framework::pir::GroupInfo>& group_info)
      : GroupScheduler(ir_sch, output_tensor_names, target, group_info) {}

  void Schedule() override;

  void MapExprSchedule();

  std::vector<std::pair<SymbolicPredicate, ir::Expr>> GetIRs() override;
  std::vector<int> GetPriorities() override;

 private:
  // Automatically align loops for each ScheduleBlock.
  void DoLoopAlignment();

  // Automatically inline some ScheduleBlock which meets the conditions.
  void DoComputeInline();

  // Make every effort to automatically merge the loops of the horizontal
  // relationship ScheduleBlockNode.
  void DoHorizontalLoopFusion();

  // Make every effort to automatically merge the loops of the vertical
  // relationship ScheduleBlockNode.
  void DoVerticalLoopFusion();

  // Automatically bind cuda axis on loops.
  void BindCudaAxis();

  // Automatically allocate storage locations for variables to optimize IO.
  void AllocateStorage();

  // Automatically optimize the reductive calculation
  void OptimizeReduction();

  // Evaluate the priority of ScheduleBlockNode.
  // The node where the performance bottleneck is located
  // has a higher priority, while the node with a lower priority
  // needs to compromise and align loops with the node with the highest
  // priority.
  NodePriority CalculateNodePriority(const ir::ScheduleBlockNode* node) const;

  // Find the highest priority ScheduleBlockNode,
  // other nodes need to align the loop with it.
  ir::ScheduleBlockNode* FindGlobalMasterNode() const;

  // Obtain the latest order of ScheduleBlock and the control structures
  // throughout the entire IR.
  void UpdateBlockOrder();

  /**
   * @brief Determine whether the graph level dependency is still maintained
   * after the schedule_block is placed in the insert position of target_loop.
   * @param schedule_block The src schedule_block to be replaced.
   * @param target_loop The target loop to be insert into the schedule_block.
   * @param insert_pos The insert position of new schedule_block in the
   * target_loop.
   */
  bool IsKeepGraphDependency(Expr schedule_block,
                             Expr target_loop,
                             int insert_pos) const;

  /**
   * @brief Determine whether all feasible conditions are met
   * after the schedule_block is placed in the insert position of target_loop.
   * @param schedule_block The src schedule_block to be replaced.
   * @param target_loop The target loop to be insert into the schedule_block.
   * @param insert_pos The insert position of new schedule_block in the
   * target_loop.
   */
  bool MeetConditions(Expr schedule_block,
                      Expr target_loop,
                      int insert_pos) const;

 private:
  /**
   * @brief Interface of feasibility condition.
   * @param schedule_block The src schedule_block to be replaced.
   * @param target_loop The target loop to be insert into the schedule_block.
   * @param insert_pos The insert position of new schedule_block in the
   * target_loop.
   */
  using FeasibleCondition = bool (StaticShapeGroupScheduler::*)(
      Expr schedule_block, Expr target_loop, int insert_pos) const;
  // All feasible conditions.
  std::vector<FeasibleCondition> feasible_conditions_;

  /**
   * The order of blocks and their control statements,
   * only For, IfThenElse and ScheduleBlock is considered.
   *
   * Example:
   * for0:
   *   for1:
   *     block0
   *     block1
   *   block2
   *   for2:
   *     block3
   *     block4
   *
   * the result is:
   *   [0]: for0
   *   [0, 0]: for1
   *   [0, 0, 0]: block0
   *   [0, 0, 1]: block1
   *   [0, 1]: block2
   *   [0, 2]: for2
   *   [0, 2, 0]: block3
   *   [0, 2, 1]: block4
   */
  std::map<std::vector<int>, ir::Expr> blocks_order_with_ctrl_stmt_;
};

}  // namespace ir
}  // namespace cinn
