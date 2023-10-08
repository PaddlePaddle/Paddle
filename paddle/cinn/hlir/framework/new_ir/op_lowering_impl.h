// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <memory>
#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/new_ir/group.h"
#include "paddle/cinn/hlir/framework/op_lowering_impl_base.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/pir/core/operation.h"

// Fusion Op lowering, there are four kinds of lowering function:
// Elementwise/Broadcast/Injective,Reduce,OutEWiseFusable,NonFusible.
// Elementwise/Broadcast/Injective Ops is with same shcedule.
// Reduce,OutEWiseFusable,NonFusible are using different schedule.

namespace cinn {
namespace hlir {
namespace framework {
namespace newir {

using GroupPtr = std::shared_ptr<Group>;

using common::Target;
class OpLowererImpl;

typedef bool (OpLowererImpl::*ScheduleDetermineFunction)(::pir::Operation*);

class OpLowererImpl : public OpLowererImplBase<GroupPtr> {
 public:
  explicit OpLowererImpl(const Target&);

  /**
   * @brief Lower a group to CINN IR.
   * @param group The group to be lowered.
   * @param apply_op_schedule Whether to schedule at Op level.
   * @param apply_group_schedule Whether to schedule at group level.
   * @return The lowered funcs.
   */
  std::vector<ir::LoweredFunc> Lower(const GroupPtr& group,
                                     bool apply_op_schedule = true,
                                     bool apply_group_schedule = true);

 private:
  /**
   * @brief Lower a group to CINN IR.
   * @param group The group to be lowered.
   * @param apply_op_schedule Whether to schedule at Op level.
   * @param apply_group_schedule Whether to schedule at group level.
   * @param schedule_determine_func Function used to determine which Ops to
   * schedule.
   * @return The lowered funcs.
   */
  std::vector<ir::LoweredFunc> LowerGroup(
      const GroupPtr& group,
      bool apply_op_schedule,
      bool apply_group_schedule,
      ScheduleDetermineFunction schedule_determine_func);

  /**
   * @brief Lower a group composed of CustomCall Op.
   * @param group The group to be lowered.
   * @return The lowered funcs.
   */
  std::vector<ir::LoweredFunc> LowerCustomCall(const GroupPtr& group);

  /**
   * @brief Post processing, including preparing function args and temporary
   * variables, applying low-level optimization passes, etc.
   * @param group The group to be lowered.
   * @param tensor_map All tensors used for calculating the group.
   * @param done_op_schedule Mark whether the Op level schedule has been
   * applied.
   * @param ir_sch The IRSchedule object of group.
   * @param group_func_arg_tensors Tensors used as the group function arguments.
   * @return The lowered funcs after the post processing.
   */
  std::vector<ir::LoweredFunc> PostProcess(
      const GroupPtr& group,
      const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
      bool done_op_schedule,
      ir::IRSchedule* ir_sch,
      std::vector<ir::Tensor>* group_func_arg_tensors);

  /**
   * @brief Lower an Op set to CINN IR.
   * Compute, Lower and optional Schedule will be performed one by one
   * for each Op.
   * @param ops The Op to be lowered.
   * @param apply_op_schedule Whether to schedule at Op level.
   * @param schedule_determine_func Function used to determine which Ops to
   * schedule.
   * @param group_func_arg_tensors Tensors used as the group function arguments.
   * @param tensor_map All tensors used for calculating the group.
   * @return The lowered func bodies of Op set.
   */
  std::vector<ir::Expr> LowerOps(
      const std::vector<::pir::Operation*>& ops,
      bool apply_op_schedule,
      ScheduleDetermineFunction schedule_determine_func,
      std::vector<ir::Tensor>* group_func_arg_tensors,
      std::unordered_map<::pir::Value, ir::Tensor>* tensor_map);

  /**
   * @brief Lower an Op to CINN IR. The Compute and Lower processes will be
   * called sequentially.
   * @param op_impl The Op implementation defining Compute and Schedule.
   * @param op The Op to be lowered.
   * @param tensor_map All tensors used for calculating the group.
   * @param op_func_arg_tensors Tensors used as the Op function arguments.
   * @return The lowered func of the Op.
   */
  std::vector<ir::LoweredFunc> DoOpLower(
      std::shared_ptr<hlir::framework::OpImpl> op_impl,
      ::pir::Operation* op,
      std::unordered_map<::pir::Value, ir::Tensor>* tensor_map,
      std::vector<ir::Tensor>* op_func_arg_tensors);

  /**
   * @brief Apply schedule on an Op.
   * @param op_impl The Op implementation defining Compute and Schedule.
   * @param op_func_arg_tensors Tensors used as the Op function arguments.
   * @param lowered_funcs The lowered funcs of an Op to be scheduled.
   * @return The lowered func body after schedule of the Op.
   */
  ir::Expr DoOpSchedule(std::shared_ptr<hlir::framework::OpImpl> op_impl,
                        const std::vector<ir::Tensor>& op_func_arg_tensors,
                        const std::vector<ir::LoweredFunc>& lowered_funcs);

  // Functions used to determine which Ops to schedule at op level, define a
  // policy for each type of group.
  inline bool ReduceScheduleDetermineFunction(::pir::Operation* op);
  inline bool ElementwiseScheduleDetermineFunction(::pir::Operation* op);
  inline bool NonFusibleScheduleDetermineFunction(::pir::Operation* op);

 private:
  Target target_;
};

}  // namespace newir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
