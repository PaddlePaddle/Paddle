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
#include "paddle/cinn/hlir/framework/op_lowering_impl_base.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/pir/include/core/operation.h"

// Fusion Op lowering, there are four kinds of lowering function:
// Elementwise/Broadcast/Injective,Reduce,OutEWiseFusible,NonFusible.
// Elementwise/Broadcast/Injective Ops is with same schedule.
// Reduce,OutEWiseFusible,NonFusible are using different schedule.

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

class PrettyNamer;
using OpLoweringGroupPtr = std::shared_ptr<OpLoweringGroup>;

using cinn::common::Target;
class OpLowererImpl;

typedef bool (OpLowererImpl::*ScheduleDetermineFunction)(::pir::Operation*);

struct GroupInfo {
  std::vector<int64_t> data_space;
  std::vector<int64_t> reduce_axis;
  int64_t raw_data_rank;
  std::vector<int64_t> raw_reduce_axis;
  std::set<std::string> reduce_var_names;
  std::set<std::string> shared_var_names;
  std::set<std::string> direct_output_var_names;
  std::vector<std::string> broadcast_output_names;
};

class OpLowererImpl : public OpLowererImplBase<OpLoweringGroupPtr> {
 public:
  explicit OpLowererImpl(const Target&);

  /**
   * @brief Lower a group to CINN IR.
   * @param group The group to be lowered.
   * @param apply_op_schedule Whether to schedule at Op level.
   * @param apply_group_schedule Whether to schedule at group level.
   * @return The lowered funcs.
   */
  std::vector<ir::LoweredFunc> Lower(const OpLoweringGroupPtr& group,
                                     bool apply_op_schedule = true,
                                     bool apply_group_schedule = true,
                                     bool apply_pass = true);

  /**
   * @brief Lower a dynamic shape group to CINN IR.
   * @param group The group to be lowered.
   * @param apply_op_schedule Whether to schedule at Op level.
   * @param apply_group_schedule Whether to schedule at group level.
   * @return The lowered funcs.
   */
  BucketLoweredFuncsWrapper BucketLower(const OpLoweringGroupPtr& group,
                                        bool apply_op_schedule = false,
                                        bool apply_group_schedule = true,
                                        bool apply_pass = true);

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
      const OpLoweringGroupPtr& group,
      bool apply_op_schedule,
      bool apply_group_schedule,
      ScheduleDetermineFunction schedule_determine_func);

  /**
   * @brief Lower a group composed of CustomCall Op.
   * @param group The group to be lowered.
   * @return The lowered funcs.
   */
  std::vector<ir::LoweredFunc> LowerCustomCall(const OpLoweringGroupPtr& group);

  /**
   * @brief Post processing, including preparing function args and temporary
   * variables, applying low-level optimization passes, etc.
   * @param group The group to be lowered.
   * @param tensor_map All tensors used for calculating the group.
   * @param done_op_schedule Mark whether the Op level schedule has been
   * applied.
   * @param func_bodies The scheduled func bodies of group.
   * @param group_func_arg_tensors Tensors used as the group function arguments.
   * @param group_func_args Arguments used as the group function arguments.
   * @return The lowered funcs after the post processing.
   */
  std::vector<ir::LoweredFunc> PostProcess(
      const OpLoweringGroupPtr& group,
      const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
      bool done_op_schedule,
      std::vector<ir::Expr> func_bodies,
      std::vector<ir::Tensor>* group_func_arg_tensors,
      std::vector<ir::Argument>* group_func_args,
      std::vector<ir::Tensor>* infer_shape_arg_tensor);

  /**
   * @brief Lower an Op set to CINN IR.
   * Compute and Lower will be performed one by one for each Op.
   * @param group The group to be lowered.
   * @param ops The Op to be lowered.
   * @param group_func_arg_tensors Tensors used as the group function arguments.
   * @param tensor_map All tensors used for calculating the group.
   * @return The lowered func bodies of Op set.
   */
  void LowerOpsForMapExpr(
      const OpLoweringGroupPtr& group,
      const std::vector<::pir::Operation*>& ops,
      std::vector<ir::Tensor>* group_func_arg_tensors,
      std::unordered_map<::pir::Value, ir::Tensor>* tensor_map);

  /**
   * @brief Generate MapExpr and Lower it to std::vector<ir::LoweredFunc>
   * @param group The group to be lowered.
   * @param ops The Op to be lowered.
   * @param apply_op_schedule Whether to schedule at Op level.
   * @param apply_group_schedule Whether to schedule at group level.
   * @param group_func_arg_tensors Tensors used as the group function arguments.
   * @param tensor_map All tensors used for calculating the group.
   * @return The lowered funcs after the post processing.
   */
  std::vector<ir::LoweredFunc> LowerMapExpr(
      const OpLoweringGroupPtr& group,
      const std::vector<::pir::Operation*>& ops,
      bool apply_op_schedule,
      bool apply_group_schedule,
      std::vector<ir::Tensor>* group_func_arg_tensors,
      std::unordered_map<::pir::Value, ir::Tensor>* tensor_map);

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
      const OpLoweringGroupPtr& group,
      const std::vector<::pir::Operation*>& ops,
      bool apply_op_schedule,
      ScheduleDetermineFunction schedule_determine_func,
      std::vector<ir::Tensor>* group_func_arg_tensors,
      std::unordered_map<::pir::Value, ir::Tensor>* tensor_map,
      std::unordered_map<std::string, ir::Tensor>* tmp_tensor_info);

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
      std::unordered_map<std::string, ir::Tensor>* tmp_tensor_info,
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

  /**
   * @brief Apply schedule on a group.
   * @param ir_sch The IRSchedule containing the entire group's lowered func
   * bodies.
   * @param group The group to be scheduled.
   * @param tensor_map All tensors used for calculating the group.
   * @return The lowered func body after schedule of the group.
   */
  ir::Expr DoGroupSchedule(
      ir::IRSchedule& ir_sch,  // NOLINT
      const OpLoweringGroupPtr& group,
      const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
      const std::unordered_map<std::string, ir::Tensor>& tmp_tensor_info);

  /**
   * @brief  Generates the output tensor infer shape function.
   * @param group The group to be lowered.
   * @param group_func_arg_tensors Tensors used as the group function arguments.
   * @param group_func_args Arguments used as the group function arguments.
   * @return The lowered func to infer output tensor's shape.
   */
  ir::LoweredFunc GenerateInferShapeFunc(
      const OpLoweringGroupPtr& group,
      const std::vector<ir::Tensor> group_func_arg_tensors,
      const std::vector<ir::Argument> group_func_args);

  // Functions used to determine which Ops to schedule at op level, define a
  // policy for each type of group.
  inline bool ReduceScheduleDetermineFunction(::pir::Operation* op);
  inline bool ElementwiseScheduleDetermineFunction(::pir::Operation* op);
  inline bool NonFusibleScheduleDetermineFunction(::pir::Operation* op);
  inline bool DyShapeScheduleDetermineFunction(::pir::Operation* op);

 private:
  std::vector<ir::Tensor> CollectInputTensor(
      const OpLoweringGroupPtr& group,
      const ::pir::Operation* op,
      std::vector<ir::Tensor>* func_args,
      std::unordered_map<::pir::Value, ir::Tensor>* tensor_map);

  ir::Tensor GetTensor(const OpLoweringGroupPtr& group,
                       const ::pir::Value& value);
  ir::Tensor GetTensorSymbolic(const OpLoweringGroupPtr& group,
                               const ::pir::Value& value);

  std::shared_ptr<GroupInfo> GetGroupInfo(
      const OpLoweringGroupPtr& group,
      const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map);

  std::shared_ptr<GroupInfo> GetGroupInfo(
      const FusionGroupInfo& fusion_group_info,
      const OpLoweringGroupPtr& group,
      const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map);

  void CollectOutputInfo(::pir::Operation* op,
                         std::vector<Type>* out_types,
                         std::vector<std::vector<int>>* out_shapes,
                         const OpLoweringGroupPtr& group);

  void CollectOutputInfo(::pir::Operation* op,
                         std::vector<Type>* out_types,
                         std::vector<std::vector<ir::Dim>>* out_shapes,
                         const OpLoweringGroupPtr& group);

  std::string ValueName(::pir::Value value);

  common::Type GetTensorDtype(
      const std::string& name,
      const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map);

  bool IsInTensorMap(
      const std::string& name,
      const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map);

  common::Type GetTensorDtype(const ::pir::Value& value);

  void BuildBroadcastInfo(const OpLoweringGroupPtr& group,
                          std::shared_ptr<GroupInfo> group_info);
  Target target_;
  ir::Expr LowerX86(const OpLoweringGroupPtr& group,
                    const std::vector<::pir::Operation*>& ops,
                    bool apply_op_schedule);
  PrettyNamer* name_gene_;
};

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
