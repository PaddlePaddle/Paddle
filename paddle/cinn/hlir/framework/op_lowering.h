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

#include <string>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/ir_schedule_util.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/packed_func.h"

// Fusion Op lowering, there are four kinds of lowering function:
// Elementwise/Broadcast/Injective,Reduce,OutEWiseFusable,NonFusible.
// Elementwise/Broadcast/Injective Ops is with same shcedule.
// Reduce,OutEWiseFusable,NonFusible are using different schedule.

namespace cinn {
namespace hlir {
namespace framework {

using GroupPtr = std::shared_ptr<Graph::Group>;
using common::Target;

class OpLowerer;
typedef std::vector<Expr> (OpLowerer::*IRComputeFunction)(
    poly::StageMap&,
    std::vector<ir::Tensor>&,
    std::unordered_map<std::string, ir::Tensor>&,
    const GroupPtr&,
    const GroupPtr&,
    bool);
typedef void (OpLowerer::*IRScheduleFunction)(
    ir::IRSchedule& ir_sch,
    std::unordered_map<std::string, ir::Tensor>&,
    const GroupPtr&,
    const GroupPtr&,
    Node*&,
    Node*&);

class OpLowerer {
 public:
  OpLowerer(const absl::flat_hash_map<std::string, Type>&,
            const absl::flat_hash_map<std::string, shape_t>&,
            const Target&);
  std::vector<ir::LoweredFunc> Lower(GroupPtr& group);
  std::vector<ir::LoweredFunc> LowerWithoutSchedule(GroupPtr& group);

 private:
  std::vector<ir::LoweredFunc> IRLowerOp(IRComputeFunction,
                                         IRScheduleFunction,
                                         GroupPtr&);
  std::vector<ir::LoweredFunc> IRLowerNonFusibleOp(GroupPtr&, bool);
  std::vector<ir::LoweredFunc> IRLowerOpWithoutSchedule(IRComputeFunction,
                                                        GroupPtr&);
#define DEFINE_IR_COMPUTE_SCHDULE(type)                        \
  std::vector<Expr> IR##type##Compute(                         \
      poly::StageMap& stages,                                  \
      std::vector<ir::Tensor>& func_args,                      \
      std::unordered_map<std::string, ir::Tensor>& tensor_map, \
      const GroupPtr& group,                                   \
      const GroupPtr& sub_group,                               \
      bool apply_impl_schedule = false);                       \
  void IR##type##Schedule(                                     \
      ir::IRSchedule& ir_sch,                                  \
      std::unordered_map<std::string, ir::Tensor>& tensor_map, \
      const GroupPtr& group,                                   \
      const GroupPtr& sub_group,                               \
      Node*& first,                                            \
      Node*& second);

  // compute and schedule
  DEFINE_IR_COMPUTE_SCHDULE(Elementwise);
  DEFINE_IR_COMPUTE_SCHDULE(Reduce);
  DEFINE_IR_COMPUTE_SCHDULE(OutEWiseFusable);

  void IRSchedule(
      ir::IRSchedule& ir_sch,
      const GroupPtr& group,
      const std::unordered_map<std::string, ir::Tensor>& tensor_map);

  Target target_;
  const absl::flat_hash_map<std::string, Type>& type_dict_;
  const absl::flat_hash_map<std::string, shape_t>& shape_dict_;

  // fucntion name prefix
  const std::string func_name_prefix = "fn_";
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
