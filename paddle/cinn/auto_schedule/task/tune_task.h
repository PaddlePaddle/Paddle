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

#include <absl/container/flat_hash_map.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class TuneTask {
  using GroupPtr = hlir::framework::GroupPtr;

 public:
  TuneTask() = default;
  explicit TuneTask(GroupPtr group) : subgraph(group) {}
  // Initialize a task
  void Initialize(
      const absl::flat_hash_map<std::string, hlir::framework::shape_t>&
          shape_dict,
      const absl::flat_hash_map<std::string, cinn::common::Type>& dtype_dict,
      hlir::framework::OpLowerer<GroupPtr>* lower_handler);
  // Extract bodies in lowered_funcs() and return
  std::vector<ir::Expr> GetLoweredFuncBodyExprs() const;

  // In CINN, we use hlir::framework::Graph::Group to represent a fused
  // sub-graph (if an op won't be fused, it will be a Group with size=1).
  std::shared_ptr<hlir::framework::Graph::Group> subgraph;
  // Lower handler, Not owned
  hlir::framework::OpLowerer<GroupPtr>* op_lowerer;
  // target of this task
  common::Target target;
  // stores the initial (un-optimized) LoweredFuncs
  std::vector<ir::LoweredFunc> lowered_funcs;
  // names of the output arguments of lowered_funcs_
  std::unordered_set<std::string> output_names;
  // serialized string of this task, it contains struct,shape,dtype,input/output
  // variable name of the subgraph and can be further used to hash
  std::string serialized_key;

 private:
  // Serialize this task as a string contains specific fields of it
  std::string SerializeToString(
      const absl::flat_hash_map<std::string, hlir::framework::shape_t>&
          shape_dict,
      const absl::flat_hash_map<std::string, cinn::common::Type>& dtype_dict);
};

}  // namespace auto_schedule
}  // namespace cinn
