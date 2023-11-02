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

#include "paddle/cinn/auto_schedule/task/tune_task.h"

#include <glog/logging.h>

#include <iostream>
#include <vector>

#include "paddle/cinn/auto_schedule/analysis/analyze_ir.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

void TuneTask::Initialize(
    const absl::flat_hash_map<std::string, hlir::framework::shape_t>&
        shape_dict,
    const absl::flat_hash_map<std::string, cinn::common::Type>& dtype_dict,
    hlir::framework::OpLowerer<GroupPtr>* lower_handler) {
  CHECK(lower_handler != nullptr) << "op_lowerer can't be nullptr";
  op_lowerer = lower_handler;

  // Set lowered_funcs and analyze output names.
  this->lowered_funcs = op_lowerer->Lower(
      subgraph, /*apply_op_schedule = */ false, /*apply_group_schedule=*/false);
  this->output_names = GetOutputNamesFromLoweredFunc(this->lowered_funcs);
  this->serialized_key = SerializeToString(shape_dict, dtype_dict);
}

std::vector<ir::Expr> TuneTask::GetLoweredFuncBodyExprs() const {
  std::vector<ir::Expr> result;
  for (const ir::LoweredFunc& func : lowered_funcs) {
    result.push_back(func->body);
  }
  return result;
}

std::string TuneTask::SerializeToString(
    const absl::flat_hash_map<std::string, hlir::framework::shape_t>&
        shape_dict,
    const absl::flat_hash_map<std::string, cinn::common::Type>& dtype_dict) {
  std::stringstream ss;
  ss << target << "\n\n";  // print target

  // local function to print dtype,shape of out/in variables of the specified
  // node
  auto print_node_links_fn =
      [&](const std::vector<common::Shared<common::GraphEdge>>& links,
          bool is_input) {
        int printed_num = 0;
        for (auto&& edge : links) {
          const auto* var_node =
              is_input ? edge->source()->safe_as<hlir::framework::NodeData>()
                       : edge->sink()->safe_as<hlir::framework::NodeData>();
          CHECK(var_node) << "var node invalid";
          auto sit = shape_dict.find(var_node->id());
          CHECK(sit != shape_dict.end())
              << "can't find shape of variable:" << var_node->id();
          auto dit = dtype_dict.find(var_node->id());
          CHECK(dit != dtype_dict.end())
              << "can't find dtype of variable:" << var_node->id();
          if (printed_num > 0) {
            ss << ", ";
          }
          ++printed_num;
          // TODO(CtfGo): CINN uses the names of input/output NodeData ids as
          // arguments of the LoweredFunc in the Lower process, so it will
          // result in different LoweredFuncs for two Nodes even though they
          // represents the same operator. Here we add `var_node->id()` into the
          // serialized_key to distinguish them, otherwise AutoTuner will get
          // wrong TuningRecords when querying cached results from database.  In
          // the future, we should remove name-related limit in Lower process,
          // to avoid duplicate tuning tasks with same operators.
          ss << var_node->id() << "->" << cinn::common::Type2Str(dit->second)
             << "[" + utils::Join(sit->second, ",") << "]";
        }
      };

  // print each node of the subgraph
  ss << "Group {\n";
  for (auto&& node : subgraph->CollectNodes()) {
    ss << "  (";
    print_node_links_fn(node->outlinks_in_order(), false);
    ss << ") = " << node->op()->name << "(";
    print_node_links_fn(node->inlinks_in_order(), true);
    ss << ")\n";
  }
  ss << "}\n";

  return ss.str();
}

}  // namespace auto_schedule
}  // namespace cinn
