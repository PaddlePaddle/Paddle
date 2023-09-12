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

#include "paddle/cinn/auto_schedule/task/task_registry.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>

#include "paddle/cinn/auto_schedule/task/task_creator.h"
#include "paddle/cinn/auto_schedule/task/tune_task.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/cinn/utils/type_defs.h"

PD_DECLARE_bool(auto_schedule_use_cost_model);

namespace cinn {
namespace auto_schedule {

std::vector<TuneTask> CreateTasks(hlir::framework::Graph* graph,
                                  const common::Target& target) {
  // create tasks
  TaskCreator task_creator;
  std::vector<TuneTask> tasks = task_creator.CreateTuneTaskOpLevel(graph);

  const auto& dtype_dict =
      graph->GetAttrs<absl::flat_hash_map<std::string, common::Type>>(
          "inferdtype");
  const auto& shape_dict = graph->GetAttrs<
      absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");

  auto op_lowerer =
      hlir::framework::CreateOpLowerer(dtype_dict, shape_dict, target);
  for (TuneTask& task : tasks) {
    task.Initialize(shape_dict, dtype_dict, &op_lowerer);
    VLOG(3) << "Add a task with serialized_key:\n" << task.serialized_key;
  }

  return tasks;
}

std::shared_ptr<hlir::framework::Graph> CreateAddProgram(
    const common::Target& target) {
  frontend::NetBuilder builder("test");

  auto a = builder.CreateInput(Float(32), {1, 64, 112, 112}, "A");
  auto b = builder.CreateInput(Float(32), {64}, "B");
  auto c = builder.Add(a, b, 1);

  return std::make_shared<hlir::framework::Graph>(builder.Build(), target);
}

TEST(TestTaskRegistry, basic) {
  FLAGS_auto_schedule_use_cost_model = true;

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::shared_ptr<hlir::framework::Graph> graph = CreateAddProgram(target);
  std::vector<TuneTask> tasks = CreateTasks(graph.get(), target);

  InitialTaskRegistry* task_registry = InitialTaskRegistry::Global();

  std::vector<ir::ModuleExpr> module_exprs;
  for (const TuneTask& task : tasks) {
    module_exprs.emplace_back(task.GetLoweredFuncBodyExprs());
    task_registry->Regist(task.serialized_key, module_exprs.back());
  }

  for (int i = 0; i < tasks.size(); ++i) {
    std::string key = tasks[i].serialized_key;
    VLOG(3) << "serialized_key = " << key;
    ir::ModuleExpr new_expr = task_registry->Get(key)->module_expr;

    ASSERT_EQ(new_expr.GetExprs().size(), module_exprs[i].GetExprs().size());
    for (int j = 0; j < new_expr.GetExprs().size(); ++j) {
      VLOG(3) << "expr " << j << " of task " << key << " : "
              << new_expr.GetExprs().at(j);
      ASSERT_EQ(utils::GetStreamCnt(new_expr.GetExprs().at(j)),
                utils::GetStreamCnt(module_exprs[i].GetExprs().at(j)));
    }
  }

  bool flag = task_registry->Has(tasks[0].serialized_key);
  ASSERT_EQ(flag, true);

  flag = task_registry->Has("not_exist");
  ASSERT_EQ(flag, false);
}

}  // namespace auto_schedule
}  // namespace cinn
