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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/auto_schedule/task/tune_task.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace auto_schedule {

// The input to a measurer
struct MeasureInput {
  // The task object related to this measurement.
  const TuneTask* task;
  // lowered Exprs to be measured
  std::vector<ir::LoweredFunc> lowered_funcs;
  // It is used to pass for some arguments that maybe
  // specified value in advance. default is null
  const std::map<std::string, cinn_pod_value_t>* execution_args = nullptr;
};

// The result of a measurement
struct MeasureResult {
  // The time cost of execution in average of running
  // with a specific repeated times.
  double execution_cost = 0.0;  // unit: us
  // The time cost of the whole measurement process including
  // building and running
  double elapsed_time = 0.0;  // unit: us
  // used to return detail messages once an error occurred during measurement,
  // empty if nothing goes wrong
  std::string error_msg;
};

// The result of building with input schedule
struct BuildResult {
  // The scope that owns detail compilation infos of parameters in the runtime
  // program
  const hlir::framework::Scope* compiled_scope;
  // The executable program
  std::unique_ptr<hlir::framework::Program> runtime_program;
};

// This interface defines how to generate executable objects
// with input schedule. A builder should not contain stateful data
// related to any task so it can be called parallelly among multiple
// processes of task tuning.
class ScheduleBuilder {
 public:
  virtual BuildResult Build(const MeasureInput& input) = 0;
};

// This interface defines how to run the built result. Like above
// ScheduleBuilder, a runner shoule be implemented with not bound to a specific
// task.
class ScheduleRunner {
 public:
  virtual MeasureResult Run(const MeasureInput& input,
                            const BuildResult& build_result) = 0;
};

}  // namespace auto_schedule
}  // namespace cinn
