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

#include "paddle/cinn/auto_schedule/measure/simple_builder.h"

namespace cinn {
namespace auto_schedule {

using hlir::framework::CompilationContext;
using hlir::framework::CompilationResult;
using hlir::framework::GraphCompiler;

SimpleBuilder::SimpleBuilder(hlir::framework::GraphCompiler* graph_compiler)
    : graph_compiler_(graph_compiler) {}

BuildResult SimpleBuilder::Build(const MeasureInput& input) {
  CHECK_NE(graph_compiler_, static_cast<GraphCompiler*>(nullptr))
      << "empty handle to GraphCompiler";
  CompilationContext& context = graph_compiler_->GetCompilationContext();
  context.groups.emplace_back(input.task->subgraph);
  context.lowered_funcs.emplace_back(input.lowered_funcs);
  context.remove_unused_variables = false;
  VLOG(5) << "call GraphCompiler to Build with Graph::Group size="
          << context.groups.size()
          << ", lowered_funcs group size=" << context.lowered_funcs.size();
  CompilationResult compiled_result = graph_compiler_->Build(&context);

  BuildResult build_result;
  build_result.compiled_scope = graph_compiler_->GetScope().get();
  build_result.runtime_program = std::move(compiled_result.RuntimeProgram());
  return build_result;
}

}  // namespace auto_schedule
}  // namespace cinn
