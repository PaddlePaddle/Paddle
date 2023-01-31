// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/passes/ir_graph_to_program_pass.h"

#include "paddle/fluid/framework/ir/graph_to_program_pass.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace inference {
namespace analysis {

void IrGraphToProgramPass::RunImpl(Argument *argument) {
  auto cache_pass =
      framework::ir::PassRegistry::Instance().Get("runtime_context_cache_pass");
  auto pass =
      framework::ir::PassRegistry::Instance().Get("graph_to_program_pass");

  if (argument->memory_optim_sort_kind_valid()) {
    pass->Set(framework::ir::kGraphToProgramSortKind,
              new int(argument->memory_optim_sort_kind()));
  }

  // Direct using ProgramDesc desc(argument->main_program()) may cause
  // incomplete copies of information.
  framework::ProgramDesc desc;
  desc.CopyFrom(*argument->main_program().Proto());
  pass->SetNotOwned("program", &desc);
  pass->Apply(cache_pass->Apply(argument->main_graph_ptr()));

  argument->SetIrAnalyzedProgram(
      new framework::proto::ProgramDesc(*desc.Proto()));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
