/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/graph_to_program_pass.h"

#include <algorithm>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle::framework {
class ProgramDesc;
}  // namespace paddle::framework

namespace paddle::framework::ir {

void GraphToProgramPass::ApplyImpl(ir::Graph* graph) const {
  auto& program = Get<ProgramDesc>("program");
  if (Has(kGraphToProgramSortKind)) {
    auto sort_kind = static_cast<SortKind>(Get<int>(kGraphToProgramSortKind));
    GraphToProgram(*graph, &program, &sort_kind);
  } else {
    GraphToProgram(*graph, &program, nullptr);
  }
}

}  // namespace paddle::framework::ir

REGISTER_PASS(graph_to_program_pass, paddle::framework::ir::GraphToProgramPass);
