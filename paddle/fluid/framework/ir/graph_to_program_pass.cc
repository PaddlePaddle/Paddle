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

#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<Graph> GraphToProgramPass::ApplyImpl(
    std::unique_ptr<Graph> graph) const {
  ProgramDesc& program = Get<ProgramDesc>("program");

  std::unique_ptr<proto::ProgramDesc> program_pb(
      new proto::ProgramDesc(*program.Proto()));

  auto block = program_pb->mutable_blocks(kRootBlockIndex);
  block->set_idx(kRootBlockIndex);
  block->clear_vars();
  std::unordered_set<std::string> visited_vars;
  for (ir::Node* n : graph->Nodes()) {
    if (n->IsVar()) {
      if (n->Var() && visited_vars.count(n->Var()->Name()) == 0) {
        visited_vars.insert(n->Var()->Name());
        block->add_vars()->MergeFrom(*n->Var()->Proto());
      }
    }
  }

  block->clear_ops();
  std::vector<ir::Node*> nodes = TopologySortOperations(*graph);
  for (ir::Node* n : nodes) {
    if (!n->Op()) {
      continue;
    }
    block->add_ops()->MergeFrom(*n->Op()->Proto());
  }

  program.CopyFrom(*program_pb);
  return graph;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(graph_to_program_pass, paddle::framework::ir::GraphToProgramPass);
