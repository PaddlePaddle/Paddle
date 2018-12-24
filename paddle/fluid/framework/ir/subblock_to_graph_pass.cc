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

#include "paddle/fluid/framework/ir/subblock_to_graph_pass.h"
#include "paddle/fluid/framework/ir/infer_clean_graph_pass.h"

namespace paddle {
namespace framework {
namespace ir {

void TransformSubblockToProgram(framework::BlockDesc* block_desc,
                                proto::ProgramDesc* program_desc) {
  *program_desc->add_blocks() = *block_desc->Proto();
}

std::unique_ptr<ir::Graph> SubblockToGraphPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  // Init sub graph
  if (!graph->Has(kSubblockGraphAttr)) {
    graph->Set(kSubblockGraphAttr, new subgraphs_t);
  }

  auto& sub_graphs = graph->Get<subgraphs_t>(kSubblockGraphAttr);

  InferCleanGraphPass clean_pass;

  // Filter out the nodes that has sub-block
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op() && node->Op()->HasAttr("sub_block")) {
      auto* sub_block_desc =
          boost::get<framework::BlockDesc*>(node->Op()->GetAttr("sub_block"));
      // sub-block to program desc
      for (auto op : sub_block_desc->Proto()->ops()) {
        VLOG(4) << op.type() << " in subblock";
      }

      framework::proto::ProgramDesc fake_proto;
      TransformSubblockToProgram(sub_block_desc, &fake_proto);
      framework::ProgramDesc fake_program_desc(fake_proto);
      // Create a graph
      sub_graphs[node] = std::unique_ptr<Graph>(new Graph(fake_program_desc));
      LOG(INFO) << "get sub-graph size " << sub_graphs[node]->Nodes().size();
      auto ptr = std::move(sub_graphs[node]);
      sub_graphs[node] = clean_pass.Apply(std::move(ptr));
    }
  }
  // Build a graph for the sub-block
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(subblock_to_graph_pass,
              paddle::framework::ir::SubblockToGraphPass);
