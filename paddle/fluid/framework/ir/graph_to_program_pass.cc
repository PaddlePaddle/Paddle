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

#include <gflags/gflags.h>
#include <algorithm>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_proto_maker.h"

DECLARE_bool(convert_all_blocks);

namespace paddle {
namespace framework {
class ProgramDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

void GraphToProgramPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_EQ(graph->IsMainGraph(), true,
                    platform::errors::InvalidArgument(
                        "This graph is a sub_graph, "
                        "and can't convert to program individually"));

  ProgramDesc& program = Get<ProgramDesc>("program");

  std::unique_ptr<proto::ProgramDesc> program_pb(
      new proto::ProgramDesc(*program.Proto()));

  auto block = program_pb->mutable_blocks(kRootBlockIndex);
  block->set_idx(kRootBlockIndex);

  if (FLAGS_convert_all_blocks) {
    GraphToBlock(graph->GetSubGraph(kRootBlockIndex), block);

    VLOG(3) << "Graph to program need convert " << graph->SubGraphsSize()
            << " sub graph";
    for (size_t idx = 0; idx < graph->SubGraphsSize(); ++idx) {
      // avoid kRootBlockIndex not 0
      if (idx == kRootBlockIndex) continue;

      block = program_pb->add_blocks();
      block->set_idx(idx);
      GraphToBlock(graph->GetSubGraph(idx), block);
    }
  } else {
    GraphToBlock(graph, block);
  }

  program.CopyFrom(*program_pb);
}

OpDesc* ReplaceScaleLossGradOp(ir::Node* node, OpDesc* desc) {
  desc->SetType("fill_constant");
  desc->SetAttr(
      OpProtoAndCheckerMaker::OpRoleAttrName(),
      (static_cast<int>(OpRole::kBackward) | static_cast<int>(OpRole::kLoss)));
  desc->SetAttr("value", 1.0f);
  std::vector<std::string> output_names;
  for (auto out : node->outputs) {
    output_names.emplace_back(out->Name());
  }
  desc->SetOutput("Out", output_names);
  return desc;
}

std::vector<OpDesc>* GetGraphOpDesc(const std::vector<ir::Node*>& nodes,
                                    std::vector<OpDesc>* ops) {
  for (ir::Node* n : nodes) {
    // if node is not Op, skip
    if (!n->IsOp()) continue;

    // create fill_constant op
    if (n->Name() == "scale_loss_grad") {
      ops->emplace_back();
      auto& desc = ops->back();
      ReplaceScaleLossGradOp(n, &desc);
    } else if (n->Op()) {
      ops->emplace_back(*n->Op());
    } else {
      // delete no OpDesc op
    }
  }
  return ops;
}

void GraphToProgramPass::GraphToBlock(const Graph* graph,
                                      proto::BlockDesc* block) const {
  // Remove the unneeded variables after memory optimization.
  std::unordered_set<std::string> vars2remove;
  if (graph->Has(kGraphToProgramVarsToRemove)) {
    vars2remove = graph->Get<std::unordered_set<std::string>>(
        kGraphToProgramVarsToRemove);
    VLOG(2) << "graph (id: " << block->idx() << ") to program remove "
            << vars2remove.size() << " nodes";
  }

  block->clear_vars();
  std::unordered_set<std::string> visited_vars;
  for (ir::Node* n : graph->Nodes()) {
    if (n->IsVar()) {
      if (n->Var() && visited_vars.count(n->Var()->Name()) == 0 &&
          !vars2remove.count(n->Var()->Name())) {
        visited_vars.insert(n->Var()->Name());
        block->add_vars()->MergeFrom(*n->Var()->Proto());
      }
    }
  }
  block->clear_ops();

  std::vector<ir::Node*> nodes;
  if (Has(kGraphToProgramSortKind)) {
    // Inference Memory Optimize relays on this branch.
    int sort_kind = Get<int>(kGraphToProgramSortKind);
    nodes = TopologyVarientSort(
        *graph, static_cast<framework::ir::SortKind>(sort_kind));
  } else {
    if (FLAGS_convert_all_blocks) {
      nodes = TopologySortGraphByDescOrder(*graph);
    } else {
      nodes = TopologySortOperations(*graph);
    }
  }

  std::vector<OpDesc> ops;
  GetGraphOpDesc(nodes, &ops);
  for (auto& op : ops) {
    block->add_ops()->MergeFrom(*op.Proto());
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(graph_to_program_pass, paddle::framework::ir::GraphToProgramPass);
