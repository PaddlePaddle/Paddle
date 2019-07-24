// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/ngraph_subgraph_pass.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"
#include "paddle/fluid/operators/ngraph/ngraph_bridge.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

namespace ANAT = paddle::inference::analysis;

std::string GenerateEngineKey(const std::set<std::string> &engine_inputs,
                              const std::set<std::string> &engine_outputs,
                              const std::string &size) {
  std::string engine_hash_key = "";
  for (auto name : engine_inputs) {
    engine_hash_key += name;
  }
  for (auto name : engine_outputs) {
    engine_hash_key += name;
  }
  engine_hash_key += size;
  auto engine_key = std::to_string(std::hash<std::string>()(engine_hash_key));
  return engine_key;
}

void NgraphSubgraphPass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE(graph);
  FusePassBase::Init("ngraph_subgraph_pass", graph);

  std::unordered_set<Node *> nodes2delete;

  auto teller = [](const Node *node) {
    if (!node->IsOp() || !node->Op()) return false;
    auto op_type = node->Op()->Type();
    return !paddle::operators::NgraphBridge::isRegister(op_type);
  };

  ANAT::SubGraphFuser fuser(graph, teller, 0, "ngraph_engine");
  fuser();

  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !ANAT::Agent(node).subgraph()->empty()) {
      OpDesc *op_desc = node->Op();
      op_desc->SetType("ngraph_engine");
      for (auto it = ANAT::Agent(node).subgraph()->begin();
           it != ANAT::Agent(node).subgraph()->end(); ++it) {
      }

      CreateNgraphEngineOp(node, graph);

      std::unordered_set<const Node *> nodes2remove(
          ANAT::Agent(node).subgraph()->begin(),
          ANAT::Agent(node).subgraph()->end());
      GraphSafeRemoveNodes(graph, nodes2remove);
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && ANAT::Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
  std::vector<ir::Node *> nodes = ir::TopologySortOperations(*graph);
}

void NgraphSubgraphPass::CreateNgraphEngineOp(framework::ir::Node *node,
                                              Graph *graph) const {
  auto *op_desc = node->Op();
  auto &subgraph = *ANAT::Agent(node).subgraph();
  PADDLE_ENFORCE(!subgraph.empty());

  framework::ProgramDesc *program_desc =
      Get<framework::ProgramDesc *>("program");
  const framework::BlockDesc &main_block =
      program_desc->Block(framework::kRootBlockIndex);
  framework::BlockDesc *new_block = program_desc->AppendBlock(main_block);

  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  for (auto *node : subgraph) {
    auto *new_block_op = new_block->AppendOp();
    auto *op = block_desc.AppendOp();
    *new_block_op->Proto() = *node->Op()->Proto();
    *op->Proto() = *node->Op()->Proto();
  }

  std::set<std::string> input_names;
  std::set<std::string> input_names_with_id;
  for (auto *x : node->inputs) {
    input_names.insert(x->Name());
    input_names_with_id.insert(x->Name() + std::to_string(x->id()));
  }
  op_desc->SetInput(
      "Xs", std::vector<std::string>(input_names.begin(), input_names.end()));

  std::set<std::string> output_names;
  std::set<std::string> output_names_with_id;

  for (auto *x : node->outputs) {
    output_names.insert(x->Name());
    output_names_with_id.insert(x->Name() + std::to_string(x->id()));
  }
  op_desc->SetOutput(
      "Ys", std::vector<std::string>(output_names.begin(), output_names.end()));
  auto *vars = block_desc.Proto()->mutable_vars();
  for (framework::ir::Node *node : graph->Nodes()) {
    if (node->IsVar() && node->Var()) {
      *vars->Add() = *node->Var()->Proto();
    }
  }

  PADDLE_ENFORCE(!block_desc.Proto()->vars().empty(),
                 "the block has no var-desc");

  op_desc->SetType("ngraph_engine");

  int sgs = subgraph.size();
  std::string engine_key = GenerateEngineKey(
      input_names_with_id, output_names_with_id, std::to_string(sgs));
  std::vector<int> interval{0, sgs};
  op_desc->SetAttr("interval", interval);
  op_desc->SetAttr("graph", block_desc.Proto()->SerializeAsString());
  op_desc->SetAttr("engine_key", engine_key);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(ngraph_subgraph_pass, paddle::framework::ir::NgraphSubgraphPass);
