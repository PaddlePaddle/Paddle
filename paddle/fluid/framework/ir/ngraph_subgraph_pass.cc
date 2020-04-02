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
#include "paddle/fluid/framework/ir/subgraph_detector.h"
#include "paddle/fluid/operators/ngraph/ngraph_bridge.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

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

void NgraphSubgraphPass::ApplyImpl(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph);
  FusePassBase::Init("ngraph_subgraph_pass", graph);

  std::unordered_set<Node *> nodes2delete;

  auto teller = [](const Node *node) {
    if (!node->IsOp() || !node->Op()) return false;
    auto op_type = node->Op()->Type();
    return !paddle::operators::NgraphBridge::isRegister(op_type);
  };

  SubGraphFuser fuser(graph, teller, 0, "ngraph_engine");
  fuser();

  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !Agent(node).subgraph()->empty()) {
      OpDesc *op_desc = node->Op();
      op_desc->SetType("ngraph_engine");

      CreateNgraphEngineOp(node, graph);

      std::unordered_set<const Node *> nodes2remove(
          Agent(node).subgraph()->begin(), Agent(node).subgraph()->end());

      GraphSafeRemoveNodes(graph, nodes2remove);
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }

  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
  // std::vector<ir::Node *> nodes = ir::TopologySortOperations(*graph);
}

bool IsValid(std::string name) {
  return name.find(Node::kControlDepVarName) == std::string::npos;
}

void UpdateNgraphIO(Node *node, Graph *graph,
                    std::vector<std::string> *input_names,
                    std::vector<std::string> *output_names) {
  bool is_test = true, has_fetch = false;
  for (Node *node : graph->Nodes()) {
    if (node->IsOp() && node->Name().find("_grad") != std::string::npos) {
      is_test = false;
    }
    if (node->IsVar() && node->Var()) {
      for (auto out : node->outputs) {
        if (out->Name() == "fetch") has_fetch = true;
      }
    }
  }
  if (is_test && has_fetch) {
    for (auto *x : node->inputs) {
      (*input_names).emplace_back(x->Name());
    }
    for (auto *x : node->outputs) {
      (*output_names).emplace_back(x->Name());
    }
    return;
  }

  auto &subgraph = *Agent(node).subgraph();
  std::unordered_set<std::string> inputs;
  std::unordered_set<std::string> outputs;
  for (auto *node : subgraph) {
    for (auto in : node->inputs) {
      auto name = in->Name();
      if (!IsValid(name)) continue;
      if (!outputs.count(name) && !inputs.count(name)) {
        (*input_names).emplace_back(name);
        inputs.insert(name);
      }
    }
    for (auto out : node->outputs) {
      auto name = out->Name();
      if (!IsValid(name)) continue;
      outputs.insert(name);
      (*output_names).emplace_back(name);
    }
  }
}

void NgraphSubgraphPass::CreateNgraphEngineOp(Node *node, Graph *graph) const {
  auto &subgraph = *Agent(node).subgraph();
  PADDLE_ENFORCE_NE(subgraph.empty(), true, "subgraph cannot be empty");

  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  for (auto *node : subgraph) {
    auto *op = block_desc.AppendOp();
    *op->Proto() = *node->Op()->Proto();
  }
  auto *vars = block_desc.Proto()->mutable_vars();
  for (Node *node : graph->Nodes()) {
    if (node->IsVar() && node->Var()) {
      *vars->Add() = *node->Var()->Proto();
    }
  }
  PADDLE_ENFORCE_NE(block_desc.Proto()->vars().empty(), true,
                    "the block has no var-desc");

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  UpdateNgraphIO(node, graph, &input_names, &output_names);
  auto *op_desc = node->Op();
  op_desc->SetInput(
      "Xs", std::vector<std::string>(input_names.begin(), input_names.end()));
  op_desc->SetOutput(
      "Ys", std::vector<std::string>(output_names.begin(), output_names.end()));

  int sgs = subgraph.size();
  std::string subgraph_str = block_desc.Proto()->SerializeAsString();
  std::string engine_key =
      std::to_string(std::hash<std::string>()(subgraph_str));
  std::vector<int> interval{0, sgs};
  op_desc->SetType("ngraph_engine");
  op_desc->SetAttr("interval", interval);
  op_desc->SetAttr("graph", subgraph_str);
  op_desc->SetAttr("engine_key", engine_key);
  op_desc->SetAttr("op_role", 0);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(ngraph_subgraph_pass, paddle::framework::ir::NgraphSubgraphPass);
