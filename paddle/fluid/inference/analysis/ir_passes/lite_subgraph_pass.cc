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

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <iostream>
#include <fstream>

#include "paddle/fluid/inference/lite/op_teller.h"
#include "paddle/fluid/framework/lod_tensor.h"

#include "paddle/fluid/inference/analysis/ir_passes/lite_subgraph_pass.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/string/pretty_log.h"

#include "paddle/fluid/inference/lite/engine.h"

namespace paddle {
namespace inference {
namespace analysis {

std::vector<std::string> IOVarsFilter(const std::vector<framework::ir::Node*>& nodes) {
  std::vector<std::string> names;
  for (const auto& node: nodes) {
    if (node->IsVar() && !node->Var()->Persistable()) { 
      names.push_back(node->Name());
    }
  }
  return names;
}

void StrToBinaryFile(const std::string& path, const std::string& str) {
  std::ofstream file(path.c_str(), std::ios::binary);
  file.write(str.c_str(), str.size());
  file.close();
}

void LiteSubgraphPass::ApplyImpl(
    framework::ir::Graph *graph) const {

  framework::ir::FusePassBase::Init("lite_subgraph_pass", graph);

  // auto &lite_ops_filter = Get<std::vector<std::string>>("lite_ops_filter");
  std::vector<std::string> lite_ops_filter = {};

  auto teller = [&lite_ops_filter](const framework::ir::Node *node) {
    if (!node->IsOp() || !node->Op())
      return false;
    else if (std::find(lite_ops_filter.begin(), lite_ops_filter.end(),
                       node->Op()->Type()) != lite_ops_filter.end())
      return false;
    return lite::OpTeller::Global().Tell(node->Op()->Type(), *node->Op());
  };

  SubGraphFuser fuser(graph, teller, 0 /* min_subgraph_size */, "lite_engine");
  fuser();

  std::vector<std::string> graph_param_names =
      ExtractParameters(graph->Nodes());

  // those parameter already exist in anakin, and should not have another copy
  // in fluid.
  std::vector<std::string> repetitive_params;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !Agent(node).subgraph()->empty()) {
      CreateOp(node, graph, graph_param_names, &repetitive_params);
      std::unordered_set<const Node *> nodes2remove(
          Agent(node).subgraph()->begin(), Agent(node).subgraph()->end());
      framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
  graph->Set(framework::ir::kRepetitiveParamAttr,
             new std::vector<std::string>(repetitive_params));
}

void LiteSubgraphPass::CreateOp(
    framework::ir::Node *node, Graph *graph,
    const std::vector<std::string> &graph_params,
    std::vector<std::string> *repetitive_params) const {
  framework::ProgramDesc *global_program = Get<framework::ProgramDesc *>("program");
  framework::ProgramDesc engine_program;

  AppendBlocks(node, global_program, &engine_program, repetitive_params);
  CreateEngine(&engine_program, repetitive_params);

  auto *op_desc = node->Op();
  op_desc->SetInput("Xs", IOVarsFilter(node->inputs));
  op_desc->SetOutput("Ys", IOVarsFilter(node->outputs));
  op_desc->SetType("lite_engine");
  op_desc->SetAttr("engine_key", std::string());
}

void LiteSubgraphPass::AppendBlocks(framework::ir::Node *node,
  framework::ProgramDesc* global_program,
  framework::ProgramDesc* engine_program,
  std::vector<std::string> *repetitive_params) const {
    auto &subgraph = *Agent(node).subgraph();
  PADDLE_ENFORCE(!subgraph.empty());
  const framework::BlockDesc &global_block = global_program->Block(framework::kRootBlockIndex);
  framework::BlockDesc *sub_block = global_program->AppendBlock(global_block);

  framework::BlockDesc* engine_global_block = engine_program->MutableBlock(framework::kRootBlockIndex);
  string::PrettyLogDetail("---  detect a sub-graph with %d nodes",
                          subgraph.size());

  std::unordered_set<Node *> io_var_nodes = GetRelatedIOVarNodes(subgraph);
  *repetitive_params = ExtractParameters(io_var_nodes);
  for (auto *var_node: io_var_nodes) {
    auto *sub_block_var = sub_block->Var(var_node->Name());
    auto *var = engine_global_block->Var(var_node->Name());
    *sub_block_var->Proto() = *var_node->Var()->Proto();
    *var->Proto() = *var_node->Var()->Proto();
  }
  for (auto *op_node : subgraph) {
    auto *sub_block_op = sub_block->AppendOp();
    auto *op = engine_global_block->AppendOp();
    *sub_block_op->Proto() = *op_node->Op()->Proto();
    *op->Proto() = *op_node->Op()->Proto();
  }
  PrependFeedOps(engine_global_block, IOVarsFilter(node->inputs));
  PrependFetchOps(engine_global_block, IOVarsFilter(node->outputs));
}

void LiteSubgraphPass::CreateEngine(framework::ProgramDesc* program,
  std::vector<std::string> *repetitive_params) const {
  inference::lite::EngineConfig config;
  auto *scope = param_scope();

  auto serialize_params = [] (std::string* str, framework::Scope* scope,
       const std::vector<std::string>& params) {
    std::ostringstream os;
    platform::CPUDeviceContext ctx;
    for (const auto& param: params) {
      auto* tensor = scope->FindVar(param)->GetMutable<framework::LoDTensor>();
      LOG(INFO) << "SerializeToStream: " << param;
      framework::SerializeToStream(os, *tensor, ctx);
    }
    *str = os.str();
  };

  serialize_params(&config.param, scope, *repetitive_params);
  config.model = program->Proto()->SerializeAsString();
  config.prefer_place = paddle::lite::Place({TARGET(kCUDA), PRECISION(kFloat)});
  config.valid_places = {
      paddle::lite::Place({TARGET(kHost), PRECISION(kFloat)}),
      paddle::lite::Place({TARGET(kCUDA), PRECISION(kFloat)}),
  };
  inference::Singleton<inference::lite::EngineManager>::Global()
      .Create("engine_key", config);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(lite_subgraph_pass,
              paddle::inference::analysis::LiteSubgraphPass);
