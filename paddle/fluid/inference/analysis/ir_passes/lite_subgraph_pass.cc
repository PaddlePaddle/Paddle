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
#include "google/protobuf/text_format.h"
#include "google/protobuf/message.h"

#include "paddle/fluid/framework/lod_tensor.h"

#include "paddle/fluid/inference/analysis/ir_passes/lite_subgraph_pass.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/string/pretty_log.h"


namespace paddle {
namespace inference {
namespace analysis {

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
    LOG(INFO) << "======== [node ] ========"; 
    LOG(INFO) << "name: " << node->Name();
    for (auto* in: node->inputs) {
      LOG(INFO) << "   inputs: " << in->Name();
    }
    for (auto* out: node->outputs) {
      LOG(INFO) << "   outputs: " << out->Name();
    }
    if (node->IsOp() && !Agent(node).subgraph()->empty()) {
      CreateLiteOp(node, graph, graph_param_names, &repetitive_params);
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

void LiteSubgraphPass::CreateLiteOp(
    framework::ir::Node *node, Graph *graph,
    const std::vector<std::string> &graph_params,
    std::vector<std::string> *repetitive_params) const {
  for (auto param: graph_params) {
    LOG(INFO) << "graph_param: " << param;
  }

  auto &subgraph = *Agent(node).subgraph();
  PADDLE_ENFORCE(!subgraph.empty());

  framework::ProgramDesc *program_desc =
      Get<framework::ProgramDesc *>("program");

  const framework::BlockDesc &global_block =
      program_desc->Block(framework::kRootBlockIndex);
  framework::BlockDesc *sub_block = program_desc->AppendBlock(global_block);

  framework::ProgramDesc engine_program;
  framework::BlockDesc* engine_global_block = engine_program.MutableBlock(framework::kRootBlockIndex);
  string::PrettyLogDetail("---  detect a sub-graph with %d nodes",
                          subgraph.size());

  std::unordered_set<Node *> io_var_nodes = GetRelatedIOVarNodes(subgraph);

  auto serialize_params = [] (std::string* str, framework::Scope* scope,
       const std::vector<std::string>& params) {
    std::ostringstream os;
    platform::CPUDeviceContext ctx;
      //std::ofstream os("param.bin", std::ios::binary);
    for (const auto& param: params) {
      auto* tensor = scope->FindVar(param)->GetMutable<framework::LoDTensor>();
      LOG(INFO) << "SerializeToStream: " << param;
      framework::SerializeToStream(os, *tensor, ctx);
    }
      //os.close();
    *str = os.str();

    /*
    for (const auto& param: params) {
      
      std::ifstream is(param + ".bin", std::ios::in | std::ios::binary);
      auto* tensor = scope->FindVar(param)->GetMutable<framework::LoDTensor>();
      LOG(INFO) << "DeserializeFromStream: " << param;
      framework::DeserializeFromStream(is, tensor, ctx);
      is.close();
      
    }
    */
  };

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

  auto target_names = [](const std::vector<framework::ir::Node*>& nodes) {
    std::vector<std::string> names;
    for (const auto& node: nodes) {
      if (node->IsVar() && !node->Var()->Persistable()) { 
        names.push_back(node->Name());
      }
    }
    return names;
  };

  const std::vector<std::string> input_names = target_names(node->inputs);
  const std::vector<std::string> output_names = target_names(node->outputs);

  PrependFeedOps(engine_global_block, input_names);
  PrependFetchOps(engine_global_block, output_names);

  auto *op_desc = node->Op();
  op_desc->SetInput("Xs", input_names);
  op_desc->SetOutput("Ys", output_names);
  op_desc->SetType("lite_engine");
  op_desc->SetAttr("engine_key", std::string());

  auto *scope = param_scope();
  std::string param_string;
  *repetitive_params = ExtractParameters(io_var_nodes);
  serialize_params(&param_string, scope, *repetitive_params);
  // StrToBinaryFile("./param.bin", param_string);

  std::string model_string = engine_program.Proto()->SerializeAsString();
  // StrToBinaryFile("./model.bin", model_string);

  std::string str;
  google::protobuf::TextFormat::PrintToString(*(engine_program.Proto()), &str);
  std::cout << "=====" << std::endl;
  std::cout << str;

}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(lite_subgraph_pass,
              paddle::inference::analysis::LiteSubgraphPass);
