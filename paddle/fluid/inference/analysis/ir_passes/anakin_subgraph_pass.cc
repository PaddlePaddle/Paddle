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

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/op_teller.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/ir_passes/anakin_subgraph_pass.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Node;

void analysis::AnakinSubgraphPass::ApplyImpl(
    framework::ir::Graph *graph) const {
  framework::ir::FusePassBase::Init("anakin_subgraph_pass", graph);

  auto teller = [](const framework::ir::Node *node) {
    if (!node->IsOp() || !node->Op()) return false;
    return anakin::OpTeller::Global().Tell(node->Op()->Type(), *node->Op());
  };

  SubGraphFuser fuser(graph, teller, 6 /* min_subgraph_size */);
  fuser();

  std::vector<std::string> graph_param_names =
      ExtractParameters(graph->Nodes());

  // those parameter already exist in anakin, and should not have another copy
  // in fluid.
  std::vector<std::string> repetitive_params;

  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !Agent(node).subgraph()->empty()) {
      CreateAnakinOp(node, graph, graph_param_names, &repetitive_params);
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

std::string GenerateAnakinEngineKey(const std::set<std::string> &engine_inputs,
                                    const std::set<std::string> &engine_outputs,
                                    std::string id) {
  std::string engine_hash_key = "";
  for (auto name : engine_inputs) {
    engine_hash_key += name;
  }
  for (auto name : engine_outputs) {
    engine_hash_key += name;
  }
  engine_hash_key += id;
  auto engine_key = std::to_string(std::hash<std::string>()(engine_hash_key));
  return engine_key;
}

void AnakinSubgraphPass::CreateAnakinOp(
    framework::ir::Node *node, Graph *graph,
    const std::vector<std::string> &graph_params,
    std::vector<std::string> *repetitive_params) const {
  auto *op_desc = node->Op();
  auto &subgraph = *Agent(node).subgraph();
  PADDLE_ENFORCE(!subgraph.empty());

  framework::ProgramDesc *program_desc =
      Get<framework::ProgramDesc *>("program");
  // Add new block for TensorRTEngineOP
  const framework::BlockDesc &main_block =
      program_desc->Block(framework::kRootBlockIndex);
  // const framework::BlockDesc& main_block = program_desc->Block(0);
  framework::BlockDesc *new_block = program_desc->AppendBlock(main_block);

  // An fake block desc.
  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  string::PrettyLogDetail("---  detect a sub-graph with %d nodes",
                          subgraph.size());

  for (auto *node : subgraph) {
    auto *new_block_op = new_block->AppendOp();
    auto *op = block_desc.AppendOp();
    *new_block_op->Proto() = *node->Op()->Proto();
    *op->Proto() = *node->Op()->Proto();
  }

  // Then, we will use the input_names_with_id and output_names_with_id to
  // generate the eigine key.
  // So, We use set instead of unordered_set here to ensure that the engine key
  // is unique.
  std::set<std::string> input_names;
  std::set<std::string> input_names_with_id;
  std::vector<std::string> params;
  for (auto *x : node->inputs) {
    input_names.insert(x->Name());
    input_names_with_id.insert(x->Name() + std::to_string(x->id()));
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0) {
      params.push_back(x->Name());
    }
  }
  std::copy(params.begin(), params.end(),
            std::back_inserter(*repetitive_params));
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
  op_desc->SetType("anakin_engine");

  std::unordered_map<std::string, std::string> output_name_map;
  auto &subgraph_nodes = *Agent(node).subgraph();

  // The following procedure is used to rename all the intermediate
  // variables and the output variables of the subgraph.
  RenameAndGetOutputs(subgraph_nodes, &block_desc, input_names_with_id,
                      &output_names_with_id, &output_names, &output_name_map,
                      false);

  // When anakin engine runs at the end of the operation,
  // output_mapping help us copy the data from the renamed ITensor
  // to Tensor.
  std::vector<std::string> output_mapping;
  for (auto name : output_names) {
    PADDLE_ENFORCE(output_name_map.count(name) != 0);
    output_mapping.push_back(output_name_map[name]);
  }

  auto *vars = block_desc.Proto()->mutable_vars();
  for (framework::ir::Node *node : graph->Nodes()) {
    if (node->IsVar() && node->Var()) {
      *vars->Add() = *node->Var()->Proto();
    }
  }

  PADDLE_ENFORCE(!block_desc.Proto()->vars().empty(),
                 "the block has no var-desc");
  PADDLE_ENFORCE(!output_mapping.empty());
  op_desc->SetBlockAttr("sub_block", new_block);
  SetAttr(op_desc->Proto(), "subgraph",
          block_desc.Proto()->SerializeAsString());
  // Set attrs
  SetAttr(op_desc->Proto(), "parameters", params);
  SetAttr(op_desc->Proto(), "output_name_mapping", output_mapping);
  int predictor_id = Get<int>("predictor_id");
  auto engine_key = GenerateAnakinEngineKey(
      input_names_with_id, output_names_with_id, std::to_string(predictor_id));

  SetAttr(op_desc->Proto(), "engine_key", engine_key);
  auto max_input_shape =
      Get<std::map<std::string, std::vector<int>>>("max_input_shape");
  auto max_batch_size = Get<int>("max_batch_size");

  auto *anakin_engine =
      inference::Singleton<
          anakin::AnakinEngineManager<::anakin::saber::NV>>::Global()
          .Create(true, Get<int>("gpu_device_id"), max_batch_size,
                  max_input_shape, engine_key);

  auto *scope = param_scope();
  std::unordered_set<std::string> param_set(params.begin(), params.end());
  framework::BlockDesc block_desc_temp(nullptr, block_desc.Proto());

  inference::Singleton<inference::anakin::AnakinOpConverter>::Global()
      .ConvertBlockToAnakinEngine(
          &block_desc_temp, scope,
          std::vector<std::string>(input_names.begin(), input_names.end()),
          param_set, output_mapping, anakin_engine);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(anakin_subgraph_pass,
              paddle::inference::analysis::AnakinSubgraphPass);
