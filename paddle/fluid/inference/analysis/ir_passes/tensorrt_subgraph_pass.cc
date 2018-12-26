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
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"
#include "paddle/fluid/inference/analysis/ir_passes/tensorrt_subgraph_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Node;

std::vector<std::string> ExtractParameters(
    const std::unordered_set<Node *> &nodes);

std::unique_ptr<framework::ir::Graph> analysis::TensorRtSubgraphPass::ApplyImpl(

    std::unique_ptr<framework::ir::Graph> graph) const {
  framework::ir::FusePassBase::Init("tensorrt_subgraph_pass", graph.get());

  auto teller =
      Get<SubgraphDetector::NodeInsideSubgraphTeller>("tensorrt_node_teller");

  SubGraphFuser fuser(graph.get(), teller,
                      Get<int>("min_subgraph_size") /*min subgraph size*/);
  fuser();

  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !Agent(node).subgraph()->empty()) {
      CreateTensorRTOp(node, graph.get());

      std::unordered_set<const Node *> nodes2remove(
          Agent(node).subgraph()->begin(), Agent(node).subgraph()->end());
      framework::ir::GraphSafeRemoveNodes(graph.get(), nodes2remove);
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph.get(), nodes2remove);

  return graph;
}

void TensorRtSubgraphPass::CreateTensorRTOp(framework::ir::Node *node,
                                            Graph *graph) const {
  auto *op_desc = node->Op();
  auto &subgraph = *Agent(node).subgraph();
  PADDLE_ENFORCE(!subgraph.empty());

  // An fake block desc.
  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  for (auto *node : subgraph) {
    auto *op = block_desc.AppendOp();
    *op->Proto() = *node->Op()->Proto();
  }

  // collect inputs
  std::unordered_set<std::string> input_names;
  std::unordered_set<std::string> input_names_with_id;
  for (auto *x : node->inputs) {
    input_names.insert(x->Name());
    input_names_with_id.insert(x->Name() + std::to_string(x->id()));
  }
  op_desc->SetInput(
      "Xs", std::vector<std::string>(input_names.begin(), input_names.end()));

  std::unordered_set<std::string> output_names;
  std::unordered_set<std::string> output_names_with_id;
  for (auto *x : node->outputs) {
    output_names.insert(x->Name());
    output_names_with_id.insert(x->Name() + std::to_string(x->id()));
  }

  op_desc->SetOutput(
      "Ys", std::vector<std::string>(output_names.begin(), output_names.end()));
  op_desc->SetType("tensorrt_engine");

  std::unordered_map<std::string, std::string> output_name_map;

  // The following procedure is used to rename all the intermediate
  // variables and the output variables of the subgraph.
  // Why we do this?
  // During the transition from fluid OP to tensorrt OP, we map
  // the input and output Tensor(fluid data structure) of fluid OP
  // to the corresponding ITensor (trt data structure) through the
  // Tensor name. When we set up ITensor for an variable, we must
  // ensure that it has not been set before.
  // If there is variable in the fluid graph, which is not only the
  // input of a OP, but also the output of a Op, there will be problems.
  // So we have to rename the variable in the subgraph to make sure
  // it is either an OP's input or an OP's output.

  auto &subgraph_nodes = *Agent(node).subgraph();
  for (size_t index = 0; index < block_desc.OpSize(); ++index) {
    framework::proto::OpDesc *op = block_desc.Op(index)->Proto();
    auto correspond_node = subgraph_nodes[index];
    PADDLE_ENFORCE_EQ(correspond_node->Name(), op->type());

    std::unordered_map<std::string, size_t> var2id;
    for (auto *in_var : correspond_node->inputs) {
      var2id[in_var->Name()] = in_var->id();
    }
    // rename for the input variables of op inside subgraph
    for (int i = 0; i < op->inputs_size(); i++) {
      // one input
      auto *in_var = op->mutable_inputs(i);
      std::vector<std::string> replaced_names;
      for (int k = 0; k < in_var->arguments_size(); k++) {  // all the arguments
        std::string arg_value = in_var->arguments(k);
        std::string arg_value_with_id =
            arg_value + std::to_string(var2id[arg_value]);
        if (input_names_with_id.count(arg_value_with_id)) {
          replaced_names.push_back(arg_value);
        } else {
          replaced_names.push_back(arg_value_with_id);
        }
      }
      in_var->clear_arguments();
      for (size_t k = 0; k < replaced_names.size(); k++) {
        in_var->add_arguments(replaced_names[k]);
      }
    }
    var2id.clear();
    for (auto out_var : correspond_node->outputs) {
      var2id[out_var->Name()] = out_var->id();
    }

    // rename for the output variables of op inside subgraph
    for (int i = 0; i < op->outputs_size(); i++) {
      framework::proto::OpDesc_Var *out_var = op->mutable_outputs(i);
      std::vector<std::string> replaced_names;
      for (int k = 0; k < out_var->arguments_size(); k++) {
        std::string arg_value = out_var->arguments(k);
        std::string arg_value_with_id =
            arg_value + std::to_string(var2id[arg_value]);
        if (output_names_with_id.count(arg_value_with_id)) {
          output_name_map[arg_value] = arg_value_with_id;
        }
        replaced_names.push_back(arg_value_with_id);
      }
      out_var->clear_arguments();
      for (size_t k = 0; k < replaced_names.size(); k++) {
        out_var->add_arguments(replaced_names[k]);
      }
    }
  }

  // When tensorrt engine runs at the end of the operation,
  // output_mapping help us copy the data from the renamed ITensor
  // to Tensor.
  std::vector<std::string> output_mapping;
  for (auto name : output_names) {
    // LOG(INFO) << name << " " << output_name_map.size();
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
  // Set attrs
  SetAttr(op_desc->Proto(), "subgraph",
          block_desc.Proto()->SerializeAsString());
  SetAttr(op_desc->Proto(), "max_batch_size", Get<int>("max_batch_size"));
  SetAttr(op_desc->Proto(), "workspace_size", Get<int>("workspace_size"));
  SetAttr(op_desc->Proto(), "parameters", ExtractParameters(graph->Nodes()));
  SetAttr(op_desc->Proto(), "output_name_mapping", output_mapping);
}

std::vector<std::string> ExtractParameters(
    const std::unordered_set<Node *> &nodes) {
  // We can judge whether a variable is a parameter by
  // its presistable property, but sometimes the presistable
  // of the feed op output is true, so we have to identify it.
  std::vector<std::string> feed_outputs;
  for (const auto &node : nodes) {
    if (!node->IsOp()) continue;
    std::string op_type = node->Op()->Type();
    if (op_type == "feed") {
      std::vector<std::string> output_names = node->Op()->OutputArgumentNames();
      std::copy(output_names.begin(), output_names.end(),
                std::back_inserter(feed_outputs));
    }
  }

  std::vector<std::string> parameters;
  for (const auto &node : nodes) {
    if (!node->IsVar()) continue;
    if (node->Var()->Persistable() &&
        std::find(feed_outputs.begin(), feed_outputs.end(), node->Name()) ==
            feed_outputs.end()) {
      parameters.push_back(node->Name());
    }
  }
  return parameters;
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(tensorrt_subgraph_pass,
              paddle::inference::analysis::TensorRtSubgraphPass)
    .RequirePassAttr("tensorrt_node_teller")
    .RequirePassAttr("max_batch_size")
    .RequirePassAttr("workspace_size")
    .RequirePassAttr("min_subgraph_size");
