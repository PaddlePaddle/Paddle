/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/fusion_group/fusion_group_pass.h"
#include <memory>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/ir/fusion_group/code_generator.h"
#include "paddle/fluid/framework/ir/fusion_group/elementwise_group_detector.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device_code.h"

namespace paddle {
namespace framework {
namespace ir {

void FusionGroupPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph);
  if (Get<bool>("use_gpu")) {
    fusion_group::OperationMap::Init();
    int num_elementwise_groups = DetectFusionGroup(graph, 0);
    LOG(INFO) << "Detect " << num_elementwise_groups
              << " elementwise fusion groups.";
  }
}

int FusionGroupPass::DetectFusionGroup(Graph* graph, int type) const {
  std::vector<fusion_group::SubGraph> subgraphs;
  std::vector<Node*> begin_of_forward_subgraph;

  // Detect subgraph of forward ops.
  fusion_group::ElementwiseGroupDetector forward_detector(graph, false);
  std::unordered_set<Node*> all_nodes = graph->Nodes();
  // TODO(liuyiqun): supported different places
  platform::CUDAPlace place = platform::CUDAPlace(0);
  int index = platform::DeviceCodePool::Init({place}).size(place);
  for (Node* n : all_nodes) {
    bool is_found = false;
    for (auto& subgraph : subgraphs) {
      if (subgraph.Has(n)) {
        is_found = true;
        break;
      }
    }
    if (is_found) {
      continue;
    }

    if (type == 0) {
      fusion_group::SubGraph subgraph = forward_detector(n);
      if (subgraph.GetNumOperations() >= 2) {
        std::string func_name = "fused_elementwise_" + std::to_string(index++);
        subgraph.SetFuncName(func_name);
        subgraphs.push_back(subgraph);
        LOG(INFO) << "subgraph: {\n" << DebugString(subgraph.Nodes()) << "}\n";
        begin_of_forward_subgraph.push_back(n);
      }
    }
  }
  // Detect subgraph of backward ops.
  fusion_group::ElementwiseGroupDetector backward_detector(graph, true);
  for (auto* begin : begin_of_forward_subgraph) {
    if (type == 0) {
      fusion_group::SubGraph subgraph = backward_detector(begin);
      if (subgraph.GetNumOperations() >= 2) {
        std::string func_name =
            "fused_elementwise_grad_" + std::to_string(index++);
        subgraph.SetFuncName(func_name);
        subgraphs.push_back(subgraph);
      }
    }
  }

  // TODO(liuyiqun): check whether there are intersection between subgraphs
  for (size_t i = 0; i < subgraphs.size(); ++i) {
    GenerateCode(&subgraphs[i]);
    InsertFusionGroupOp(graph, &subgraphs[i]);
  }
  return subgraphs.size();
}

void FusionGroupPass::GenerateCode(fusion_group::SubGraph* subgraph) const {
  fusion_group::CodeGenerator code_generator;
  std::string code_str = code_generator.Generate(subgraph);
  VLOG(3) << code_str;

  // TODO(liuyiqun): supported different places
  platform::CUDAPlace place = platform::CUDAPlace(0);
  std::unique_ptr<platform::CUDADeviceCode> device_code(
      new platform::CUDADeviceCode(place, subgraph->GetFuncName(), code_str));
  device_code->Compile();

  platform::DeviceCodePool& pool = platform::DeviceCodePool::Init({place});
  pool.Set(std::move(device_code));
}

void FusionGroupPass::InsertFusionGroupOp(
    Graph* graph, fusion_group::SubGraph* subgraph) const {
  const std::vector<Node*>& input_vars_of_subgraph =
      subgraph->GetInputVarNodes();
  const std::vector<Node*>& output_vars_of_subgraph =
      subgraph->GetOutputVarNodes();
  std::unordered_set<Node*> external_nodes;

  OpDesc op_desc;
  op_desc.SetType("fusion_group");

  std::vector<std::string> input_names;
  for (auto* n : input_vars_of_subgraph) {
    input_names.push_back(n->Name());
    external_nodes.insert(n);
  }
  op_desc.SetInput("Inputs", input_names);

  std::vector<std::string> output_names;
  for (auto* n : output_vars_of_subgraph) {
    output_names.push_back(n->Name());
    external_nodes.insert(n);
  }
  op_desc.SetOutput("Outs", output_names);
  op_desc.SetAttr("type", subgraph->GetType());
  op_desc.SetAttr("func_name", subgraph->GetFuncName());

  auto fusion_group_node = graph->CreateOpNode(&op_desc);
  for (auto* in : input_vars_of_subgraph) {
    IR_NODE_LINK_TO(in, fusion_group_node);
  }
  for (auto* out : output_vars_of_subgraph) {
    IR_NODE_LINK_TO(fusion_group_node, out);
  }

  std::unordered_set<const Node*> internal_nodes;
  for (auto* n : subgraph->Nodes()) {
    if (external_nodes.find(n) == external_nodes.end()) {
      internal_nodes.insert(n);
    }
  }
  GraphSafeRemoveNodes(graph, internal_nodes);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fusion_group_pass, paddle::framework::ir::FusionGroupPass)
    .RequirePassAttr("use_gpu");
