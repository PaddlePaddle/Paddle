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
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/platform/device_code.h"

namespace paddle {
namespace framework {
namespace ir {

void FusionGroupPass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init("fusion_group_pass", graph);
  if (Get<bool>("use_gpu")) {
    fusion_group::OperationMap::Init();
    int num_elementwise_groups = DetectFusionGroup(graph, 0);
    AddStatis(num_elementwise_groups);
  }
}

int FusionGroupPass::DetectFusionGroup(Graph* graph, int type) const {
  // TODO(liuyiqun): supported different places
  platform::CUDAPlace place = platform::CUDAPlace(0);
  int index = platform::DeviceCodePool::Init({place}).size(place);

  std::vector<std::vector<Node*>> subgraphs =
      fusion_group::ElementwiseGroupDetector()(graph);

  int num_subgraphs = 0;
  size_t min_subgraph_size = 2;
  bool save_intermediate_out = true;
  for (auto& vec : subgraphs) {
    fusion_group::SubGraph subgraph(
        type, "", save_intermediate_out,
        std::unordered_set<Node*>(vec.begin(), vec.end()));
    VLOG(3) << "subgraph: {\n" << DebugString(subgraph.SortedNodes()) << "}\n";

    if (subgraph.IsValid(min_subgraph_size)) {
      subgraph.SetFuncName("fused_elementwise_" + std::to_string(index++));
      if (GenerateCode(&subgraph)) {
        InsertFusionGroupOp(graph, &subgraph);
        num_subgraphs++;
      }
    }
  }
  return num_subgraphs;
}

bool FusionGroupPass::GenerateCode(fusion_group::SubGraph* subgraph) const {
  fusion_group::CodeGenerator code_generator;
  std::string code_str = code_generator.Generate(subgraph);
  VLOG(3) << code_str;

  // TODO(liuyiqun): supported different places
  platform::CUDAPlace place = platform::CUDAPlace(0);
  std::unique_ptr<platform::CUDADeviceCode> device_code(
      new platform::CUDADeviceCode(place, subgraph->GetFuncName(), code_str));
  bool is_compiled = device_code->Compile();
  if (is_compiled) {
    platform::DeviceCodePool& pool = platform::DeviceCodePool::Init({place});
    pool.Set(std::move(device_code));
  }
  return is_compiled;
}

static int ExtractOpRole(fusion_group::SubGraph* subgraph) {
  std::unordered_set<int> op_roles;
  std::string attr_name = OpProtoAndCheckerMaker::OpRoleAttrName();
  for (auto* n : subgraph->Nodes()) {
    if (n && n->IsOp() && n->Op()) {
      if (n->Op()->HasAttr(attr_name)) {
        op_roles.insert(boost::get<int>(n->Op()->GetAttr(attr_name)));
      }
    }
  }
  if (op_roles.size() == 1U) {
    return *(op_roles.begin());
  } else {
    return static_cast<int>(OpRole::kNotSpecified);
  }
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
  op_desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                  ExtractOpRole(subgraph));

  Node* fusion_group_node = graph->CreateOpNode(&op_desc);
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
