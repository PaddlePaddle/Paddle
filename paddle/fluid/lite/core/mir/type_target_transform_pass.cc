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

#include "paddle/fluid/lite/core/mir/type_target_transform_pass.h"
#include <list>
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/mir/graph_visualize_pass.h"
#include "paddle/fluid/lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void TypeTargetTransformPass::Apply(std::unique_ptr<mir::SSAGraph>& graph) {
  // Start from inputs of the graph, those should have place set.
  std::list<Node*> nodes;
  for (auto& node : graph->mutable_nodes()) {
    nodes.push_back(&node);
  }

  CHECK(!valid_places_.empty());

  for (auto& node : nodes) {
    if (!node->IsInstruct()) continue;
    auto inlinks = node->inlinks;
    for (auto* in : inlinks) {
      ComplementInputs(graph.get(), node, in);
    }
  }
  VLOG(3) << "\n" << Visualize(graph.get());
}

void TypeTargetTransformPass::ComplementInputs(SSAGraph* graph, Node* inst_node,
                                               Node* in) {
  // If this input is out of date.
  if (inst_node->inlinks.end() ==
      std::find(inst_node->inlinks.begin(), inst_node->inlinks.end(), in))
    return;

  CHECK(inst_node->IsInstruct());
  auto& inst = inst_node->AsInstruct();
  CHECK(in->IsRoleSet());
  CHECK(in->IsArgument());
  auto in_arg_name = in->AsArgument().name;
  std::string tmp;
  CHECK(inst.op_info()->GetInputArgname(in_arg_name, &tmp));
  auto decl_arg_type = inst.picked_kernel().GetInputDeclType(tmp);
  CHECK(in->AsArgument().type);
  if (!TargetCompatibleTo(*in->AsArgument().type, *decl_arg_type)) {
    LOG(INFO) << "found Target unmatched tensor: " << in->AsArgument().name
              << " for kernel " << inst.op->DebugString() << " "
              << *in->AsArgument().type << " -> " << *decl_arg_type;
    // Add an IoCopy instruction to make the input compatible with other dist.
    AddIoCopyInst(*in->AsArgument().type, *decl_arg_type, in->AsArgument().name,
                  graph, inst_node, valid_places_);
  }
}

void TypeTargetTransformPass::AddIoCopyInst(
    const Type& from, const Type& to, const std::string& var, SSAGraph* graph,
    Node* inst_node, const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty()) << "valid_place should be set";
  // var -> new_transform_op -> new_var -> inst
  // So there will be a new Argument node and a new IoCopy Instruct Node.

  auto node_id = [&] { return graph->nodes().size(); };
  auto io_copy_output_name = var + "/trans/" + std::to_string(node_id());
  auto* io_copy_output_arg = graph->NewArgumentNode(io_copy_output_name);
  auto* io_copy_inst = graph->NewInstructNode();

  // create Op and kernels.
  auto io_copy_op = LiteOpRegistry::Global().Create("io_copy");
  CHECK(io_copy_op) << "create op [" << io_copy_op << "] failed";
  // CHECK(io_copy_op);
  // Create the new var manually.
  inst_node->AsInstruct().op->scope()->Var(io_copy_output_name);

  // Create IoCopy Instruction.
  lite::OpDesc op_desc;
  op_desc.SetType("io_copy");
  op_desc.SetInput("Input", {var});
  op_desc.SetOutput("Out", {io_copy_output_name});

  io_copy_op->Attach(op_desc, inst_node->AsInstruct().op->scope());
  auto kernels = io_copy_op->CreateKernels(valid_places);
  io_copy_inst->AsInstruct("io_copy", std::move(kernels), io_copy_op);

  // Remove the old link
  RemoveDirectedLink(graph->Argument(var), inst_node);

  // Update the original instruction OpDesc.
  // Update its input to the io_copy_output_name
  auto& inst = inst_node->AsInstruct();
  auto inst_program_desc = inst.op_info()->desc();

  // Add new link, var -> new_inst, new_inst->newarg, newarg->inst
  DirectedLink(graph->Argument(var), io_copy_inst);
  DirectedLink(io_copy_inst, io_copy_output_arg);
  DirectedLink(io_copy_output_arg, inst_node);

  // reset opdesc and update kernel information
  auto desc_dummy = inst_node->AsInstruct().op->op_info()->desc();
  UpdateInputTo(&desc_dummy, var, io_copy_output_name);

  lite::OpDesc desc_fake(desc_dummy);
  inst_node->AsInstruct().op->Attach(desc_fake,
                                     inst_node->AsInstruct().op->scope());

  std::string tmp;
  if (inst_node->AsInstruct().op_info()->GetInputArgname("a", &tmp)) {
    CHECK(false) << "get old a " << tmp;
  }

  for (auto& kernel : inst_node->AsInstruct().valid_kernels) {
    inst_node->AsInstruct().op->AttachKernel(kernel.get());
  }

  graph->CheckValid();
}

void TypeTargetTransformPass::SetValidPlaces(
    const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty());
  valid_places_ = valid_places;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(type_target_transform_pass,
                  paddle::lite::mir::TypeTargetTransformPass);
