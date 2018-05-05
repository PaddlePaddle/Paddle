//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/data_flow_graph_to_fluid_pass.h"
#include "paddle/fluid/framework/proto_desc.h"

namespace paddle {
namespace inference {
namespace analysis {

bool DataFlowGraphToFluidPass::Initialize(framework::proto::ProgramDesc* desc) {
  desc_ = desc;
  // Here some logic from program_desc.cc and will not add new interfaces into
  // framework::ProgramDesc class, use some UT to assure the correctness.
  auto* block = desc_->mutable_blocks()->Add();
  block->set_idx(framework::kRootBlockIndex);
  block->set_parent_idx(framework::kNoneBlockIndex);
  return true;
}

bool DataFlowGraphToFluidPass::Finalize() { return true; }

void DataFlowGraphToFluidPass::Run(DataFlowGraph* graph) {
  auto traits = GraphTraits<DataFlowGraph>(graph);
  for (auto it = traits.nodes().begin(); it != traits.nodes().end(); ++it) {
    if (it->deleted()) continue;
    switch (it->type()) {
      case Node::Type::kFunction:
        AddFluidOp(&(*it));
        break;
      case Node::Type::kFunctionBlock:
        AddEngineOp(&(*it));
        break;
      default:
        continue;
    }
  }
}

void DataFlowGraphToFluidPass::AddFluidOp(Node* node) {
  auto* ori_op = static_cast<framework::proto::OpDesc*>(node->extra_info());
  // currently only the main block is analyzed.
  auto* main_block = desc_->mutable_blocks(framework::kRootBlockIndex);
  auto* op = main_block->add_ops();
  *op = *ori_op;  // copy the attributes, by default, these will not be changed
                  // by analysis phrase.
  // Rewrite the inputs and outputs of the op, for that after analysis, the
  // inputs and outputs might be changed.
  op->mutable_inputs()->Clear();
  for (const auto& v : node->inlinks) {
    *(op->mutable_inputs()->Add()) =
        *(static_cast<framework::proto::OpDesc_Var*>(v->extra_info()));
  }
  op->mutable_outputs()->Clear();
  for (const auto& v : node->outlinks) {
    *(op->mutable_outputs()->Add()) =
        *(static_cast<framework::proto::OpDesc_Var*>(v->extra_info()));
  }
}

void DataFlowGraphToFluidPass::AddEngineOp(Node* node) {
  // auto* ori_op = static_cast<framework::proto::OpDesc*>(node->extra_info());
  // auto* main_block = desc_->mutable_blocks(framework::kRootBlockIndex);
  // auto* op = main_block->add_ops();
  // TODO(Superjomn) Here need to expose some arguments for default setting.
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
