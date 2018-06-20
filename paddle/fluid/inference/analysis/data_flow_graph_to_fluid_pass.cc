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
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/proto_desc.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::proto::ProgramDesc;

std::vector<std::string> ExtractParameters(
    const std::vector<std::unique_ptr<Node>>& nodes);

bool DataFlowGraphToFluidPass::Initialize(Argument* argument) {
  ANALYSIS_ARGUMENT_CHECK_FIELD(argument)
  ANALYSIS_ARGUMENT_CHECK_FIELD(argument->origin_program_desc)
  PADDLE_ENFORCE(!argument->transformed_program_desc);
  // The transformed_program_desc should inherit all the VarDesc and BlockDesc
  // from the original program desc. The operators of the main block(the first
  // block) should rewritten by data flow graph.
  argument->transformed_program_desc.reset(
      new ProgramDesc(*argument->origin_program_desc));
  argument->transformed_program_desc->mutable_blocks(0)->clear_ops();
  desc_ = argument->transformed_program_desc.get();
  // Here some logic from program_desc.cc and will not add new interfaces into
  // framework::ProgramDesc class, use some UT to assure the correctness.
  auto* block = desc_->mutable_blocks()->Add();
  block->set_idx(framework::kRootBlockIndex);
  block->set_parent_idx(framework::kNoneBlockIndex);
  argument_ = argument;
  return true;
}

bool DataFlowGraphToFluidPass::Finalize() { return true; }

void DataFlowGraphToFluidPass::Run(DataFlowGraph* graph) {
  auto traits = GraphTraits<DataFlowGraph>(graph);
  for (auto it = traits.nodes().begin(); it != traits.nodes().end(); ++it) {
    if (it->deleted()) continue;

    switch (it->type()) {
      case Node::Type::kFunction: {
        LOG(INFO) << "add function " << it->repr();
        AddFluidOp(&(*it));
      } break;
      case Node::Type::kFunctionBlock: {
        LOG(INFO) << "add engine op " << it->repr();
        AddEngineOp(&(*it));
      } break;
      default:
        continue;
    }
  }
}

void DataFlowGraphToFluidPass::AddFluidOp(Node* node) {
  auto* ori_op = static_cast<framework::proto::OpDesc*>(node->pb_desc());
  // currently only the main block is analyzed.
  auto* main_block = desc_->mutable_blocks(framework::kRootBlockIndex);
  auto* op = main_block->add_ops();
  *op = *ori_op;  // copy the attributes, by default, these will not be changed
                  // by analysis phrase.
  // The inputs and outputs of the existing ops are not changed by tensorrt
  // subgraph pass.
  // NOTE It might be changed by other passes in the long run.
}

void CreateTrtEngineOp(Node* node, const DataFlowGraph& graph,
                       const framework::proto::BlockDesc& block) {
  static int counter{0};
  PADDLE_ENFORCE(node->IsFunction());
  framework::OpDesc desc;
  auto* func = static_cast<Function*>(node);

  // collect inputs
  std::vector<std::string> io;
  for (auto* x : func->inlinks) {
    io.push_back(x->name());
  }
  desc.SetInput("Xs", io);

  // collect outputs
  io.clear();
  for (auto* x : func->outlinks) {
    io.push_back(x->name());
  }
  desc.SetOutput("Ys", io);

  // Set attrs
  SetAttr(desc.Proto(), "subgraph", block.SerializeAsString());
  SetAttr(desc.Proto(), "engine_unique_key",
          "trt-" + std::to_string(counter++));
  SetAttr(desc.Proto(), "max_batch", 100);  // TODO(Superjomn) add config latter
  SetAttr(desc.Proto(), "max_workspace",
          1024);  // TODO(Superjomn) add config latter
  SetAttr(desc.Proto(), "parameters", ExtractParameters(graph.nodes.nodes()));
  node->SetPbMsg(desc.Proto()->SerializeAsString());
}

std::vector<std::string> ExtractParameters(
    const std::vector<std::unique_ptr<Node>>& nodes) {
  std::vector<std::string> parameters;
  for (const auto& node : nodes) {
    framework::proto::VarDesc var;
    var.ParseFromString(node->pb_msg());
    if (var.persistable()) {
      parameters.push_back(var.name());
    }
  }
  return parameters;
}

void DataFlowGraphToFluidPass::AddEngineOp(Node* node) {
  // auto* ori_op =
  // static_cast<framework::proto::OpDesc*>(node->extra_info());
  // auto* main_block = desc_->mutable_blocks(framework::kRootBlockIndex);
  // auto* op = main_block->add_ops();
  // TODO(Superjomn) Here need to expose some arguments for default setting.
  PADDLE_ENFORCE(node->IsFunctionBlock());
  auto* block_node = static_cast<FunctionBlock*>(node);
  framework::proto::BlockDesc proto;
  framework::BlockDesc block_desc(nullptr, &proto);
  // copy ops.
  for (auto* node : block_node->subgraph) {
    auto* op = block_desc.AppendOp();
    PADDLE_ENFORCE(!node->pb_msg().empty());
    op->Proto()->ParseFromString(node->pb_msg());
  }
  CreateTrtEngineOp(node, *argument_->main_dfg, *block_desc.Proto());
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
