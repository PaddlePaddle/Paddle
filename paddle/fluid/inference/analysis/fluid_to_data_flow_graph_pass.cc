/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"
#include <vector>

namespace paddle {
namespace inference {
namespace analysis {

bool FluidToDataFlowGraphPass::Initialize() { return Pass::Initialize(); }

bool FluidToDataFlowGraphPass::Initialize(const framework::ProgramDesc &desc) {
  desc_ = &desc;
  return true;
}

bool FluidToDataFlowGraphPass::Finalize() { return Pass::Finalize(); }

void FluidToDataFlowGraphPass::Run(DataFlowGraph *graph) {
  // insert vars
  std::unordered_map<std::string, size_t> var2id;
  for (const auto &var : desc_->Block(0).AllVars()) {
    auto *v = graph->nodes.Create(Node::Type::kValue);
    v->SetName(var->Name());
    v->SetExtraInfo(var);
    var2id[var->Name()] = v->id();
  }

  // insert ops
  for (const auto &op : desc_->Block(0).AllOps()) {
    auto *o = graph->nodes.Create(Node::Type::kFunction);
    o->SetName(op->Type());
    // Link to the original protobuf message's memory, make it easier to
    // generate from a data flow graph to fluid ProgramDesc.
    o->SetExtraInfo(op);
    // set inputs and outputs
    for (auto &in : op->InputNames()) {
      o->inlinks.push_back(graph->nodes.Get(var2id[in]));
    }
    for (auto &out : op->OutputNames()) {
      o->outlinks.push_back(graph->nodes.Get(var2id[out]));
    }
  }
}
Pass *FluidToDataFlowGraphPass::CreatePrinterPass(
    std::ostream &os, const std::string &banner) const {
  return nullptr;
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
