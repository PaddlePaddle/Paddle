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

#include "paddle/fluid/lite/core/mir/io_complement_pass.h"
#include "paddle/fluid/lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void IoComplementPass::Apply(std::unique_ptr<mir::SSAGraph>& graph) {
  // Start from inputs of the graph, those should have place set.
  for (auto& node : graph->mutable_nodes()) {
    if (!node.IsInstruct()) continue;
    auto& inst = node.AsInstruct();

    // inputs
    for (auto* in : node.inlinks) {
      CHECK(in->IsArgument());
      auto name = in->AsArgument().name;
      std::string tmp;
      CHECK(inst.op_info->GetInputArgname(name, &tmp));
      auto type =
          ParamTypeRegistry::Global().Retrieve<ParamTypeRegistry::IO::kInput>(
              inst.place, inst.op_type, tmp);
      CHECK(type) << "no param type found for " << inst.op_type << ":" << name
                  << " " << inst.place;
      if (type->tensor_place != inst.place) {
        LOG(INFO) << "found IO unmatched tensor";
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(io_complement_pass, paddle::lite::mir::IoComplementPass);
