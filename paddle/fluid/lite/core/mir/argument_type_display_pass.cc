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

#include "paddle/fluid/lite/core/mir/pass.h"
#include "paddle/fluid/lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

class ArgumentTypeDisplayPass : public DebugPass {
 public:
  void Apply(std::unique_ptr<mir::SSAGraph>& graph) override {
    LOG(INFO) << "== Argument types ==";
    for (auto& node : graph->mutable_nodes()) {
      if (!node.IsArg()) continue;

      auto* type = node.AsArg().type;
      if (type) {
        LOG(INFO) << "* ARG " << node.AsArg().name << " type: " << *type;
      } else {
        LOG(INFO) << "* ARG " << node.AsArg().name << " type: UNK";
      }
    }
    LOG(INFO) << "---------------------";
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(argument_type_display_pass,
                  paddle::lite::mir::ArgumentTypeDisplayPass);
