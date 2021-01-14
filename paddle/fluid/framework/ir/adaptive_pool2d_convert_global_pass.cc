/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/adaptive_pool2d_convert_global_pass.h"

#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void AdaptivePool2dConvertGlobalPass::ApplyImpl(ir::Graph* graph) const {
  std::string name_scope = "adaptive_pool2d_convert_global_pass";
  FusePassBase::Init(name_scope, graph);
  int num = 0;
  for (const Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (op->HasAttr("adaptive") && op->HasAttr("ksize")) {
        bool adaptive = BOOST_GET_CONST(bool, op->GetAttr("adaptive"));
        std::vector<int> ksize =
            BOOST_GET_CONST(std::vector<int>, op->GetAttr("ksize"));
        if (adaptive && ksize.size() == 2 && ksize[0] == 1 && ksize[1] == 1) {
          op->SetAttr("adaptive", false);
          op->SetAttr("global_pooling", true);
          ++num;
        }
      }
    }
  }
  // LOG(INFO) << "---  processed " << num << " nodes";
  AddStatis(num);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(adaptive_pool2d_convert_global_pass,
              paddle::framework::ir::AdaptivePool2dConvertGlobalPass);

REGISTER_PASS_CAPABILITY(adaptive_pool2d_convert_global_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "pool2d", 0));
