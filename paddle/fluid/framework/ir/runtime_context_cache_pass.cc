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

#include "paddle/fluid/framework/ir/runtime_context_cache_pass.h"
#include <memory>
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
namespace ir {

void RuntimeContextCachePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Applies Runtime Context Cache strategy.";
  for (const Node* n : graph->Nodes()) {
    if (n->IsOp() && n->Op()) {
      n->Op()->SetAttr(kEnableCacheRuntimeContext, true);
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(runtime_context_cache_pass,
              paddle::framework::ir::RuntimeContextCachePass);
