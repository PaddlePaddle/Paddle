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

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

class SyncBatchNormPass : public Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override {
#if defined(_WIN32)
    VLOG(3) << "Not use synchronize batch norm on windows";
    return;
#endif
    VLOG(3) << "Use synchronize batch norm";
    for (const Node *n : graph->Nodes()) {
      if (n->IsOp() && n->Op()) {
        auto *op = n->Op();
        // process synchronize in batch_norm
        if (op->Type() == "batch_norm") {
          op->SetType("sync_batch_norm");
        }
        if (op->Type() == "batch_norm_grad") {
          op->SetType("sync_batch_norm_grad");
        }
      }
    }
  }
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(sync_batch_norm_pass, paddle::framework::ir::SyncBatchNormPass);
