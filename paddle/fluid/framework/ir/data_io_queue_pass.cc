/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include <map>

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

static int MAX_VARS_LEN = 100;

class DataIOQueuePass: public Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override {
    // VLOG(3) << "Change inputs/outputs of data ops to queue";
    LOG(ERROR) << "Change inputs/outputs of data ops to queue";
    std::vector<std::string> var_names;
    var_names.reserve(MAX_VARS_LEN);
    for (const Node *n : graph->Nodes()) {
      if (n->IsOp() && n->Op()) {
        auto *op = n->Op();
        if (op->Type() == "file_label_reader"
            || op->Type() == "batch_decode"
            || op->Type() == "map") {
          auto& outputs = op->Outputs();
          for (auto iter = outputs.begin(); iter != outputs.end(); iter++) {
            auto vars = iter->second;
            std::copy(vars.begin(), vars.end(), std::back_inserter(var_names));
          }
        }
      }
    }

    for (const Node *n : graph->Nodes()) {
      if (n->IsVar() && n->Var()) {
        auto *var = n->Var();
        auto iter = std::find(var_names.begin(), var_names.end(), var->Name());
        if (iter != var_names.end()) {
          var->SetType(framework::proto::VarType::LOD_TENSOR_BLOCKING_QUEUE);
          var->SetPersistable(true);
        }
      }
    }
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(data_io_queue_pass, paddle::framework::ir::DataIOQueuePass);
