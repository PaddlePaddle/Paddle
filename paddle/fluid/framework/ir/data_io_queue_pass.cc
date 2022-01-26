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

#include <set>
#include <map>

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

std::set<std::string> output_queue_holder_ops = {
  "file_label_reader",
  "map",
  "data_reader",
};

std::set<std::string> input_array_ops = {
  "random_crop_and_resize",
  "batch_decode",
};

static bool IsOutputQueueHolderOp(std::string op_type) {
  return output_queue_holder_ops.find(op_type) != output_queue_holder_ops.end();
}

static bool IsInputArrayOp(std::string op_type) {
  return input_array_ops.find(op_type) != input_array_ops.end();
}

static void ProcessOutputQueueHolderOp(ir::Graph* graph) {
  std::set<std::string> var_names;
  for (const Node *n : graph->Nodes()) {
    if (n->IsOp() && n->Op()) {
      auto *op = n->Op();
      if (IsOutputQueueHolderOp(op->Type())) {
        auto& outputs = op->Outputs();
        for (auto iter = outputs.begin(); iter != outputs.end(); iter++) {
          for (auto var: iter->second) var_names.insert(var);
        }
      }
    }
  }

  for (const Node *n : graph->Nodes()) {
    if (n->IsVar() && n->Var()) {
      auto *var = n->Var();
      if (var_names.find(var->Name()) != var_names.end()) {
        VLOG(3) << "Change output variable type of " << var->Name() << " to queue holder";
        var->SetType(framework::proto::VarType::LOD_TENSOR_BLOCKING_QUEUE);
        var->SetPersistable(true);
      }
    }
  }
}

static void ProcessInputArrayOp(ir::Graph* graph) {
  std::set<std::string> var_names;
  for (const Node *n : graph->Nodes()) {
    if (n->IsOp() && n->Op()) {
      auto *op = n->Op();
      if (IsInputArrayOp(op->Type())) {
        auto& inputs = op->Inputs();
        for (auto iter = inputs.begin(); iter != inputs.end(); iter++) {
          for (auto var: iter->second) var_names.insert(var);
        }
      }
    }
  }

  for (const Node *n : graph->Nodes()) {
    if (n->IsVar() && n->Var()) {
      auto *var = n->Var();
      if (var_names.find(var->Name()) != var_names.end()) {
        VLOG(3) << "Change output variable type of " << var->Name() << " to queue holder";
        var->SetType(framework::proto::VarType::LOD_TENSOR_ARRAY);
      }
    }
  }
}

class DataIOQueuePass: public Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override {
    ProcessOutputQueueHolderOp(graph);
    ProcessInputArrayOp(graph);
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(data_io_queue_pass, paddle::framework::ir::DataIOQueuePass);
