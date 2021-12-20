// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/ipu/inference_postprocess_pass.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"
#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void InferencePostprocessPass::ApplyImpl(ir::Graph *graph) const {
  VLOG(10) << "enter InferencePostprocessPass::ApplyImpl";

  std::vector<std::string> feed_list;
  feed_list = Get<std::vector<std::string>>("feed_list");
  std::vector<std::string> fetch_list;
  fetch_list = Get<std::vector<std::string>>("fetch_list");

  auto *feed_var = new paddle::framework::VarDesc("feed");
  feed_var->SetType(proto::VarType::FEED_MINIBATCH);
  auto *feed_var_node = graph->CreateVarNode(feed_var);

  auto *fetch_var = new paddle::framework::VarDesc("fetch");
  fetch_var->SetType(proto::VarType::FETCH_LIST);
  auto *fetch_var_node = graph->CreateVarNode(fetch_var);

  for (int i = 0; i < feed_list.size(); i++) {
    for (auto node : graph->Nodes()) {
      if (node->Name() == feed_list[i]) {
        auto *op = new paddle::framework::OpDesc();
        op->SetType("feed");
        op->SetInput("X", {"feed"});
        op->SetOutput("Out", {node->Name()});
        op->SetAttr("col", i);
        auto *op_node = graph->CreateOpNode(op);
        node->inputs.push_back(op_node);
        op_node->outputs.push_back(node);
        feed_var_node->outputs.push_back(op_node);
        op_node->inputs.push_back(feed_var_node);
        break;
      }
    }
  }

  for (int i = 0; i < fetch_list.size(); i++) {
    for (auto node : graph->Nodes()) {
      if (node->Name() == fetch_list[i]) {
        auto *op = new paddle::framework::OpDesc();
        op->SetType("fetch");
        op->SetInput("X", {node->Name()});
        op->SetOutput("Out", {"fetch"});
        op->SetAttr("col", i);
        auto *op_node = graph->CreateOpNode(op);
        node->outputs.push_back(op_node);
        op_node->inputs.push_back(node);
        fetch_var_node->inputs.push_back(op_node);
        op_node->outputs.push_back(fetch_var_node);
        break;
      }
    }
  }

  VLOG(10) << "leave InferencePostprocessPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inference_postprocess_pass,
              paddle::framework::ir::InferencePostprocessPass)
    .RequirePassAttr("feed_list")
    .RequirePassAttr("fetch_list");
