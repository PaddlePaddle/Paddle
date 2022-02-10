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

#include "paddle/fluid/framework/ir/ipu/ipu_inplace_pass.h"

#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

std::string GenerateVarName(Node *node) {
  return node->Name() + "_" + std::to_string(node->id());
}

void IpuInplacePass::ApplyImpl(ir::Graph *graph) const {
  // use this pass after forward_graph_extract_pass
  // raise error if the inplaced var both in feed_list & fetch_list
  VLOG(10) << "enter IpuInplacePass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  std::vector<std::string> feed_list;
  feed_list = Get<std::vector<std::string>>("feed_list");
  std::vector<std::string> fetch_list;
  fetch_list = Get<std::vector<std::string>>("fetch_list");

  std::map<std::string, int> var_name;
  for (auto *node : graph->Nodes()) {
    if (node->IsVar()) {
      if (var_name.find(node->Name()) == var_name.end()) {
        var_name.emplace(node->Name(), 1);
      } else {
        var_name[node->Name()]++;
      }
    }
  }

  for (auto *node : graph->Nodes()) {
    if (node->IsVar()) {
      if (var_name[node->Name()] > 1) {
        auto is_feed = (std::find(feed_list.begin(), feed_list.end(),
                                  node->Name()) != feed_list.end()) &&
                       (node->inputs.size() == 0);
        auto is_fetch = (std::find(fetch_list.begin(), fetch_list.end(),
                                   node->Name()) != fetch_list.end()) &&
                        (node->outputs.size() == 0);
        if (!is_feed && !is_fetch && !node->Var()->Persistable()) {
          auto old_name = node->Name();
          auto new_name = GenerateVarName(node);
          node->RenameVar(new_name);
          for (auto *op_in : node->inputs) {
            op_in->Op()->RenameOutput(old_name, new_name);
          }
          for (auto *op_out : node->outputs) {
            op_out->Op()->RenameInput(old_name, new_name);
          }
        }
      }
    }
  }

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave IpuInplacePass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(ipu_inplace_pass, paddle::framework::ir::IpuInplacePass)
    .RequirePassAttr("feed_list")
    .RequirePassAttr("fetch_list");
