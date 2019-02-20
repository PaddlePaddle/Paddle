//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

class FuseParametersPass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override {
    ir::Graph &result = *graph;

    // Step 1: Find Parameters
    std::vector<ir::Node *> param_nodes;
    std::vector<std::string> param_names;
    for (auto &node : result.Nodes()) {
      if (node->IsVar() && node->Var() && node->Var()->Persistable()) {
        param_nodes.emplace_back(node);
        param_names.emplace_back(node->Var()->Name());
        VLOG(10) << "Find : " << node->Var()->Name();
      }
    }

    // Step 2: Insert fused_var_name to FusedVars
    if (!result.Has(kFusedVars)) {
      result.Set(kFusedVars, new FusedVars);
    }

    auto fused_var_name = "@FUSED_PARAMETERS@";
    result.Get<FusedVars>(kFusedVars).emplace_back(fused_var_name);

    if (!result.Has(kRunOnlyOnceProgram)) {
      result.Set(kRunOnlyOnceProgram, new RunOnlyOnceProgram);
    }
    result.Get<RunOnlyOnceProgram>(kRunOnlyOnceProgram).emplace_back();
    auto &program_desc =
        result.Get<RunOnlyOnceProgram>(kRunOnlyOnceProgram).back();
    auto *global_block = program_desc.MutableBlock(0);
    AppendAllocContinuousSpace(param_names, fused_var_name, true, global_block);
    return std::move(graph);
  }

 private:
  void AppendAllocContinuousSpace(const std::vector<std::string> &args,
                                  const std::string &out_arg, bool copy_data,
                                  BlockDesc *global_block) const {
    auto op_desc = global_block->AppendOp();
    op_desc->SetType("alloc_continuous_space");
    op_desc->SetInput("Input", args);
    op_desc->SetOutput("Output", args);
    op_desc->SetOutput("FusedOutput", {out_arg});
    op_desc->SetAttr("copy_data", copy_data);
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_parameters_pass,
              paddle::framework::details::FuseParametersPass);
