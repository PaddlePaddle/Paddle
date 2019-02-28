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

#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
namespace details {

class FuseParametersPass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override {
    ir::Graph &result = *graph;

    auto &places = Get<const std::vector<platform::Place>>(kPlaces);
    auto &local_scopes = Get<const std::vector<Scope *>>(kLocalScopes);

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

    if (param_names.size() == 0) {
      return std::move(graph);
    }

    // Step 2: Insert fused_var_name to FusedVars
    if (!result.Has(kFusedVars)) {
      result.Set(kFusedVars, new FusedVars);
    }

    auto fused_var_name = std::string("@FUSED_PARAMETERS@") + param_names[0];
    auto &fused_var_set = result.Get<FusedVars>(kFusedVars);
    PADDLE_ENFORCE_EQ(fused_var_set.count(fused_var_name), 0);
    fused_var_set.insert(fused_var_name);

    // Step 2: Init fused_var_name and AllocSpaceForVar
    InitFusedVarsAndAllocSpaceForVars(places, local_scopes, param_names,
                                      fused_var_name);

    return std::move(graph);
  }

  void InitFusedVarsAndAllocSpaceForVars(
      const std::vector<platform::Place> &places,
      const std::vector<Scope *> &local_scopes,
      const std::vector<std::string> &param_names,
      const std::string &fused_var_name) const {
    VLOG(10) << "Init FusedVars.";
    // NOTE: Because of Scope's inheritance structure, we should traverse
    // backwards.
    size_t idx = local_scopes.size() - 1;
    for (auto iter = local_scopes.rbegin(); iter != local_scopes.rend();
         ++iter, --idx) {
      auto &scope = *iter;
      VLOG(10) << "Find var in scope " << idx << " " << fused_var_name;
      Variable *var = scope->FindVar(fused_var_name);
      if (var) {
        PADDLE_THROW("%s has exist in scope[%d]", fused_var_name, idx);
      }
      scope->Var(fused_var_name)->GetMutable<LoDTensor>();
    }

    ProgramDesc program_desc;
    auto *global_block = program_desc.MutableBlock(0);
    AppendAllocContinuousSpace(param_names, fused_var_name, true, global_block);

    for (size_t i = 0; i < local_scopes.size(); ++i) {
      for (auto &op_desc : global_block->AllOps()) {
        auto op = OpRegistry::CreateOp(*op_desc);
        VLOG(4) << op->DebugStringEx(local_scopes[i]);
        op->Run(*local_scopes[i], places[i]);
        VLOG(3) << op->DebugStringEx(local_scopes[i]);
      }
    }
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
              paddle::framework::details::FuseParametersPass)
    .RequirePassAttr(paddle::framework::details::kPlaces)
    .RequirePassAttr(paddle::framework::details::kLocalScopes);
