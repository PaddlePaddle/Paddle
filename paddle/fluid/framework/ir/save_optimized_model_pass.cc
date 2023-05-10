/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/save_optimized_model_pass.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace framework {
class ProgramDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

void SaveOptimizedModelPass::ApplyImpl(ir::Graph* graph) const {
  if (!Has("save_optimized_model") || !Get<bool>("save_optimized_model"))
    return;

  std::string model_opt_cache_dir = Get<std::string>("model_opt_cache_dir");
  auto& scope = graph->Get<Scope>(kParamScopeAttr);
  framework::ProgramDesc optimized_program_desc;
  framework::ir::GraphToProgram(*graph, &optimized_program_desc);

  auto IsPersistable = [](const framework::VarDesc* var) {
    if (var->Persistable() &&
        var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
        var->GetType() != framework::proto::VarType::FETCH_LIST &&
        var->GetType() != framework::proto::VarType::RAW) {
      return true;
    }
    return false;
  };

  auto SerializeParams = [&](const std::string& path) {
    framework::ProgramDesc save_program;
    auto* save_block = save_program.MutableBlock(0);
    std::set<std::string> save_var_set;
    for (size_t i = 0; i < optimized_program_desc.Size(); ++i) {
      const auto& global_block = optimized_program_desc.Block(i);
      for (framework::VarDesc* var : global_block.AllVars()) {
        if (IsPersistable(var)) {
          framework::VarDesc* new_var = save_block->Var(var->Name());
          new_var->SetShape(var->GetShape());
          new_var->SetDataType(var->GetDataType());
          new_var->SetType(var->GetType());
          new_var->SetLoDLevel(var->GetLoDLevel());
          new_var->SetPersistable(true);
          save_var_set.insert(new_var->Name());
        }
      }
    }

    std::string save_params_path = path + "/" + "optimized.pdiparams";
    std::vector<std::string> save_var_list(save_var_set.begin(),
                                           save_var_set.end());
    std::sort(save_var_list.begin(), save_var_list.end());
    auto* op = save_block->AppendOp();
    op->SetType("save_combine");
    op->SetInput("X", save_var_list);
    op->SetAttr("file_path", save_params_path);
    op->CheckAttrs();

    framework::Executor exe(platform::CPUPlace{});
    exe.Run(save_program, &scope, 0, true, true);
  };
  // TODO(shentanyue01): Setting hardware and version identification for
  // optimized models.
  auto SerializeProg = [&](const std::string& path) {
    // All persistable var need to be moved to global block
    auto* global_block = optimized_program_desc.MutableBlock(0);
    for (size_t i = 1; i < optimized_program_desc.Size(); ++i) {
      const auto& sub_block = optimized_program_desc.Block(i);
      for (framework::VarDesc* var : sub_block.AllVars()) {
        if (IsPersistable(var) && !global_block->HasVar(var->Name())) {
          framework::VarDesc* new_var = global_block->Var(var->Name());
          new_var->SetShape(var->GetShape());
          new_var->SetDataType(var->GetDataType());
          new_var->SetType(var->GetType());
          new_var->SetLoDLevel(var->GetLoDLevel());
          new_var->SetPersistable(true);
        }
      }
    }
    std::string save_model_path = path + "/" + "optimized.pdmodel";
    auto str = optimized_program_desc.Proto()->SerializeAsString();
    std::ofstream file(save_model_path.c_str(), std::ios::binary);
    file.write(str.c_str(), str.size());
    file.close();
  };

  SerializeProg(model_opt_cache_dir);
  SerializeParams(model_opt_cache_dir);
  LOG(INFO) << "Optimized model saved to " << model_opt_cache_dir;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(save_optimized_model_pass,
              paddle::framework::ir::SaveOptimizedModelPass);
