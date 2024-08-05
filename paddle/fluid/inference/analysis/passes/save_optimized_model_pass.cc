/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/analysis/passes/save_optimized_model_pass.h"

#include <unordered_set>
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle::inference::analysis {

void SaveOptimizedModelPass::SaveOptimizedModel(Argument* argument) {
  std::string model_opt_cache_dir = argument->optimized_model_save_path();

  auto& scope = argument->scope();
  auto* graph = argument->main_graph_ptr();

  framework::ProgramDesc optimized_program_desc;

  // NOTE(liuyuanle): If the following line of code is not added, an error
  // [SegmentFault] may occur!
  optimized_program_desc.CopyFrom(*argument->main_program().Proto());

  framework::ir::GraphToProgram(*graph, &optimized_program_desc);

  // TODO(minghaipeng): Move the following code to a separate clean pass.
  // Remove the scale and zero point parameters from optimized program.
  auto scale_and_zero_point_param = graph->GetOrInit<std::vector<std::string>>(
      framework::ir::kScaleAndZeroPointParamAttr);
  framework::BlockDesc* block = optimized_program_desc.MutableBlock(0);
  for (auto& var_desc : block->AllVars()) {
    auto var_name = var_desc->Name();
    if (var_desc->Persistable() && scope.FindVar(var_name) &&
        std::count(scale_and_zero_point_param.begin(),
                   scale_and_zero_point_param.end(),
                   var_name) > 0) {
      scope.EraseVars({var_name});
      block->RemoveVar(var_desc->Name());
    }
  }

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
    std::unordered_set<std::string> save_var_set;
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

    std::string save_params_path = path + "/" + "_optimized.pdiparams";
    std::vector<std::string> save_var_list(save_var_set.begin(),
                                           save_var_set.end());
    std::sort(save_var_list.begin(), save_var_list.end());
    auto* op = save_block->AppendOp();
    op->SetType("save_combine");
    op->SetInput("X", save_var_list);
    op->SetAttr("file_path", save_params_path);
    op->CheckAttrs();

    framework::Executor exe(phi::CPUPlace{});
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
    std::string save_model_path = path + "/" + "_optimized.pdmodel";
    auto str = optimized_program_desc.Proto()->SerializeAsString();
    std::ofstream file(save_model_path.c_str(), std::ios::binary);
    file.write(str.c_str(), str.size());  // NOLINT
    file.close();
  };

  SerializeProg(model_opt_cache_dir);
  SerializeParams(model_opt_cache_dir);
  LOG(INFO) << "Optimized model saved to " << model_opt_cache_dir;
}

void SaveOptimizedModelPass::RunImpl(Argument* argument) {
  if (!argument->save_optimized_model() || !argument->enable_ir_optim()) {
    return;
  }
  SaveOptimizedModel(argument);
}

std::string SaveOptimizedModelPass::repr() const {
  return "save_optimized_model_pass";
}

}  // namespace paddle::inference::analysis
