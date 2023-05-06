// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/passes/save_optimized_model_pass.h"

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/phi/common/backend.h"

namespace paddle {
namespace inference {
namespace analysis {

SaveOptimizedModelPass::SaveOptimizedModelPass(
    const std::string& model_file,
    const std::string& params_file,
    const std::string& output_model_file,
    const std::string& output_params_file,
    phi::Backend backend,
    const std::unordered_set<std::string>& black_list)
    : model_file_(model_file),
      params_file_(params_file),
      output_model_file_(output_model_file),
      output_params_file_(output_params_file),
      backend_(backend),
      black_list_(black_list) {
  switch (backend_) {
    case phi::Backend::XPU:
    case phi::Backend::GPU:
    case phi::Backend::CUSTOM:
    case phi::Backend::CPU:
      break;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "save optimized model currently supported place XPU, CPU, GPU or "
          "CUSTOM DEVICE "
          "not support %s.",
          experimental::BackendToString(backend_)));
      break;
  }
}

void SaveOptimizedModelPass::LoadModel() {
  framework::Executor exe{platform::CPUPlace{}};
  // If we did not find the provided weight path,
  // we assume that the model to be converted only has a model file and no
  // params file, we believe this situation is reasonable. In this case, weight
  // data may not be loaded.
  bool load_params = !params_file_.empty();
  auto program_desc =
      inference::Load(&exe, &scope_, model_file_, params_file_, load_params);
  main_graph_ = std::unique_ptr<framework::ir::Graph>(
      new framework::ir::Graph(*program_desc));
  main_graph_->SetNotOwned<framework::Scope>(framework::ir::kParamScopeAttr,
                                             &scope_);
}

void SaveOptimizedModelPass::SaveModel() {
  framework::ProgramDesc optimized_program_desc;
  framework::ir::GraphToProgram(*main_graph_, &optimized_program_desc);

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
    bool has_persistable_var = false;
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
          has_persistable_var = true;
        }
      }
    }

    std::string save_params_path = path;
    if (save_params_path.empty() && has_persistable_var) {
      LOG(WARNING)
          << "The [SerializeParams] function did not find the provided weight "
             "path, "
             "so we assume that the model to be converted only has a model "
             "file and no params file, "
             "we believe this situation is reasonable. After constant folding, "
             "a weight file will be generated, which is saved in the same "
             "level file directory "
             "as the model file by default and ends in pdiparams.";
      save_params_path = output_model_file_;
      std::string::size_type pos = save_params_path.rfind(".pdmodel");
      if (pos != std::string::npos) {
        save_params_path.replace(pos, 8, ".pdiparams");
        LOG(WARNING)
            << " The storage path of the optimized params has been created: ["
            << save_params_path << "]";
      }
    }

    std::vector<std::string> save_var_list(save_var_set.begin(),
                                           save_var_set.end());
    std::sort(save_var_list.begin(), save_var_list.end());
    auto* op = save_block->AppendOp();
    op->SetType("save_combine");
    op->SetInput("X", save_var_list);
    op->SetAttr("file_path", save_params_path);
    op->CheckAttrs();

    framework::Executor exe(platform::CPUPlace{});
    exe.Run(save_program, &scope_, 0, true, true);
  };

  auto SerializeProg = [&](const std::string& path) {
    // TODO(shentanyue01): Setting hardware and version identification for
    // optimized models. All persistable var need to be moved to global block
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
    auto str = optimized_program_desc.Proto()->SerializeAsString();
    std::ofstream file(path.c_str(), std::ios::binary);
    file.write(str.c_str(), str.size());
    file.close();
  };

  SerializeProg(output_model_file_);
  SerializeParams(output_params_file_);
}

void SaveOptimizedModelPass::Run() {
  LoadModel();
  // main graph optimize pass
  std::unique_ptr<PassStrategy> pass_builder_;
  // sub graph optimize pass
  std::vector<std::string> subgraph_passes =
      framework::ir::support_subgraph_passes;
  switch (backend_) {
    case phi::Backend::XPU:
      LOG(INFO) << "Create XPU IR passes";
      pass_builder_.reset(new XpuPassStrategy);
      subgraph_passes = framework::ir::xpu_support_subgraph_passes;
      break;
    case phi::Backend::GPU:
      LOG(INFO) << "Create GPU IR passes";
      pass_builder_.reset(new GpuPassStrategy);
      break;
    case phi::Backend::CUSTOM:
      LOG(INFO) << "Create CUSTOM DEVICE IR passes";
      pass_builder_.reset(new CustomDevicePassStrategy);
      break;
    case phi::Backend::CPU:
      LOG(INFO) << "Create CPU IR passes";
      pass_builder_.reset(new CpuPassStrategy);
      break;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "save optimized model currently supported place XPU, CPU, GPU or "
          "CUSTOM DEVICE "
          "not support %s.",
          experimental::BackendToString(backend_)));
      break;
  }

  auto passes_ = pass_builder_->AllPasses();
  for (const auto& pass_name : passes_) {
    if (black_list_.find(pass_name) != black_list_.end()) {
      continue;
    }
    auto pass = framework::ir::PassRegistry::Instance().Get(pass_name);
    LOG(INFO) << "--- Running IR pass: " << pass->Type();
    main_graph_.reset(pass->Apply(main_graph_.release()));
    if (main_graph_->IsMainGraph() &&
        std::count(
            subgraph_passes.begin(), subgraph_passes.end(), pass->Type())) {
      for (size_t i = 1; i < main_graph_->SubGraphsSize(); i++) {
        auto* sub_graph = main_graph_->GetSubGraph(i);
        if (!sub_graph->Has(framework::ir::kParamScopeAttr)) {
          sub_graph->SetNotOwned(framework::ir::kParamScopeAttr, &scope_);
        }
        pass->Apply(sub_graph);
        if (!sub_graph->Has(framework::ir::kPassRecorder)) {
          sub_graph->Set<framework::ir::PassRecorder>(
              framework::ir::kPassRecorder, new framework::ir::PassRecorder);
        }
        sub_graph
            ->Get<framework::ir::PassRecorder>(framework::ir::kPassRecorder)
            .insert(pass->Type());
      }
    }
  }
  SaveModel();
}

void SaveOptimizedModel(const std::string& model_file,
                        const std::string& params_file,
                        const std::string& output_model_file,
                        const std::string& output_params_file,
                        phi::Backend backend,
                        const std::unordered_set<std::string>& black_list) {
  SaveOptimizedModelPass pass(model_file,
                              params_file,
                              output_model_file,
                              output_params_file,
                              backend,
                              black_list);
  pass.Run();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
