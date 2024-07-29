// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/passes/convert_to_mixed_precision.h"

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/ir/auto_mixed_precision_pass.h"
#include "paddle/fluid/framework/ir/constant_folding_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/identity_op_clean_pass.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/phi/common/backend.h"

namespace paddle {
namespace inference {
namespace analysis {

ConvertToMixedPrecisionPass::ConvertToMixedPrecisionPass(
    const std::string& model_file,
    const std::string& params_file,
    const std::string& mixed_model_file,
    const std::string& mixed_params_file,
    phi::DataType mixed_precision,
    phi::Backend backend,
    bool keep_io_types,
    const std::unordered_set<std::string>& black_list,
    const std::unordered_set<std::string>& white_list)
    : model_file_(model_file),
      params_file_(params_file),
      mixed_model_file_(mixed_model_file),
      mixed_params_file_(mixed_params_file),
      mixed_precision_(mixed_precision),
      backend_(backend),
      keep_io_types_(keep_io_types),
      black_list_(black_list),
      white_list_(white_list) {
  switch (backend_) {
    case phi::Backend::GPU:
      PADDLE_ENFORCE(mixed_precision_ == phi::DataType::FLOAT16 ||
                         mixed_precision_ == phi::DataType::BFLOAT16,
                     common::errors::InvalidArgument(
                         "mixed_precision of %s currently only supported fp16 "
                         "and bf16, not support %s.",
                         experimental::BackendToString(backend_),
                         phi::DataTypeToString(mixed_precision_)));
      break;
    case phi::Backend::XPU:
    case phi::Backend::CUSTOM:
      PADDLE_ENFORCE(mixed_precision_ == phi::DataType::FLOAT16,
                     common::errors::InvalidArgument(
                         "mixed_precision of %s currently only supported fp16 "
                         "and bf16, not support %s.",
                         experimental::BackendToString(backend_),
                         phi::DataTypeToString(mixed_precision_)));
      break;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "mixed_precision currently not supported place GPU or XPU or CUSTOM, "
          "not support %s.",
          experimental::BackendToString(backend_)));
      break;
  }
}

void ConvertToMixedPrecisionPass::LoadModel() {
  framework::Executor exe{phi::CPUPlace{}};
  // If we did not find the provided weight path,
  // we assume that the model to be converted only has a model file and no
  // params file, we believe this situation is reasonable. In this case, weight
  // data may not be loaded.
  bool load_params = !params_file_.empty();
  auto program_desc =
      inference::Load(&exe, &scope_, model_file_, params_file_, load_params);
  main_graph_ = std::make_unique<framework::ir::Graph>(*program_desc);
  main_graph_->SetNotOwned(framework::ir::kParamScopeAttr, &scope_);
}

void ConvertToMixedPrecisionPass::Run() {
  LoadModel();

  framework::ir::ConstantFoldingPass constant_folding_pass;
  constant_folding_pass.Apply(main_graph_.get());

  framework::ir::AutoMixedPrecisionPass auto_mixed_precision_pass;
  auto_mixed_precision_pass.Set("mixed_precision_mode",
                                new int{static_cast<int>(mixed_precision_)});
  if (backend_ == phi::Backend::GPU) {
    auto_mixed_precision_pass.Set("enable_gpu_mixed", new bool{true});
  } else if (backend_ == phi::Backend::XPU) {
    auto_mixed_precision_pass.Set("enable_xpu_mixed", new bool{true});
  } else if (backend_ == phi::Backend::CUSTOM) {
    auto_mixed_precision_pass.Set("enable_custom_device_mixed", new bool{true});
  }
  auto_mixed_precision_pass.Set(
      "mixed_black_list", new std::unordered_set<std::string>{black_list_});
  auto_mixed_precision_pass.Set(
      "mixed_white_list", new std::unordered_set<std::string>{white_list_});
  auto_mixed_precision_pass.Set("enable_low_precision_io",
                                new bool{!keep_io_types_});
  auto_mixed_precision_pass.Apply(main_graph_.get());

  framework::ir::IdentityOpCleanPass identity_op_clean_pass;
  identity_op_clean_pass.Apply(main_graph_.get());

  SaveMixedModel();
}

void ConvertToMixedPrecisionPass::SaveMixedModel() {
  framework::ProgramDesc mixed_program_desc;
  framework::ir::GraphToProgram(*main_graph_, &mixed_program_desc);

  auto SerializeParams = [&](const std::string& path) {
    auto IsPersistable = [](const framework::VarDesc* var) {
      if (var->Persistable() &&
          var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
          var->GetType() != framework::proto::VarType::FETCH_LIST &&
          var->GetType() != framework::proto::VarType::RAW) {
        return true;
      }
      return false;
    };
    framework::ProgramDesc save_program;
    auto* save_block = save_program.MutableBlock(0);

    const auto& global_block = mixed_program_desc.Block(0);
    std::vector<std::string> save_var_list;
    bool has_persistable_var = false;
    for (framework::VarDesc* var : global_block.AllVars()) {
      if (IsPersistable(var)) {
        framework::VarDesc* new_var = save_block->Var(var->Name());
        new_var->SetShape(var->GetShape());
        new_var->SetDataType(var->GetDataType());
        new_var->SetType(var->GetType());
        new_var->SetLoDLevel(var->GetLoDLevel());
        new_var->SetPersistable(true);

        save_var_list.push_back(new_var->Name());
        has_persistable_var = true;
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
      save_params_path = mixed_model_file_;
      std::string::size_type pos = save_params_path.rfind(".pdmodel");
      if (pos != std::string::npos) {
        save_params_path.replace(pos, 8, ".pdiparams");
        LOG(WARNING) << " The storage path of the converted mixed-precision "
                        "params has been created: ["
                     << save_params_path << "]";
      }
    }

    std::sort(save_var_list.begin(), save_var_list.end());
    auto* op = save_block->AppendOp();
    op->SetType("save_combine");
    op->SetInput("X", save_var_list);
    op->SetAttr("file_path", save_params_path);
    op->CheckAttrs();

    framework::Executor exe(phi::CPUPlace{});
    exe.Run(save_program, &scope_, 0, true, true);
  };

  auto SerializeProg = [&](const std::string& path) {
    auto str = mixed_program_desc.Proto()->SerializeAsString();
    std::ofstream file(path.c_str(), std::ios::binary);
    file.write(str.c_str(), str.size());  // NOLINT
    file.close();
  };

  SerializeProg(mixed_model_file_);
  SerializeParams(mixed_params_file_);
}

bool OpSupportPrecision(const std::string& op_type,
                        phi::Backend backend,
                        phi::DataType precision,
                        const std::unordered_set<std::string>& black_list,
                        const std::unordered_set<std::string>& white_list) {
  return framework::ir::OpSupportPrecision(
      op_type, backend, precision, black_list, white_list);
}

void InsertCastOp(
    framework::ir::Graph* graph,
    framework::ir::Node* var_node,
    framework::ir::Node* op_node,
    framework::proto::VarType::Type from_type,
    framework::proto::VarType::Type to_type,
    framework::BlockDesc* block_desc,
    int* suffix,
    std::unordered_map<framework::ir::Node*, framework::ir::Node*>* visited) {
  framework::ir::DoInsertCastOp(graph,
                                var_node,
                                op_node,
                                from_type,
                                to_type,
                                block_desc,
                                suffix,
                                visited);
}

void ConvertToMixedPrecision(
    const std::string& model_file,
    const std::string& params_file,
    const std::string& mixed_model_file,
    const std::string& mixed_params_file,
    phi::DataType mixed_precision,
    phi::Backend backend,
    bool keep_io_types,
    const std::unordered_set<std::string>& black_list,
    const std::unordered_set<std::string>& white_list) {
  ConvertToMixedPrecisionPass pass(model_file,
                                   params_file,
                                   mixed_model_file,
                                   mixed_params_file,
                                   mixed_precision,
                                   backend,
                                   keep_io_types,
                                   black_list,
                                   white_list);
  pass.Run();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
