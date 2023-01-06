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
#include "paddle/fluid/framework/ir/graph_helper.h"
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
    const std::unordered_set<std::string>& black_list)
    : model_file_(model_file),
      params_file_(params_file),
      mixed_model_file_(mixed_model_file),
      mixed_params_file_(mixed_params_file),
      mixed_precision_(mixed_precision),
      backend_(backend),
      keep_io_types_(keep_io_types),
      black_list_(black_list) {
  if (mixed_precision_ != phi::DataType::FLOAT16 &&
      mixed_precision_ != phi::DataType::BFLOAT16) {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "mixed_precision currently not supported dtype %d, we now only "
        "support fp16 and bf16.",
        static_cast<int>(mixed_precision_)));
  }
  if (backend_ != phi::Backend::GPU) {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "mixed_precision currently not supported place %d, we now only "
        "support gpu.",
        static_cast<int>(backend_)));
  }
}

void ConvertToMixedPrecisionPass::LoadModel() {
  framework::Executor exe{platform::CPUPlace{}};

  auto program_desc = inference::Load(&exe, &scope_, model_file_, params_file_);
  main_graph_ = std::unique_ptr<framework::ir::Graph>(
      new framework::ir::Graph(*program_desc));
  main_graph_->SetNotOwned(framework::ir::kParamScopeAttr, &scope_);
}

void ConvertToMixedPrecisionPass::Run() {
  LoadModel();

  framework::ir::AutoMixedPrecisionPass pass;
  pass.Set("mixed_precision_mode", new int{static_cast<int>(mixed_precision_)});
  pass.Set("mixed_black_list",
           new std::unordered_set<std::string>{black_list_});
  pass.Set("enable_gpu_mixed", new bool{true});
  pass.Set("keep_io_types", new bool{keep_io_types_});

  pass.Apply(main_graph_.get());

  SaveMixedModel();
}

void ConvertToMixedPrecisionPass::SaveMixedModel() {
  framework::ProgramDesc mixed_program_desc;
  framework::ir::GraphToProgram(*main_graph_, &mixed_program_desc);

  auto parameters = scope_.LocalVarNames();
  std::sort(parameters.begin(), parameters.end());

  auto SerializeParams = [&]() -> std::string {
    std::ostringstream os;
    phi::CPUContext ctx;
    for (const auto& param : parameters) {
      PADDLE_ENFORCE_NOT_NULL(
          scope_.FindVar(param),
          platform::errors::NotFound(
              "Block should already have a '%s' variable", param));
      auto* tensor = scope_.FindVar(param)->GetMutable<phi::DenseTensor>();
      framework::SerializeToStream(os, *tensor, ctx);
    }
    return os.str();
  };

  auto StrToBinary = [](const std::string& path, const std::string& str) {
    std::ofstream file(path.c_str(), std::ios::binary);
    file.write(str.c_str(), str.size());
    file.close();
  };

  StrToBinary(mixed_model_file_,
              mixed_program_desc.Proto()->SerializeAsString());
  StrToBinary(mixed_params_file_, SerializeParams());
}

bool OpSupportPrecision(const std::string& op_type,
                        phi::Backend backend,
                        phi::DataType precision,
                        const std::unordered_set<std::string>& black_list) {
  return framework::ir::OpSupportPrecision(
      op_type, backend, precision, black_list);
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
    const std::unordered_set<std::string>& black_list) {
  ConvertToMixedPrecisionPass pass(model_file,
                                   params_file,
                                   mixed_model_file,
                                   mixed_params_file,
                                   mixed_precision,
                                   backend,
                                   keep_io_types,
                                   black_list);
  pass.Run();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
