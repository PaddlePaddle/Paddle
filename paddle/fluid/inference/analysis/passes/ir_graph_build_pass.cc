// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/passes/ir_graph_build_pass.h"

#include <memory>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::inference {

extern void ReadBinaryFile(const std::string &filename, std::string *contents);

}  // namespace paddle::inference
namespace paddle::inference::analysis {

void IrGraphBuildPass::RunImpl(Argument *argument) {
  if (!argument->scope_valid()) {
    argument->SetScope(new framework::Scope);
  }
  PADDLE_ENFORCE_EQ(
      argument->use_gpu_valid(),
      true,
      common::errors::PreconditionNotMet("The use_gpu field should be valid"));

  // The load program should run on the same device with the inference program,
  // so that the parameters will on the same device, or they will keep copying
  // between difference devices.
  phi::Place place;
  place = phi::CPUPlace();

  if (argument->model_dir_valid()) {
    auto program =
        LoadModel(argument->model_dir(), argument->scope_ptr(), place);
    argument->SetMainProgram(program.release());
  } else if (argument->model_program_path_valid() &&
             argument->model_params_path_valid()) {
    auto program = LoadModel(
        argument->model_program_path(),
        argument->model_params_path(),
        argument->scope_ptr(),
        place,
        argument->model_from_memory_valid() && argument->model_from_memory(),
        argument->skip_load_params());
    argument->SetMainProgram(program.release());
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "either model_dir or (program path and parameter path) should be "
        "set."));
  }

  auto graph = std::make_unique<framework::ir::Graph>(argument->main_program());
  argument->SetMainGraph(graph.release());
  auto *scope_ptr = argument->scope_ptr();
  PADDLE_ENFORCE_NOT_NULL(scope_ptr,
                          common::errors::PreconditionNotMet(
                              "The scope ptr should not be nullptr."));
  argument->main_graph().SetNotOwned(framework::ir::kParamScopeAttr, scope_ptr);

// ipu related
#ifdef PADDLE_WITH_IPU
  if (argument->Has("use_ipu")) {
    if (argument->use_ipu()) {
      argument->main_graph().SetNotOwned("num_ipus",
                                         &argument->ipu_device_num());
      argument->main_graph().SetNotOwned("micro_batch_size",
                                         &argument->ipu_micro_batch_size());
      argument->main_graph().SetNotOwned("enable_pipelining",
                                         &argument->ipu_enable_pipelining());
      argument->main_graph().SetNotOwned("batches_per_step",
                                         &argument->ipu_batches_per_step());
      argument->main_graph().SetNotOwned("enable_fp16",
                                         &argument->ipu_enable_fp16());
      argument->main_graph().SetNotOwned("replica_num",
                                         &argument->ipu_replica_num());
      argument->main_graph().SetNotOwned(
          "available_memory_proportion",
          &argument->ipu_available_memory_proportion());
      argument->main_graph().SetNotOwned("enable_half_partial",
                                         &argument->ipu_enable_half_partial());
      argument->main_graph().SetNotOwned("custom_ops_info",
                                         &argument->ipu_custom_ops_info());
      argument->main_graph().SetNotOwned("custom_patterns",
                                         &argument->ipu_custom_patterns());
      argument->main_graph().SetNotOwned(
          "enable_model_runtime_executor",
          &argument->ipu_enable_model_runtime_executor());
    }
  }
#endif
}

std::unique_ptr<framework::ProgramDesc> IrGraphBuildPass::LoadModel(
    const std::string &path, framework::Scope *scope, const phi::Place &place) {
  framework::Executor exe(place);
  return Load(&exe, scope, path);
}

std::unique_ptr<framework::ProgramDesc> IrGraphBuildPass::LoadModel(
    const std::string &program_path,
    const std::string &params_path,
    framework::Scope *scope,
    const phi::Place &place,
    bool model_from_memory,
    bool skip_load_params) {
  framework::Executor exe(place);
  if (!model_from_memory) {  // NOLINT
    return Load(&exe, scope, program_path, params_path, !skip_load_params);
  } else {
    return LoadFromMemory(&exe, scope, program_path, params_path);
  }
}

std::string IrGraphBuildPass::repr() const { return "ir_graph_build_pass"; }

}  // namespace paddle::inference::analysis
