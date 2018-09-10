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

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/inference/analysis/analysis_pass.h"
#include "paddle/fluid/inference/analysis/flags.h"
#include "paddle/fluid/inference/analysis/ir_pass_manager.h"

namespace paddle {
namespace inference {
namespace analysis {

static const char kFluidToIrPassesAttr[] = "__fluid_to_ir_passes__";

class FluidToIrPass final : public DataFlowGraphPass {
 public:
  FluidToIrPass() = default;

  bool Initialize(Argument *argument) override {
    ANALYSIS_ARGUMENT_CHECK_FIELD(argument);
    PADDLE_ENFORCE(argument->Has(kFluidToIrPassesAttr),
                   "argument need the attr %s", kFluidToIrPassesAttr);
    argument_ = argument;
    if (argument->origin_program_desc) {
      LOG(WARNING) << "argument's origin_program_desc is already set, might "
                      "duplicate called";
    }
    // set fluid model program path
    if (!argument->fluid_model_program_path) {
      ANALYSIS_ARGUMENT_CHECK_FIELD(argument->fluid_model_dir);
      argument->fluid_model_program_path.reset(
          new std::string(*argument->fluid_model_dir + "/__model__"));
    }
    ANALYSIS_ARGUMENT_CHECK_FIELD(argument->fluid_model_program_path);
    // Load program.
    auto program = LoadProgramDesc(*argument->fluid_model_program_path);
    argument->origin_program_desc.reset(
        new framework::proto::ProgramDesc(program));
    // Create main data flow graph.
    if (!argument->main_dfg) {
      argument->main_dfg.reset(new DataFlowGraph);
    }
    argument->Set("ir_program_desc", new ProgramDesc(program));

    LOG(INFO) << "Loading parameters";
    // Load parameters to argument if needed.
    if (argument->fluid_model_dir || (argument->fluid_model_program_path &&
                                      argument->fluid_model_param_path)) {
#define SAFE_GET(ATTR) std::string ATTR = argument->ATTR ? *argument->ATTR : "";
      SAFE_GET(fluid_model_dir);
      SAFE_GET(fluid_model_program_path);
      SAFE_GET(fluid_model_param_path);
#undef SAFE_GET
      EnableParamModify(fluid_model_dir, fluid_model_program_path,
                        fluid_model_param_path);
    }

    return true;
  }

  bool Finalize() override { return true; }

  void Run(DataFlowGraph *graph) override {
    // Call all the IR Passes
    IRPassManager ir_passes(argument_->Get<ProgramDesc>("ir_program_desc"),
                            nullptr);
    // Pass the scope from analysis to IR if needed.
    if (argument_->Has(framework::ir::kParamScopeAttr)) {
      // Here the address is passed, attention that IR doesn't own the scope, so
      // the real scope in analysis should live during the IR phase.
      ir_passes.graph().Set(
          framework::ir::kParamScopeAttr,
          new framework::Scope *(&argument_->Get<framework::Scope>(
              framework::ir::kParamScopeAttr)));
    }

    if (FLAGS_IA_enable_ir) {
      const auto &ir_passes_to_apply =
          argument_->Get<std::vector<std::string>>(kFluidToIrPassesAttr);
      ir_passes.Apply(ir_passes_to_apply);
    }

    PADDLE_ENFORCE(argument_->main_dfg.get());
    argument_->main_dfg->Build(ir_passes.graph());
    // inherit the arguments from ir.
    if (ir_passes.graph().Has(framework::ir::kFuseStatisAttr)) {
      argument_->Set(
          framework::ir::kFuseStatisAttr,
          new std::unordered_map<std::string, int>(
              ir_passes.graph().Get<std::unordered_map<std::string, int>>(
                  framework::ir::kFuseStatisAttr)));
    }
  }

  void EnableParamModify(const std::string &model_dir,
                         const std::string &prog_file,
                         const std::string &param_file);

  std::string repr() const override { return "fluid-to-ir-pass"; }

 private:
  // Load parameters from a single file or from a directory.
  bool LoadParams(framework::Scope *scope, const std::string &dir,
                  const std::string &prog_file, const std::string &param_file);

 private:
  Argument *argument_{nullptr};
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
