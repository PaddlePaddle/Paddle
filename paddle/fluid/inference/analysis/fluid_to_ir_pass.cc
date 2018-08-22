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

#include "paddle/fluid/inference/analysis/fluid_to_ir_pass.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace inference {
namespace analysis {

void FluidToIrPass::EnableParamModify(const std::string &model_dir,
                                      const std::string &prog_file,
                                      const std::string &param_file) {
  argument_->Set("param_scope",
                 new std::unique_ptr<framework::Scope>(new framework::Scope));
  // Load parameters.
  VLOG(3) << "Loading parameters from " << model_dir;
  LoadParams(&argument_->Get<framework::Scope>("param_scope"), model_dir,
             prog_file, param_file);
}

void FluidToIrPass::LoadParams(framework::Scope *scope, const std::string &dir,
                               const std::string &prog_file,
                               const std::string &param_file) {
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  framework::Executor executor(place);
  PADDLE_ENFORCE(argument_->origin_program_desc.get());
  framework::ProgramDesc program(*argument_->origin_program_desc);
  LoadPersistables(&executor, scope, program, dir, param_file);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
