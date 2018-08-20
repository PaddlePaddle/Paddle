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

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/fluid/inference/analysis/model_store_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

void ModelStorePass::Run(DataFlowGraph *x) {
  if (!argument_->fluid_model_param_path) {
    PADDLE_ENFORCE_NOT_NULL(argument_->fluid_model_dir);
    argument_->fluid_model_param_path.reset(
        new std::string(*argument_->fluid_model_dir + "param"));
  }
  PADDLE_ENFORCE_NOT_NULL(argument_->model_output_store_path);
  // Directly copy param file to destination.
  std::stringstream ss;
  // NOTE these commands only works on linux.
  ss << "mkdir -p " << *argument_->model_output_store_path;
  VLOG(3) << "run command: " << ss.str();
  PADDLE_ENFORCE_EQ(system(ss.str().c_str()), 0);
  ss.str("");

  ss << "cp " << *argument_->fluid_model_dir << "/*"
     << " " << *argument_->model_output_store_path;
  VLOG(3) << "run command: " << ss.str();
  PADDLE_ENFORCE_EQ(system(ss.str().c_str()), 0);

  // Store program
  PADDLE_ENFORCE_NOT_NULL(argument_->transformed_program_desc,
                          "program desc is not transformed, should call "
                          "DataFlowGraphToFluidPass first.");
  VLOG(3) << "store analyzed program to "
          << *argument_->model_output_store_path;
  const std::string program_output_path =
      *argument_->model_output_store_path + "/__model__";
  std::ofstream file(program_output_path, std::ios::binary);
  PADDLE_ENFORCE(file.is_open(), "failed to open %s to write.",
                 program_output_path);
  const std::string serialized_message =
      argument_->transformed_program_desc->SerializeAsString();
  file.write(serialized_message.c_str(), serialized_message.size());
}

bool ModelStorePass::Finalize() { return true; }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
