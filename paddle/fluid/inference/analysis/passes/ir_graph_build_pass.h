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

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/analysis_pass.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Load program and parameter to memory from the disk or directly from memory.
 */
class IrGraphBuildPass : public AnalysisPass {
 public:
  void RunImpl(Argument *argument) override;

  std::string repr() const override;

 private:
  std::unique_ptr<framework::ProgramDesc> LoadModel(const std::string &path,
                                                    framework::Scope *scope,
                                                    const phi::Place &place);
  std::unique_ptr<framework::ProgramDesc> LoadModel(
      const std::string &program_path,
      const std::string &params_path,
      framework::Scope *scope,
      const phi::Place &place,
      bool model_from_memory,
      bool skip_load_params);

  std::string model_binary_str_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
