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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include <string>
#include <vector>
#include "paddle/fluid/inference/analysis/passes/passes.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {
namespace analysis {

Analyzer::Analyzer() {}

void Analyzer::Run(Argument *argument) { RunAnalysis(argument); }

void Analyzer::RunAnalysis(Argument *argument) {
  PADDLE_ENFORCE(argument->analysis_passes_valid(),
                 "analsis_passes is not valid in the argument.");
  for (auto &pass : argument->analysis_passes()) {
    string::PrettyLogH1("--- Running analysis [%s]", pass);
    auto *ptr = PassRegistry::Global().Retreive(pass);
    PADDLE_ENFORCE_NOT_NULL(ptr, "no analysis pass called %s", pass);
    ptr->Run(argument);
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
