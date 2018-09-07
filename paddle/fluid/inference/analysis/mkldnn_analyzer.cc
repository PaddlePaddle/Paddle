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

#include <string>
#include <vector>
#include "paddle/fluid/inference/analysis/fluid_to_ir_pass.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/mkldnn_analyzer.h"

namespace paddle {
namespace inference {
namespace analysis {

void MKLDNNAnalyzer::Run(Argument* argument) {
  std::vector<std::string> passes;
  for (auto& pass : ir_passes_) {
    passes.push_back(pass);
    passes.push_back("graph_viz_pass");  // add graphviz for debug.
  }
  passes.push_back("graph_viz_pass");
  // Ugly support fluid-to-ir-pass
  argument->Set(kFluidToIrPassesAttr, new std::vector<std::string>(passes));

  for (auto& x : data_) {
    PADDLE_ENFORCE(x->Initialize(argument));
    x->RunAll();
    PADDLE_ENFORCE(x->Finalize());
  }
}

MKLDNNAnalyzer& MKLDNNAnalyzer::SetIrPasses(
		const std::vector<std::string>& passes) {
  ir_passes_ = passes;
  return *this;
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
