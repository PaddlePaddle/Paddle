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
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {

using inference::analysis::Argument;
using inference::analysis::Analyzer;
using framework::proto::ProgramDesc;

/* This predictor is based on the original native predictor with IR and Analysis
 * support. It will optimize IR and Parameters in the runtime.
 * TODO(Superjomn) Replace the Navive predictor?
 */
class AnalysisPredictor : public NativePaddlePredictor {
 public:
  explicit AnalysisPredictor(const contrib::AnalysisConfig& config)
      : NativePaddlePredictor(config), config_(config) {}

  bool Init(const std::shared_ptr<framework::Scope>& parent_scope);

  bool Run(const std::vector<PaddleTensor>& inputs,
           std::vector<PaddleTensor>* output_data,
           int batch_size = -1) override {
    return NativePaddlePredictor::Run(inputs, output_data, batch_size);
  }

  void OptimizeInferenceProgram();

  Argument& analysis_argument() { return argument_; }

 private:
  contrib::AnalysisConfig config_;
  Argument argument_;
};

}  // namespace paddle
