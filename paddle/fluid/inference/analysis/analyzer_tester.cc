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
#include <google/protobuf/text_format.h>
#include "paddle/fluid/inference/analysis/ut_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

TEST(Analyzer, analysis_without_tensorrt) {
  FLAGS_inference_analysis_enable_tensorrt_subgraph_engine = false;
  Argument argument;
  argument.fluid_model_dir.reset(new std::string(FLAGS_inference_model_dir));
  Analyzer analyser;
  analyser.Run(&argument);
}

TEST(Analyzer, analysis_with_tensorrt) {
  FLAGS_inference_analysis_enable_tensorrt_subgraph_engine = true;
  Argument argument;
  argument.fluid_model_dir.reset(new std::string(FLAGS_inference_model_dir));
  Analyzer analyser;
  analyser.Run(&argument);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
