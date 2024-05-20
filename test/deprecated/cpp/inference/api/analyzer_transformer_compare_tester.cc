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

#include "test/cpp/inference/api/analyzer_transformer_tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {
namespace transformer_tester {

void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (!use_mkldnn) {
    cfg.DisableMKLDNN();
  }

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

TEST(Analyzer_Transformer, compare) { compare(); }
#ifdef PADDLE_WITH_DNNL
TEST(Analyzer_Transformer, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

}  // namespace transformer_tester
}  // namespace analysis
}  // namespace inference
}  // namespace paddle
