/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/tests/api/analyzer_seq_pool1_tester_helper.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {
namespace seq_pool1_tester {

void profile(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg, use_mkldnn);

  std::vector<std::vector<PaddleTensor>> outputs;
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, FLAGS_num_threads);
}

TEST(Analyzer_seq_pool1_profile, profile) { profile(); }

}  // namespace seq_pool1_tester
}  // namespace analysis
}  // namespace inference
}  // namespace paddle
