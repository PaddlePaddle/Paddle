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

#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->param_file = FLAGS_infer_model + "/params";
  cfg->prog_file = FLAGS_infer_model + "/model";
  cfg->use_gpu = false;
  cfg->device = 0;
  cfg->enable_ir_optim = true;
  cfg->specify_input_name = true;
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  SetFakeImageInput(inputs, FLAGS_infer_model);
}

// Easy for profiling independently.
TEST(Analyzer_face, profile) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  std::vector<PaddleTensor> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(cfg, input_slots_all, &outputs, FLAGS_num_threads);
}

// Check the fuse status
TEST(Analyzer_face, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
}

// Compare result of NativeConfig and AnalysisConfig
TEST(Analyzer_face, compare) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(cfg, input_slots_all);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
