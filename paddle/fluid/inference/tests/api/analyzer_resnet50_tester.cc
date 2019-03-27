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

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/model", FLAGS_infer_model + "/params");
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->SwitchSpecifyInputNames();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  SetFakeImageInput(inputs, FLAGS_infer_model);
}

// Easy for profiling independently.
void profile(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  cfg.EnableUseGpu(2000);

  if (use_mkldnn) {
    cfg.EnableMKLDNN();
  }
  std::vector<PaddleTensor> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, FLAGS_num_threads);
  LOG(INFO) << "outputs " << outputs.size();
  LOG(INFO) << DescribeTensor(outputs[0], 8);
}

TEST(Analyzer_resnet50, profile) { profile(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_resnet50, profile_mkldnn) { profile(true /* use_mkldnn */); }
#endif

// Check the fuse status
TEST(Analyzer_resnet50, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  LOG(INFO) << "num_ops: " << num_ops;
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
  }
  cfg.EnableUseGpu(100);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

TEST(Analyzer_resnet50, compare) { compare(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_resnet50, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

// Compare Deterministic result
TEST(Analyzer_resnet50, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

TEST(test, raw_mat) {

}

TEST(test, test) {
  AnalysisConfig config("/home/chunwei/project2/models/model");
  config.EnableUseGpu(100);
  //config.pass_builder()->DeletePass("identity_scale_op_clean_pass");
  config.SwitchIrDebug();

  std::vector<PaddleTensor> inputs, outputs;
  inputs.resize(2);

  const int batch_size = 1000;
  std::srand(0);

  auto fill_tensor = [&](PaddleTensor *tensor) {
    tensor->shape.assign({batch_size, 100});
    tensor->data.Resize(batch_size * 100 * sizeof(float));
    tensor->dtype = PaddleDType::FLOAT32;
    for (int i = 0; i < 100 * batch_size; i++) {
      static_cast<float *>(tensor->data.data())[i] =
          std::rand() * 1. / (RAND_MAX);
    }
  };

  fill_tensor(&(inputs[0]));
  fill_tensor(&(inputs[1]));

  auto pr = CreatePaddlePredictor(config);
  ASSERT_TRUE(pr->Run(inputs, &outputs));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
