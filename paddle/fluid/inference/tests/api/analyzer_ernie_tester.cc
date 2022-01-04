// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tests/api/analyzer_ernie_tester.h"

namespace paddle {
namespace inference {

using paddle::PaddleTensor;

void profile(bool use_mkldnn = false, bool use_gpu = false) {
  AnalysisConfig config;

  SetConfig(&config, use_mkldnn, use_gpu);

  std::vector<std::vector<PaddleTensor>> outputs;
  std::vector<std::vector<PaddleTensor>> inputs;
  LoadInputData(&inputs);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&config),
                 inputs, &outputs, FLAGS_num_threads);
}

TEST(Analyzer_ernie, profile) { profile(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_ernie, profile_mkldnn) { profile(true, false); }
#endif

// Check the model by gpu
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(Analyzer_ernie, profile_gpu) { profile(false, true); }
#endif

// Check the fuse status
TEST(Analyzer_Ernie, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  ASSERT_TRUE(fuse_statis.count("fc_fuse"));
  LOG(INFO) << "num_ops: " << num_ops;
  if (FLAGS_ernie_large) {
    ASSERT_EQ(fuse_statis.at("fc_fuse"), 146);
    EXPECT_EQ(num_ops, 859);
  } else {
    ASSERT_EQ(fuse_statis.at("fc_fuse"), 74);
    EXPECT_EQ(num_ops, 295);
  }
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  std::vector<std::vector<PaddleTensor>> inputs;
  LoadInputData(&inputs);

  AnalysisConfig cfg;
  SetConfig(&cfg, use_mkldnn, false);

  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), inputs);
}

TEST(Analyzer_ernie, compare) { compare(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_ernie, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

// Compare Deterministic result
TEST(Analyzer_Ernie, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  LoadInputData(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

// Compare results
TEST(Analyzer_Ernie, compare_results) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  LoadInputData(&input_slots_all);

  std::ifstream fin(FLAGS_refer_result);
  std::string line;
  std::vector<float> ref;

  while (std::getline(fin, line)) {
    Split(line, ' ', &ref);
  }

  auto predictor = CreateTestPredictor(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
      FLAGS_use_analysis);

  std::vector<PaddleTensor> outputs;
  for (size_t i = 0; i < input_slots_all.size(); i++) {
    outputs.clear();
    predictor->Run(input_slots_all[i], &outputs);
    auto outputs_size = outputs.front().data.length() / (sizeof(float));
    for (size_t j = 0; j < outputs_size; ++j) {
      EXPECT_NEAR(ref[i * outputs_size + j],
                  static_cast<float *>(outputs[0].data.data())[j],
                  FLAGS_accuracy);
    }
  }
}

#ifdef PADDLE_WITH_IPU
// IPU: Compare Deterministic result
TEST(Analyzer_Ernie_ipu, ipu_compare_determine) {
  AnalysisConfig cfg;
  SetIpuConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  LoadInputData(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

// IPU: Compare results
TEST(Analyzer_Ernie_ipu, ipu_compare_results) {
  AnalysisConfig cfg;
  SetIpuConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  LoadInputData(&input_slots_all);

  std::ifstream fin(FLAGS_refer_result);
  std::string line;
  std::vector<float> ref;

  while (std::getline(fin, line)) {
    Split(line, ' ', &ref);
  }

  auto predictor = CreateTestPredictor(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
      FLAGS_use_analysis);

  std::vector<PaddleTensor> outputs;
  for (size_t i = 0; i < input_slots_all.size(); i++) {
    outputs.clear();
    predictor->Run(input_slots_all[i], &outputs);
    auto outputs_size = outputs.front().data.length() / (sizeof(float));
    for (size_t j = 0; j < outputs_size; ++j) {
      EXPECT_NEAR(ref[i * outputs_size + j],
                  static_cast<float *>(outputs[0].data.data())[j],
                  FLAGS_accuracy);
    }
  }
}
#endif

}  // namespace inference
}  // namespace paddle
