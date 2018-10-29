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
  PADDLE_ENFORCE_EQ(FLAGS_test_all_data, 0, "Only have single batch of data.");

  PaddleTensor input;
  // channel=3, height/width=318
  std::vector<int> shape({FLAGS_batch_size, 3, 318, 318});
  input.shape = shape;
  input.dtype = PaddleDType::FLOAT32;

  // fill input data, for profile easily, do not use random data here.
  size_t size = FLAGS_batch_size * 3 * 318 * 318;
  input.data.Resize(size * sizeof(float));
  float *input_data = static_cast<float *>(input.data.data());
  for (size_t i = 0; i < size; i++) {
    *(input_data + i) = static_cast<float>(i) / size;
  }

  std::vector<PaddleTensor> input_slots;
  input_slots.assign({input});
  (*inputs).emplace_back(input_slots);
}

// Easy for profiling independently.
void profile(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  cfg._use_mkldnn = use_mkldnn;
  std::vector<PaddleTensor> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(cfg, input_slots_all, &outputs, FLAGS_num_threads);

  if (FLAGS_num_threads == 1 && !FLAGS_test_all_data) {
    PADDLE_ENFORCE_EQ(outputs.size(), 1UL);
    size_t size = GetSize(outputs[0]);
    // output is a 512-dimension feature
    EXPECT_EQ(size, 512 * FLAGS_batch_size);
  }
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
  ASSERT_TRUE(fuse_statis.count("fc_fuse"));
  EXPECT_EQ(fuse_statis.at("fc_fuse"), 1);
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  cfg._use_mkldnn = use_mkldnn;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(cfg, input_slots_all);
}

TEST(Analyzer_resnet50, compare) { compare(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_resnet50, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
