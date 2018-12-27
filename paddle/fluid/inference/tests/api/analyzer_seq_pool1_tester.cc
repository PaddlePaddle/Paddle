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
  cfg->SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  std::vector<std::string> feed_names = {
      "slot10000_embed", "slot10001_embed", "slot10004_embed",
      "slot10005_embed", "slot10008_embed", "slot10009_embed",
      "slot10012_embed", "slot10013_embed", "slot10108_embed",
      "slot13324_embed", "slot13325_embed", "slot13326_embed",
      "slot13327_embed", "slot13328_embed", "slot13329_embed",
      "slot13330_embed", "slot13331_embed", "slot15501_embed",
      "slot15502_embed", "slot15503_embed", "slot15504_embed",
      "slot15505_embed", "slot15506_embed", "slot15507_embed",
      "slot15508_embed", "slot15516_embed", "slot15519_embed",
      "slot15523_embed", "slot15531_embed", "slot15533_embed",
      "slot15548_embed", "slot15564_embed", "slot15565_embed",
      "slot15566_embed", "slot15570_embed", "slot15571_embed",
      "slot15572_embed", "slot15573_embed", "slot15574_embed",
      "slot15575_embed", "slot15576_embed", "slot15577_embed",
      "slot15579_embed", "slot15581_embed", "slot15582_embed",
      "slot15583_embed", "slot15584_embed", "slot5016_embed",
      "slot5021_embed",  "slot6002_embed",  "slot6003_embed",
      "slot6004_embed",  "slot6005_embed",  "slot6006_embed",
      "slot6007_embed",  "slot6008_embed",  "slot6009_embed",
      "slot6011_embed",  "slot6014_embed",  "slot6015_embed",
      "slot6023_embed",  "slot6024_embed",  "slot6025_embed",
      "slot6027_embed",  "slot6029_embed",  "slot6031_embed",
      "slot6034_embed",  "slot6035_embed",  "slot6036_embed",
      "slot6037_embed",  "slot6039_embed",  "slot6048_embed",
      "slot6050_embed",  "slot6058_embed",  "slot6059_embed",
      "slot6060_embed",  "slot6066_embed",  "slot6067_embed",
      "slot6068_embed",  "slot6069_embed",  "slot6070_embed",
      "slot6071_embed",  "slot6072_embed",  "slot6073_embed",
      "slot6182_embed",  "slot6183_embed",  "slot6184_embed",
      "slot6185_embed",  "slot6186_embed",  "slot6188_embed",
      "slot6189_embed",  "slot6190_embed",  "slot6201_embed",
      "slot6202_embed",  "slot6203_embed",  "slot6247_embed",
      "slot6248_embed",  "slot6250_embed",  "slot6251_embed",
      "slot6807_embed",  "slot6808_embed",  "slot6809_embed",
      "slot6810_embed",  "slot6811_embed",  "slot6812_embed",
      "slot6813_embed",  "slot6814_embed",  "slot6815_embed",
      "slot6816_embed",  "slot6817_embed",  "slot6818_embed",
      "slot6819_embed",  "slot6820_embed",  "slot6822_embed",
      "slot6823_embed",  "slot6826_embed",  "slot7002_embed",
      "slot7003_embed",  "slot7004_embed",  "slot7005_embed",
      "slot7006_embed",  "slot7008_embed",  "slot7009_embed",
      "slot7010_embed",  "slot7011_embed",  "slot7013_embed",
      "slot7014_embed",  "slot7015_embed",  "slot7016_embed",
      "slot7017_embed",  "slot7019_embed",  "slot7100_embed",
      "slot7506_embed",  "slot7507_embed",  "slot7514_embed",
      "slot7515_embed",  "slot7516_embed"};
  SetFakeImageInput(inputs, FLAGS_infer_model, true, "model", "params",
                    &feed_names);
}

// Easy for profiling independently.
void profile(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  if (use_mkldnn) {
    cfg.EnableMKLDNN();
  }
  std::vector<PaddleTensor> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, FLAGS_num_threads);
}

TEST(Analyzer_seq_pool1, profile) { profile(); }

// Check the fuse status
TEST(Analyzer_seq_pool1, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  LOG(INFO) << "num_ops: " << num_ops;
  EXPECT_EQ(num_ops, 314);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
