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

#include "paddle/contrib/inference/paddle_inference_api.h"
#include <gtest/gtest.h>

DEFINE_string(dirname, "", "Directory of the inference model.");

namespace paddle {

AnakinConfig GetConfig() {
  AnakinConfig config;
  config.model_file = FLAGS_dirname + "xxx.anakin.bin";
  LOG(INFO) << "dirname  " << config.model_file;
  config.device = 0;
  config.max_batch_size = 1;
  return config;
}

TEST(inference, anakin) {
  AnakinConfig config;

  auto predictor =
      CreatePaddlePredictor<AnakinConfig, PaddleEngineKind::kAnakin>(config);

  int batch_size = 2;
  bool repeat = false;

  AnakinConfig config = GetConfig();

  const bool is_combined = false;
  std::vector<std::vector<int64_t>> feed_target_shapes =
      GetFeedTargetShapes(config.model_dir, is_combined);

  framework::LoDTensor input;
  // Use normilized image pixels as input data,
  // which should be in the range [0.0, 1.0].
  feed_target_shapes[0][0] = batch_size;
  framework::DDim input_dims = framework::make_ddim(feed_target_shapes[0]);
  SetupTensor<float>(
      &input, input_dims, static_cast<float>(0), static_cast<float>(1));
  std::vector<framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&input);

  framework::LoDTensor output1;
  std::vector<framework::LoDTensor*> cpu_fetchs1;
  cpu_fetchs1.push_back(&output1);

  TestInference<platform::CPUPlace, false, true>(config.model_dir,
                                                 cpu_feeds,
                                                 cpu_fetchs1,
                                                 repeat,
                                                 is_combined,
                                                 use_mkldnn);

  std::vector<PaddleTensor> paddle_tensor_feeds;
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&input));

  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));
  ASSERT_EQ(outputs.size(), 1UL);
  size_t len = outputs[0].data.length;
  float* data = static_cast<float*>(outputs[0].data.data);
  float* lod_data = output1.data<float>();
  for (size_t j = 0; j < len / sizeof(float); ++j) {
    EXPECT_NEAR(lod_data[j], data[j], 1e-3);
  }
  free(data);
}

}  // namespace paddle
