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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/contrib/inference/paddle_inference_api.h"

DEFINE_string(model, "", "Directory of the inference model.");

namespace paddle {

AnakinConfig GetConfig() {
  AnakinConfig config;
  config.model_file = FLAGS_model;
  config.device = 0;
  config.max_batch_size = 1;
  return config;
}

TEST(inference, anakin) {
  AnakinConfig config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<AnakinConfig, PaddleEngineKind::kAnakin>(config);

  float data[1 * 3 * 224 * 224] = {1.0f};

  PaddleTensor tensor{.name = "input_0",
                      .shape = std::vector<int>({1, 3, 224, 224}),
                      .data = PaddleBuf(data, sizeof(data)),
                      .dtype = PaddleDType::FLOAT32};

  // For simplicity, we set all the slots with the same data.
  std::vector<PaddleTensor> paddle_tensor_feeds;
  paddle_tensor_feeds.emplace_back(std::move(tensor));

  PaddleTensor tensor_out{.name = "prob_out",
                          .shape = std::vector<int>({1000, 1}),
                          .data = PaddleBuf(),
                          .dtype = PaddleDType::FLOAT32};

  std::vector<PaddleTensor> outputs;
  outputs.emplace_back(std::move(tensor_out));

  ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));

  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < 1000; ++j) {
    LOG(INFO) << "output[" << j << "]: " << data_o[j];
  }
}

}  // namespace paddle
