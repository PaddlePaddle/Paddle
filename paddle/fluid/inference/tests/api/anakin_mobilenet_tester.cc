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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "gflags/gflags.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(model, "", "Directory of the inference model(mobile_v2).");

namespace paddle {

contrib::AnakinConfig GetConfig() {
  contrib::AnakinConfig config;
  // using AnakinConfig::X86 if you need to use cpu to do inference
  config.target_type = contrib::AnakinConfig::NVGPU;
  config.model_file = FLAGS_model;
  config.device_id = 0;
  config.init_batch_size = 1;
  return config;
}

TEST(inference, anakin) {
  auto config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<contrib::AnakinConfig, PaddleEngineKind::kAnakin>(
          config);

  float data[1 * 3 * 224 * 224] = {1.0f};
  PaddleTensor tensor;
  tensor.name = "input_0";
  tensor.shape = std::vector<int>({1, 3, 224, 224});
  tensor.data = PaddleBuf(data, sizeof(data));
  tensor.dtype = PaddleDType::FLOAT32;

  // For simplicity, we set all the slots with the same data.
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

  PaddleTensor tensor_out;
  tensor_out.name = "prob_out";
  tensor_out.shape = std::vector<int>({});
  tensor_out.data = PaddleBuf();
  tensor_out.dtype = PaddleDType::FLOAT32;

  std::vector<PaddleTensor> outputs(1, tensor_out);

  ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));

  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length(); ++j) {
    LOG(INFO) << "output[" << j << "]: " << data_o[j];
  }
}

}  // namespace paddle
