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
#include "paddle/contrib/inference/paddle_inference_api.h"
//#include "paddle/fluid/inference/tests/test_helper.h"

//DEFINE_string(dirname, "", "Directory of the inference model (path/to/models/).");

namespace paddle {

AnakinConfig GetConfig() {
  AnakinConfig config;
  config.model_file = /*FLAGS_dirname +*/ "./mobilenet_v2.anakin.bin";
  config.device = 0;
  config.max_batch_size = 1;
  return config;
}

TEST(inference, anakin) {
  AnakinConfig config = GetConfig();
  auto predictor = CreatePaddlePredictor<AnakinConfig, PaddleEngineKind::kAnakin>(config);

  float data[1*3*224*224] = {1.0f};

  PaddleBuf buf{.data = data, .length = sizeof(data)};
  PaddleTensor tensor{.name = "input_0",
                      .shape = std::vector<int>({1, 3, 224, 224}),
                      .data = buf,
                      .dtype = PaddleDType::FLOAT32};

  // For simplicity, we set all the slots with the same data.
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);


  float data_out[1000];

  PaddleBuf buf_out{.data = data_out, .length = sizeof(data)};
  PaddleTensor tensor_out{.name = "prob_out",
                          .shape = std::vector<int>({1000,1}),
                          .data = buf_out,
                          .dtype = PaddleDType::FLOAT32};

  std::vector<PaddleTensor> outputs(1, tensor_out);

  ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));

  float* data_o = static_cast<float*>(outputs[0].data.data);
  for (size_t j = 0; j < 1000; ++j) {
    LOG(INFO) << "output[" << j << "]: " << data_o[j];
  }
}

}  // namespace paddle
