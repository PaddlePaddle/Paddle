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


DEFINE_string(dirname, "", "Directory of the inference model (path/to/models/).");
DEFINE_string(model_name, "", "Name of the inference model (xxx.anakin.bin).");

namespace paddle {

AnakinConfig GetConfig() {
  AnakinConfig config;
  config.model_file = FLAGS_dirname + FLAGS_model_name;
  config.device = 0;
  config.max_batch_size = 1;
  return config;
}

TEST(inference, anakin) {
  AnakinConfig config = GetConfig();
  auto predictor = CreatePaddlePredictor<AnakinConfig, PaddleEngineKind::kAnakin>(config);


  std::vector<PaddleTensor> paddle_tensor_feeds;

  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));
  ASSERT_EQ(outputs.size(), 1UL);

  size_t len = outputs[0].data.length;
  float* data = static_cast<float*>(outputs[0].data.data);
  for (size_t j = 0; j < len / sizeof(float); ++j) {
    LOG(INFO) << "output[" << j << "]: " << data[j];
  }
  free(data);
}

}  // namespace paddle
