// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <gtest/gtest.h>
#include <cmath>
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/timer.h"

namespace paddle {

DEFINE_string(model, "", "Directory of the inference model and data.");

bool Main(int batch_size, int repeat) {
  AnakinConfig config;
  config.model_file = FLAGS_model;
  config.device = 0;
  config.max_batch_size = batch_size;
  config.target_type = AnakinConfig::NVGPU;
  std::string line;
  std::vector<PaddleTensor> inputs;
  std::vector<std::vector<int>> shapes({{batch_size, 4, 1, 1},
                                        {batch_size, 1, 50, 12},
                                        {batch_size, 1, 50, 19},
                                        {batch_size, 1, 50, 1},
                                        {batch_size, 4, 50, 1},
                                        {batch_size, 1, 50, 1},
                                        {batch_size, 5, 50, 1},
                                        {batch_size, 7, 50, 1},
                                        {batch_size, 3, 50, 1}});

  int id = 0;
  int ids[] = {8, 0, 1, 2, 3, 4, 5, 6, 7};

  float* input_data[shapes.size()];

  int index = 0;
  for (auto& shape : shapes) {
    size_t data_size = (accumulate(shape.begin(), shape.end(), 1,
                                   [](int a, int b) { return a * b; }));
    input_data[index] = new float[data_size * batch_size];
    memset(input_data[index], 100, sizeof(float) * data_size);
    index += 1;
  }

  index = 0;
  for (auto& shape : shapes) {
    PaddleTensor feature;
    feature.name = "input_" + std::to_string(ids[id++]);
    feature.shape = shape;
    feature.data = PaddleBuf(
        static_cast<void*>(input_data[index]),
        sizeof(float) * std::accumulate(shape.begin(), shape.end(), 1,
                                        [](int a, int b) { return a * b; }));
    feature.dtype = PaddleDType::FLOAT32;
    inputs.emplace_back(std::move(feature));
    CHECK_EQ(inputs.back().shape.size(), 4UL);
    index += 1;
  }
  auto predictor =
      CreatePaddlePredictor<AnakinConfig, PaddleEngineKind::kAnakin>(config);

  PaddleTensor tensor_out;
  tensor_out.name = "outnet_con1.tmp_1_gout";
  tensor_out.shape = std::vector<int>({});
  tensor_out.data = PaddleBuf();
  tensor_out.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> outputs(1, tensor_out);

  CHECK(predictor->Run(inputs, &outputs));

  paddle::inference::Timer timer;
  timer.tic();
  for (int i = 0; i < repeat; i++) {
    CHECK(predictor->Run(inputs, &outputs));
  }
  paddle::inference::PrintTime(batch_size, repeat, 1, 0, timer.toc() / repeat);

  for (auto& tensor : outputs) {
    LOG(INFO) << "output.length: " << tensor.data.length();
    float* data = static_cast<float*>(tensor.data.data());
    for (int i = 0; i < std::min(100UL, tensor.data.length() / sizeof(float));
         i++) {
      LOG(INFO) << data[i];
      if (fabs(data[i] - 0.592781) > 1e-5) return false;
    }
  }
  return true;
}
TEST(anakin_cnn_test, map_cnn_model) {
  for (int i = 0; i < 1; i++) {
    ASSERT_TRUE(Main(1 << i, 1));
  }
}
}  // namespace paddle
