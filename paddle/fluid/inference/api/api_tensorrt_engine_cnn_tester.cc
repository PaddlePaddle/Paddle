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
#include <numeric>
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {
DEFINE_string(dirname, "", "Directory of the inference model.");

TensorRTConfig GetConfig() {
  TensorRTConfig config;
  config.prog_file = FLAGS_dirname + "/__model__";
  config.param_file = FLAGS_dirname + "/__param__";
  config.use_gpu = true;
  config.fraction_of_gpu_memory = 0.15;
  config.device = 0;
  // Specific the variable's name of each input.
  config.specify_input_name = true;
  config.max_batch_size = 1;
  return config;
}

bool test_map_cnn(int batch_size, int repeat) {
  TensorRTConfig config = GetConfig();
  config.max_batch_size = batch_size;
  std::vector<std::string> input_names(
      {"t_feature_f", "realtime_feature_f", "static_feature_f", "eta_cost_f",
       "lukuang_f", "length_f", "path_f", "speed_f", "lane_f"});
  std::vector<std::vector<int>> shapes({{batch_size, 4},
                                        {batch_size, 1, 50, 12},
                                        {batch_size, 1, 50, 19},
                                        {batch_size, 1, 50, 1},
                                        {batch_size, 4, 50, 1},
                                        {batch_size, 1, 50, 1},
                                        {batch_size, 5, 50, 1},
                                        {batch_size, 7, 50, 1},
                                        {batch_size, 3, 50, 1}});
  float* input_data[shapes.size()];

  int index = 0;
  for (auto& shape : shapes) {
    size_t data_size = (accumulate(shape.begin(), shape.end(), 1,
                                   [](int a, int b) { return a * b; }));
    input_data[index] = new float[data_size];
    memset(input_data[index], 100, sizeof(float) * data_size);
    index += 1;
  }

  std::vector<PaddleTensor> inputs;
  LOG(INFO) << "inputs  ";

  index = 0;
  for (auto& shape : shapes) {
    // For simplicity, max_batch as the batch_size
    // shape.insert(shape.begin(), max_batch);
    // shape.insert(shape.begin(), 1);
    PaddleTensor feature;
    feature.name = input_names[index];
    feature.shape = shape;
    // feature.lod = std::vector<std::vector<size_t>>();
    size_t data_size =
        (sizeof(float) * accumulate(shape.begin(), shape.end(), 1,
                                    [](int a, int b) { return a * b; }));

    feature.data = PaddleBuf(static_cast<void*>(input_data[index]), data_size);
    feature.dtype = PaddleDType::FLOAT32;
    inputs.emplace_back(feature);
    index += 1;
  }
  LOG(INFO) << "predcit:  ";

  // warm-up
  auto predictor =
      CreatePaddlePredictor<TensorRTConfig,
                            PaddleEngineKind::kAutoMixedTensorRT>(config);
  // { batch begin
  std::vector<PaddleTensor> outputs;
  CHECK(predictor->Run(inputs, &outputs, batch_size));

  for (int i = 0; i < repeat; i++) {
    CHECK(predictor->Run(inputs, &outputs, batch_size));
  }

  float* data_o = static_cast<float*>(outputs[0].data.data());

  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); ++j) {
    LOG(INFO) << "output[" << j << "]: " << data_o[j];
    if (fabs(data_o[j] - 0.592781) >= 1e-5) return false;
  }
  return true;
}

TEST(map_cnn, tensorrt) {
  for (int i = 0; i < 1; i++) {
    ASSERT_TRUE(test_map_cnn(1 << i, 1));
  }
}
}  // namespace paddle
