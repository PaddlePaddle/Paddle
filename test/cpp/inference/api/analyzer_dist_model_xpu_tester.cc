// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {

TEST(test_dist_model_xpu, dist_model_xpu) {
  std::cout << "Analysis Predictor DistModel XPU test." << std::endl;
  AnalysisConfig config;
  config.SetModel(FLAGS_infer_model + "/__model__",
                  FLAGS_infer_model + "/__params__");
  config.EnableXpu();
  config.SetXpuDeviceId(0);

  auto predictor = paddle_infer::CreatePredictor(config);
  int batch_size = 1;
  int channels = 1;
  int height = 48;
  int width = 512;
  int nums = batch_size * channels * height * width;
  std::cout << "Created predictor." << std::endl;

  float* input = new float[nums];
  for (int i = 0; i < nums; ++i) input[i] = 0;
  auto input_names = predictor->GetInputNames();

  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->CopyFromCpu(input);
  std::cout << "Input data." << std::endl;

  predictor->Run();
  std::cout << "Zero Copy Run." << std::endl;

  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());
  std::cout << "Output data." << std::endl;
  delete[] input;
}

}  // namespace inference
}  // namespace paddle
