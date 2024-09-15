// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {

TEST(test_zerocopy_tensor, zerocopy_tensor) {
  AnalysisConfig config;
  config.SetModel(FLAGS_infer_model + "/inference.pdmodel",
                  FLAGS_infer_model + "/inference.pdiparams");

  auto predictor = CreatePaddlePredictor(config);
  int batch_size = 1;
  int channels = 3;
  int height = 224;
  int width = 224;
  int nums = batch_size * channels * height * width;

  float* input = new float[nums];
  for (int i = 0; i < nums; ++i) input[i] = 0;
  auto input_names = predictor->GetInputNames();
  PaddlePlace p = PaddlePlace::kCPU;
  PaddlePlace* place = &p;
  int size;

  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->copy_from_cpu<float>(input);
  input_t->data<float>(place, &size);
  input_t->mutable_data<float>(p);

  predictor->ZeroCopyRun();

  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  out_data.resize(out_num);
  output_t->copy_to_cpu<float>(out_data.data());
}

}  // namespace inference
}  // namespace paddle
