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

#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstring>
#include <numeric>

#include "paddle/fluid/inference/tests/api/trt_test_helper.h"

namespace paddle_infer {

TEST(Predictor, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/model";
  Config config;
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableUseGpu(100, 0);
  auto predictor = CreatePredictor(config);
  auto pred2 = predictor->Clone();
  std::vector<int> in_shape = {1, 3, 318, 318};
  int in_num = std::accumulate(in_shape.begin(), in_shape.end(), 1,
                               [](int &a, int &b) { return a * b; });

  std::vector<float> input(in_num, 0);

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);

  input_t->Reshape(in_shape);
  input_t->CopyFromCpu(input.data());
  predictor->Run();

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  std::vector<float> out_data;
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());
  predictor->ClearIntermediateTensor();

  auto in_names2 = pred2->GetInputNames();
  auto input_t2 = pred2->GetInputHandle(in_names2[0]);
  input_t2->name();
  input_t2->Reshape(in_shape);
  auto *input_t2_tensor_data = input_t2->mutable_data<float>(PaddlePlace::kGPU);

  cudaMemcpy(input_t2_tensor_data, input.data(), in_num * sizeof(float),
             cudaMemcpyHostToDevice);
  input_t2->CopyFromCpu(input.data());
  pred2->Run();
  auto out_names2 = pred2->GetOutputNames();
  auto output_t2 = pred2->GetOutputHandle(out_names2[0]);
  auto out2_type = output_t2->type();
  if (out2_type == PaddleDType::FLOAT32) {
    PaddlePlace place;
    int size;
    output_t2->data<float>(&place, &size);
  }
}
}  // namespace paddle_infer
