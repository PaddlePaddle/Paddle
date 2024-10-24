/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/common/flags.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle_infer {

TEST(Predictor, use_gpu) {
  LOG(INFO) << GetVersion();
  UpdateDllFlag("conv_workspace_size_limit", "4000");
  std::string model_dir = FLAGS_infer_model + "/model";
  Config config;
  config.EnableNewIR(false);
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableUseGpu(100, 0);

  auto predictor = CreatePredictor(config);
  auto pred_clone = predictor->Clone();

  std::vector<int> in_shape = {1, 3, 318, 318};
  int in_num =
      std::accumulate(in_shape.begin(), in_shape.end(), 1, [](int &a, int &b) {
        return a * b;
      });

  std::vector<float> input(in_num, 0);

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);

  input_t->Reshape(in_shape);
  input_t->CopyFromCpu(input.data());
  predictor->Run();

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  std::vector<float> out_data;
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());
  predictor->ClearIntermediateTensor();
}

TEST(PredictorPool, basic) {
  LOG(INFO) << GetVersion();
  UpdateDllFlag("conv_workspace_size_limit", "4000");
  std::string model_dir = FLAGS_infer_model + "/model";
  Config config;
  config.EnableNewIR(false);
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableUseGpu(100, 0);

  services::PredictorPool pred_pool(config, 4);
  auto pred = pred_pool.Retrieve(2);

  std::vector<int> in_shape = {1, 3, 318, 318};
  int in_num =
      std::accumulate(in_shape.begin(), in_shape.end(), 1, [](int &a, int &b) {
        return a * b;
      });
  std::vector<float> input(in_num, 0);

  auto in_names = pred->GetInputNames();
  auto input_t = pred->GetInputHandle(in_names[0]);
  input_t->name();
  input_t->Reshape(in_shape);
  input_t->CopyFromCpu(input.data());
  pred->Run();
  auto out_names = pred->GetOutputNames();
  auto output_t = pred->GetOutputHandle(out_names[0]);
  auto out_type = output_t->type();
  LOG(INFO) << GetNumBytesOfDataType(out_type);
  if (out_type == DataType::FLOAT32) {
    PlaceType place;
    int size;
    output_t->data<float>(&place, &size);
  }
}

}  // namespace paddle_infer
