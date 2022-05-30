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

/*
 * This file contains a simple demo for how to take a model for inference with
 * IPUs.
 * Model: wget -q
 * http://paddle-inference-dist.bj.bcebos.com/word2vec.inference.model.tar.gz
 */

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(infer_model, "", "Directory of the inference model.");

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

void inference(std::string model_path, bool use_ipu,
               std::vector<float> *out_data) {
  //# 1. Create Predictor with a config.
  Config config;
  config.SetModel(FLAGS_infer_model);
  if (use_ipu) {
    // ipu_device_num, ipu_micro_batch_size
    config.EnableIpu(1, 4);
  }
  auto predictor = CreatePredictor(config);

  //# 2. Prepare input/output tensor.
  auto input_names = predictor->GetInputNames();
  std::vector<int64_t> data{1, 2, 3, 4};
  // For simplicity, we set all the slots with the same data.
  for (auto input_name : input_names) {
    auto input_tensor = predictor->GetInputHandle(input_name);
    input_tensor->Reshape({4, 1});
    input_tensor->CopyFromCpu(data.data());
  }

  //# 3. Run
  predictor->Run();

  //# 4. Get output.
  auto output_names = predictor->GetOutputNames();
  auto output_tensor = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_tensor->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
  out_data->resize(out_num);
  output_tensor->CopyToCpu(out_data->data());
}

int main(int argc, char *argv[]) {
  ::GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<float> ipu_result;
  std::vector<float> cpu_result;
  inference(FLAGS_infer_model, true, &ipu_result);
  inference(FLAGS_infer_model, false, &cpu_result);
  for (size_t i = 0; i < ipu_result.size(); i++) {
    CHECK_NEAR(ipu_result[i], cpu_result[i], 1e-6);
  }
  LOG(INFO) << "Finished";
}
