/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <numeric>

#include "paddle_inference_api.h"  //NOLINT

DEFINE_string(modeldir, "", "Directory of the inference model.");

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;

void run(Predictor *predictor,
         const std::vector<float> &input,
         const std::vector<int> &input_shape,
         std::vector<float> *out_data) {
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->CopyFromCpu(input.data());

  CHECK(predictor->Run());

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  out_data->resize(out_num);
  output_t->CopyToCpu(out_data->data());
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  Config config;
  config.EnableUseGpu(100, 0);
  config.SetModel(FLAGS_modeldir + "/custom_relu.pdmodel",
                  FLAGS_modeldir + "/custom_relu.pdiparams");
  config.EnableNewExecutor(true);
  config.EnableNewIR(true);
  auto predictor = CreatePredictor(config);
  std::vector<int> input_shape = {1, 1, 28, 28};
  std::vector<float> input_data(1 * 1 * 28 * 28, 1);
  std::vector<float> out_data;
  run(predictor.get(), input_data, input_shape, &out_data);
  for (auto e : out_data) {
    LOG(INFO) << e << '\n';
  }
  return 0;
}
