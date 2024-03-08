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

#include "paddle/extension.h"
#include "paddle_inference_api.h"  //NOLINT

DEFINE_string(modeldir, "", "Directory of the inference model.");

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  Config config;
  config.EnableUseGpu(100, 0);
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  config.EnableNewExecutor(true);
  config.EnableNewIR(true);
  config.SwitchIrDebug(true);
  auto predictor = CreatePredictor(config);

  auto input_names = predictor->GetInputNames();
  auto input_shapes = predictor->GetInputTensorShape();

  for (const auto &input_name : input_names) {
    // update input shape's batch size
    input_shapes[input_name][0] = 1;
  }

  std::vector<paddle::Tensor> inputs, outputs;
  for (const auto &input_name : input_names) {
    auto input_tensor = paddle::full(input_shapes[input_name],
                                     0.5,
                                     paddle::DataType::FLOAT32,
                                     paddle::GPUPlace{});
    input_tensor.set_name(input_name);
    inputs.emplace_back(std::move(input_tensor));
  }
  CHECK(predictor->Run(inputs, &outputs));

  return 0;
}
