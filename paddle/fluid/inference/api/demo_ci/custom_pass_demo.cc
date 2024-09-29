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
#include <cmath>
#include <memory>
#include <numeric>

#include "paddle/extension.h"
#include "paddle_inference_api.h"  //NOLINT

DEFINE_string(modeldir, "", "Directory of the inference model.");

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;

std::shared_ptr<Predictor> InitPredictor(bool use_custom_pass) {
  Config config;
  config.EnableUseGpu(100, 0);
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  config.EnableNewExecutor(true);
  config.EnableNewIR(true);
  // config.SwitchIrDebug(true);
  if (use_custom_pass) {
    config.EnableCustomPasses({"relu_replace_pass"});
  }

  return CreatePredictor(config);
}

std::vector<float> GetOutputData(const std::shared_ptr<Predictor> &predictor) {
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
  PADDLE_ENFORCE_EQ(
      predictor->Run(inputs, &outputs),
      true,
      common::errors::ExecutionTimeout("Sorry, predictor run failed"));
  PADDLE_ENFORCE_EQ(outputs[0].place(),
                    paddle::GPUPlace{},
                    common::errors::InvalidArgument(
                        "Sorry, output tensor place is not GPUPlace"));
  PADDLE_ENFORCE_EQ(outputs[0].dtype(),
                    paddle::DataType::FLOAT32,
                    common::errors::InvalidArgument(
                        "Sorry, output tensor dtype is not FLOAT32"));
  auto output = outputs[0].copy_to(paddle::CPUPlace{}, true);

  std::vector<float> output_data;
  for (int64_t i = 0; i < output.numel(); i++) {
    output_data.push_back(output.data<float>()[i]);
  }
  return output_data;
}

bool AreEqual(const std::vector<float> &vec1,
              const std::vector<float> &vec2,
              float epsilon) {
  if (vec1.size() != vec2.size()) {
    return false;
  }
  for (size_t i = 0; i < vec1.size(); ++i) {
    if (std::fabs(vec1[i] - vec2[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  auto base_data = GetOutputData(InitPredictor(false));
  auto custom_data = GetOutputData(InitPredictor(true));

  PADDLE_ENFORCE_EQ(AreEqual(base_data, custom_data, 1e-3),
                    true,
                    common::errors::InvalidArgument(
                        "Sorry, base_data and custom_data are not equal"));
  return 0;
}
