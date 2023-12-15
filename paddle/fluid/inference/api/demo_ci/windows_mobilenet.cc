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

#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "paddle_inference_api.h"  // NOLINT

DEFINE_string(modeldir, "", "Directory of the inference model.");
DEFINE_bool(use_gpu, false, "Whether use gpu.");

namespace paddle {
namespace demo {

void RunAnalysis() {
  // 1. create AnalysisConfig
  AnalysisConfig config;
  if (FLAGS_modeldir.empty()) {
    LOG(INFO) << "Usage: path\\mobilenet --modeldir=path/to/your/model";
    exit(1);
  }

  // CreateConfig(&config);
  if (FLAGS_use_gpu) {
    config.EnableUseGpu(100, 0);
  }
  config.SetModel(FLAGS_modeldir + "/__model__",
                  FLAGS_modeldir + "/__params__");

  // 2. create predictor, prepare input data
  std::unique_ptr<PaddlePredictor> predictor = CreatePaddlePredictor(config);
  int batch_size = 1;
  int channels = 3;
  int height = 300;
  int width = 300;
  int nums = batch_size * channels * height * width;

  float* input = new float[nums];
  for (int i = 0; i < nums; ++i) input[i] = 0;

  // 3. create input tensor, use ZeroCopyTensor
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->copy_from_cpu(input);

  // 4. run predictor
  predictor->ZeroCopyRun();

  // 5. get out put
  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());
  delete[] input;
}

}  // namespace demo
}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  paddle::demo::RunAnalysis();
  std::cout << "=========================Runs successfully===================="
            << std::endl;
  return 0;
}
