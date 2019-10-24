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

#include <fstream>
#include <ios>

#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"

#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/platform/enforce.h"

int main() {
  LOG(INFO) << "leaky_relu";
  paddle::AnalysisConfig config;
  config.SetModel("/shixiaowei02/Paddle_lite/xingzhaolong/leaky_relu_model");
  // config.SetModel("/Paddle/models/lite/leaky_relu");
  config.SwitchUseFeedFetchOps(false);
  config.EnableUseGpu(10, 0);
  config.EnableLiteEngine(paddle::AnalysisConfig::Precision::kFloat32);
  config.pass_builder()->TurnOnDebug();

  auto predictor = CreatePaddlePredictor(config);
  PADDLE_ENFORCE_NOT_NULL(predictor.get());

  const int batch_size = 1;
  const int channels = 1;
  const int height = 3;
  const int width = 3;
  // float *data = new float[batch_size * channels * height * width];
  float data[batch_size * channels * height * width] = {0.5, -0.5, 0,  -0, 1,
                                                        -1,  2,    -2, 3};

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->copy_from_cpu(data);

  CHECK(predictor->ZeroCopyRun());

  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
  LOG(INFO) << "out_num is " << out_num;
  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());
  return 0;

  /*
    // for yolov3
    LOG(INFO) << "yolo_v3";
    paddle::AnalysisConfig config;
    config.SetModel("/Paddle/models/lite/yolov3_infer/__model__",
    "/Paddle/models/lite/yolov3_infer/__params__");
    config.SwitchUseFeedFetchOps(false);
    config.EnableUseGpu(10, 1);
    config.EnableLiteEngine(paddle::AnalysisConfig::Precision::kFloat32);
    config.pass_builder()->TurnOnDebug();

    auto predictor = CreatePaddlePredictor(config);
    PADDLE_ENFORCE_NOT_NULL(predictor.get());

    const int batch_size = 1;
    const int channels = 3;
    const int height = 608;
    const int width = 608;
    // float *data = new float[batch_size * channels * height * width];
    float data[batch_size * channels * height * width];
    memset(data, 0, sizeof(float) * batch_size * channels * height * width);

    auto input_names = predictor->GetInputNames();
    LOG(INFO) << input_names[0];
    LOG(INFO) << input_names[1];
    auto input_image = predictor->GetInputTensor(input_names[0]);
    input_image->Reshape({batch_size, channels, height, width});
    input_image->copy_from_cpu(data);

    int im_size_data[2] = {608, 608};
    auto input_size = predictor->GetInputTensor(input_names[1]);
    input_size->Reshape({1, 2});
    input_size->copy_from_cpu(im_size_data);

    CHECK(predictor->ZeroCopyRun());

    std::vector<float> out_data;
    auto output_names = predictor->GetOutputNames();
    auto output_t = predictor->GetOutputTensor(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    LOG(INFO) << "out_num is " << out_num;
    out_data.resize(out_num);
    output_t->copy_to_cpu(out_data.data());
    return 0;
  */
}
