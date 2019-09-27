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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {

const char* GetModelPath(std::string a) { return a.c_str(); }

/*TEST(TensorRT_mobilenet, compare) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  compare(model_dir, true);
  // Open it when need.
  // profile(model_dir, true, FLAGS_use_tensorrt);
}*/

TEST(PD_AnalysisPredictor, compare) {
  // std::string a = FLAGS_infer_model;
  LOG(INFO) << FLAGS_infer_model;
  const char* model_dir = GetModelPath(FLAGS_infer_model + "/mobilenet");
  // const char* model_dir = GetModelPath(
  //     "/paddle/Paddle/build/third_party/inference_demo/trt_tests_models/"
  //     "trt_inference_test_models/mobilenet/");
  PD_AnalysisConfig* config = PD_NewAnalysisConfig();
  config = PD_SetModel(config, model_dir);
  config = PD_DisableGpu(config);
  // PD_SetCpuMathLibraryNumThreads(config, 10);
  config = PD_SwitchUseFeedFetchOps(config, false);
  // PD_SwitchSpecifyInputNames(config, true);
  // PD_SwitchIrDebug(config, true);
  LOG(INFO) << "before here! ";

  const int batch_size = 1;
  const int channels = 3;
  const int height = 224;
  const int width = 224;
  float input[batch_size * channels * height * width] = {0};

  int shape[4] = {batch_size, channels, height, width};
  // float* out;
  // int* out_size;
  // PD_PredictorZeroCopyRun(config, input, batch_size * channels * height *
  // width,
  //                         &out, &out_size, shape, 4);
  int shape_size = 4;
  auto predictor = CreatePaddlePredictor(config->config);
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  std::vector<int> tensor_shape;
  tensor_shape.assign(shape, shape + shape_size);
  input_t->Reshape(tensor_shape);
  input_t->copy_from_cpu(input);
  CHECK(predictor->ZeroCopyRun());

  // PD_Predictor* predictor = PD_NewPredictor(config);
  /*auto predictor = CreatePaddlePredictor(config->config);

  int* size;
  char** input_names = PD_GetPredictorInputNames(predictor, &size);
  LOG(INFO) << input_names[0];
  PD_DataType data_type = PD_FLOAT32;
  PD_ZeroCopyTensor* tensor =
      PD_GetPredictorInputTensor(predictor, input_names[0]);
  PD_ZeroCopyTensorReshape(tensor, shape, 4);
  PD_ZeroCopyFromCpu(tensor, input, data_type);
  CHECK(PD_PredictorZeroCopyRun(predictor));*/

  /*PD_Tensor* ten = PD_NewPaddleTensor();
  ten->tensor = inputs_all[0][0];
  PD_Tensor* out = PD_NewPaddleTensor();
  int* outsize;
  int insize = 1;
  PD_PredictorRun(predictor, ten, insize, out, &outsize, 1);*/

  /*std::vector<PaddleTensor> outputs;
  for (auto& input : inputs_all) {
    ASSERT_TRUE(predictor->Run(input, &outputs));
  }*/
}

}  // namespace inference
}  // namespace paddle
