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
// #include "paddle/fluid/inference/capi/c_api_internal.h"
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

TEST(Analysis_capi, compare) {
  std::string a = FLAGS_infer_model;
  const char* model_dir =
      GetModelPath(FLAGS_infer_model + "/mobilenet/__model__");
  const char* params_file =
      GetModelPath(FLAGS_infer_model + "/mobilenet/__params__");
  LOG(INFO) << model_dir;
  PD_AnalysisConfig* config = PD_NewAnalysisConfig();
  PD_SetModel(config, model_dir, params_file);
  LOG(INFO) << PD_ModelDir(config);
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  PD_SwitchUseFeedFetchOps(config, false);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  LOG(INFO) << "before here! ";

  const int batch_size = 1;
  const int channels = 3;
  const int height = 224;
  const int width = 224;
  float input[batch_size * channels * height * width] = {0};

  int shape[4] = {batch_size, channels, height, width};

  AnalysisConfig c;
  c.SetModel(model_dir, params_file);
  LOG(INFO) << c.model_dir();
  c.DisableGpu();
  c.SwitchUseFeedFetchOps(false);
  int shape_size = 4;
  auto predictor = CreatePaddlePredictor(c);
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  std::vector<int> tensor_shape;
  tensor_shape.assign(shape, shape + shape_size);
  input_t->Reshape(tensor_shape);
  input_t->copy_from_cpu(input);
  CHECK(predictor->ZeroCopyRun());
}

}  // namespace inference
}  // namespace paddle
