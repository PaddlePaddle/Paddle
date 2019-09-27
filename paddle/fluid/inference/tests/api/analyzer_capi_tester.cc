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

#include "paddle/fluid/inference/capi/c_api.h"
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

TEST(PD_AnalysisPredictor, use_gpu) {
  std::string a = FLAGS_infer_model;
  const char* model_dir = GetModelPath(FLAGS_infer_model + "/mobilenet");
  PD_AnalysisConfig* config = PD_NewAnalysisConfig();
  PD_SetModel(config, model_dir);
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  PD_SwitchUseFeedFetchOps(config, false);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);

  const int batch_size = 1;
  const int channels = 3;
  const int height = 224;
  const int width = 224;
  float input[batch_size * channels * height * width] = {0};
  // std::vector<std::vector<PaddleTensor>> inputs_all;
  // SetFakeImageInput(&inputs_all, model_dir, false, "__model__", "");
  int shape[4] = {batch_size, channels, height, width};
  PD_Predictor* predictor = PD_NewPredictor(config);
  int* size;
  char** input_names = PD_GetPredictorInputNames(predictor, &size);
  PD_ZeroCopyTensor** tensor = new PD_ZeroCopyTensor*[*size];
  PD_DataType data_type = PD_FLOAT32;
  tensor[0] = PD_GetPredictorInputTensor(predictor, input_names[0]);
  PD_ZeroCopyTensorReshape(tensor[0], shape, 4);
  PD_ZeroCopyFromCpu(tensor[0], input, data_type);
  CHECK(PD_PredictorZeroCopyRun(predictor));

  /*std::vector<PaddleTensor> outputs;
  for (auto& input : inputs_all) {
    ASSERT_TRUE(predictor->Run(input, &outputs));
  }*/
}

}  // namespace inference
}  // namespace paddle
