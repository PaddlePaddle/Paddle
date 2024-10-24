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

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "paddle/fluid/inference/capi/paddle_c_api.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void zero_copy_run() {
  std::string model_dir = FLAGS_infer_model;
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  PD_SetModel(config, model_dir.c_str(), nullptr);
  bool use_feed_fetch = PD_UseFeedFetchOpsEnabled(config);
  PADDLE_ENFORCE_EQ(
      use_feed_fetch, false, common::errors::PreconditionNotMet("NO"));
  bool specify_input_names = PD_SpecifyInputName(config);
  PADDLE_ENFORCE_EQ(
      specify_input_names, true, common::errors::PreconditionNotMet("NO"));
  const int batch_size = 1;
  const int channels = 3;
  const int height = 224;
  const int width = 224;
  float input[batch_size * channels * height * width] = {0};
  int shape[4] = {batch_size, channels, height, width};
  int shape_size = 4;
  int in_size = 2;
  int out_size;
  PD_ZeroCopyData *inputs = new PD_ZeroCopyData[2];
  PD_ZeroCopyData *outputs = nullptr;
  inputs[0].data = static_cast<void *>(input);
  inputs[0].dtype = PD_FLOAT32;
  inputs[0].name = new char[6];
  inputs[0].name[0] = 'i';
  inputs[0].name[1] = 'm';
  inputs[0].name[2] = 'a';
  inputs[0].name[3] = 'g';
  inputs[0].name[4] = 'e';
  inputs[0].name[5] = '\0';
  inputs[0].shape = shape;
  inputs[0].shape_size = shape_size;

  int *label = new int[1];
  label[0] = 0;
  inputs[1].data = static_cast<void *>(label);
  inputs[1].dtype = PD_INT64;
  inputs[1].name = new char[6];
  inputs[1].name[0] = 'l';
  inputs[1].name[1] = 'a';
  inputs[1].name[2] = 'b';
  inputs[1].name[3] = 'e';
  inputs[1].name[4] = 'l';
  inputs[1].name[5] = '\0';
  int label_shape[2] = {1, 1};
  int label_shape_size = 2;
  inputs[1].shape = label_shape;
  inputs[1].shape_size = label_shape_size;

  PD_PredictorZeroCopyRun(config, inputs, in_size, &outputs, &out_size);

  LOG(INFO) << "output size is: " << out_size;
  LOG(INFO) << outputs[0].name;
  for (int j = 0; j < out_size; ++j) {
    LOG(INFO) << "output[" << j
              << "]'s shape_size is: " << outputs[j].shape_size;
    for (int i = 0; i < outputs[0].shape_size; ++i) {
      LOG(INFO) << "output[" << j << "]'s shape is: " << outputs[j].shape[i];
    }
    LOG(INFO) << "output[" << j
              << "]'s DATA is: " << *(static_cast<float *>(outputs[j].data));
  }
  delete[] outputs;
  delete[] inputs;
}

#ifdef PADDLE_WITH_DNNL
TEST(PD_ZeroCopyRun, zero_copy_run) { zero_copy_run(); }
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
