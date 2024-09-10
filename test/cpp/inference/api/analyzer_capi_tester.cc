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
  std::string prog_file = model_dir + "/model";
  std::string params_file = model_dir + "/params";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  PD_SetModel(config, prog_file.c_str(), params_file.c_str());
  bool use_feed_fetch = PD_UseFeedFetchOpsEnabled(config);
  EXPECT_FALSE(use_feed_fetch);
  bool specify_input_names = PD_SpecifyInputName(config);
  EXPECT_TRUE(specify_input_names);

  const int batch_size = 1;
  const int channels = 3;
  const int height = 318;
  const int width = 318;
  float *input = new float[batch_size * channels * height * width]();

  int shape[4] = {batch_size, channels, height, width};
  int shape_size = 4;
  int in_size = 1;
  int out_size;
  PD_ZeroCopyData *inputs = new PD_ZeroCopyData;
  PD_ZeroCopyData *outputs = new PD_ZeroCopyData;
  inputs->data = static_cast<void *>(input);
  inputs->dtype = PD_FLOAT32;
  inputs->name = new char[5];
  inputs->name[0] = 'd';
  inputs->name[1] = 'a';
  inputs->name[2] = 't';
  inputs->name[3] = 'a';
  inputs->name[4] = '\0';
  inputs->shape = shape;
  inputs->shape_size = shape_size;

  PD_PredictorZeroCopyRun(config, inputs, in_size, &outputs, &out_size);

  delete[] input;
  delete[] inputs;
  delete[] outputs;
}

TEST(PD_PredictorZeroCopyRun, zero_copy_run) { zero_copy_run(); }

#ifdef PADDLE_WITH_DNNL
TEST(PD_AnalysisConfig, profile_mkldnn) {
  std::string model_dir = FLAGS_infer_model;
  std::string prog_file = model_dir + "/model";
  std::string params_file = model_dir + "/params";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  PD_EnableMKLDNN(config);
  bool mkldnn_enable = PD_MkldnnEnabled(config);
  EXPECT_TRUE(mkldnn_enable);
  PD_EnableMkldnnBfloat16(config);
  PD_SetMkldnnCacheCapacity(config, 0);
  PD_SetModel(config, prog_file.c_str(), params_file.c_str());
  PD_DeleteAnalysisConfig(config);
}
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
