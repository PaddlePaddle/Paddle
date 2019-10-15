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
#include <fstream>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

template <typename T>
void zero_copy_run() {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  PD_SwitchUseFeedFetchOps(config, false);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  PD_SetModel(config, model_dir.c_str());  //, params_file1.c_str());
  bool use_feed_fetch = PD_UseFeedFetchOpsEnabled(config);
  CHECK(!use_feed_fetch) << "NO";
  bool specify_input_names = PD_SpecifyInputName(config);
  CHECK(specify_input_names) << "NO";

  const int batch_size = 1;
  const int channels = 3;
  const int height = 224;
  const int width = 224;
  T input[batch_size * channels * height * width] = {0};

  int shape[4] = {batch_size, channels, height, width};
  int shape_size = 4;
  int in_size = 1;
  int *out_size;
  PD_ZeroCopyData *inputs = new PD_ZeroCopyData;
  PD_ZeroCopyData *outputs = new PD_ZeroCopyData;
  inputs->data = static_cast<void *>(input);
  std::string nm = typeid(T).name();
  if ("f" == nm) {
    inputs->dtype = PD_FLOAT32;
  } else if ("i" == nm) {
    inputs->dtype = PD_INT32;
  } else if ("x" == nm) {
    inputs->dtype = PD_INT64;
  } else if ("h" == nm) {
    inputs->dtype = PD_UINT8;
  } else {
    CHECK(false) << "Unsupport dtype. ";
  }
  inputs->name = new char[2];
  inputs->name[0] = 'x';
  inputs->name[1] = '\0';
  LOG(INFO) << inputs->name;
  inputs->shape = shape;
  inputs->shape_size = shape_size;

  PD_PredictorZeroCopyRun(config, inputs, in_size, &outputs, &out_size);
}

TEST(PD_ZeroCopyRun, zero_copy_run) { zero_copy_run<float>(); }

#ifdef PADDLE_WITH_MKLDNN
TEST(PD_AnalysisConfig, profile_mkldnn) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  PD_SwitchUseFeedFetchOps(config, false);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  PD_EnableMKLDNN(config);
  bool mkldnn_enable = PD_MkldnnEnabled(config);
  CHECK(mkldnn_enable) << "NO";
  PD_EnableMkldnnQuantizer(config);
  bool quantizer_enable = PD_MkldnnQuantizerEnabled(config);
  CHECK(quantizer_enable) << "NO";
  PD_SetMkldnnCacheCapacity(config, 0);
  PD_SetModel(config, model_dir.c_str());
  PD_EnableAnakinEngine(config);
  bool anakin_enable = PD_AnakinEngineEnabled(config);
  LOG(INFO) << anakin_enable;
  PD_DeleteAnalysisConfig(config);
}
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
