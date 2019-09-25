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

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void zero_copy_run() {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig config;
  PD_DisableGpu(&config);
  PD_SetCpuMathLibraryNumThreads(&config, 10);
  PD_SwitchUseFeedFetchOps(&config, false);
  PD_SwitchSpecifyInputNames(&config, true);
  PD_SwitchIrDebug(&config, true);
  PD_SetModel(&config, model_dir.c_str());  //, params_file1.c_str());
  bool use_feed_fetch = PD_UseFeedFetchOpsEnabled(&config);
  CHECK(!use_feed_fetch) << "NO";
  bool specify_input_names = PD_SpecifyInputName(&config);
  CHECK(specify_input_names) << "NO";

  const int batch_size = 1;
  const int channels = 3;
  const int height = 224;
  const int width = 224;
  float input[batch_size * channels * height * width] = {0};

  int shape[4] = {batch_size, channels, height, width};
  int shape_size = 4;

  int in_size = 1;
  int *out_size;
  PD_ZeroCopyData *inputs = new PD_ZeroCopyData;
  PD_ZeroCopyData *outputs = new PD_ZeroCopyData;
  inputs->data = static_cast<void *>(input);
  inputs->dtype = PD_FLOAT32;
  inputs->name = new char[2];
  inputs->name[0] = 'x';
  inputs->name[1] = '\0';
  LOG(INFO) << inputs->name;
  inputs->shape = shape;
  inputs->shape_size = shape_size;

  PD_PredictorZeroCopyRun(&config, inputs, in_size, &outputs, &out_size);
}

TEST(PD_ZeroCopyRun, zero_copy_run) { zero_copy_run(); }

TEST(PD_AnalysisConfig, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  int num_thread = PD_CpuMathLibraryNumThreads(config);
  CHECK(10 == num_thread) << "NO";
  PD_SwitchUseFeedFetchOps(config, false);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  PD_SetModel(config, model_dir.c_str());
  PD_SetOptimCacheDir(config, (FLAGS_infer_model + "/OptimCacheDir").c_str());
  const char *model_dir_ = PD_ModelDir(config);
  LOG(INFO) << model_dir_;
  PD_EnableUseGpu(config, 100, 0);
  bool use_gpu = PD_UseGpu(config);
  CHECK(use_gpu) << "NO";
  int device = PD_GpuDeviceId(config);
  CHECK(0 == device) << "NO";
  int init_size = PD_MemoryPoolInitSizeMb(config);
  CHECK(100 == init_size) << "NO";
  float frac = PD_FractionOfGpuMemoryForPool(config);
  LOG(INFO) << frac;
  PD_EnableCUDNN(config);
  bool cudnn = PD_CudnnEnabled(config);
  CHECK(cudnn) << "NO";
  PD_SwitchIrOptim(config, true);
  bool ir_optim = PD_IrOptim(config);
  CHECK(ir_optim) << "NO";
  PD_EnableTensorRtEngine(config);
  bool trt_enable = PD_TensorrtEngineEnabled(config);
  CHECK(trt_enable) << "NO";
  PD_EnableNgraph(config);
  bool ngraph_enable = PD_NgraphEnabled(config);
  LOG(INFO) << ngraph_enable << " Ngraph";
  PD_EnableMemoryOptim(config);
  bool memory_optim_enable = PD_MemoryOptimEnabled(config);
  CHECK(memory_optim_enable) << "NO";
  PD_EnableProfile(config);
  bool profiler_enable = PD_ProfileEnabled(config);
  CHECK(profiler_enable) << "NO";
  PD_SetInValid(config);
  bool is_valid = PD_IsValid(config);
  CHECK(!is_valid) << "NO";
  PD_DeleteAnalysisConfig(config);
}

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
