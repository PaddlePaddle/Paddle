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
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

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
  PD_SetModel(config, model_dir.c_str(), nullptr);
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
  PD_EnableTensorRtEngine(config, 1 << 20, 1, 3, Precision::kFloat32, false,
                          false);
  bool trt_enable = PD_TensorrtEngineEnabled(config);
  CHECK(trt_enable) << "NO";
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

TEST(PD_AnalysisConfig, trt_int8) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_EnableUseGpu(config, 100, 0);
  PD_EnableTensorRtEngine(config, 1 << 20, 1, 3, Precision::kInt8, false, true);
  bool trt_enable = PD_TensorrtEngineEnabled(config);
  CHECK(trt_enable) << "NO";
  PD_DeleteAnalysisConfig(config);
}

TEST(PD_AnalysisConfig, trt_fp16) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_EnableUseGpu(config, 100, 0);
  PD_EnableTensorRtEngine(config, 1 << 20, 1, 3, Precision::kHalf, false,
                          false);
  bool trt_enable = PD_TensorrtEngineEnabled(config);
  CHECK(trt_enable) << "NO";
  PD_Predictor *predictor = PD_NewPredictor(config);
  PD_DeletePredictor(predictor);
  PD_DeleteAnalysisConfig(config);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
