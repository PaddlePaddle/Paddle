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

TEST(PD_AnalysisConfig, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();

  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  int num_thread = PD_CpuMathLibraryNumThreads(config);
  PADDLE_ENFORCE_EQ(
      10,
      num_thread,
      common::errors::InvalidArgument("The num of thread should be"
                                      "equal to 10, but got %d.",
                                      num_thread));
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  PD_SetModel(config, model_dir.c_str(), nullptr);
  PD_SetOptimCacheDir(config, (FLAGS_infer_model + "/OptimCacheDir").c_str());
  const char *model_dir_ = PD_ModelDir(config);
  LOG(INFO) << model_dir_;
  PD_EnableUseGpu(config, 100, 0);
  bool use_gpu = PD_UseGpu(config);
  PADDLE_ENFORCE_EQ(use_gpu,
                    true,
                    common::errors::InvalidArgument(
                        "GPU is not enabled. "
                        "The configuration indicates that GPU should be used, "
                        "but it is currently disabled. "
                        "Please check your configuration settings and ensure "
                        "that GPU is properly enabled."));
  int device = PD_GpuDeviceId(config);
  PADDLE_ENFORCE_EQ(device,
                    0,
                    common::errors::InvalidArgument(
                        "The device ID is incorrect. "
                        "Expected device ID is 0, but received %d. "
                        "Please check your device configuration and "
                        "ensure the correct device ID is used.",
                        device));
  int init_size = PD_MemoryPoolInitSizeMb(config);
  PADDLE_ENFORCE_EQ(init_size,
                    100,
                    common::errors::InvalidArgument(
                        "The initial size of the memory pool is incorrect. "
                        "Expected size is 100 MB, but received %d MB. "
                        "Please check your configuration settings and ensure "
                        "the correct memory pool size is set.",
                        init_size));
  float frac = PD_FractionOfGpuMemoryForPool(config);
  LOG(INFO) << frac;
  PD_EnableCUDNN(config);
  bool cudnn = PD_CudnnEnabled(config);
  PADDLE_ENFORCE_EQ(cudnn,
                    true,
                    common::errors::InvalidArgument(
                        "cuDNN is not enabled. "
                        "The configuration indicates that cuDNN should be "
                        "enabled, but it is currently disabled. "
                        "Please check your configuration settings and ensure "
                        "that cuDNN is properly enabled."));
  PD_SwitchIrOptim(config, true);
  bool ir_optim = PD_IrOptim(config);
  PADDLE_ENFORCE_EQ(ir_optim,
                    true,
                    common::errors::InvalidArgument(
                        "IR optimization is not enabled. "
                        "The configuration indicates that IR optimization "
                        "should be enabled, but it is currently disabled. "
                        "Please check your configuration settings and ensure "
                        "that IR optimization is properly enabled."));
  PD_EnableTensorRtEngine(
      config, 1 << 20, 1, 3, Precision::kFloat32, false, false);
  bool trt_enable = PD_TensorrtEngineEnabled(config);
  PADDLE_ENFORCE_EQ(trt_enable,
                    true,
                    common::errors::InvalidArgument(
                        "TensorRT engine is not enabled. "
                        "The configuration indicates that TensorRT engine "
                        "should be enabled, but it is currently disabled. "
                        "Please check your configuration settings and ensure "
                        "that TensorRT engine is properly enabled."));
  PD_EnableMemoryOptim(config);
  bool memory_optim_enable = PD_MemoryOptimEnabled(config);
  PADDLE_ENFORCE_EQ(memory_optim_enable,
                    true,
                    common::errors::InvalidArgument(
                        "Memory optimization is not enabled. "
                        "The configuration indicates that memory optimization "
                        "should be enabled, but it is currently disabled. "
                        "Please check your configuration settings and ensure "
                        "that memory optimization is properly enabled."));
  PD_EnableProfile(config);
  bool profiler_enable = PD_ProfileEnabled(config);
  PADDLE_ENFORCE_EQ(profiler_enable,
                    true,
                    common::errors::InvalidArgument(
                        "Profiler is not enabled. "
                        "The configuration indicates that the profiler should "
                        "be enabled, but it is currently disabled. "
                        "Please check your configuration settings and ensure "
                        "that the profiler is properly enabled."));
  PD_SetInValid(config);
  bool is_valid = PD_IsValid(config);
  PADDLE_ENFORCE_EQ(
      is_valid,
      true,
      common::errors::InvalidArgument(
          "Configuration is not valid. "
          "The configuration should be valid, but it is currently invalid. "
          "Please check your configuration settings and ensure they are "
          "correct."));
  PD_DeleteAnalysisConfig(config);
}

TEST(PD_AnalysisConfig, trt_int8) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_EnableUseGpu(config, 100, 0);
  PD_EnableTensorRtEngine(config, 1 << 20, 1, 3, Precision::kInt8, false, true);
  bool trt_enable = PD_TensorrtEngineEnabled(config);
  PADDLE_ENFORCE_EQ(trt_enable,
                    true,
                    common::errors::InvalidArgument(
                        "TensorRT engine is not enabled. "
                        "The configuration indicates that TensorRT engine "
                        "should be enabled, but it is currently disabled. "
                        "Please check your configuration settings and ensure "
                        "that TensorRT engine is properly enabled."));
  PD_DeleteAnalysisConfig(config);
}

TEST(PD_AnalysisConfig, trt_fp16) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_EnableUseGpu(config, 100, 0);
  PD_EnableTensorRtEngine(
      config, 1 << 20, 1, 3, Precision::kHalf, false, false);
  bool trt_enable = PD_TensorrtEngineEnabled(config);
  PADDLE_ENFORCE_EQ(trt_enable,
                    true,
                    common::errors::InvalidArgument(
                        "TensorRT engine is not enabled. "
                        "The configuration indicates that TensorRT engine "
                        "should be enabled, but it is currently disabled. "
                        "Please check your configuration settings and ensure "
                        "that TensorRT engine is properly enabled."));
  PD_Predictor *predictor = PD_NewPredictor(config);
  PD_DeletePredictor(predictor);
  PD_DeleteAnalysisConfig(config);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
